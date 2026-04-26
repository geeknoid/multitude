//! `ArenaInner<A>` — the heap-allocated state shared between an [`Arena`]
//! and its outstanding chunks.
//!
//! `ArenaInner` owns the per-arena configuration, the chunk cache, and
//! the live-stats counters. Each surviving chunk holds a back-pointer
//! to the `ArenaInner` so it can register itself in the cache when its
//! refcount drops to zero (via [`crate::chunk_header::teardown_chunk`]).
//!
//! **Lifetime**: `ArenaInner` is heap-allocated and may outlive the
//! parent `Arena` handle. When the parent `Arena` is dropped, it sets
//! [`ArenaInner::arena_dropped`]. The actual `ArenaInner` storage is
//! freed only when the last surviving chunk drops AND the parent has
//! been dropped — see [`ArenaInner::register_chunk_freed`]. This avoids
//! a use-after-free when handles outlive the arena.

use core::cell::Cell;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use alloc::boxed::Box;
use allocator_api2::alloc::Allocator;

use crate::chunk_header::ChunkHeader;
use crate::stats::StatsStorage;

pub(crate) struct ArenaInner<A: Allocator + Clone> {
    /// Current `Local`-flavor normal chunk used for bump allocation.
    pub(crate) current_local: Cell<Option<NonNull<ChunkHeader<A>>>>,
    /// Current `Shared`-flavor normal chunk used for bump allocation.
    pub(crate) current_shared: Cell<Option<NonNull<ChunkHeader<A>>>>,
    /// Head of the singly-linked free-list cache (each cached chunk's
    /// `bump` field is reused as the next-pointer).
    pub(crate) chunk_cache_head: Cell<Option<NonNull<ChunkHeader<A>>>>,
    /// Number of chunks currently in the cache.
    pub(crate) chunk_cache_len: Cell<usize>,

    // ---- per-arena configuration (set by ArenaBuilder, immutable after) -
    /// Cap on `chunk_cache_len`.
    pub(crate) chunk_cache_capacity: usize,
    /// Per-arena nominal chunk size, in bytes. Power of 2,
    /// `MIN_CHUNK_SIZE..=CHUNK_ALIGN`.
    pub(crate) chunk_size: usize,
    /// Per-arena threshold (in worst-case bytes) above which an
    /// allocation is routed to its own oversized chunk.
    pub(crate) max_normal_alloc: usize,
    /// Optional lifetime fuel cap on total chunk bytes ever allocated
    /// via this arena. `None` means unlimited.
    pub(crate) byte_budget: Option<usize>,

    // ---- per-arena runtime accounting -----------------------------------
    /// Running sum of every chunk's `total_size` ever allocated through
    /// this arena. Compared against `byte_budget` before each chunk
    /// allocation.
    pub(crate) bytes_used_against_budget: Cell<usize>,
    /// Lifetime statistics; per-field `Cell<u64>` so each counter bump
    /// only touches 8 bytes (vs 48 bytes for a `Cell<ArenaStats>`).
    pub(crate) stats: StatsStorage,

    // ---- handles-outlive-arena bookkeeping (cross-thread atomics) -------
    /// Number of chunk allocations whose backing memory has not yet been
    /// freed. Incremented in `try_alloc_fresh_chunk`; decremented in
    /// [`crate::chunk_header::free_chunk`] right before deallocation.
    /// Cached chunks count as "outstanding"; they're decremented when
    /// the cache is drained or when they're popped+freed.
    pub(crate) outstanding_chunks: AtomicUsize,
    /// `true` once the parent `Arena` has been dropped. Read by
    /// `teardown_chunk` to skip cache push (the cache is gone) and to
    /// decide whether to free `ArenaInner`'s own storage.
    pub(crate) arena_dropped: AtomicBool,

    /// The backing allocator. Cloned into each chunk.
    pub(crate) allocator: A,
}

impl<A: Allocator + Clone> ArenaInner<A> {
    /// Construct a fresh inner state with the given configuration.
    pub(crate) fn new_with_config(
        allocator: A,
        chunk_size: usize,
        max_normal_alloc: usize,
        byte_budget: Option<usize>,
        chunk_cache_capacity: usize,
    ) -> Self {
        Self {
            current_local: Cell::new(None),
            current_shared: Cell::new(None),
            chunk_cache_head: Cell::new(None),
            chunk_cache_len: Cell::new(0),
            chunk_cache_capacity,
            chunk_size,
            max_normal_alloc,
            byte_budget,
            bytes_used_against_budget: Cell::new(0),
            stats: StatsStorage::default(),
            outstanding_chunks: AtomicUsize::new(0),
            arena_dropped: AtomicBool::new(false),
            allocator,
        }
    }

    /// Try to push `chunk` (which has refcount==0 and an empty drop list)
    /// onto the cache. Returns `true` if cached, `false` if the cache is
    /// full (caller should free).
    ///
    /// # Safety
    ///
    /// `chunk` must be empty (refcount==0, drop list walked) and must be
    /// a `Normal`-class chunk that this arena created. Caller must be on
    /// the arena's owner thread.
    pub(crate) unsafe fn try_push_to_cache(&self, chunk: NonNull<ChunkHeader<A>>) -> bool {
        let len = self.chunk_cache_len.get();
        if len >= self.chunk_cache_capacity {
            return false;
        }
        // Reuse the chunk's `bump` cell as the cache next-pointer.
        let next = self.chunk_cache_head.get();
        // SAFETY: caller owns the chunk; no other access in flight.
        unsafe {
            let next_addr = next.map_or(0_usize, |p| p.as_ptr() as usize);
            chunk.as_ref().bump.set(next_addr);
        }
        self.chunk_cache_head.set(Some(chunk));
        self.chunk_cache_len.set(len + 1);
        true
    }

    /// Pop a chunk from the cache. Returns `None` if the cache is empty.
    ///
    /// The returned chunk is in an undefined state — caller must
    /// re-initialize it via [`crate::chunk_header::revive_cached_chunk`]
    /// or free it directly.
    pub(crate) fn try_pop_cache(&self) -> Option<NonNull<ChunkHeader<A>>> {
        let head = self.chunk_cache_head.get()?;
        // SAFETY: head was previously installed by us with bump as
        // next-pointer.
        let next_addr = unsafe { head.as_ref().bump.get() };
        let next = if next_addr == 0 {
            None
        } else {
            // SAFETY: a non-zero next_addr came from a NonNull cached
            // chunk pointer stored on push.
            Some(unsafe { NonNull::new_unchecked(next_addr as *mut ChunkHeader<A>) })
        };
        self.chunk_cache_head.set(next);
        self.chunk_cache_len.set(self.chunk_cache_len.get() - 1);
        Some(head)
    }

    // ---- counter helpers (bump individual stats fields directly via
    // ----                   `self.stats.<field>.set(self.stats.<field>.get() + n)`)

    /// Note that one chunk's backing memory has been freed back to the
    /// underlying allocator.
    ///
    /// Returns `true` iff the parent `Arena` has been dropped AND this
    /// was the last outstanding chunk; in that case the caller is
    /// responsible for freeing `ArenaInner`'s own storage (via
    /// [`Self::free_storage`] on the pointer they captured before this
    /// call).
    #[inline]
    pub(crate) fn register_chunk_freed(&self) -> bool {
        let prev = self.outstanding_chunks.fetch_sub(1, Ordering::Release);
        debug_assert!(prev > 0, "outstanding_chunks underflow");
        prev == 1 && self.arena_dropped.load(Ordering::Acquire)
    }

    /// Free this `ArenaInner` storage via `Box::from_raw`. Called by
    /// `Arena::drop` (when no chunks survive) and by the last surviving
    /// chunk's free path (when the arena dropped first).
    ///
    /// # Safety
    ///
    /// `inner` must be a pointer originally returned by
    /// `Box::leak(Box::new(ArenaInner::new_with_config(...)))` and must
    /// not have been freed previously. There must be no outstanding
    /// references to the `ArenaInner` (no chunks holding back-pointers).
    pub(crate) unsafe fn free_storage(inner: NonNull<Self>) {
        // SAFETY: contract documented on the function.
        unsafe {
            let _ = Box::from_raw(inner.as_ptr());
        }
    }
}

// SAFETY: `ArenaInner` contains `Cell<...>` fields that are `!Sync`.
// We unsafe-impl `Sync` here because all such fields are only accessed
// on the parent `Arena`'s owner thread (the `Arena` itself is `!Send`,
// so its lifetime is owner-thread-pinned). Cross-thread accesses (from
// chunk teardown on a non-owner thread for `Shared`-flavor chunks)
// only touch the atomic fields (`outstanding_chunks`, `arena_dropped`).
unsafe impl<A: Allocator + Clone + Send> Send for ArenaInner<A> {}
// SAFETY: see above.
unsafe impl<A: Allocator + Clone + Sync> Sync for ArenaInner<A> {}

// Note: `ArenaInner` deliberately has NO `Drop` impl. Cleanup is owned
// by `Arena::drop`, which then conditionally frees `ArenaInner`'s
// storage via `free_storage`. The reason: a `Drop` impl on `ArenaInner`
// would create a Stacked-Borrows `Unique` retag covering the entire
// `ArenaInner` for the duration of the drop, which conflicts with
// chunk-back-pointer accesses from `teardown_chunk`. By doing cleanup
// via `&self` inside `Arena::drop`, only the necessary
// `SharedReadWrite` retag is in play and chunks can freely deref their
// back-pointer.
