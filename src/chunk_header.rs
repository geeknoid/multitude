//! `ChunkHeader<A>` — per-chunk metadata stored at the chunk's base.
//!
//! This is the heart of the implementation. It owns:
//! - The atomic refcount and the bump cursor.
//! - The doubly-linked drop list head.
//! - The clone of the backing allocator that will free the chunk.
//!
//! The header lives at offset 0 of every chunk, so the address-mask trick
//! (`p & !0xFFFF`) recovers it from any pointer in the chunk.

use core::alloc::Layout;
use core::cell::Cell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering, fence};

use allocator_api2::alloc::Allocator;

use crate::arena_inner::ArenaInner;
use crate::chunk_sharing::ChunkSharing;
use crate::chunk_size_class::ChunkSizeClass;
use crate::constants::{CHUNK_ALIGN, MAX_INITIAL_ALIGN, align_up, checked_align_up};
use crate::drop_entry::DropEntry;

/// Per-chunk metadata, stored at the very start of every chunk allocation.
///
/// `#[repr(C)]` with a deliberate field order — see PLAN.md §4.19.1.
#[repr(C)]
pub(crate) struct ChunkHeader<A: Allocator + Clone> {
    /// Live handle references into this chunk (plus the arena's own
    /// reference while it's the current chunk). Accessed with cheap
    /// `Relaxed` load+store pairs for `Local` chunks and with full
    /// Acquire/Release atomic RMWs for `Shared` chunks.
    pub(crate) ref_count: AtomicUsize,
    /// Current bump offset, measured from the chunk's base address (NOT
    /// from the start of usable space). Touched only by the owner thread,
    /// never after retirement.
    ///
    /// When the chunk sits in the free-list cache, this field is reused
    /// as the next-pointer (see `ArenaInner::cache_next` accessors).
    pub(crate) bump: Cell<usize>,
    /// Head of the intrusive doubly-linked drop list. Owner-thread-only
    /// writes; read once at teardown (after Acquire fence on Shared).
    pub(crate) drop_head: Cell<Option<NonNull<DropEntry>>>,
    /// Total chunk size, in bytes. For a normal chunk this is the
    /// arena's configured `chunk_size`; for an oversized chunk it's the
    /// rounded-up size needed to fit the single oversized allocation.
    pub(crate) total_size: usize,
    /// `Local` (refcount touched only by owner thread) vs `Shared`
    /// (refcount may be touched by any thread). Set at chunk birth and
    /// only re-written when a chunk is pulled from the cache to be
    /// repurposed (owner-thread-only operation), hence `Cell`.
    pub(crate) sharing: Cell<ChunkSharing>,
    /// `Normal` vs `Oversized`. Determines whether this chunk can be
    /// returned to the free-list cache.
    pub(crate) size_class: ChunkSizeClass,
    /// Back-pointer to the owning arena. Only safe to dereference from
    /// the owner thread. Used for free-list reuse on owner-thread frees.
    pub(crate) arena: NonNull<ArenaInner<A>>,
    /// A clone of the backing allocator. Each chunk holds its own clone so
    /// it can free itself even after the arena is dropped. For ZST
    /// allocators (e.g., `Global`) this costs zero bytes.
    pub(crate) allocator: A,
}

// SAFETY: ChunkHeader is only Send/Sync to the extent A is. Shared chunks
// require A: Send + Sync (enforced at the alloc_shared call sites). Local
// chunks never cross threads. So we don't need explicit Send/Sync impls
// here — they fall out of the field types.

impl<A: Allocator + Clone> ChunkHeader<A> {
    /// Compute the padded size of `ChunkHeader<A>` (i.e., the offset at
    /// which the bump cursor starts).
    #[inline]
    #[must_use]
    pub(crate) fn header_padded_size() -> usize {
        align_up(size_of::<Self>(), MAX_INITIAL_ALIGN)
    }

    /// Layout for a normal chunk of the given `chunk_size`. The chunk is
    /// always aligned to [`CHUNK_ALIGN`] regardless of `chunk_size` so
    /// that the address-mask trick continues to work.
    #[inline]
    #[must_use]
    pub(crate) fn normal_layout(chunk_size: usize) -> Layout {
        // SAFETY: CHUNK_ALIGN is a power of 2 and ≤ isize::MAX. The
        // builder validates that chunk_size is a power of 2 in
        // [MIN_CHUNK_SIZE, CHUNK_ALIGN], so size ≤ align here, which is
        // a valid Layout.
        unsafe { Layout::from_size_align_unchecked(chunk_size, CHUNK_ALIGN) }
    }

    /// Layout for an oversized chunk that holds one allocation of
    /// `payload`, optionally preceded by a `DropEntry`. Returns `None` on
    /// arithmetic overflow.
    #[must_use]
    pub(crate) fn oversized_layout(payload: Layout, has_drop: bool) -> Option<Layout> {
        let header = Self::header_padded_size();
        let pre_value = if has_drop { size_of::<DropEntry>() } else { 0 };
        let after_pre = header.checked_add(pre_value)?;
        let value_offset = checked_align_up(after_pre, payload.align())?;
        let needed = value_offset.checked_add(payload.size())?;
        let total = checked_align_up(needed, CHUNK_ALIGN)?;
        Layout::from_size_align(total, CHUNK_ALIGN).ok()
    }

    /// Try to bump-allocate `layout` bytes from this chunk. Returns the
    /// allocated pointer, or `None` if the request doesn't fit.
    ///
    /// Takes `chunk: NonNull<Self>` (rather than `&self`) so the
    /// returned pointer carries the chunk allocation's full provenance,
    /// not the `SharedRO` retag of an `&self` borrow. Writing to the
    /// returned pointer through an `&self`-derived address would be UB
    /// under Stacked Borrows.
    pub(crate) fn try_alloc(chunk: NonNull<Self>, layout: Layout) -> Option<NonNull<u8>> {
        if layout.align() > CHUNK_ALIGN {
            return None;
        }
        // SAFETY: chunk is a valid live ChunkHeader (caller's contract).
        let h = unsafe { chunk.as_ref() };
        let cur = h.bump.get();
        let aligned = checked_align_up(cur, layout.align())?;
        let end = aligned.checked_add(layout.size())?;
        if end > h.total_size {
            return None;
        }
        h.bump.set(end);
        // Use the chunk's full-allocation pointer (preserves Unique
        // provenance through to the returned bytes).
        let base = chunk.as_ptr().cast::<u8>();
        // SAFETY: aligned < total_size, both are byte offsets within the
        // chunk allocation; provenance is the original chunk allocation.
        Some(unsafe { NonNull::new_unchecked(base.add(aligned)) })
    }

    /// Try to bump-allocate a `DropEntry` plus a `T` value. Returns
    /// `(entry_ptr, value_ptr)` or `None` on overflow.
    ///
    /// Takes `chunk: NonNull<Self>` for the same provenance reason as
    /// [`Self::try_alloc`].
    pub(crate) fn try_alloc_with_drop_entry<T>(chunk: NonNull<Self>) -> Option<(NonNull<DropEntry>, NonNull<T>)> {
        // SAFETY: chunk is valid live ChunkHeader.
        let h = unsafe { chunk.as_ref() };
        let cur = h.bump.get();
        let entry_addr = checked_align_up(cur, align_of::<DropEntry>())?;
        let after_entry = entry_addr.checked_add(size_of::<DropEntry>())?;
        let value_addr = checked_align_up(after_entry, align_of::<T>())?;
        let end = value_addr.checked_add(size_of::<T>())?;
        if end > h.total_size {
            return None;
        }
        h.bump.set(end);
        let base = chunk.as_ptr().cast::<u8>();
        // SAFETY: entry_addr and value_addr are byte offsets within the
        // chunk; both are well-aligned; provenance is the original chunk.
        unsafe {
            let entry = NonNull::new_unchecked(base.add(entry_addr).cast::<DropEntry>());
            let value = NonNull::new_unchecked(base.add(value_addr).cast::<T>());
            Some((entry, value))
        }
    }

    /// Increment the chunk refcount.
    #[inline]
    pub(crate) fn inc_ref(&self) {
        match self.sharing.get() {
            ChunkSharing::Local => {
                let v = self.ref_count.load(Ordering::Relaxed);
                self.ref_count.store(v + 1, Ordering::Relaxed);
            }
            ChunkSharing::Shared => {
                let _ = self.ref_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Decrement the chunk refcount. Returns `true` iff the count reached
    /// zero (and the caller should run teardown).
    #[inline]
    pub(crate) fn dec_ref(&self) -> bool {
        let was = match self.sharing.get() {
            ChunkSharing::Local => {
                let v = self.ref_count.load(Ordering::Relaxed);
                self.ref_count.store(v - 1, Ordering::Relaxed);
                v
            }
            ChunkSharing::Shared => self.ref_count.fetch_sub(1, Ordering::Release),
        };
        if was == 1 {
            if matches!(self.sharing.get(), ChunkSharing::Shared) {
                fence(Ordering::Acquire);
            }
            true
        } else {
            false
        }
    }

    /// Read the current refcount.
    #[must_use]
    pub(crate) fn current_ref_count(&self) -> usize {
        match self.sharing.get() {
            ChunkSharing::Local => self.ref_count.load(Ordering::Relaxed),
            ChunkSharing::Shared => self.ref_count.load(Ordering::Acquire),
        }
    }

    /// Push `entry` onto the front of the doubly-linked drop list.
    ///
    /// `slice_len` is `0` for single-value entries; for slice entries it
    /// is the number of elements (so the slice drop shim knows how many
    /// to drop). Initializing it as part of the same atomic struct write
    /// (rather than a separate post-link write) avoids a brief window
    /// in which the entry is observable in the list with stale data.
    ///
    /// # Safety
    ///
    /// `entry` must be a freshly-allocated, uninitialized `DropEntry`
    /// inside this chunk. After this call its `prev` is `None`, `next`
    /// is the previous head, and the previous head's `prev` (if any)
    /// points back at `entry`.
    pub(crate) unsafe fn link_drop_entry(&self, entry: NonNull<DropEntry>, drop_fn: unsafe fn(*mut DropEntry), slice_len: usize) {
        let old_head = self.drop_head.get();
        // SAFETY: caller guarantees the entry slot is writable.
        unsafe {
            entry.as_ptr().write(DropEntry {
                drop_fn,
                prev: None,
                next: old_head,
                slice_len,
            });
        }
        if let Some(old) = old_head {
            // SAFETY: old_head was previously installed by us; it's a valid
            // entry in this chunk and still alive (chunk hasn't been torn
            // down).
            unsafe {
                (*old.as_ptr()).prev = Some(entry);
            }
        }
        self.drop_head.set(Some(entry));
    }

    /// Unlink `entry` from the drop list.
    ///
    /// # Safety
    ///
    /// `entry` must currently be in this chunk's drop list. Used by
    /// [`ArenaBox::drop`](crate::ArenaBox).
    pub(crate) unsafe fn unlink_drop_entry(&self, entry: NonNull<DropEntry>) {
        // SAFETY: caller guarantees entry is in our list.
        unsafe {
            let prev = (*entry.as_ptr()).prev;
            let next = (*entry.as_ptr()).next;
            if let Some(p) = prev {
                (*p.as_ptr()).next = next;
            } else {
                // entry was the head.
                self.drop_head.set(next);
            }
            if let Some(n) = next {
                (*n.as_ptr()).prev = prev;
            }
        }
    }

    /// The byte offset of the bump cursor's initial position (i.e., the
    /// first usable byte after the padded header).
    #[inline]
    #[must_use]
    pub(crate) fn data_start_offset() -> usize {
        Self::header_padded_size()
    }

    /// Reset this chunk's bump cursor and drop list to "fresh" state. Used
    /// when pulling a chunk from the cache.
    pub(crate) fn reset(&self) {
        self.bump.set(Self::data_start_offset());
        self.drop_head.set(None);
        // ref_count will be set to 1 by the caller before publishing.
    }
}

/// Recover the [`ChunkHeader<A>`] address for any pointer allocated out
/// of a 64 KiB-aligned chunk by masking off the low bits.
///
/// Preserves provenance: derives the header pointer from `ptr` via
/// `byte_sub`, not via integer cast round-trip.
///
/// # Safety
///
/// `ptr` must be a pointer returned by an allocation from an
/// `Arena<A>` (so the offset-back address points to a real
/// `ChunkHeader<A>`).
#[inline]
#[must_use]
pub(crate) unsafe fn header_for<T: ?Sized, A: Allocator + Clone>(ptr: NonNull<T>) -> NonNull<ChunkHeader<A>> {
    let raw = ptr.as_ptr().cast::<u8>();
    let offset_within_chunk = (raw as usize) & (CHUNK_ALIGN - 1);
    // SAFETY: The chunk allocation spans [base, base + chunk_size); `raw`
    // lies inside it, and `offset_within_chunk` is `raw - base`. Subtracting
    // the offset returns to the chunk base, which holds a valid
    // `ChunkHeader<A>`.
    unsafe {
        let header = raw.byte_sub(offset_within_chunk).cast::<ChunkHeader<A>>();
        NonNull::new_unchecked(header)
    }
}

/// Initialize a freshly-allocated chunk's header in place and return a
/// pointer to it.
///
/// The chunk's `ref_count` starts at `initial_refcount`. Use:
/// - `1` for **normal** chunks (the parent arena's transient hold; the
///   handle adds another via `inc_ref` and the arena releases its hold
///   when the chunk rotates out of `current_*` or the arena is dropped).
/// - `0` for **oversized** chunks (no arena hold — they don't go in
///   `current_*`; the handle adds the only refcount via `inc_ref`).
///
/// # Safety
///
/// `chunk_ptr` must point to an allocation of at least `total_size` bytes,
/// aligned to `CHUNK_ALIGN`. `arena` must outlive the chunk's owner-side
/// usage.
pub(crate) unsafe fn init_chunk<A: Allocator + Clone>(
    chunk_ptr: NonNull<u8>,
    total_size: usize,
    sharing: ChunkSharing,
    size_class: ChunkSizeClass,
    arena: NonNull<ArenaInner<A>>,
    allocator: A,
    initial_refcount: usize,
) -> NonNull<ChunkHeader<A>> {
    let header_ptr = chunk_ptr.cast::<MaybeUninit<ChunkHeader<A>>>();
    // SAFETY: the chunk allocation is at least `size_of::<ChunkHeader<A>>()`
    // bytes (any chunk's total_size includes the padded header).
    unsafe {
        header_ptr.as_ptr().write(MaybeUninit::new(ChunkHeader {
            ref_count: AtomicUsize::new(initial_refcount),
            bump: Cell::new(ChunkHeader::<A>::data_start_offset()),
            drop_head: Cell::new(None),
            total_size,
            sharing: Cell::new(sharing),
            size_class,
            arena,
            allocator,
        }));
        NonNull::new_unchecked(header_ptr.as_ptr().cast::<ChunkHeader<A>>())
    }
}

/// Tear down a chunk: walk its drop list, then either return it to the
/// owner arena's cache or free its memory back to `A`.
///
/// # Safety
///
/// The caller must have observed `ref_count == 0` (after the appropriate
/// fence on Shared chunks). The chunk must not be touched after this
/// returns.
pub(crate) unsafe fn teardown_chunk<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>, on_owner_thread: bool) {
    // SAFETY: we own the chunk exclusively now (refcount==0).
    unsafe {
        // 1. Walk the drop list (LIFO from drop_head, follow `next`).
        let mut node = (*chunk.as_ptr()).drop_head.get();
        while let Some(entry) = node {
            let next = (*entry.as_ptr()).next;
            ((*entry.as_ptr()).drop_fn)(entry.as_ptr());
            node = next;
        }

        // 2. Cache or free decision. Only `Normal` chunks and owner-thread
        //    frees can be cached. Additionally, if the parent arena has
        //    been dropped, the cache is unreachable (and ArenaInner may
        //    be queued to be freed) — skip the cache attempt.
        let header = chunk.as_ref();
        let can_cache_class =
            matches!(header.size_class, ChunkSizeClass::Normal) && (matches!(header.sharing.get(), ChunkSharing::Local) || on_owner_thread);

        if can_cache_class {
            // SAFETY: chunk's back-pointer was set at init_chunk; the
            // arena_dropped check below ensures we don't read other
            // fields when the parent has been dropped.
            let arena = header.arena.as_ref();
            if !arena.arena_dropped.load(Ordering::Acquire) {
                // SAFETY: we hold the chunk exclusively; on owner thread.
                if arena.try_push_to_cache(chunk) {
                    return;
                }
            }
        }

        // 3. Free via the chunk's own allocator clone.
        free_chunk(chunk);
    }
}

/// Free a chunk's backing memory unconditionally. Decrements the parent
/// `ArenaInner`'s outstanding-chunks counter; if this was the last chunk
/// after the arena was dropped, also frees the `ArenaInner` storage.
///
/// # Safety
///
/// The caller owns the chunk exclusively (refcount is zero, no handles
/// reference it). The chunk's drop list must already have been walked.
pub(crate) unsafe fn free_chunk<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>) {
    // SAFETY: caller owns chunk exclusively.
    unsafe {
        let header_ptr = chunk.as_ptr();
        let total_size = (*header_ptr).total_size;
        // Capture the back-pointer to ArenaInner BEFORE freeing the
        // chunk (after free, the header memory is gone).
        let inner_ptr: NonNull<ArenaInner<A>> = (*header_ptr).arena;
        // Move the allocator out of the chunk before deallocating it.
        let allocator: A = core::ptr::read(&raw const (*header_ptr).allocator);
        // The other fields (`AtomicUsize`, `Cell`, `usize`, enums,
        // `NonNull`) are trivially droppable — no `drop_in_place` needed.
        let layout = Layout::from_size_align_unchecked(total_size, CHUNK_ALIGN);
        allocator.deallocate(chunk.cast::<u8>(), layout);
        // `allocator` drops here.

        // Now that the chunk is gone, decrement the outstanding count.
        // If we were the last chunk and the arena was dropped, free
        // the ArenaInner storage (it was kept alive on our behalf).
        let inner = inner_ptr.as_ref();
        if inner.register_chunk_freed() {
            ArenaInner::free_storage(inner_ptr);
        }
    }
}

/// Update the cache's reuse state on a chunk pulled from the cache:
/// reset bump, `drop_head`, sharing flavor, and refcount=1.
///
/// # Safety
///
/// Caller has just popped this chunk from the free-list and holds it
/// exclusively.
pub(crate) unsafe fn revive_cached_chunk<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>, sharing: ChunkSharing) {
    // SAFETY: caller has exclusive access.
    let h = unsafe { chunk.as_ref() };
    h.reset();
    // Cached chunks may have been Local before; repurpose to the requested
    // sharing via the `Cell`. Both refcount access patterns are valid on a
    // fresh AtomicUsize so the flavor change is sound.
    h.sharing.set(sharing);
    h.ref_count.store(1, Ordering::Relaxed);
}
