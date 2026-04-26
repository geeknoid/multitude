use core::alloc::Layout;
use core::cell::Cell;
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::{AllocError, Allocator, Global};

use crate::arc::Arc;
use crate::arc_str::ArcStr;
use crate::arena_builder::ArenaBuilder;
use crate::arena_inner::ArenaInner;
#[cfg(feature = "stats")]
use crate::arena_stats::{ArenaStats, StatsStorage, bump_stat};
use crate::arena_str_helpers::try_reserve_str_in_chunk;
use crate::r#box::Box;
use crate::box_str::BoxStr;
#[cfg(feature = "builders")]
use crate::builders::String;
#[cfg(feature = "builders")]
use crate::builders::Vec;
#[cfg(feature = "builders")]
use crate::chunk_header::{ChunkHeader, header_for, init_chunk, release_chunk_ref, teardown_chunk};
#[cfg(not(feature = "builders"))]
use crate::chunk_header::{ChunkHeader, init_chunk, release_chunk_ref};
use crate::chunk_ref::ChunkRef;
use crate::chunk_sharing::ChunkSharing;
use crate::chunk_size_class::ChunkSizeClass;
use crate::constants::{CHUNK_ALIGN, LARGE_INITIAL_SHARED_REFCOUNT, MIN_MAX_NORMAL_ALLOC};
use crate::drop_entry::{DropEntry, drop_shim, value_ptr_after_entry};
use crate::entry_layout::worst_case_extra_with_entry;
use crate::owned_chunk::{ChunkSlot, OwnedChunk};
use crate::rc::Rc;
use crate::rc_str::RcStr;

/// A flexible bump allocator.
///
/// Allocates large chunks of memory from an underlying allocator and parcels them out
/// efficiently in response to allocation requests.
///
/// # Configuration
///
/// [`Arena::new`] uses sensible defaults (64 KiB chunks, oversized
/// cutover at 16 KiB, cache up to 8 chunks, no byte budget). For
/// non-default configuration, use
/// [`Arena::builder`].
///
/// # Example
///
/// ```
/// use multitude::Arena;
///
/// let arena = Arena::new();
/// let value = arena.alloc_rc(42_u32);
/// assert_eq!(*value, 42);
/// ```
pub struct Arena<A: Allocator + Clone = Global> {
    /// Cross-thread state (heap-allocated, accessed by chunks via the
    /// back-pointer in their header). Holds only fields that need to
    /// survive the [`Arena`] handle's drop.
    inner: NonNull<ArenaInner<A>>,

    // Owner-only state (read on every alloc; on the `Arena<A>` handle
    // itself to avoid an indirection).
    current_local: ChunkSlot<A>,
    current_shared: ChunkSlot<A>,
    pinned_chunks_head: Cell<Option<NonNull<ChunkHeader<A>>>>,

    max_normal_alloc: usize,
    chunk_size: usize,
    byte_budget: Option<usize>,
    bytes_used_against_budget: Cell<usize>,
    #[cfg(feature = "stats")]
    stats: StatsStorage,

    allocator: A,

    /// `Arena` is single-threaded only.
    _not_sync: PhantomData<*mut ()>,
}

impl Arena<Global> {
    /// Create a new, empty arena backed by [`Global`] with default
    /// configuration.
    ///
    /// No chunk is allocated up front: the alloc fast path's "is there
    /// a current chunk?" check is folded into the bump fit-check via
    /// per-arena sentinel headers (see [`ChunkSlot`](crate::owned_chunk::ChunkSlot)),
    /// so the first allocation lazily pulls in a chunk on the slow path.
    ///
    /// For non-default configuration, use [`Self::builder`].
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// # #[cfg(feature = "stats")]
    /// assert_eq!(arena.stats().chunks_allocated, 0);
    /// let _ = arena.alloc_rc(42_u32);
    /// # #[cfg(feature = "stats")]
    /// assert_eq!(arena.stats().chunks_allocated, 1);
    /// ```
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        ArenaBuilder::new().build()
    }

    /// Create an [`ArenaBuilder`](crate::ArenaBuilder) using [`Global`]
    /// as the backing allocator.
    #[must_use]
    #[inline]
    pub const fn builder() -> ArenaBuilder<Global> {
        ArenaBuilder::new()
    }
}

impl<A: Allocator + Clone> Arena<A> {
    /// Create an [`ArenaBuilder`](crate::ArenaBuilder) backed by a custom
    /// `allocator`.
    #[must_use]
    #[inline]
    pub const fn builder_in(allocator: A) -> ArenaBuilder<A> {
        ArenaBuilder::new_in(allocator)
    }

    /// Create a new, empty arena backed by `allocator` with default
    /// configuration.
    ///
    /// For non-default configuration, use [`Self::builder_in`].
    #[must_use]
    #[inline]
    pub fn new_in(allocator: A) -> Self {
        ArenaBuilder::new_in(allocator).build()
    }

    /// Construct an arena from its configuration. Used by
    /// [`ArenaBuilder`](crate::ArenaBuilder).
    pub(crate) fn from_config(
        allocator: A,
        chunk_size: usize,
        max_normal_alloc: usize,
        byte_budget: Option<usize>,
        chunk_cache_capacity: usize,
    ) -> Self {
        let inner = ArenaInner::<A>::new(chunk_cache_capacity);
        let boxed = alloc::boxed::Box::new(inner);
        let inner_ptr = NonNull::from(alloc::boxed::Box::leak(boxed));
        // SAFETY: `inner_ptr` is at its final heap address; init the
        // embedded sentinel headers so the slots can point at them.
        unsafe {
            ArenaInner::init_sentinels(inner_ptr, &allocator);
        }
        // SAFETY: sentinels are initialized; the slots can hold raw
        // pointers to them for the lifetime of `inner` (and `inner`
        // strictly outlives `Arena` because chunk teardown decrements
        // `outstanding_chunks` to free `inner` only after this `Arena`
        // is dropped).
        let (sentinel_local, sentinel_shared) = unsafe {
            let r = inner_ptr.as_ref();
            (r.sentinel_local_ptr(), r.sentinel_shared_ptr())
        };
        let arena = Self {
            inner: inner_ptr,
            // SAFETY: sentinel pointers are initialized and outlive
            // `Arena`; their `bump = total_size = 0` configuration
            // makes the alloc fast path's fit-check fail naturally.
            current_local: unsafe { ChunkSlot::new(sentinel_local) },
            // SAFETY: see `current_local` above.
            current_shared: unsafe { ChunkSlot::new(sentinel_shared) },
            pinned_chunks_head: Cell::new(None),
            max_normal_alloc,
            chunk_size,
            byte_budget,
            bytes_used_against_budget: Cell::new(0),
            #[cfg(feature = "stats")]
            stats: StatsStorage::default(),
            allocator,
            _not_sync: PhantomData,
        };
        // The Local and Shared slots both start pointing at their
        // sentinel headers (`bump = total_size = 0`). The first
        // allocation falls into the slow path, allocates a fresh chunk,
        // and installs it. No upfront 64 KiB allocation is needed.
        arena
    }

    /// Borrow the inner state.
    ///
    /// Always sound: `Arena` is `!Send`/`!Sync`, so any `&self` access is
    /// confined to the owner thread, and `ArenaInner` is `Box::leak`'d
    /// at construction and only freed inside `Arena::drop` (which holds
    /// `&mut self`).
    #[inline]
    #[must_use]
    pub(crate) const fn inner_ref(&self) -> &ArenaInner<A> {
        // SAFETY: live `ArenaInner` for the lifetime of `&self`.
        unsafe { self.inner.as_ref() }
    }

    /// Push `chunk` onto the front of the pinned-chunks list,
    /// transferring ownership. The pinned list keeps chunks alive
    /// (with their +1 refcount) until [`Arena::drop`] releases them.
    #[inline]
    pub(crate) fn push_pinned(&self, chunk: OwnedChunk<A>) {
        chunk.as_ref().push_into_intrusive_list(&self.pinned_chunks_head);
        let _ = chunk.into_raw();
    }

    /// Pop a chunk from the pinned-chunks list, transferring ownership.
    #[inline]
    pub(crate) fn pop_pinned(&self) -> Option<OwnedChunk<A>> {
        let head = crate::chunk_ref::pop_from_intrusive_list(&self.pinned_chunks_head)?;
        // SAFETY: the pinned list owned a +1 hold on `head`; transfer it.
        Some(unsafe { OwnedChunk::from_raw(head) })
    }

    /// Borrow the backing allocator.
    #[must_use]
    #[inline]
    pub const fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Snapshot of the arena's lifetime statistics.
    ///
    /// Only available with the `stats` Cargo feature enabled.
    #[cfg(feature = "stats")]
    #[cfg_attr(docsrs, doc(cfg(feature = "stats")))]
    #[must_use]
    #[inline]
    pub const fn stats(&self) -> ArenaStats {
        self.stats.snapshot()
    }

    /// Reset the arena to a fresh state, ready for a new allocation phase.
    ///
    /// This moves any unused chunks either to the chunk cache or back to the system
    /// if the cache is full. The arena's lifetime statistics (`Arena::stats`,
    /// available with the `stats` Cargo feature) are preserved across the reset.
    ///
    /// # Behavior
    ///
    /// - Every value the arena currently owns has its destructor run.
    /// - The arena's chunks return to its internal cache for reuse
    ///   (oversized chunks are freed instead).
    /// - The byte budget ([`ArenaBuilder::byte_budget`](crate::ArenaBuilder::byte_budget))
    ///   resets to "currently committed in the cache" so the next phase
    ///   gets a fresh budget.
    /// - Outstanding smart pointers ([`Rc`], [`Arc`],
    ///   [`RcStr`](crate::RcStr),
    ///   [`ArcStr`](crate::ArcStr), [`Box`],
    ///   [`BoxStr`](crate::BoxStr)) keep working — each holds
    ///   a +1 chunk refcount, so its chunk lives until the smart
    ///   pointer drops, exactly as if the arena itself had been
    ///   dropped.
    ///
    /// `reset` takes `&mut self`, so the borrow checker statically
    /// prevents you from calling it while [`Arena::alloc`]-style
    /// references, [`String`](crate::builders::String), or
    /// [`Vec`](crate::builders::Vec) borrow the arena.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "&mut self is the API contract — it statically excludes outstanding borrows from `alloc` etc."
    )]
    #[cold]
    pub fn reset(&mut self) {
        let inner = self.inner_ref();

        let mut made_progress;
        loop {
            made_progress = false;
            if self.current_local.take().is_some() {
                made_progress = true;
            }
            // For Shared chunks, `OwnedChunk::drop` reconciles the
            // refcount via `LARGE - arcs_issued` — no separate credit
            // release needed.
            if self.current_shared.take().is_some() {
                made_progress = true;
            }
            while self.pop_pinned().is_some() {
                made_progress = true;
            }
            if !made_progress {
                break;
            }
        }

        let cache_bytes = inner.chunk_cache_len.get().saturating_mul(self.chunk_size);
        self.bytes_used_against_budget.set(cache_bytes);
    }

    /// Fused fast path: try to allocate `request_layout` (optionally
    /// with a co-allocated `DropEntry`) from the current `Local`-flavor
    /// chunk in one shot — fit-check + bump-update + (optional pin)
    /// without re-reading the bump cell. Returns `None` on miss
    /// (no current chunk, oversized, or doesn't fit).
    ///
    /// Caller must fall back to the slow path on miss.
    #[expect(
        clippy::inline_always,
        reason = "callers supply constant has_drop/pin_for_bump; constant-folding requires inlining"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "fused fast-path return is internal; a type alias would obscure the structure"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw pointer projection + NonNull::new_unchecked share one in-bounds invariant"
    )]
    #[expect(
        clippy::if_then_some_else_none,
        reason = "explicit `if has_drop` keeps the const-foldable branch obvious to LLVM"
    )]
    #[inline(always)]
    fn try_bump_alloc_in_current_local(
        &self,
        request_layout: Layout,
        has_drop: bool,
        pin_for_bump: bool,
    ) -> Option<(ChunkRef<'_, A>, NonNull<u8>, Option<NonNull<DropEntry>>)> {
        // SAFETY: builder rejects max_normal_alloc < MIN_MAX_NORMAL_ALLOC.
        // For typed allocations whose worst-case footprint fits in any
        // minimum-sized chunk (e.g. `T = u64`, `Layout::new::<T>()`),
        // this lets LLVM const-fold the oversized check away entirely.
        unsafe { core::hint::assert_unchecked(self.max_normal_alloc >= MIN_MAX_NORMAL_ALLOC) };
        let worst_case_size = compute_worst_case_size(request_layout, has_drop);
        if worst_case_size > self.max_normal_alloc {
            return None;
        }
        let chunk_ref = self.current_local.peek();
        let cur = chunk_ref.bump_get();
        let total_size = chunk_ref.total_size();
        let align = request_layout.align();
        let (alloc_offset, value_offset, end) = if has_drop {
            let entry_align_eff = align_of::<DropEntry>().max(align);
            let entry_addr = (cur + (entry_align_eff - 1)) & !(entry_align_eff - 1);
            let after_entry = entry_addr + size_of::<DropEntry>();
            let value_addr = (after_entry + (align - 1)) & !(align - 1);
            let end = value_addr + request_layout.size();
            (entry_addr, value_addr, end)
        } else {
            let aligned = (cur + (align - 1)) & !(align - 1);
            let end = aligned + request_layout.size();
            (aligned, aligned, end)
        };
        if end > total_size {
            return None;
        }
        chunk_ref.bump_set(end);
        if pin_for_bump {
            // Unconditional store. The conditional-pin trick used in
            // `try_alloc_str_inner` (load + cmp + skip-store) is only
            // a net win when paired with also dropping the
            // worst-case-size early-out — the cost balance hinges on
            // the layout being dynamic (str). For const-layout typed
            // allocs (e.g. `alloc::<u64>()`) LLVM already const-folds
            // the worst-case check, so the conditional pin is pure
            // ~2-instruction-per-call overhead and regresses
            // `alloc_u64` benches measurably. Keep the unconditional
            // store here.
            if !chunk_ref.pinned() {
                chunk_ref.set_pinned();
            }
        }
        let base = chunk_ref.base();
        // SAFETY: end <= total_size; offsets are in-bounds within the chunk allocation.
        let value_ptr = unsafe { NonNull::new_unchecked(base.as_ptr().add(value_offset)) };
        let entry_ptr = if has_drop {
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "DropEntry pointer is bump-aligned to align_of::<DropEntry>() before the cast"
            )]
            // SAFETY: entry slot is in-bounds; the bump cursor was aligned to
            // `entry_align_eff` (which is at least `align_of::<DropEntry>()`),
            // so the cast is sound.
            let entry = unsafe { NonNull::new_unchecked(base.as_ptr().add(alloc_offset).cast::<DropEntry>()) };
            Some(entry)
        } else {
            None
        };
        Some((chunk_ref, value_ptr, entry_ptr))
    }

    /// Same as [`Self::try_bump_alloc_in_current_local`] for `Shared`
    /// chunks. `Shared` chunks never participate in bump-style pinning.
    #[expect(
        clippy::inline_always,
        reason = "callers supply constant has_drop; constant-folding requires inlining"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "fused fast-path return is internal; a type alias would obscure the structure"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw pointer projection + NonNull::new_unchecked share one in-bounds invariant"
    )]
    #[expect(
        clippy::if_then_some_else_none,
        reason = "explicit `if has_drop` keeps the const-foldable branch obvious to LLVM"
    )]
    #[inline(always)]
    fn try_bump_alloc_in_current_shared(
        &self,
        request_layout: Layout,
        has_drop: bool,
    ) -> Option<(ChunkRef<'_, A>, NonNull<u8>, Option<NonNull<DropEntry>>)> {
        // SAFETY: see `try_bump_alloc_in_current_local`.
        unsafe { core::hint::assert_unchecked(self.max_normal_alloc >= MIN_MAX_NORMAL_ALLOC) };
        let worst_case_size = compute_worst_case_size(request_layout, has_drop);
        if worst_case_size > self.max_normal_alloc {
            return None;
        }
        let chunk_ref = self.current_shared.peek();
        let cur = chunk_ref.bump_get();
        let total_size = chunk_ref.total_size();
        let align = request_layout.align();
        let (alloc_offset, value_offset, end) = if has_drop {
            let entry_align_eff = align_of::<DropEntry>().max(align);
            let entry_addr = (cur + (entry_align_eff - 1)) & !(entry_align_eff - 1);
            let after_entry = entry_addr + size_of::<DropEntry>();
            let value_addr = (after_entry + (align - 1)) & !(align - 1);
            let end = value_addr + request_layout.size();
            (entry_addr, value_addr, end)
        } else {
            let aligned = (cur + (align - 1)) & !(align - 1);
            let end = aligned + request_layout.size();
            (aligned, aligned, end)
        };
        if end > total_size {
            return None;
        }
        chunk_ref.bump_set(end);
        let base = chunk_ref.base();
        // SAFETY: end <= total_size; offsets are in-bounds.
        let value_ptr = unsafe { NonNull::new_unchecked(base.as_ptr().add(value_offset)) };
        let entry_ptr = if has_drop {
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "DropEntry pointer is bump-aligned to align_of::<DropEntry>() before the cast"
            )]
            // SAFETY: see `try_bump_alloc_in_current_local`.
            let entry = unsafe { NonNull::new_unchecked(base.as_ptr().add(alloc_offset).cast::<DropEntry>()) };
            Some(entry)
        } else {
            None
        };
        Some((chunk_ref, value_ptr, entry_ptr))
    }

    /// Dispatcher for the fused fast path. With a constant `sharing`,
    /// the match collapses at the call site.
    #[expect(
        clippy::inline_always,
        reason = "trivial dispatcher; const-folding the match requires inlining at every call site"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "fused fast-path return is internal; a type alias would obscure the structure"
    )]
    #[inline(always)]
    fn try_bump_alloc_in_current(
        &self,
        sharing: ChunkSharing,
        request_layout: Layout,
        has_drop: bool,
        pin_for_bump: bool,
    ) -> Option<(ChunkRef<'_, A>, NonNull<u8>, Option<NonNull<DropEntry>>)> {
        match sharing {
            ChunkSharing::Local => self.try_bump_alloc_in_current_local(request_layout, has_drop, pin_for_bump),
            ChunkSharing::Shared => self.try_bump_alloc_in_current_shared(request_layout, has_drop),
        }
    }

    /// Get a `Local`-flavored chunk that can satisfy `request_layout`.
    /// May return the current chunk, pull from cache, or allocate fresh.
    ///
    /// When `pin_for_bump` is `true`, the chunk is also pinned for
    /// bump-style allocation: normal chunks have their `pinned` flag
    /// set; oversized chunks get an extra refcount + push onto the
    /// pinned list. Smart-pointer callers pass `false` and handle the
    /// refcount themselves.
    #[expect(
        clippy::inline_always,
        reason = "callers supply constant has_drop/pin_for_bump; constant-folding requires inlining"
    )]
    #[inline(always)]
    fn try_get_chunk_for_local(
        &self,
        request_layout: Layout,
        has_drop: bool,
        pin_for_bump: bool,
    ) -> Result<(NonNull<ChunkHeader<A>>, EvictedChunkGuard<A>), AllocError> {
        // SAFETY: the builder rejects configurations with
        // `max_normal_alloc < MIN_MAX_NORMAL_ALLOC`, so this invariant
        // always holds once we've reached here. Asserting it lets LLVM
        // const-fold the oversized check away for typed allocations
        // whose worst-case footprint fits in any minimum-sized chunk.
        unsafe { core::hint::assert_unchecked(self.max_normal_alloc >= MIN_MAX_NORMAL_ALLOC) };
        let worst_case_size = compute_worst_case_size(request_layout, has_drop);
        if worst_case_size > self.max_normal_alloc {
            let chunk = self.try_create_oversized_chunk(ChunkSharing::Local, request_layout, has_drop)?;
            if pin_for_bump {
                #[expect(
                    clippy::multiple_unsafe_ops_per_block,
                    reason = "inc_ref + push_pinned form one tightly-coupled handoff"
                )]
                // SAFETY: fresh oversized chunk; inc_ref then transfer
                // the +1 to the pinned list.
                unsafe {
                    let chunk_ref = ChunkRef::<A>::new(chunk);
                    chunk_ref.inc_ref_local();
                    let owned = OwnedChunk::from_raw(chunk);
                    self.push_pinned(owned);
                }
            }
            return Ok((chunk, EvictedChunkGuard(None)));
        }
        let inner = self.inner_ref();
        let current_slot = &self.current_local;
        // Holds an evicted, non-pinned chunk so its `Drop` runs only after
        // the caller has finished its `*_unchecked` bump-cursor write on the
        // newly-installed chunk. Dropping the evicted chunk can release its
        // refcount to zero and execute user `Drop` impls, which may re-enter
        // `alloc_*`. The re-entrant call sees the populated `current_slot`
        // and bumps the new chunk's cursor; if the outer caller's
        // `alloc_unchecked` then runs based on the pre-re-entrancy cursor,
        // it overwrites bytes the re-entrant alloc just claimed (or, worse,
        // writes past `total_size`). By returning the guard to the caller,
        // we ensure the caller's `*_unchecked` claim happens BEFORE any
        // re-entrant user `Drop` runs.
        let mut evicted_guard: Option<OwnedChunk<A>> = None;
        let chunk_ref = current_slot.peek();
        let cur = chunk_ref.bump_get();
        let total_size = chunk_ref.total_size();
        let align = request_layout.align();
        let aligned = (cur + (align - 1)) & !(align - 1);
        let fits = if has_drop {
            let entry_align_eff = align_of::<DropEntry>().max(align);
            let entry_addr = (cur + (entry_align_eff - 1)) & !(entry_align_eff - 1);
            let after_entry = entry_addr + size_of::<DropEntry>();
            let value_addr = (after_entry + (align - 1)) & !(align - 1);
            let end = value_addr + request_layout.size();
            end <= total_size
        } else {
            let end = aligned + request_layout.size();
            end <= total_size
        };
        if fits {
            if pin_for_bump {
                chunk_ref.set_pinned();
            }
            return Ok((chunk_ref.as_non_null(), EvictedChunkGuard(None)));
        }
        let wasted = total_size.saturating_sub(cur);
        #[cfg(feature = "stats")]
        bump_stat!(self, wasted_tail_bytes, wasted as u64);
        #[cfg(not(feature = "stats"))]
        let _ = wasted;
        // `take()` returns `None` when the slot still holds the
        // sentinel — i.e. on first allocation or after a previous
        // eviction. Real chunks are either pinned (push to pinned
        // list) or evicted (drop only after the caller's
        // `*_unchecked`).
        if let Some(owned) = current_slot.take() {
            if owned.as_ref().pinned() {
                self.push_pinned(owned);
            } else {
                evicted_guard = Some(owned);
            }
        }

        if let Some(retired) = inner.try_pop_cache() {
            let owned = retired.revive(ChunkSharing::Local, 1);
            let ptr = owned.as_non_null();
            if pin_for_bump {
                owned.as_ref().set_pinned();
            }
            current_slot.set(owned);
            return Ok((ptr, EvictedChunkGuard(evicted_guard)));
        }

        let ptr = self.try_alloc_fresh_chunk_normal(ChunkSharing::Local)?;
        // SAFETY: fresh chunk with refcount=1; take ownership.
        let owned = unsafe { OwnedChunk::from_raw(ptr) };
        if pin_for_bump {
            owned.as_ref().set_pinned();
        }
        current_slot.set(owned);
        Ok((ptr, EvictedChunkGuard(evicted_guard)))
    }

    /// Get a `Shared`-flavored chunk that can satisfy `request_layout`.
    /// May return the current chunk, pull from cache, or allocate fresh.
    ///
    /// `Shared`-flavored chunks are only used by `Arc`-style smart
    /// pointers; bump-style allocation (which would need `pin_for_bump`)
    /// is `Local`-only.
    #[expect(
        clippy::inline_always,
        reason = "callers supply a constant has_drop; constant-folding requires inlining"
    )]
    #[inline(always)]
    fn try_get_chunk_for_shared(
        &self,
        request_layout: Layout,
        has_drop: bool,
    ) -> Result<(NonNull<ChunkHeader<A>>, EvictedChunkGuard<A>), AllocError> {
        // SAFETY: see `try_get_chunk_for_local`.
        unsafe { core::hint::assert_unchecked(self.max_normal_alloc >= MIN_MAX_NORMAL_ALLOC) };
        let worst_case_size = compute_worst_case_size(request_layout, has_drop);
        if worst_case_size > self.max_normal_alloc {
            let chunk = self.try_create_oversized_chunk(ChunkSharing::Shared, request_layout, has_drop)?;
            return Ok((chunk, EvictedChunkGuard(None)));
        }
        let inner = self.inner_ref();
        let current_slot = &self.current_shared;
        // See `try_get_chunk_for_local` for why this guard is returned to
        // the caller rather than dropped at function exit.
        let mut evicted_guard: Option<OwnedChunk<A>> = None;
        let chunk_ref = current_slot.peek();
        let cur = chunk_ref.bump_get();
        let total_size = chunk_ref.total_size();
        let align = request_layout.align();
        let aligned = (cur + (align - 1)) & !(align - 1);
        let fits = if has_drop {
            let entry_align_eff = align_of::<DropEntry>().max(align);
            let entry_addr = (cur + (entry_align_eff - 1)) & !(entry_align_eff - 1);
            let after_entry = entry_addr + size_of::<DropEntry>();
            let value_addr = (after_entry + (align - 1)) & !(align - 1);
            let end = value_addr + request_layout.size();
            end <= total_size
        } else {
            let end = aligned + request_layout.size();
            end <= total_size
        };
        if fits {
            return Ok((chunk_ref.as_non_null(), EvictedChunkGuard(None)));
        }
        let wasted = total_size.saturating_sub(cur);
        #[cfg(feature = "stats")]
        bump_stat!(self, wasted_tail_bytes, wasted as u64);
        #[cfg(not(feature = "stats"))]
        let _ = wasted;
        // The Shared chunk's `OwnedChunk::drop` reconciles the
        // refcount via `LARGE - arcs_issued` (deferred-reconciliation
        // scheme) — no separate credit release needed here.
        // `take()` returns `None` when the slot still holds the
        // sentinel.
        if let Some(owned) = current_slot.take() {
            if owned.as_ref().pinned() {
                self.push_pinned(owned);
            } else {
                evicted_guard = Some(owned);
            }
        }

        if let Some(retired) = inner.try_pop_cache() {
            let owned = retired.revive(ChunkSharing::Shared, LARGE_INITIAL_SHARED_REFCOUNT);
            let ptr = owned.as_non_null();
            current_slot.set(owned);
            return Ok((ptr, EvictedChunkGuard(evicted_guard)));
        }

        let ptr = self.try_alloc_fresh_chunk_normal(ChunkSharing::Shared)?;
        // SAFETY: fresh chunk with refcount=1; take ownership.
        let owned = unsafe { OwnedChunk::from_raw(ptr) };
        current_slot.set(owned);
        Ok((ptr, EvictedChunkGuard(evicted_guard)))
    }

    /// Dispatch to [`Self::try_get_chunk_for_local`] or
    /// [`Self::try_get_chunk_for_shared`] based on a runtime `sharing`.
    /// Used by mid-level helpers (e.g. [`Self::reserve_slice`]) that
    /// take `sharing` from their own caller. With inlining, a constant
    /// `sharing` collapses the dispatch.
    #[expect(
        clippy::inline_always,
        reason = "trivial dispatcher; const-folding the match requires inlining at every call site"
    )]
    #[inline(always)]
    fn try_get_chunk_for(
        &self,
        sharing: ChunkSharing,
        request_layout: Layout,
        has_drop: bool,
        pin_for_bump: bool,
    ) -> Result<(NonNull<ChunkHeader<A>>, EvictedChunkGuard<A>), AllocError> {
        match sharing {
            ChunkSharing::Local => self.try_get_chunk_for_local(request_layout, has_drop, pin_for_bump),
            ChunkSharing::Shared => self.try_get_chunk_for_shared(request_layout, has_drop),
        }
    }

    /// Allocate a fresh normal chunk and return it with refcount=1
    /// (the arena's transient hold). Increments `chunks_allocated` and
    /// charges the byte budget. Used by both `try_get_chunk_for` and
    /// the [`ArenaBuilder`](crate::ArenaBuilder)'s preallocate path.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or the
    /// arena's byte budget is exhausted.
    #[cold]
    #[inline(never)]
    pub(crate) fn try_alloc_fresh_chunk_normal(&self, sharing: ChunkSharing) -> Result<NonNull<ChunkHeader<A>>, AllocError> {
        let layout = ChunkHeader::<A>::normal_layout(self.chunk_size);
        self.try_alloc_fresh_chunk(sharing, ChunkSizeClass::Normal, layout)
    }

    /// Allocate a fresh oversized chunk for one allocation. Doesn't touch
    /// `current_*`; the chunk is returned with refcount=1 (the arena's
    /// transient hold, transferred to the resulting smart pointer).
    #[cold]
    #[inline(never)]
    fn try_create_oversized_chunk(
        &self,
        sharing: ChunkSharing,
        request_layout: Layout,
        has_drop: bool,
    ) -> Result<NonNull<ChunkHeader<A>>, AllocError> {
        let chunk_layout = ChunkHeader::<A>::oversized_layout(request_layout, has_drop).ok_or(AllocError)?;
        self.try_alloc_fresh_chunk(sharing, ChunkSizeClass::Oversized, chunk_layout)
    }

    #[cold]
    #[inline(never)]
    fn try_alloc_fresh_chunk(
        &self,
        sharing: ChunkSharing,
        size_class: ChunkSizeClass,
        chunk_layout: Layout,
    ) -> Result<NonNull<ChunkHeader<A>>, AllocError> {
        let inner = self.inner_ref();
        let chunk_bytes = chunk_layout.size();
        let prev = self.bytes_used_against_budget.get();
        let next = prev.checked_add(chunk_bytes).ok_or(AllocError)?;
        if let Some(budget) = self.byte_budget
            && next > budget
        {
            return Err(AllocError);
        }

        let raw = self.allocator.allocate(chunk_layout)?;
        self.bytes_used_against_budget.set(next);

        let total_size = chunk_layout.size();
        let initial_refcount = match (size_class, sharing) {
            // Shared+Normal chunks use deferred-reconciliation refcounting:
            // the slot's "transient hold" is implicit in the LARGE
            // initial value, and per-`alloc_arc` increments are
            // accumulated non-atomically in `chunk.arcs_issued`. On
            // eviction, one `fetch_sub(LARGE - arcs_issued)` reconciles
            // both the unused over-allocation and the slot's hold in a
            // single atomic. Eliminates the per-allocation LOCK RMW
            // entirely.
            (ChunkSizeClass::Normal, ChunkSharing::Shared) => LARGE_INITIAL_SHARED_REFCOUNT,
            (ChunkSizeClass::Normal, ChunkSharing::Local) => 1,
            (ChunkSizeClass::Oversized, _) => 0,
        };
        // SAFETY: just allocated, properly aligned per Allocator contract.
        let chunk = unsafe {
            init_chunk(
                raw.cast::<u8>(),
                total_size,
                sharing,
                size_class,
                self.inner,
                self.allocator.clone(),
                initial_refcount,
            )
        };

        #[cfg(feature = "stats")]
        match size_class {
            ChunkSizeClass::Normal => {
                bump_stat!(self, chunks_allocated, 1);
            }
            ChunkSizeClass::Oversized => {
                bump_stat!(self, oversized_chunks_allocated, 1);
            }
        }
        let _ = inner.outstanding_chunks.fetch_add(1, crate::sync::Ordering::Relaxed);
        Ok(chunk)
    }

    /// Increment the chunk refcount for a smart-pointer allocation.
    ///
    /// `Local` chunks: single non-atomic `Cell` increment on the
    /// chunk's refcount.
    ///
    /// `Shared` chunks: under the deferred-reconciliation refcount
    /// scheme, the chunk's atomic refcount was initialized to
    /// [`LARGE_INITIAL_SHARED_REFCOUNT`] at chunk creation. Per-allocation
    /// inc is just a non-atomic increment of the chunk-local
    /// `arcs_issued` counter — **no LOCK RMW on the hot path**.
    /// Reconciliation happens at chunk eviction (see
    /// `try_get_chunk_for_shared`'s eviction branch), where one
    /// `fetch_sub(LARGE - arcs_issued)` returns the unused
    /// over-allocation and brings `ref_count` to the actual
    /// outstanding-`Arc` count.
    ///
    /// Oversized `Shared` chunks bypass the scheme: they're never in
    /// `current_shared`, are created with refcount=0, and take a
    /// regular `fetch_add(1)` per smart pointer.
    ///
    /// # Safety
    ///
    /// `chunk_ref` must point at a live chunk whose flavor matches `sharing`.
    #[expect(
        clippy::inline_always,
        reason = "trivial dispatcher; const-folding the match requires inlining at every call site"
    )]
    #[inline(always)]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "match arms share one flavor-matches-sharing invariant"
    )]
    unsafe fn inc_ref_for(&self, chunk_ref: ChunkRef<'_, A>, sharing: ChunkSharing) {
        // SAFETY: caller's contract.
        unsafe {
            match sharing {
                ChunkSharing::Local => chunk_ref.inc_ref_local(),
                ChunkSharing::Shared => self.inc_ref_shared_deferred(chunk_ref),
            }
        }
    }

    /// `Shared`-flavor body of [`Self::inc_ref_for`].
    ///
    /// For Normal chunks (always in `current_shared` once installed):
    /// non-atomic increment of `chunk.arcs_issued`. The chunk's atomic
    /// refcount is unchanged.
    ///
    /// For Oversized chunks (never in `current_shared`, refcount=0
    /// at creation): plain `fetch_add(1)` — no batching applicable.
    ///
    /// # Safety
    ///
    /// `chunk_ref` must point at a live `Shared`-flavored chunk header.
    #[inline]
    #[expect(
        clippy::unused_self,
        reason = "method form keeps the call-site dispatch shape symmetric with the Local arm"
    )]
    unsafe fn inc_ref_shared_deferred(&self, chunk_ref: ChunkRef<'_, A>) {
        if chunk_ref.size_class() == ChunkSizeClass::Normal {
            // SAFETY: chunk is live; `arcs_issued` is owner-only and we're on the owner thread.
            let h = unsafe { chunk_ref.header_ref() };
            let prev = h.arcs_issued.get();
            // For non-ZST allocations, `arcs_issued` is bounded by
            // `chunk_size / min_alloc_size` (≪ LARGE). For ZST `Arc<T>`s
            // (size_of::<T>() == 0, no Drop), the bump cursor doesn't
            // advance, so a tight loop could in principle drive
            // `arcs_issued` toward `LARGE_INITIAL_SHARED_REFCOUNT` on
            // 32-bit targets. Reconcile early once we cross the
            // half-way mark: fold `arcs_issued` into `ref_count`
            // atomically and reset. This preserves the eviction
            // accounting (`fetch_sub(LARGE - arcs_issued)` still yields
            // the correct live-arc count) while removing the overflow
            // path. The branch is cold in practice — non-ZSTs evict
            // long before reaching the threshold.
            if prev >= LARGE_INITIAL_SHARED_REFCOUNT / 2 {
                // SAFETY: chunk is Shared; `ref_count` is the atomic refcount.
                unsafe { chunk_ref.fetch_add_ref_shared(prev) };
                h.arcs_issued.set(1);
            } else {
                h.arcs_issued.set(prev + 1);
            }
        } else {
            // Oversized: plain atomic.
            // SAFETY: chunk is live + Shared.
            unsafe { chunk_ref.inc_ref_shared() };
        }
    }

    /// Specialized refcount inc for the *fast path* — chunks that
    /// just came out of `try_bump_alloc_in_current_shared`. Those
    /// chunks are guaranteed `Normal` (the worst-case-size check in
    /// `try_bump_alloc_in_current_shared` rejects oversize requests
    /// before they reach the bump cursor), so we skip the `size_class`
    /// branch in [`Self::inc_ref_shared_deferred`].
    ///
    /// Saves one cmp + cond-branch on every Arc fast-path allocation.
    ///
    /// # Safety
    ///
    /// `chunk_ref` must point at a live `Shared`-flavored chunk that
    /// is `ChunkSizeClass::Normal` (i.e. came from the bump fast path
    /// against `current_shared`).
    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "single-load-store helper; inlining makes it a 2-instruction sequence per Arc fast-path call"
    )]
    #[expect(clippy::unused_self, reason = "method form mirrors `inc_ref_shared_deferred`")]
    unsafe fn inc_ref_shared_normal_deferred(&self, chunk_ref: ChunkRef<'_, A>) {
        // SAFETY: caller's contract; chunk is live + Shared + Normal,
        // and `arcs_issued` is owner-only.
        let h = unsafe { chunk_ref.header_ref() };
        let prev = h.arcs_issued.get();
        // ZST overflow guard: see `inc_ref_shared_deferred`. Cold in
        // practice; non-ZST allocs evict long before reaching the
        // threshold.
        if prev >= LARGE_INITIAL_SHARED_REFCOUNT / 2 {
            // SAFETY: chunk is Shared; `ref_count` is the atomic refcount.
            unsafe { chunk_ref.fetch_add_ref_shared(prev) };
            h.arcs_issued.set(1);
        } else {
            h.arcs_issued.set(prev + 1);
        }
    }

    /// Specialized fast-path-only twin of [`Self::inc_ref_for`]. The
    /// caller asserts the chunk just came out of
    /// `try_bump_alloc_in_current_*` and is therefore [`Normal`]; this
    /// saves the `size_class` dispatch on the Shared arm.
    ///
    /// [`Normal`]: ChunkSizeClass::Normal
    ///
    /// # Safety
    ///
    /// Same as [`Self::inc_ref_for`], plus the chunk must be `Normal`.
    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "trivial dispatcher; const-folding the match requires inlining at every call site"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "match arms share one flavor-matches-sharing invariant"
    )]
    unsafe fn inc_ref_for_normal(&self, chunk_ref: ChunkRef<'_, A>, sharing: ChunkSharing) {
        // SAFETY: caller's contract.
        unsafe {
            match sharing {
                ChunkSharing::Local => chunk_ref.inc_ref_local(),
                ChunkSharing::Shared => self.inc_ref_shared_normal_deferred(chunk_ref),
            }
        }
    }

    /// Allocate `value` and return a thread-local reference-counted
    /// smart pointer to it.
    ///
    /// If `T` needs drop, a tiny entry is added to the owning chunk's
    /// drop list so `T::drop` runs exactly once when the chunk is
    /// reclaimed.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_rc`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let v = arena.alloc_rc(vec![1, 2, 3]);
    /// assert_eq!(v.len(), 3);
    /// ```
    #[inline]
    pub fn alloc_rc<T>(&self, value: T) -> Rc<T, A> {
        self.try_alloc_rc(value).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate `value` and return a thread-local reference-counted smart pointer to
    /// it, returning Err([`AllocError`]) instead of panicking if the
    /// backing allocator fails. The supplied `value` is
    /// dropped on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_rc<T>(&self, value: T) -> Result<Rc<T, A>, AllocError> {
        self.try_alloc_rc_with(move || value)
    }

    /// Allocate the result of `f`, constructing it directly inside the
    /// arena (no stack copy of `T`).
    ///
    /// The closure may freely allocate from this arena. If it panics,
    /// the reservation is leaked (no drop is registered, no double-drop).
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_rc_with`] for a fallible variant.
    #[inline]
    pub fn alloc_rc_with<T, F: FnOnce() -> T>(&self, f: F) -> Rc<T, A> {
        self.try_alloc_rc_with(f).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate the result of `f`, constructing it directly inside the arena,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator fails. The closure is not called on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_rc_with<T, F: FnOnce() -> T>(&self, f: F) -> Result<Rc<T, A>, AllocError> {
        let value_ptr = self.try_reserve_and_init::<T, _>(ChunkSharing::Local, f)?;
        // SAFETY: from_raw wraps the just-allocated, refcount-bumped pointer.
        Ok(unsafe { Rc::from_raw(value_ptr) })
    }

    /// Allocate `value` and return a `Send + Sync` reference-counted
    /// smart pointer to it. Costs an atomic RMW per clone/drop.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if  `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_arc`] for a fallible variant.
    #[inline]
    pub fn alloc_arc<T: Send + Sync>(&self, value: T) -> Arc<T, A>
    where
        A: Send + Sync,
    {
        self.try_alloc_arc(value).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate `value` and return a `Send + Sync` [`Arc`] smart pointer to it,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator fails. The supplied `value` is dropped
    /// on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_arc<T: Send + Sync>(&self, value: T) -> Result<Arc<T, A>, AllocError>
    where
        A: Send + Sync,
    {
        self.try_alloc_arc_with(move || value)
    }

    /// Allocate the result of `f` directly inside a `Shared`-flavor chunk and
    /// return an [`Arc`] smart pointer safe for cross-thread sharing. The closure
    /// constructs the value in place — no stack copy of `T`.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_arc_with`] for a fallible variant.
    #[inline]
    pub fn alloc_arc_with<T: Send + Sync, F: FnOnce() -> T>(&self, f: F) -> Arc<T, A>
    where
        A: Send + Sync,
    {
        self.try_alloc_arc_with(f).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate the result of `f` directly inside a `Shared`-flavor chunk and
    /// return an [`Arc`] smart pointer, returning Err([`AllocError`]) instead
    /// of panicking if the backing allocator cannot satisfy the request. The closure is not called
    /// on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_arc_with<T: Send + Sync, F: FnOnce() -> T>(&self, f: F) -> Result<Arc<T, A>, AllocError>
    where
        A: Send + Sync,
    {
        let value_ptr = self.try_reserve_and_init::<T, _>(ChunkSharing::Shared, f)?;
        // SAFETY: from_raw wraps the just-allocated, refcount-bumped pointer.
        Ok(unsafe { Arc::from_raw(value_ptr) })
    }

    /// Allocate `value` and return an owned, mutable smart pointer whose `Drop`
    /// runs `T::drop` immediately when the smart pointer is dropped.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_box`] for a fallible variant.
    #[inline]
    pub fn alloc_box<T>(&self, value: T) -> Box<T, A> {
        self.try_alloc_box(value).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate `value` and return an owned, mutable [`Box`] smart pointer whose
    /// `Drop` runs `T::drop` immediately when the smart pointer is dropped, returning
    /// Err([`AllocError`]) instead of panicking if the backing allocator
    /// cannot satisfy the request. The supplied `value` is dropped on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_box<T>(&self, value: T) -> Result<Box<T, A>, AllocError> {
        self.try_alloc_box_with(move || value)
    }

    /// Allocate the result of `f` directly inside the arena and return an
    /// owned, mutable [`Box`] smart pointer whose `Drop` runs `T::drop`
    /// immediately when the smart pointer is dropped.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_box_with`] for a fallible variant.
    #[inline]
    pub fn alloc_box_with<T, F: FnOnce() -> T>(&self, f: F) -> Box<T, A> {
        self.try_alloc_box_with(f).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate the result of `f` directly inside the arena and return an
    /// owned, mutable [`Box`] smart pointer, returning Err([`AllocError`])
    /// instead of panicking if the backing allocator fails. The closure is not called on allocator failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_box_with<T, F: FnOnce() -> T>(&self, f: F) -> Result<Box<T, A>, AllocError> {
        let value_ptr = self.try_reserve_and_init::<T, _>(ChunkSharing::Local, f)?;
        // SAFETY: just-allocated, refcount-bumped pointer.
        Ok(unsafe { Box::from_raw(value_ptr) })
    }

    /// Bump-allocate `value` and return a mutable reference whose
    /// lifetime is tied to `&self`. The cheapest allocation multitude
    /// offers — no refcount, no per-pointer bookkeeping. The borrow
    /// checker bounds the returned reference to the arena's lifetime.
    ///
    /// If `T: Drop`, a drop entry is registered in the chunk's drop
    /// list; `T::drop` runs at arena drop. (For per-pointer
    /// drop-on-drop semantics, use [`Self::alloc_box`] instead.)
    ///
    /// The chunk that hosts the value is "pinned" — it lives until
    /// arena drop (other allocations into the same chunk follow normal
    /// per-chunk reclamation rules and may extend its life past the
    /// arena via [`Rc`] / [`Arc`] smart pointers).
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let x: &mut u32 = arena.alloc(42);
    /// let y: &mut u32 = arena.alloc(100);
    /// *x += 1;
    /// *y += 1;
    /// assert_eq!(*x, 43);
    /// assert_eq!(*y, 101);
    /// ```
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.try_alloc(value).unwrap_or_else(|_| panic_alloc())
    }

    /// Bump-allocate `value` and return a mutable reference, returning
    /// [`AllocError`] instead of panicking if the backing allocator
    /// cannot satisfy the request. The supplied `value` is dropped on
    /// failure.
    ///
    /// See [`Self::alloc`] for full semantics.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy
    /// the request.
    #[inline]
    pub fn try_alloc<T>(&self, value: T) -> Result<&mut T, AllocError> {
        self.try_alloc_with(move || value)
    }

    /// Bump-allocate the result of `f`, constructing it directly inside
    /// the arena (no stack copy of `T`). Returns a mutable reference
    /// whose lifetime is tied to `&self`.
    ///
    /// See [`Self::alloc`] for full semantics.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_with`] for a fallible variant.
    ///
    /// If `f` panics, the reservation is leaked in-chunk (no drop is registered, no
    /// refcount bumped) but the chunk itself reclaims normally.
    #[inline]
    pub fn alloc_with<T, F: FnOnce() -> T>(&self, f: F) -> &mut T {
        self.try_alloc_with(f).unwrap_or_else(|_| panic_alloc())
    }

    /// Bump-allocate the result of `f` directly inside the arena and
    /// return a mutable reference, returning [`AllocError`] instead of
    /// panicking if the backing allocator fails.
    /// The closure is not called on allocator failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[expect(
        clippy::mut_from_ref,
        reason = "Simple references: each call returns a fresh, disjoint &mut T; the borrow checker treats the returned reference as exclusive of its own region but harmlessly aliasing-with-shared with the &Arena borrow"
    )]
    #[inline]
    pub fn try_alloc_with<T, F: FnOnce() -> T>(&self, f: F) -> Result<&mut T, AllocError> {
        if Layout::new::<T>().align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }
        // SAFETY: alignment now known to fit.
        unsafe { self.try_alloc_with_aligned(f) }
    }

    /// Body of [`Self::try_alloc_with`] for the case where the alignment
    /// guard has already passed. Split out so the caller's frame stays
    /// `T`-alignment-independent.
    ///
    /// # Safety
    ///
    /// `Layout::new::<T>().align() <= CHUNK_ALIGN`.
    #[inline]
    #[expect(clippy::mut_from_ref, reason = "see `try_alloc_with`")]
    unsafe fn try_alloc_with_aligned<T, F: FnOnce() -> T>(&self, f: F) -> Result<&mut T, AllocError> {
        let needs_drop = core::mem::needs_drop::<T>();
        let layout = Layout::new::<T>();
        // Pin the chunk before running `f` so a panic can't leak it;
        // defer `link_drop_entry` until after `f` succeeds so a panic
        // doesn't register `drop_in_place` over uninit bytes.
        //
        // Fast path: if the current local chunk fits the request, do
        // fit-check + bump-update + (optional pin) in one shot, avoiding
        // the redundant bump re-read that splitting `try_get_chunk_for_local`
        // and `alloc_unchecked` would otherwise impose.
        let (chunk_ref, value_ptr_u8, entry_ptr) = if let Some(t) = self.try_bump_alloc_in_current_local(layout, needs_drop, true) {
            t
        } else {
            let (chunk, _evicted_guard) = self.try_get_chunk_for_local(layout, needs_drop, true)?;
            // SAFETY: live chunk.
            let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
            if needs_drop {
                // SAFETY: worst-case sizing guaranteed.
                let (entry, value) = unsafe { ChunkHeader::<A>::alloc_with_drop_entry_unchecked::<T>(chunk) };
                (chunk_ref, value.cast::<u8>(), Some(entry))
            } else {
                // SAFETY: worst-case sizing guaranteed.
                let value = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) };
                (chunk_ref, value, None)
            }
            // `_evicted_guard` drops here, AFTER the unchecked alloc has
            // claimed its bump-cursor range — see `EvictedChunkGuard`.
        };
        let mut value_ptr = value_ptr_u8.cast::<T>();
        // SAFETY: writable slot.
        unsafe { value_ptr.as_ptr().write(f()) };
        if let Some(entry) = entry_ptr {
            // SAFETY: entry slot writable; chunk live.
            unsafe { chunk_ref.link_drop_entry(entry, drop_shim::<T>, 0) };
        }
        self.charge_alloc_stats(layout.size());
        // SAFETY: T initialized and pinned for `&self`'s lifetime.
        Ok(unsafe { value_ptr.as_mut() })
    }

    /// Bump-allocate a copy of `s` and return a mutable string slice
    /// whose lifetime is tied to `&self`. Like [`Self::alloc`] but
    /// for `&str`.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_str`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let s: &mut str = arena.alloc_str("hello");
    /// s.make_ascii_uppercase();
    /// assert_eq!(s, "HELLO");
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc_str(&self, s: impl AsRef<str>) -> &mut str {
        self.try_alloc_str_inner(s.as_ref()).unwrap_or_else(|_| panic_alloc())
    }

    /// Copy `s` into the arena and return an [`RcStr`](crate::RcStr) smart pointer —
    /// a single-pointer, refcounted, `!Send`/`!Sync` immutable string.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_str_rc`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let s = arena.alloc_str_rc("hello");
    /// assert_eq!(&*s, "hello");
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc_str_rc(&self, s: impl AsRef<str>) -> RcStr<A> {
        self.try_alloc_str_rc(s).unwrap_or_else(|_| panic_alloc())
    }

    /// Copy `s` into the arena and return an [`ArcStr`](crate::ArcStr)
    /// pointing to it.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_str_arc`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let s = arena.alloc_str_arc("hello");
    /// assert_eq!(&*s, "hello");
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc_str_arc(&self, s: impl AsRef<str>) -> ArcStr<A>
    where
        A: Send + Sync,
    {
        self.try_alloc_str_arc(s).unwrap_or_else(|_| panic_alloc())
    }

    /// Copy `s` into the arena and return an [`BoxStr`](crate::BoxStr)
    /// smart pointer — a single-pointer (8 bytes) owned, mutable string.
    ///
    /// Compared to [`Self::alloc_str_rc`] / [`Self::alloc_str_arc`]:
    /// the box is `!Clone` (single-owner) but supports `&mut str` access
    /// and releases its chunk hold the moment it is dropped.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails.
    /// Use [`Self::try_alloc_str_box`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let mut s = arena.alloc_str_box("hello");
    /// s.make_ascii_uppercase();
    /// assert_eq!(&*s, "HELLO");
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc_str_box(&self, s: impl AsRef<str>) -> BoxStr<A> {
        self.try_alloc_str_box(s).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_str`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_str(&self, s: impl AsRef<str>) -> Result<&mut str, AllocError> {
        self.try_alloc_str_inner(s.as_ref())
    }

    /// Specialized fast path for `&str` allocation.
    ///
    /// Bypasses the generic [`Self::try_alloc_slice_copy`] for `T = u8`,
    /// which carries dead-but-not-LLVM-eliminated overhead specific to
    /// the `T: Copy` shape (overflow-on-`checked_mul`, `total >
    /// isize::MAX - 0`, `align >= CHUNK_ALIGN`), and additionally
    /// inlines the bump fit-check directly so we can skip:
    ///
    /// - The `worst_case_size > max_normal_alloc` early-out in
    ///   `try_bump_alloc_in_current_local` — for strings the bump
    ///   fit-check is sufficient: oversize requests fall through to the
    ///   slow path naturally because the fit-check fails.
    /// - The unconditional `set_pinned(true)` write — strings only
    ///   need to pin the chunk once per chunk (pinning is sticky), so
    ///   we read the flag and only store when it's still false. After
    ///   the first allocation in a chunk the branch is perfectly
    ///   predicted and the store is skipped.
    ///
    /// On the `alloc_str` benchmark (short strings, `T = u8`,
    /// `pin_for_bump = true`) this closes most of the gap against
    /// bumpalo's `alloc_str`.
    #[expect(clippy::mut_from_ref, reason = "simple references: see Self::try_alloc_with")]
    #[expect(
        clippy::inline_always,
        reason = "hot path; force-inlining lets LLVM see the &str fat-pointer through AsRef and skip stack materialization"
    )]
    #[inline(always)]
    fn try_alloc_str_inner(&self, s: &str) -> Result<&mut str, AllocError> {
        let bytes = s.as_bytes();
        let len = bytes.len();

        // Fully-fused fast path: peek slot, fit-check, bump, conditional pin.
        let chunk_ref = self.current_local.peek();
        let cur = chunk_ref.bump_get();
        let end = cur.wrapping_add(len);
        let dst = if end <= chunk_ref.total_size() {
            chunk_ref.bump_set(end);
            if !chunk_ref.pinned() {
                chunk_ref.set_pinned();
            }
            // SAFETY: `end <= total_size` so `cur + len <= total_size`;
            // the `cur..end` byte range is in-bounds within the chunk.
            unsafe { chunk_ref.base().as_ptr().add(cur) }
        } else {
            // Slow path. The fit-check above already ruled out the
            // sentinel (`total_size = 0`, `bump = 1`) and any chunk
            // that's too small; the slow path does its own
            // worst-case-size check and routes oversize requests to a
            // dedicated chunk.
            //
            // SAFETY: align=1 is a power of two; `len <= isize::MAX`
            // since `bytes` came from a `&str` (UTF-8 length is
            // bounded by `isize::MAX`).
            let layout = unsafe { Layout::from_size_align_unchecked(len, 1) };
            let (chunk, _evicted_guard) = self.try_get_chunk_for_local(layout, false, true)?;
            // SAFETY: worst-case sizing was guaranteed by `try_get_chunk_for_local`.
            unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) }.as_ptr()
        };
        // SAFETY: `dst` has space for `len` bytes; src and dst do not overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, len);
        }
        self.charge_alloc_stats(len);
        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            reason = "from_raw_parts_mut + from_utf8_unchecked_mut share one initialized-UTF-8 invariant"
        )]
        // SAFETY: `dst` is initialized for `len` bytes copied verbatim
        // from valid UTF-8 input; the resulting `&mut str` borrows the
        // arena for `&self`'s lifetime.
        let r = unsafe {
            let slice = core::slice::from_raw_parts_mut(dst, len);
            core::str::from_utf8_unchecked_mut(slice)
        };
        Ok(r)
    }

    /// Fallible variant of [`Self::alloc_str_rc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[expect(
        clippy::inline_always,
        reason = "callers supply constant `sharing`; const-folding requires inlining"
    )]
    #[inline(always)]
    pub fn try_alloc_str_rc(&self, s: impl AsRef<str>) -> Result<RcStr<A>, AllocError> {
        let data = try_reserve_str_in_chunk(self, s.as_ref(), ChunkSharing::Local)?;
        // SAFETY: helper reserved a length-prefixed buffer and bumped the chunk refcount.
        Ok(unsafe { RcStr::from_raw_data(data) })
    }

    /// Fallible variant of [`Self::alloc_str_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[expect(
        clippy::inline_always,
        reason = "callers supply constant `sharing`; const-folding requires inlining"
    )]
    #[inline(always)]
    pub fn try_alloc_str_arc(&self, s: impl AsRef<str>) -> Result<ArcStr<A>, AllocError>
    where
        A: Send + Sync,
    {
        let data = try_reserve_str_in_chunk(self, s.as_ref(), ChunkSharing::Shared)?;
        // SAFETY: helper reserved a length-prefixed buffer and bumped the chunk refcount.
        Ok(unsafe { ArcStr::from_raw_data(data) })
    }

    /// Fallible variant of [`Self::alloc_str_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[expect(
        clippy::inline_always,
        reason = "callers supply constant `sharing`; const-folding requires inlining"
    )]
    #[inline(always)]
    pub fn try_alloc_str_box(&self, s: impl AsRef<str>) -> Result<BoxStr<A>, AllocError> {
        let data = try_reserve_str_in_chunk(self, s.as_ref(), ChunkSharing::Local)?;
        // SAFETY: helper reserved a length-prefixed buffer and bumped the chunk refcount.
        Ok(unsafe { BoxStr::from_raw_data(data) })
    }

    /// Bump-allocate a copy of `slice` (element-by-element `Copy`) and
    /// return a mutable slice whose lifetime is tied to `&self`. Like
    /// [`Self::alloc`] but for slices of `T: Copy`.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_copy`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_slice_copy<T: Copy>(&self, slice: impl AsRef<[T]>) -> &mut [T] {
        self.try_alloc_slice_copy(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_copy`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[expect(clippy::mut_from_ref, reason = "simple references: see Self::try_alloc_with")]
    #[inline]
    pub fn try_alloc_slice_copy<T: Copy>(&self, slice: impl AsRef<[T]>) -> Result<&mut [T], AllocError> {
        let slice = slice.as_ref();
        let len = slice.len();
        let elem_size = size_of::<T>();
        let total = elem_size.checked_mul(len).ok_or(AllocError)?;
        if total > isize::MAX as usize - (align_of::<T>() - 1) {
            return Err(AllocError);
        }
        // SAFETY: `align_of::<T>()` is a power of two; `total` is bounded above.
        let layout = unsafe { Layout::from_size_align_unchecked(total, align_of::<T>()) };
        if layout.align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }
        // Fast path: bump-allocate in the current local chunk (fit-check +
        // bump-update + pin in one shot). Same trick as `try_alloc_with_aligned`,
        // and a major win for short `alloc_str` / `alloc_slice_copy` calls
        // because it avoids the redundant fit-check + bump re-read that
        // splitting `try_get_chunk_for_local` and `alloc_unchecked` imposes.
        // `T: Copy` implies `!Drop`, so `has_drop = false` is constant here.
        let dst = if let Some((_chunk_ref, ptr, _)) = self.try_bump_alloc_in_current_local(layout, false, true) {
            ptr.cast::<T>()
        } else {
            let (chunk, _evicted_guard) = self.try_get_chunk_for_local(layout, false, true)?;
            // SAFETY: worst-case sizing was guaranteed by `try_get_chunk_for_local`.
            let p = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) }.cast::<T>();
            // `_evicted_guard` drops here, AFTER `alloc_unchecked` has
            // claimed its bump-cursor range — see `EvictedChunkGuard`.
            p
        };
        // SAFETY: dst has space for `len` Ts; src and dst do not overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst.as_ptr(), len);
        }
        self.charge_alloc_stats(total);
        // SAFETY: dst initialized for `len` Ts and pinned for `&self`.
        Ok(unsafe { core::slice::from_raw_parts_mut(dst.as_ptr(), len) })
    }

    /// Bump-allocate a slice and fill it with values pulled from `f`,
    /// returning a mutable slice whose lifetime is tied to `&self`.
    ///
    /// If `T: Drop`, a drop entry is registered (drops at arena drop).
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_with`] for a fallible variant.
    ///
    /// If `f` panics, already-initialized elements are dropped (drop guard) and the
    /// panic propagates.
    #[must_use]
    #[inline]
    pub fn alloc_slice_fill_with<T, F: FnMut(usize) -> T>(&self, len: usize, f: F) -> &mut [T] {
        self.try_alloc_slice_fill_with(len, f).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_fill_with`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// If `f` panics, already-initialized elements are dropped.
    #[expect(clippy::mut_from_ref, reason = "simple references: see Self::try_alloc_with")]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "single tightly-coupled fill / link sequence with one safety invariant"
    )]
    #[inline]
    pub fn try_alloc_slice_fill_with<T, F: FnMut(usize) -> T>(&self, len: usize, mut f: F) -> Result<&mut [T], AllocError> {
        let needs_drop = core::mem::needs_drop::<T>();
        let elem_size = size_of::<T>();
        let total = elem_size.checked_mul(len).ok_or(AllocError)?;
        if total > isize::MAX as usize - (align_of::<T>() - 1) {
            return Err(AllocError);
        }
        // SAFETY: `align_of::<T>()` is a power of two; `total` is bounded above.
        let layout = unsafe { Layout::from_size_align_unchecked(total, align_of::<T>()) };
        if layout.align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }

        // Fast path: when `T: !Drop` we don't need the `SliceReservation`
        // bookkeeping (no drop entry to commit), so we can use the same
        // fused fit-check + bump-update path as `try_alloc_with_aligned`.
        // `needs_drop` is constant per monomorphization, so LLVM eliminates
        // the dead branch.
        if !needs_drop {
            let dst = if let Some((_chunk_ref, ptr, _)) = self.try_bump_alloc_in_current_local(layout, false, true) {
                ptr.cast::<T>()
            } else {
                let (chunk, _evicted_guard) = self.try_get_chunk_for_local(layout, false, true)?;
                // SAFETY: worst-case sizing was guaranteed by `try_get_chunk_for_local`.
                let p = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) }.cast::<T>();
                // `_evicted_guard` drops here, AFTER `alloc_unchecked` has
                // claimed its bump-cursor range — see `EvictedChunkGuard`.
                p
            };
            // SAFETY: drop guard catches partial-init panics from `f`.
            unsafe {
                let mut guard = SliceInitGuard::<T> { ptr: dst.as_ptr(), len: 0 };
                for i in 0..len {
                    dst.as_ptr().add(i).write(f(i));
                    guard.len += 1;
                }
                core::mem::forget(guard);
            }
            self.charge_alloc_stats(total);
            // SAFETY: dst initialized for `len` Ts and pinned for `&self`.
            return Ok(unsafe { core::slice::from_raw_parts_mut(dst.as_ptr(), len) });
        }

        // Drop path: needs the full reservation so we can register a
        // drop-in-place entry once initialization succeeds.
        let reservation = self.reserve_slice::<T>(len, ChunkSharing::Local, true, true)?;
        // SAFETY: reservation produced a live chunk.
        let chunk_ref = unsafe { ChunkRef::<A>::new(reservation.chunk_for_pin()) };
        let dst = reservation.ptr.cast::<T>();
        // SAFETY: drop guard catches partial-init panics from `f`.
        unsafe {
            let mut guard = SliceInitGuard::<T> { ptr: dst.as_ptr(), len: 0 };
            for i in 0..len {
                dst.as_ptr().add(i).write(f(i));
                guard.len += 1;
            }
            core::mem::forget(guard);
            if let Some(entry) = reservation.entry_for_pin() {
                chunk_ref.link_drop_entry(entry, slice_drop_shim::<T>, len);
            }
        }
        self.charge_alloc_stats(total);
        // SAFETY: dst initialized for `len` Ts and pinned for `&self`.
        Ok(unsafe { core::slice::from_raw_parts_mut(dst.as_ptr(), len) })
    }

    /// Bump-allocate a slice by cloning each element of `slice` and
    /// return a mutable slice whose lifetime is tied to `&self`.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_clone`] for a fallible variant.
    ///
    /// May panic if `T::clone` panics; already-cloned elements are dropped before the
    /// panic propagates.
    #[must_use]
    #[inline]
    pub fn alloc_slice_clone<T: Clone>(&self, slice: impl AsRef<[T]>) -> &mut [T] {
        self.try_alloc_slice_clone(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_clone`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// May panic if a `T::clone` impl panics; already-cloned elements
    /// are dropped before the panic propagates.
    #[expect(clippy::mut_from_ref, reason = "simple references: see Self::try_alloc_with")]
    #[inline]
    pub fn try_alloc_slice_clone<T: Clone>(&self, slice: impl AsRef<[T]>) -> Result<&mut [T], AllocError> {
        let slice = slice.as_ref();
        // Use a raw-pointer-indexed closure rather than the older
        // `iter.next().expect().clone()` chain. The latter introduces
        // an `Option::None`-branch per element that the predictor
        // mispredicted once per outer call (≈ 1000 mispredicts per
        // 1000-call bench in practice). With raw indexing the loop is
        // a straight `clone()` per i, which LLVM auto-vectorizes for
        // `T: Copy`.
        let len = slice.len();
        let src = slice.as_ptr();
        // SAFETY: `src` is valid for `len` reads of `T`; the closure
        // is called exactly `len` times by `try_alloc_slice_fill_with`,
        // each with a distinct `i` in `0..len`.
        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            reason = "raw pointer arithmetic + clone share the in-bounds invariant"
        )]
        // SAFETY: see above.
        self.try_alloc_slice_fill_with(len, |i| unsafe { (*src.add(i)).clone() })
    }

    /// Bump-allocate a slice and fill it with values pulled from `iter`,
    /// returning a mutable slice whose lifetime is tied to `&self`.
    ///
    /// If `T: Drop`, a drop entry is registered (drops at arena drop).
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_iter`] for a fallible variant.
    ///
    /// May also panic if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[must_use]
    #[inline]
    pub fn alloc_slice_fill_iter<T, I>(&self, iter: I) -> &mut [T]
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        self.try_alloc_slice_fill_iter(iter).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_fill_iter`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[inline]
    pub fn try_alloc_slice_fill_iter<T, I>(&self, iter: I) -> Result<&mut [T], AllocError>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut iter = iter.into_iter();
        let len = iter.len();
        self.try_alloc_slice_fill_with(len, |_| iter.next().expect("iterator shorter than ExactSizeIterator len"))
    }

    /// Reserve space for one `T` value (with optional `DropEntry` for
    /// `T: needs_drop`), invoke `f` to construct the value, write it,
    /// link the drop entry, and return the value pointer. The chunk's
    /// refcount is bumped by 1 (the new smart pointer's hold).
    ///
    #[inline]
    fn try_reserve_and_init<T, F: FnOnce() -> T>(&self, sharing: ChunkSharing, f: F) -> Result<NonNull<T>, AllocError> {
        if Layout::new::<T>().align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }
        // SAFETY: alignment guard above discharges the inner fn's precondition.
        unsafe { self.try_reserve_and_init_aligned(sharing, f) }
    }

    /// Body of [`Self::try_reserve_and_init`] for the case where the
    /// alignment guard has already passed. Split out so the outer
    /// frame stays `T`-alignment-independent.
    ///
    /// # Safety
    ///
    /// Same as [`Self::try_reserve_and_init`], plus
    /// `Layout::new::<T>().align() <= CHUNK_ALIGN`.
    ///
    /// The fast path (current chunk fits) is fully inlined here so it
    /// pays no `EvictedChunkGuard` stack-frame cost. The slow path is
    /// in a `#[cold]` helper that owns the guard for its full lifetime.
    ///
    /// Both paths take a *protective `+1`* on the chunk BEFORE running
    /// `f()`, then transfer that `+1` to the smart pointer on success
    /// (no extra `inc_ref_for` at the bottom). The protective `+1`
    /// keeps the chunk alive across `f()`'s body even if `f` re-enters
    /// the arena and triggers an eviction that would otherwise drop
    /// the chunk's only refcount (the slot's transient `+1`) to zero.
    /// `RefcountReleaseGuard` releases the `+1` on panic.
    #[inline]
    unsafe fn try_reserve_and_init_aligned<T, F: FnOnce() -> T>(&self, sharing: ChunkSharing, f: F) -> Result<NonNull<T>, AllocError> {
        let needs_drop = core::mem::needs_drop::<T>();
        let layout = Layout::new::<T>();

        if let Some((chunk_ref, value_ptr_u8, entry_ptr)) = self.try_bump_alloc_in_current(sharing, layout, needs_drop, false) {
            // Take the protective `+1` BEFORE running user code. The
            // chunk just came out of the bump fast path so it is
            // guaranteed `Normal`. After `f()` succeeds, this `+1`
            // becomes the smart pointer's refcount; on panic, the
            // guard releases it.
            //
            // SAFETY: chunk live + Normal; flavor matches `sharing`.
            unsafe { self.inc_ref_for_normal(chunk_ref, sharing) };
            let panic_guard = RefcountReleaseGuard::<A> {
                chunk: chunk_ref.as_non_null(),
                sharing,
                holds_oversized: false,
            };
            let value = value_ptr_u8.cast::<T>();
            // SAFETY: writable slot. Panic from `f()` is intercepted
            // by `panic_guard`.
            unsafe { value.as_ptr().write(f()) };
            core::mem::forget(panic_guard);
            if let Some(entry) = entry_ptr {
                // SAFETY: entry slot writable; chunk live (we hold +1).
                unsafe { chunk_ref.link_drop_entry(entry, drop_shim::<T>, 0) };
            }
            // No additional `inc_ref_for` — the protective +1 is the
            // smart pointer's +1.
            self.charge_alloc_stats(layout.size());
            return Ok(value);
        }

        // SAFETY: caller's contract.
        unsafe { self.try_reserve_and_init_aligned_slow::<T, F>(sharing, layout, needs_drop, f) }
    }

    /// Cold slow path of [`Self::try_reserve_and_init_aligned`]. Owns
    /// the `EvictedChunkGuard` for its full lifetime so re-entrant
    /// `Drop` from the guard's eviction can't free the freshly-
    /// installed chunk before we `inc_ref_for` it. See
    /// `slow_path_eviction_does_not_free_new_chunk_via_reentrant_drop_in_pin_false_smart_ptr_paths`
    /// in `tests/arena_drop_reentrancy.rs`.
    ///
    /// # Safety
    ///
    /// Same as [`Self::try_reserve_and_init_aligned`].
    #[cold]
    #[inline(never)]
    unsafe fn try_reserve_and_init_aligned_slow<T, F: FnOnce() -> T>(
        &self,
        sharing: ChunkSharing,
        layout: Layout,
        needs_drop: bool,
        f: F,
    ) -> Result<NonNull<T>, AllocError> {
        let (chunk, _evicted_guard) = self.try_get_chunk_for(sharing, layout, needs_drop, false)?;
        // SAFETY: live chunk.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        let (value_ptr_u8, entry_ptr) = if needs_drop {
            // SAFETY: worst-case sizing guaranteed.
            let (entry, value) = unsafe { ChunkHeader::<A>::alloc_with_drop_entry_unchecked::<T>(chunk) };
            (value.cast::<u8>(), Some(entry))
        } else {
            // SAFETY: worst-case sizing guaranteed.
            let value = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) };
            (value, None)
        };
        // Take the protective `+1` BEFORE running `f()`. For
        // Oversized this is a real refcount op (chunk started at 0);
        // for Normal it's the deferred-reconciliation `arcs_issued += 1`
        // (Shared) or a Cell increment (Local). Either way, the
        // chunk is now safe across user code execution.
        let holds_oversized = chunk_ref.size_class() == ChunkSizeClass::Oversized;
        if holds_oversized {
            // SAFETY: chunk live, flavor matches sharing.
            unsafe { self.inc_ref_for(chunk_ref, sharing) };
        } else {
            // SAFETY: chunk live + Normal, flavor matches sharing.
            unsafe { self.inc_ref_for_normal(chunk_ref, sharing) };
        }
        let panic_guard = RefcountReleaseGuard::<A> {
            chunk: chunk_ref.as_non_null(),
            sharing,
            holds_oversized,
        };
        let value = value_ptr_u8.cast::<T>();
        // SAFETY: writable slot. Panic from `f()` is intercepted by
        // `panic_guard`.
        unsafe { value.as_ptr().write(f()) };
        core::mem::forget(panic_guard);
        if let Some(entry) = entry_ptr {
            // SAFETY: entry slot writable; chunk live.
            unsafe { chunk_ref.link_drop_entry(entry, drop_shim::<T>, 0) };
        }
        // No additional `inc_ref_for` — protective +1 transfers to
        // the smart pointer.
        self.charge_alloc_stats(layout.size());
        // `_evicted_guard` drops here, AFTER all our work has run.
        Ok(value)
    }

    /// Charge a successful user allocation against `total_bytes_allocated`.
    /// No-op when the `stats` feature is disabled.
    #[inline]
    #[cfg_attr(
        not(feature = "stats"),
        expect(
            clippy::missing_const_for_fn,
            reason = "non-const under stats feature; bump_stat! invokes Cell::set"
        )
    )]
    #[cfg_attr(
        not(feature = "stats"),
        expect(clippy::unused_self, reason = "no-op stub when `stats` feature is disabled")
    )]
    pub(crate) fn charge_alloc_stats(&self, bytes: usize) {
        #[cfg(feature = "stats")]
        {
            bump_stat!(self, total_bytes_allocated, bytes as u64);
        }
        #[cfg(not(feature = "stats"))]
        let _ = bytes;
    }

    /// Bump `relocations` by 1. Called whenever a growing collection
    /// has to be moved to a fresh, larger buffer. No-op when the
    /// `stats` feature is disabled.
    #[inline]
    #[cfg_attr(
        not(feature = "stats"),
        expect(
            clippy::missing_const_for_fn,
            reason = "non-const under stats feature; bump_stat! invokes Cell::set"
        )
    )]
    #[cfg_attr(
        not(feature = "stats"),
        expect(clippy::unused_self, reason = "no-op stub when `stats` feature is disabled")
    )]
    pub(crate) fn bump_relocation(&self) {
        #[cfg(feature = "stats")]
        {
            bump_stat!(self, relocations, 1);
        }
    }
}

/// Triple returned by [`Arena::try_reserve_dst_with_entry`].
#[cfg(feature = "dst")]
type DstReservation<A> = (NonNull<DropEntry>, NonNull<u8>, NonNull<ChunkHeader<A>>);

/// A pending slice reservation: bump-allocated space (and an unlinked
/// `DropEntry` slot when needed) but no committed refcount yet.
/// Returned by `Arena::reserve_slice`; finalized by `Arena::commit_slice`
/// once the caller has fully initialized the elements.
///
/// The reserve/commit split exists so that, if the caller's element-init
/// code panics partway, neither the drop entry is linked (avoiding
/// `drop_in_place` on uninit bytes at chunk teardown) nor a smart
/// pointer constructed. The reserved bump bytes are leaked in-chunk,
/// but the chunk itself reclaims normally.
pub struct SliceReservation<A: Allocator + Clone> {
    chunk: NonNull<ChunkHeader<A>>,
    pub ptr: NonNull<u8>,
    entry: Option<NonNull<DropEntry>>,
    layout: Layout,
    /// `true` iff the chunk is oversized. `reserve_slice` always
    /// takes a protective `+1` regardless of size class; this flag
    /// is informational. Always released through `release_chunk_ref` —
    /// the dispatch on `chunk.sharing` handles both cases.
    holds_oversized: bool,
    /// Holds any chunk evicted from `current_*` during the reservation's
    /// `try_get_chunk_for` slow path. MUST live until the reservation
    /// is committed (or dropped on panic): re-entrant `Drop` impls
    /// fired while this guard drops can re-enter the arena and, with
    /// `pin_for_bump = false`, evict and free the freshly-installed
    /// chunk before the smart-pointer `inc_ref_for` brings its
    /// refcount above the slot's transient hold. See
    /// `try_reserve_and_init_aligned` for the full bug story.
    ///
    /// Declared *last* so it drops *after* this struct's manual `Drop`
    /// impl runs (which releases the protective +1) — i.e. only after
    /// any final use of `chunk` has finished.
    evicted_guard: EvictedChunkGuard<A>,
}

impl<A: Allocator + Clone> Drop for SliceReservation<A> {
    fn drop(&mut self) {
        // Release the protective `+1` taken in `reserve_slice`.
        // `release_chunk_ref` dispatches on the chunk's `sharing` and
        // runs `teardown_chunk` if ref_count hits zero — see
        // `RefcountReleaseGuard::Drop` for the full case analysis.
        let _ = self.holds_oversized;
        // SAFETY: we own the `+1` the reservation took; arena methods
        // run on the owner thread.
        unsafe { release_chunk_ref::<A>(self.chunk, true) };
    }
}

impl<A: Allocator + Clone> SliceReservation<A> {
    /// Chunk pointer for accessors that need to mark the chunk pinned
    /// (used by `Arena::alloc_slice_*`).
    pub(crate) const fn chunk_for_pin(&self) -> NonNull<ChunkHeader<A>> {
        self.chunk
    }

    /// Drop-entry pointer (if reserved) for the bump-style slice path
    /// to link into the chunk's drop list manually (replacing
    /// `commit_slice`'s call so we can skip its `inc_ref`).
    pub(crate) const fn entry_for_pin(&self) -> Option<NonNull<DropEntry>> {
        self.entry
    }
}

impl<A: Allocator + Clone> fmt::Debug for SliceReservation<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SliceReservation")
            .field("layout", &self.layout)
            .field("has_drop_entry", &self.entry.is_some())
            .finish_non_exhaustive()
    }
}

/// A deferred-drop guard for a chunk that was evicted from
/// `Arena::current_{local,shared}` during a slow-path chunk acquisition.
///
/// The evicted chunk's `OwnedChunk::drop` may release the chunk's
/// final refcount and run user-supplied `Drop` impls linked into its
/// drop list. Those `Drop` impls can re-enter `Arena::alloc_*`. If the
/// eviction's drop runs *before* the outer caller has finished its
/// post-acquisition `*_unchecked` bump-cursor write, the re-entrant
/// allocation observes the stale cursor on the *new* chunk and bumps
/// it; the outer caller then writes based on the pre-re-entrancy cursor
/// value, overwriting bytes the re-entrant alloc just claimed (or
/// writing past the chunk's `total_size`, which is undefined behavior).
///
/// `try_get_chunk_for_{local,shared}` therefore return this guard to
/// the caller. Callers must bind it to a `let _evicted_guard = ...;`
/// scope that outlives every `*_unchecked` operation on the returned
/// chunk pointer.
#[must_use = "drop only after the outer *_unchecked allocation has run; \
              dropping early can advance the chunk's bump cursor via \
              re-entrant user Drop impls and cause out-of-bounds writes"]
#[expect(
    clippy::redundant_pub_crate,
    reason = "exported under pub(crate) for symmetry with SliceReservation; arena module is private"
)]
pub(crate) struct EvictedChunkGuard<A: Allocator + Clone>(pub(crate) Option<OwnedChunk<A>>);

impl<A: Allocator + Clone> fmt::Debug for EvictedChunkGuard<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvictedChunkGuard").field("has_chunk", &self.0.is_some()).finish()
    }
}

impl<A: Allocator + Clone> Arena<A> {
    /// Reserve space for a DST + `DropEntry`. The bytes are uninitialized;
    /// the caller (a `PendingRc`) is responsible for writing the value
    /// before finalizing.
    #[cfg(feature = "dst")]
    pub(crate) fn try_reserve_dst_with_entry(&self, sharing: ChunkSharing, layout: Layout) -> Result<DstReservation<A>, AllocError> {
        if layout.align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }
        let (chunk, _evicted_guard) = self.try_get_chunk_for(sharing, layout, true, false)?;
        // SAFETY: live chunk.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        // SAFETY: worst-case sizing checked by `try_get_chunk_for`.
        let (entry, value) = unsafe { chunk_ref.alloc_entry_value_slot_unchecked(layout) };
        // SAFETY: chunk flavor matches `sharing` by construction.
        unsafe { self.inc_ref_for(chunk_ref, sharing) };
        self.charge_alloc_stats(layout.size());
        // `_evicted_guard` drops here, AFTER the unchecked bump-cursor
        // write — see `EvictedChunkGuard`.
        Ok((entry, value, chunk))
    }

    /// Internal: copy `slice` into a chunk of the given sharing flavor
    /// and return the raw `(ptr, len)` pair. Each public `_copy_*`
    /// variant wraps the result in its smart pointer type.
    ///
    /// Uses the same fused fast path as [`Self::try_alloc_slice_copy`]
    /// — bump-alloc + copy + `inc_ref` directly, bypassing the
    /// `SliceReservation`/`commit_slice` machinery whose drop-guard
    /// panic-safety isn't needed here (`T: Copy` ⇒ no `Drop`; bump +
    /// memcpy + atomic inc are all straight-line and panic-free).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy
    /// the request.
    #[inline]
    fn try_alloc_slice_copy_inner<T: Copy>(&self, slice: &[T], sharing: ChunkSharing) -> Result<(NonNull<T>, usize), AllocError> {
        let len = slice.len();
        let total = size_of::<T>().checked_mul(len).ok_or(AllocError)?;
        if total > isize::MAX as usize - (align_of::<T>() - 1) {
            return Err(AllocError);
        }
        // SAFETY: `align_of::<T>()` is a power of two; `total` is bounded.
        let layout = unsafe { Layout::from_size_align_unchecked(total, align_of::<T>()) };
        if layout.align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }

        if let Some((chunk_ref, p, _)) = self.try_bump_alloc_in_current(sharing, layout, false, false) {
            let ptr = p.cast::<T>();
            // SAFETY: `ptr` has space for `len` Ts; src and dst do not overlap.
            unsafe { core::ptr::copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), len) };
            // SAFETY: chunk is live; flavor matches `sharing` by construction.
            unsafe { self.inc_ref_for_normal(chunk_ref, sharing) };
            self.charge_alloc_stats(total);
            return Ok((ptr, len));
        }
        self.try_alloc_slice_copy_inner_slow::<T>(slice, sharing, layout, len, total)
    }

    /// Cold slow path of [`Self::try_alloc_slice_copy_inner`].
    #[cold]
    #[inline(never)]
    fn try_alloc_slice_copy_inner_slow<T: Copy>(
        &self,
        slice: &[T],
        sharing: ChunkSharing,
        layout: Layout,
        len: usize,
        total: usize,
    ) -> Result<(NonNull<T>, usize), AllocError> {
        let (chunk, _evicted_guard) = self.try_get_chunk_for(sharing, layout, false, false)?;
        // SAFETY: live chunk.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        // SAFETY: worst-case sizing was guaranteed by `try_get_chunk_for`.
        let ptr = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) }.cast::<T>();

        // SAFETY: `ptr` has space for `len` Ts; src and dst do not overlap.
        unsafe { core::ptr::copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), len) };
        // SAFETY: chunk is live; flavor matches `sharing` by construction.
        unsafe { self.inc_ref_for(chunk_ref, sharing) };
        self.charge_alloc_stats(total);
        // `_evicted_guard` drops here, AFTER `inc_ref_for`.
        Ok((ptr, len))
    }

    /// Internal: allocate a slice of `len` `T`s in a chunk of the given
    /// sharing flavor, initializing element `i` from `f(i)`. If `f`
    /// panics, already-initialized elements are dropped (drop guard) and
    /// the panic propagates. Returns the raw `(ptr, len)` pair.
    ///
    /// Uses the same fused fast path as
    /// [`Self::try_alloc_slice_copy_inner`] — bump-allocate (with
    /// optional `DropEntry` slot) directly via
    /// [`Self::try_bump_alloc_in_current`], skipping the
    /// `SliceReservation`/`commit_slice` machinery. Panic safety for
    /// partial-init is provided by an inline drop guard that also
    /// releases the protective +1 taken on oversized chunks (which
    /// start at refcount 0).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy
    /// the request.
    #[expect(clippy::inline_always, reason = "hot helper; bench callers expect this fully inlined")]
    #[inline(always)]
    fn try_alloc_slice_clone_inner<T: Clone>(&self, src: &[T], sharing: ChunkSharing) -> Result<(NonNull<T>, usize), AllocError> {
        let len = src.len();
        let (chunk_ref, ptr, entry_slot, holds_oversized, _evicted_guard) = self.acquire_slice_slot::<T>(len, sharing)?;

        // Drop guard: on panic in `T::clone`, drops already-init `T`s
        // and releases the protective +1 taken by `acquire_slice_slot`.
        let mut guard = SliceInitFailGuard::<T, A> {
            ptr: ptr.as_ptr(),
            len: 0,
            chunk: chunk_ref.as_non_null(),
            sharing,
            holds_oversized,
        };
        // Iter-based clone loop. Compiles down to `copy_nonoverlapping`
        // for `T: Copy` thanks to LLVM's auto-vectorization of
        // `iter().cloned()`. Crucially this avoids the
        // `iter.next().expect().clone()` closure pattern (used by the
        // older `alloc_slice_clone` body), which adds an
        // `Option::None`-branch per element that the predictor
        // mispredicted once per outer call (≈ 1000 mispredicts per
        // 1000-call bench).
        // SAFETY: `ptr.add(i)` is in-bounds; `guard` covers panics.
        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            reason = "raw write + pointer arithmetic share the in-bounds invariant"
        )]
        // SAFETY: see comment above.
        unsafe {
            for (i, val) in src.iter().cloned().enumerate() {
                ptr.as_ptr().add(i).write(val);
                guard.len += 1;
            }
        }
        core::mem::forget(guard);

        self.commit_slice_init::<T>(chunk_ref, entry_slot, len, sharing, holds_oversized);
        Ok((ptr, len))
    }

    /// Internal: allocate a slice of `len` `T`s in a chunk of the given
    /// sharing flavor, initializing element `i` from `f(i)`. If `f`
    /// panics, already-initialized elements are dropped (drop guard) and
    /// the panic propagates. Returns the raw `(ptr, len)` pair.
    ///
    /// Uses the same fused fast path as
    /// [`Self::try_alloc_slice_copy_inner`] — bump-allocate (with
    /// optional `DropEntry` slot) directly via
    /// [`Self::try_bump_alloc_in_current`], skipping the
    /// `SliceReservation`/`commit_slice` machinery. Panic safety for
    /// partial-init is provided by an inline drop guard that also
    /// releases the protective +1 taken on oversized chunks (which
    /// start at refcount 0).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy
    /// the request.
    #[expect(clippy::inline_always, reason = "hot helper; bench callers expect this fully inlined")]
    #[inline(always)]
    fn try_alloc_slice_fill_with_inner<T, F: FnMut(usize) -> T>(
        &self,
        len: usize,
        mut f: F,
        sharing: ChunkSharing,
    ) -> Result<(NonNull<T>, usize), AllocError> {
        let (chunk_ref, ptr, entry_slot, holds_oversized, _evicted_guard) = self.acquire_slice_slot::<T>(len, sharing)?;

        // Drop guard: on panic in `f`, drop already-init `T`s and
        // release the protective +1 taken by `acquire_slice_slot`.
        // On success, the guard is `mem::forget`-ed.
        let mut guard = SliceInitFailGuard::<T, A> {
            ptr: ptr.as_ptr(),
            len: 0,
            chunk: chunk_ref.as_non_null(),
            sharing,
            holds_oversized,
        };
        // SAFETY: `ptr.add(i)` is in-bounds within the reservation;
        // a panic in `f(i)` is intercepted by `guard`.
        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            reason = "raw write + pointer arithmetic share the in-bounds invariant"
        )]
        // SAFETY: see comment above.
        unsafe {
            for i in 0..len {
                ptr.as_ptr().add(i).write(f(i));
                guard.len += 1;
            }
        }
        core::mem::forget(guard);

        self.commit_slice_init::<T>(chunk_ref, entry_slot, len, sharing, holds_oversized);
        Ok((ptr, len))
    }

    /// Acquire space for a `[T; len]` slice via the fused fast path
    /// (or its slow-path fallback), returning the chunk handle, value
    /// pointer, optional drop-entry slot (only for `T: Drop`), and a
    /// flag indicating whether the chunk is oversized (in which case
    /// the helper has taken a protective +1 on the chunk's refcount,
    /// to be released on init-panic via `SliceInitFailGuard` or
    /// transferred to the smart pointer via [`Self::commit_slice_init`]).
    ///
    /// Shared by [`Self::try_alloc_slice_clone_inner`],
    /// [`Self::try_alloc_slice_fill_with_inner`], and
    /// [`Self::try_reserve_uninit_slice`].
    #[expect(
        clippy::inline_always,
        reason = "hot helper; inlining critical for fast-path bump+slot construction"
    )]
    #[inline(always)]
    /// Acquire a (chunk, value-slot, optional drop-entry) triple for a
    /// slice of `len` `T`s.
    ///
    /// Returns the [`EvictedChunkGuard`] alongside the slot so the
    /// caller can keep it alive until **after** they have committed
    /// the slice (linked the drop entry, called `inc_ref_for`). See
    /// `try_reserve_and_init_aligned` for why this lifetime extension
    /// is required: re-entrant `Drop` impls fired during the guard
    /// drop can re-enter the arena and, with `pin_for_bump = false`,
    /// evict and **free** the freshly-installed chunk before the
    /// caller's commit work runs.
    #[expect(
        clippy::type_complexity,
        reason = "internal slot-acquisition return shape; collapsing into a struct hurts inlining and codegen"
    )]
    fn acquire_slice_slot<T>(
        &self,
        len: usize,
        sharing: ChunkSharing,
    ) -> Result<(ChunkRef<'_, A>, NonNull<T>, Option<NonNull<DropEntry>>, bool, EvictedChunkGuard<A>), AllocError> {
        let needs_drop = core::mem::needs_drop::<T>();
        let total = size_of::<T>().checked_mul(len).ok_or(AllocError)?;
        if total > isize::MAX as usize - (align_of::<T>() - 1) {
            return Err(AllocError);
        }
        // SAFETY: `align_of::<T>()` is a power of two; `total` is bounded.
        let layout = unsafe { Layout::from_size_align_unchecked(total, align_of::<T>()) };
        if layout.align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }

        // Fast path: the bump fit-check succeeds. Returns
        // `holds_oversized=false` and an empty `EvictedChunkGuard` —
        // both inhabited by zero-cost `None` discriminants that LLVM
        // promotes to compile-time constants in the caller.
        if let Some((chunk_ref, p, entry_slot)) = self.try_bump_alloc_in_current(sharing, layout, needs_drop, false) {
            // Take a protective +1 BEFORE returning so the init loop's
            // user code (`T::clone()` / `f(i)`) can re-enter the arena
            // and evict the slot without the chunk's refcount falling
            // to zero. See `acquire_slice_slot_slow` for the full
            // commentary; the successful-commit path transfers this
            // +1 to the smart pointer (so `commit_slice_init` does
            // NOT add another inc), and `SliceInitFailGuard` releases
            // it on panic.
            //
            // SAFETY: chunk is `Normal` (came from
            // `try_bump_alloc_in_current` which never returns Oversized);
            // flavor matches `sharing`.
            unsafe { self.inc_ref_for_normal(chunk_ref, sharing) };
            return Ok((chunk_ref, p.cast::<T>(), entry_slot, false, EvictedChunkGuard(None)));
        }
        self.acquire_slice_slot_slow::<T>(sharing, layout, needs_drop)
    }

    /// Cold slow path of [`Self::acquire_slice_slot`].
    #[cold]
    #[inline(never)]
    #[expect(
        clippy::type_complexity,
        reason = "internal slot-acquisition return shape; collapsing into a struct hurts inlining and codegen"
    )]
    fn acquire_slice_slot_slow<T>(
        &self,
        sharing: ChunkSharing,
        layout: Layout,
        needs_drop: bool,
    ) -> Result<(ChunkRef<'_, A>, NonNull<T>, Option<NonNull<DropEntry>>, bool, EvictedChunkGuard<A>), AllocError> {
        let (chunk, evicted_guard) = self.try_get_chunk_for(sharing, layout, needs_drop, false)?;
        // SAFETY: live chunk.
        let cr = unsafe { ChunkRef::<A>::new(chunk) };
        let (ptr, entry_slot) = if needs_drop {
            // SAFETY: worst-case sizing covered.
            let (e, p) = unsafe { cr.alloc_entry_value_slot_unchecked(layout) };
            (p.cast::<T>(), Some(e))
        } else {
            // SAFETY: worst-case sizing covered.
            let p = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) }.cast::<T>();
            (p, None)
        };

        // Take a protective +1 BEFORE returning. The init loop in the
        // caller can run user code (`T::clone()` / `f(i)`) which may
        // re-enter and evict the slot; that eviction would otherwise
        // drop the chunk's refcount to zero (Normal: the slot's
        // transient +1 is the only ref; Oversized: refcount started at
        // 0). The protective +1 keeps the chunk alive across user code;
        // the init guard releases it on panic, and successful commit
        // simply transfers it to the smart pointer (no extra
        // `inc_ref_for` call needed).
        let holds_oversized = cr.size_class() == ChunkSizeClass::Oversized;
        if holds_oversized {
            // SAFETY: live chunk, flavor matches sharing.
            // Oversized: full atomic-or-cell inc (no deferred-reconciliation
            // pre-payment to draw against; refcount was 0).
            unsafe { self.inc_ref_for(cr, sharing) };
        } else {
            // SAFETY: live chunk, flavor matches sharing; chunk is
            // Normal so the deferred-reconciliation `arcs_issued`
            // path is safe (LARGE pre-payment is already in place
            // for Shared, slot's transient +1 for Local).
            unsafe { self.inc_ref_for_normal(cr, sharing) };
        }

        Ok((cr, ptr, entry_slot, holds_oversized, evicted_guard))
    }

    /// Companion to [`Self::acquire_slice_slot`]: link the slice's
    /// drop entry (if registered) and apply the smart-pointer
    /// refcount accounting after the init loop has completed
    /// successfully.
    #[expect(clippy::inline_always, reason = "hot helper; inlining critical for fast-path commit")]
    #[inline(always)]
    fn commit_slice_init<T>(
        &self,
        chunk_ref: ChunkRef<'_, A>,
        entry_slot: Option<NonNull<DropEntry>>,
        len: usize,
        sharing: ChunkSharing,
        holds_oversized: bool,
    ) {
        // Init succeeded: link the drop entry now (so chunk teardown
        // calls `slice_drop_shim::<T>` with the actual element count
        // — for `T: !Drop` the entry slot was never reserved).
        if let Some(entry) = entry_slot {
            // SAFETY: entry slot writable; chunk live.
            unsafe { chunk_ref.link_drop_entry(entry, slice_drop_shim::<T>, len) };
        }
        self.finalize_slice_alloc::<T>(chunk_ref, len, sharing, holds_oversized);
    }

    /// Refcount + stats finalization shared by every smart-pointer
    /// slice alloc path.
    ///
    /// Call AFTER linking any required drop entry. For oversized
    /// chunks, the protective +1 taken in [`Self::acquire_slice_slot`]
    /// transfers to the smart pointer (no extra inc); for Normal
    /// chunks, take a fresh +1 alongside the slot's transient hold.
    #[expect(clippy::inline_always, reason = "hot helper; inlining critical for fast-path commit")]
    #[inline(always)]
    fn finalize_slice_alloc<T>(&self, _chunk_ref: ChunkRef<'_, A>, len: usize, _sharing: ChunkSharing, _holds_oversized: bool) {
        // The smart-pointer's `+1` is already on the chunk: it was
        // taken in `acquire_slice_slot{,_slow}` BEFORE the init loop
        // ran, to keep the chunk alive across user-supplied
        // `T::clone()` / `f(i)` code that may re-enter and evict the
        // slot. The `SliceInitFailGuard` releases that +1 on panic;
        // success simply transfers it to the smart pointer here, so
        // there is no additional refcount bump to perform.
        self.charge_alloc_stats(len.saturating_mul(size_of::<T>()));
    }

    /// Copy `slice` into the arena, returning an immutable smart pointer.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_copy_rc`] for a fallible variant.
    #[inline]
    pub fn alloc_slice_copy_rc<T: Copy>(&self, slice: impl AsRef<[T]>) -> Rc<[T], A> {
        self.try_alloc_slice_copy_rc(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Copy `slice` into the arena and return an immutable [`Rc`] smart pointer,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator fails.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_slice_copy_rc<T: Copy>(&self, slice: impl AsRef<[T]>) -> Result<Rc<[T], A>, AllocError> {
        let (ptr, len) = self.try_alloc_slice_copy_inner::<T>(slice.as_ref(), ChunkSharing::Local)?;
        // SAFETY: ptr valid; refcount bumped above.
        Ok(unsafe { Rc::from_raw_slice(ptr, len) })
    }

    /// Clone every element of `slice` into the arena, returning an
    /// immutable smart pointer.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_clone_rc`] for a fallible variant.
    #[inline]
    pub fn alloc_slice_clone_rc<T: Clone>(&self, slice: impl AsRef<[T]>) -> Rc<[T], A> {
        self.try_alloc_slice_clone_rc(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Clone every element of `slice` into the arena and return an immutable
    /// [`Rc`] smart pointer, returning Err([`AllocError`]) instead of
    /// panicking if the backing allocator cannot satisfy the request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// Cannot panic from the iterator length mismatch — `slice.len()`
    /// matches the iterator's length by construction. May panic if a
    /// `T::clone()` impl panics; in that case already-initialized
    /// elements are dropped via the slice init guard before the panic
    /// propagates.
    #[inline]
    pub fn try_alloc_slice_clone_rc<T: Clone>(&self, slice: impl AsRef<[T]>) -> Result<Rc<[T], A>, AllocError> {
        let (ptr, len) = self.try_alloc_slice_clone_inner::<T>(slice.as_ref(), ChunkSharing::Local)?;
        // SAFETY: ptr valid; refcount bumped above.
        Ok(unsafe { Rc::from_raw_slice(ptr, len) })
    }

    /// Allocate a slice of `len` elements, with element `i` produced by
    /// `f(i)`.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_with_rc`] for a fallible variant.
    ///
    /// If `f` panics, already-initialized elements are dropped (drop guard) and the
    /// panic propagates.
    #[inline]
    pub fn alloc_slice_fill_with_rc<T, F: FnMut(usize) -> T>(&self, len: usize, f: F) -> Rc<[T], A> {
        self.try_alloc_slice_fill_with_rc(len, f).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate a slice of `len` elements with element `i` produced by `f(i)`,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator fails. If `f` panics, already-initialized
    /// elements are dropped and the panic propagates.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_slice_fill_with_rc<T, F: FnMut(usize) -> T>(&self, len: usize, f: F) -> Result<Rc<[T], A>, AllocError> {
        let (ptr, len) = self.try_alloc_slice_fill_with_inner::<T, F>(len, f, ChunkSharing::Local)?;
        // SAFETY: ptr valid; refcount bumped above.
        Ok(unsafe { Rc::from_raw_slice(ptr, len) })
    }

    /// Allocate a slice and fill it with values pulled from `iter`.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_iter_rc`] for a fallible variant.
    ///
    /// May also panic if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[inline]
    pub fn alloc_slice_fill_iter_rc<T, I>(&self, iter: I) -> Rc<[T], A>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        self.try_alloc_slice_fill_iter_rc(iter).unwrap_or_else(|_| panic_alloc())
    }

    /// Allocate a slice and fill it with values pulled from `iter`,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator fails.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[inline]
    pub fn try_alloc_slice_fill_iter_rc<T, I>(&self, iter: I) -> Result<Rc<[T], A>, AllocError>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut iter = iter.into_iter();
        let len = iter.len();
        self.try_alloc_slice_fill_with_rc(len, |_| iter.next().expect("iterator shorter than ExactSizeIterator len"))
    }

    /// Copy `slice` into a `Shared`-flavor chunk and return an [`Arc`]
    /// smart pointer safe for cross-thread sharing.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_copy_arc`] for a fallible variant.
    #[inline]
    pub fn alloc_slice_copy_arc<T: Copy + Send + Sync>(&self, slice: impl AsRef<[T]>) -> Arc<[T], A>
    where
        A: Send + Sync,
    {
        self.try_alloc_slice_copy_arc(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Copy `slice` into a `Shared`-flavor chunk and return an [`Arc`]
    /// smart pointer, returning Err([`AllocError`]) instead of panicking if the
    /// backing allocator fails.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_slice_copy_arc<T: Copy + Send + Sync>(&self, slice: impl AsRef<[T]>) -> Result<Arc<[T], A>, AllocError>
    where
        A: Send + Sync,
    {
        let (ptr, len) = self.try_alloc_slice_copy_inner::<T>(slice.as_ref(), ChunkSharing::Shared)?;
        // SAFETY: fully initialized slice in Shared chunk with Send+Sync bounds
        Ok(unsafe { Arc::from_raw_slice(ptr, len) })
    }

    /// Clone every element of `slice` into a `Shared`-flavor chunk and
    /// return an [`Arc`] smart pointer safe for cross-thread sharing.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_clone_arc`] for a fallible variant.
    ///
    /// May panic if `T::clone` panics; already-cloned elements are dropped before the
    /// panic propagates.
    #[inline]
    pub fn alloc_slice_clone_arc<T: Clone + Send + Sync>(&self, slice: impl AsRef<[T]>) -> Arc<[T], A>
    where
        A: Send + Sync,
    {
        self.try_alloc_slice_clone_arc(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_clone_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// May panic if `T::clone` panics; already-cloned elements are
    /// dropped before the panic propagates.
    #[inline]
    pub fn try_alloc_slice_clone_arc<T: Clone + Send + Sync>(&self, slice: impl AsRef<[T]>) -> Result<Arc<[T], A>, AllocError>
    where
        A: Send + Sync,
    {
        let (ptr, len) = self.try_alloc_slice_clone_inner::<T>(slice.as_ref(), ChunkSharing::Shared)?;
        // SAFETY: ptr valid; refcount bumped above.
        Ok(unsafe { Arc::from_raw_slice(ptr, len) })
    }

    /// Allocate a slice of `len` elements in a `Shared`-flavor chunk,
    /// with element `i` produced by `f(i)`, returning an [`Arc`]
    /// smart pointer safe for cross-thread sharing.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_with_arc`] for a fallible variant.
    ///
    /// If `f` panics, already-initialized elements are dropped (drop guard) and the
    /// panic propagates.
    #[inline]
    pub fn alloc_slice_fill_with_arc<T, F>(&self, len: usize, f: F) -> Arc<[T], A>
    where
        T: Send + Sync,
        F: FnMut(usize) -> T,
        A: Send + Sync,
    {
        self.try_alloc_slice_fill_with_arc(len, f).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_fill_with_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// If `f` panics, already-initialized elements are dropped and the
    /// panic propagates.
    #[inline]
    pub fn try_alloc_slice_fill_with_arc<T, F>(&self, len: usize, f: F) -> Result<Arc<[T], A>, AllocError>
    where
        T: Send + Sync,
        F: FnMut(usize) -> T,
        A: Send + Sync,
    {
        let (ptr, len) = self.try_alloc_slice_fill_with_inner::<T, F>(len, f, ChunkSharing::Shared)?;
        // SAFETY: fully initialized slice in Shared chunk with Send+Sync bounds
        Ok(unsafe { Arc::from_raw_slice(ptr, len) })
    }

    /// Allocate a slice in a `Shared`-flavor chunk and fill it with
    /// values pulled from `iter`, returning an [`Arc`] smart pointer safe
    /// for cross-thread sharing.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_iter_arc`] for a fallible variant.
    ///
    /// May also panic if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[inline]
    pub fn alloc_slice_fill_iter_arc<T, I>(&self, iter: I) -> Arc<[T], A>
    where
        T: Send + Sync,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
        A: Send + Sync,
    {
        self.try_alloc_slice_fill_iter_arc(iter).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_fill_iter_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[inline]
    pub fn try_alloc_slice_fill_iter_arc<T, I>(&self, iter: I) -> Result<Arc<[T], A>, AllocError>
    where
        T: Send + Sync,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
        A: Send + Sync,
    {
        let mut iter = iter.into_iter();
        let len = iter.len();
        self.try_alloc_slice_fill_with_arc(len, |_| iter.next().expect("iterator shorter than ExactSizeIterator len"))
    }

    // ---- _box slice constructors (T: ?Sized via dst feature) -------------

    /// Copy `slice` into the arena and return an [`Box<[T], A>`](crate::Box)
    /// — an owned, mutable smart pointer whose `Drop` runs `T::drop` on each
    /// element immediately when the smart pointer is dropped.
    ///
    /// Available only with the `dst` Cargo feature, which pulls in the
    /// `ptr_meta` crate to polyfill stable `core::ptr::metadata`.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_copy_box`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_slice_copy_box<T: Copy>(&self, slice: impl AsRef<[T]>) -> Box<[T], A> {
        self.try_alloc_slice_copy_box(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_copy_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_slice_copy_box<T: Copy>(&self, slice: impl AsRef<[T]>) -> Result<Box<[T], A>, AllocError> {
        let (ptr, len) = self.try_alloc_slice_copy_inner::<T>(slice.as_ref(), ChunkSharing::Local)?;
        let fat = NonNull::slice_from_raw_parts(ptr, len);
        // SAFETY: fully initialized slice in Local chunk with refcount already bumped
        Ok(unsafe { Box::from_raw_unsized(fat) })
    }

    /// Clone every element of `slice` into the arena and return an
    /// owned, mutable [`Box<[T], A>`](crate::Box).
    ///
    /// Available only with the `dst` Cargo feature.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_clone_box`] for a fallible variant.
    ///
    /// May panic if `T::clone` panics; already-cloned elements are dropped before the
    /// panic propagates.
    #[inline]
    pub fn alloc_slice_clone_box<T: Clone>(&self, slice: impl AsRef<[T]>) -> Box<[T], A> {
        self.try_alloc_slice_clone_box(slice).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_clone_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// May panic if `T::clone` panics; already-cloned elements are
    /// dropped before the panic propagates.
    #[inline]
    pub fn try_alloc_slice_clone_box<T: Clone>(&self, slice: impl AsRef<[T]>) -> Result<Box<[T], A>, AllocError> {
        let (ptr, len) = self.try_alloc_slice_clone_inner::<T>(slice.as_ref(), ChunkSharing::Local)?;
        let fat = NonNull::slice_from_raw_parts(ptr, len);
        // SAFETY: fully initialized slice in Local chunk; refcount bumped.
        Ok(unsafe { Box::from_raw_unsized(fat) })
    }

    /// Allocate a slice of `len` elements, with element `i` produced by
    /// `f(i)`, returning an owned, mutable [`Box<[T], A>`](crate::Box).
    ///
    /// Available only with the `dst` Cargo feature.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_with_box`] for a fallible variant.
    ///
    /// If `f` panics, already-initialized elements are dropped (drop guard) and the
    /// panic propagates.
    #[inline]
    pub fn alloc_slice_fill_with_box<T, F: FnMut(usize) -> T>(&self, len: usize, f: F) -> Box<[T], A> {
        self.try_alloc_slice_fill_with_box(len, f).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_fill_with_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// If `f` panics, already-initialized elements are dropped and the
    /// panic propagates.
    #[inline]
    pub fn try_alloc_slice_fill_with_box<T, F: FnMut(usize) -> T>(&self, len: usize, f: F) -> Result<Box<[T], A>, AllocError> {
        let (ptr, len) = self.try_alloc_slice_fill_with_inner::<T, F>(len, f, ChunkSharing::Local)?;
        let fat = NonNull::slice_from_raw_parts(ptr, len);
        // SAFETY: fully initialized slice in Local chunk with refcount already bumped
        Ok(unsafe { Box::from_raw_unsized(fat) })
    }

    /// Allocate a slice and fill it with values pulled from `iter`,
    /// returning an owned, mutable [`Box<[T], A>`](crate::Box).
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if the `align_of::<T>()` is at least 64 KiB.
    /// Use [`Self::try_alloc_slice_fill_iter_box`] for a fallible variant.
    ///
    /// May also panic if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[inline]
    pub fn alloc_slice_fill_iter_box<T, I>(&self, iter: I) -> Box<[T], A>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        self.try_alloc_slice_fill_iter_box(iter).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_slice_fill_iter_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    #[inline]
    pub fn try_alloc_slice_fill_iter_box<T, I>(&self, iter: I) -> Result<Box<[T], A>, AllocError>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut iter = iter.into_iter();
        let len = iter.len();
        self.try_alloc_slice_fill_with_box(len, |_| iter.next().expect("iterator shorter than ExactSizeIterator len"))
    }

    /// Reserve `layout`-sized space for a slice in a chunk of the given
    /// `sharing` flavor, optionally also reserving a `DropEntry` slot
    /// just before the slice. Returns a `SliceReservation` that the
    /// caller must finalize via [`Self::commit_slice`] AFTER successful
    /// element initialization. Dropping the reservation without
    /// committing leaks the bump bytes but is sound.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] on overflow, alignment-too-large, or any
    /// underlying allocator failure during chunk acquisition.
    #[inline]
    pub(crate) fn reserve_slice<T>(
        &self,
        len: usize,
        sharing: ChunkSharing,
        register_drop: bool,
        pin_for_bump: bool,
    ) -> Result<SliceReservation<A>, AllocError> {
        let elem_size = size_of::<T>();
        let total = elem_size.checked_mul(len).ok_or(AllocError)?;
        if total > isize::MAX as usize - (align_of::<T>() - 1) {
            return Err(AllocError);
        }
        // SAFETY: align_of::<T>() is a power of two; total is bounded above.
        let layout = unsafe { Layout::from_size_align_unchecked(total, align_of::<T>()) };
        if layout.align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }
        let (chunk, evicted_guard) = self.try_get_chunk_for(sharing, layout, register_drop, pin_for_bump)?;
        // SAFETY: live chunk.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        // Take a protective `+1` BEFORE returning. Between `reserve_slice`
        // and `commit_slice`, the caller writes data into the
        // reservation; that write may invoke user code (the `Vec`'s
        // backing-buffer `Drop` calls `Allocator::deallocate` →
        // `release_chunk_ref_local` on its OWN chunk, whose teardown
        // can run user `T::drop` impls that re-enter the arena). A
        // re-entrant alloc with `pin_for_bump = false` can evict the
        // reservation chunk; without our `+1`, the chunk would be
        // freed before `commit_slice` runs, causing UAF.
        //
        // The protective `+1` always lives here (not just for
        // Oversized), and is transferred to the smart pointer by
        // `commit_slice` (no second inc). On `Drop` (panic before
        // commit), `SliceReservation` releases it via `release_chunk_ref`.
        let holds_oversized = chunk_ref.size_class() == ChunkSizeClass::Oversized;
        if holds_oversized {
            // SAFETY: chunk flavor matches `sharing`.
            unsafe { self.inc_ref_for(chunk_ref, sharing) };
        } else {
            // SAFETY: chunk live + Normal, flavor matches `sharing`.
            unsafe { self.inc_ref_for_normal(chunk_ref, sharing) };
        }
        if register_drop {
            // SAFETY: worst-case sizing checked by `try_get_chunk_for`.
            let (entry, ptr) = unsafe { chunk_ref.alloc_entry_value_slot_unchecked(layout) };
            Ok(SliceReservation {
                chunk,
                ptr,
                entry: Some(entry),
                layout,
                holds_oversized,
                evicted_guard,
            })
        } else {
            // SAFETY: worst-case sizing fits.
            let ptr = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) };
            Ok(SliceReservation {
                chunk,
                ptr,
                entry: None,
                layout,
                holds_oversized,
                evicted_guard,
            })
        }
    }

    /// Commit a previously-reserved slice once initialization has
    /// succeeded: link the drop entry (if reserved) with the actual
    /// element count, transfer the reservation's protective `+1` to
    /// the smart pointer, and account the allocation in stats.
    ///
    /// # Safety
    ///
    /// Caller must have written `len` valid `T` values starting at
    /// `reservation.ptr` (typed as `T`). `len` must be the same length
    /// originally requested in [`Self::reserve_slice`].
    #[inline]
    pub(crate) unsafe fn commit_slice<T>(&self, reservation: SliceReservation<A>, len: usize) {
        let chunk = reservation.chunk;
        let alloc_size = reservation.layout.size();
        let entry = reservation.entry;
        // SAFETY: chunk alive (held by the reservation's protective +1).
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        if let Some(entry) = entry {
            // SAFETY: entry slot writable; chunk live.
            unsafe { chunk_ref.link_drop_entry(entry, slice_drop_shim::<T>, len) };
        }
        // The reservation already took a protective `+1` in
        // `reserve_slice`; transfer it to the smart pointer (no fresh
        // `inc_ref_for`) by suppressing the reservation's `Drop`.
        // We must still drop `evicted_guard` so it releases its
        // hold on any chunk that was evicted from `current_*` during
        // reservation.
        //
        // SAFETY: `reservation` is owned by us; raw-read the field
        // before `mem::forget` to extract the guard's bits, then
        // suppress the reservation's `Drop` (which would otherwise
        // release the protective +1 we want to transfer).
        let evicted_guard = unsafe { core::ptr::read(&raw const reservation.evicted_guard) };
        core::mem::forget(reservation);
        drop(evicted_guard);
        self.charge_alloc_stats(alloc_size);
    }

    /// Bump-allocate `layout` from a chunk of the requested sharing flavor
    /// and bump the chunk refcount by 1 for the upcoming string smart pointer,
    /// returning `AllocError` instead of panicking on allocator failure.
    /// The returned bytes are uninitialized; callers wrap the pointer in
    /// a smart pointer that releases the refcount on drop.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy
    /// the request.
    #[expect(
        clippy::inline_always,
        reason = "callers supply constant `sharing`; const-folding the dispatcher chain requires inlining"
    )]
    #[inline(always)]
    pub(crate) fn try_bump_alloc_for_str(&self, layout: Layout, sharing: ChunkSharing) -> Result<NonNull<u8>, AllocError> {
        if let Some((chunk_ref, ptr, _)) = self.try_bump_alloc_in_current(sharing, layout, false, false) {
            // SAFETY: chunk flavor matches `sharing`.
            unsafe { self.inc_ref_for_normal(chunk_ref, sharing) };
            self.charge_alloc_stats(layout.size());
            return Ok(ptr);
        }
        self.try_bump_alloc_for_str_slow(layout, sharing)
    }

    /// Cold slow path of [`Self::try_bump_alloc_for_str`].
    #[cold]
    #[inline(never)]
    fn try_bump_alloc_for_str_slow(&self, layout: Layout, sharing: ChunkSharing) -> Result<NonNull<u8>, AllocError> {
        let (chunk, _evicted_guard) = self.try_get_chunk_for(sharing, layout, false, false)?;
        // SAFETY: live chunk.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        // SAFETY: layout fits.
        let ptr = unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) };
        // SAFETY: chunk flavor matches `sharing`.
        unsafe { self.inc_ref_for(chunk_ref, sharing) };
        self.charge_alloc_stats(layout.size());
        Ok(ptr)
    }

    #[cfg(feature = "builders")]
    /// Create a new, empty growable [`String`](crate::builders::String) backed by this
    /// arena. No allocation is performed until the first push.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let mut s = arena.alloc_string();
    /// s.push_str("hello");
    /// assert_eq!(&*s, "hello");
    /// ```
    #[must_use]
    #[inline]
    pub const fn alloc_string(&self) -> String<'_, A> {
        String::new_in(self)
    }

    #[cfg(feature = "builders")]
    /// Create a new growable [`String`](crate::builders::String) backed by this arena, with
    /// at least `cap` bytes of pre-allocated capacity.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use
    /// [`Self::try_alloc_string_with_capacity`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let mut s = arena.alloc_string_with_capacity(64);
    /// s.push_str("preallocated");
    /// assert!(s.capacity() >= 64);
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc_string_with_capacity(&self, cap: usize) -> String<'_, A> {
        String::with_capacity_in(cap, self)
    }

    #[cfg(feature = "builders")]
    /// Fallible variant of [`Self::alloc_string_with_capacity`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_string_with_capacity(&self, cap: usize) -> Result<String<'_, A>, AllocError> {
        String::try_with_capacity_in(cap, self)
    }

    #[cfg(feature = "builders")]
    /// Create a new, empty growable [`Vec`](crate::builders::Vec) backed by this arena.
    /// No allocation is performed until the first push.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let mut v = arena.alloc_vec::<u32>();
    /// v.push(1);
    /// v.push(2);
    /// assert_eq!(v.as_slice(), &[1, 2]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn alloc_vec<T>(&self) -> Vec<'_, T, A> {
        Vec::new_in(self)
    }

    #[cfg(feature = "builders")]
    /// Create a new growable [`Vec`](crate::builders::Vec) backed by this arena, with
    /// capacity for at least `cap` elements pre-allocated.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_vec_with_capacity`] for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let mut v = arena.alloc_vec_with_capacity::<u32>(100);
    /// for i in 0..50 { v.push(i); }
    /// assert!(v.capacity() >= 100);
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc_vec_with_capacity<T>(&self, cap: usize) -> Vec<'_, T, A> {
        Vec::with_capacity_in(cap, self)
    }

    #[cfg(feature = "builders")]
    /// Fallible variant of [`Self::alloc_vec_with_capacity`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_vec_with_capacity<T>(&self, cap: usize) -> Result<Vec<'_, T, A>, AllocError> {
        Vec::try_with_capacity_in(cap, self)
    }

    #[cfg(feature = "builders")]
    /// Internal grow helper used by [`String`](crate::builders::String).
    ///
    /// # Safety
    ///
    /// Caller must follow the same rules as `Allocator::grow`: `ptr`
    /// must have come from a previous allocation through this arena
    /// with size `old_size` and alignment `align_of::<usize>()`;
    /// `new_size >= old_size`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails on the
    /// slow-path relocation or arithmetic overflows.
    pub(crate) unsafe fn grow_for_string(&self, ptr: NonNull<u8>, old_size: usize, new_size: usize) -> Result<NonNull<u8>, AllocError> {
        // SAFETY: caller's contract.
        let chunk: NonNull<ChunkHeader<A>> = unsafe { header_for(ptr) };
        // SAFETY: chunk alive (caller holds an allocation in it).
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        let extra = new_size - old_size;
        if chunk_ref.try_grow_in_place(ptr, old_size, extra) {
            self.charge_alloc_stats(extra);
            return Ok(ptr);
        }

        debug_assert!(isize::try_from(new_size).is_ok());
        // SAFETY: align_of::<usize>() is a power of two.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, align_of::<usize>()) };
        let new_ptr = self.try_bump_alloc_for_str(new_layout, ChunkSharing::Local)?;
        // SAFETY: source initialized; dst non-overlapping.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), old_size);
        }
        if chunk_ref.dec_ref() {
            // SAFETY: refcount=0; we own the old chunk.
            unsafe { teardown_chunk(chunk, true) };
        }
        self.bump_relocation();
        Ok(new_ptr)
    }

    /// Reserve uninitialized space for a single `T` and return a pointer
    /// typed as `MaybeUninit<T>`. If `T` needs drop, also reserves a
    /// `DropEntry` slot linked with a no-op shim — `assume_init` will
    /// later rewrite the shim to `drop_shim::<T>`. Bumps the chunk
    /// refcount by 1 for the upcoming smart pointer.
    #[inline]
    fn try_reserve_uninit<T>(&self, sharing: ChunkSharing) -> Result<NonNull<MaybeUninit<T>>, AllocError> {
        if Layout::new::<T>().align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }
        // SAFETY: alignment guard above discharges the inner fn's precondition.
        unsafe { self.try_reserve_uninit_aligned::<T>(sharing) }
    }

    /// # Safety
    ///
    /// `Layout::new::<T>().align() <= CHUNK_ALIGN`.
    #[inline]
    unsafe fn try_reserve_uninit_aligned<T>(&self, sharing: ChunkSharing) -> Result<NonNull<MaybeUninit<T>>, AllocError> {
        let needs_drop = core::mem::needs_drop::<T>();
        let layout = Layout::new::<T>();

        if let Some((chunk_ref, value_u8, entry_ptr)) = self.try_bump_alloc_in_current(sharing, layout, needs_drop, false) {
            if let Some(entry) = entry_ptr {
                // SAFETY: entry slot writable; chunk live. Placeholder shim;
                // assume_init rewrites it to drop_shim::<T> once initialized.
                unsafe { chunk_ref.link_drop_entry(entry, drop_shim::<MaybeUninit<T>>, 0) };
            }
            let value = value_u8.cast::<MaybeUninit<T>>();
            // SAFETY: chunk flavor matches `sharing`.
            unsafe { self.inc_ref_for_normal(chunk_ref, sharing) };
            self.charge_alloc_stats(layout.size());
            return Ok(value);
        }
        // SAFETY: caller's contract.
        unsafe { self.try_reserve_uninit_aligned_slow::<T>(sharing, layout, needs_drop) }
    }

    /// Cold slow path of [`Self::try_reserve_uninit_aligned`].
    ///
    /// # Safety
    ///
    /// Same as [`Self::try_reserve_uninit_aligned`].
    #[cold]
    #[inline(never)]
    unsafe fn try_reserve_uninit_aligned_slow<T>(
        &self,
        sharing: ChunkSharing,
        layout: Layout,
        needs_drop: bool,
    ) -> Result<NonNull<MaybeUninit<T>>, AllocError> {
        let (chunk, _evicted_guard) = self.try_get_chunk_for(sharing, layout, needs_drop, false)?;
        // SAFETY: live chunk.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        let value = if needs_drop {
            // SAFETY: worst-case sizing guaranteed.
            let (entry, value) = unsafe { ChunkHeader::<A>::alloc_with_drop_entry_unchecked::<T>(chunk) };
            // SAFETY: entry slot writable; chunk live.
            unsafe { chunk_ref.link_drop_entry(entry, drop_shim::<MaybeUninit<T>>, 0) };
            value
        } else {
            // SAFETY: worst-case sizing guaranteed.
            unsafe { ChunkHeader::<A>::alloc_unchecked(chunk, layout) }.cast::<T>()
        };
        // SAFETY: chunk flavor matches `sharing`.
        unsafe { self.inc_ref_for(chunk_ref, sharing) };
        self.charge_alloc_stats(layout.size());
        Ok(value.cast::<MaybeUninit<T>>())
    }

    /// Like [`Self::try_reserve_uninit`] but zeroes the value bytes.
    #[inline]
    fn try_reserve_zeroed<T>(&self, sharing: ChunkSharing) -> Result<NonNull<MaybeUninit<T>>, AllocError> {
        let ptr = self.try_reserve_uninit::<T>(sharing)?;
        // SAFETY: reservation owns `size_of::<T>()` bytes.
        unsafe { core::ptr::write_bytes(ptr.as_ptr().cast::<u8>(), 0, size_of::<T>()) };
        Ok(ptr)
    }

    /// Reserve `len` uninitialized `T` slots in a chunk of the given
    /// sharing flavor. If `T` needs drop, links a slice-drop entry with
    /// a no-op shim and `slice_len = len`; `assume_init` rewrites the
    /// shim to `slice_drop_shim::<T>`. Bumps the chunk refcount by 1.
    ///
    /// Uses the same fused fast path as
    /// [`Self::try_alloc_slice_copy_inner`] / `_fill_with_inner`.
    #[expect(clippy::inline_always, reason = "hot helper; bench callers expect this fully inlined")]
    #[inline(always)]
    fn try_reserve_uninit_slice<T>(&self, len: usize, sharing: ChunkSharing) -> Result<(NonNull<MaybeUninit<T>>, usize), AllocError> {
        let (chunk_ref, ptr, entry_slot, holds_oversized, _evicted_guard) = self.acquire_slice_slot::<T>(len, sharing)?;

        // Link drop entry with a placeholder shim (no-op for
        // MaybeUninit<T>); `assume_init` rewrites it to
        // `slice_drop_shim::<T>` once the slice is initialized.
        if let Some(entry) = entry_slot {
            // SAFETY: entry slot writable; chunk live.
            unsafe { chunk_ref.link_drop_entry(entry, slice_drop_shim::<MaybeUninit<T>>, len) };
        }

        self.finalize_slice_alloc::<T>(chunk_ref, len, sharing, holds_oversized);
        Ok((ptr.cast::<MaybeUninit<T>>(), len))
    }

    /// Like [`Self::try_reserve_uninit_slice`] but zeroes the slice bytes.
    #[inline]
    fn try_reserve_zeroed_slice<T>(&self, len: usize, sharing: ChunkSharing) -> Result<(NonNull<MaybeUninit<T>>, usize), AllocError> {
        let (ptr, len) = self.try_reserve_uninit_slice::<T>(len, sharing)?;
        let bytes = len.checked_mul(size_of::<T>()).ok_or(AllocError)?;
        // SAFETY: reservation owns `bytes` bytes.
        unsafe { core::ptr::write_bytes(ptr.as_ptr().cast::<u8>(), 0, bytes) };
        Ok((ptr, len))
    }

    /// Allocate uninitialized space for a `T` and return an
    /// [`Box<MaybeUninit<T>, A>`](crate::Box). The caller must
    /// initialize the value (e.g., via [`MaybeUninit::write`]) before
    /// calling [`Box::<MaybeUninit<T>, A>::assume_init`].
    ///
    /// For types that need drop, this allocation reserves a `DropEntry`
    /// slot in addition to the value bytes. Dropping the box without
    /// `assume_init` is sound (no destructor runs).
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_uninit_box`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_uninit_box<T>(&self) -> Box<MaybeUninit<T>, A> {
        self.try_alloc_uninit_box::<T>().unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_uninit_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_uninit_box<T>(&self) -> Result<Box<MaybeUninit<T>, A>, AllocError> {
        let ptr = self.try_reserve_uninit::<T>(ChunkSharing::Local)?;
        // SAFETY: ptr lives in a Local chunk with refcount bumped by 1.
        Ok(unsafe { Box::from_raw(ptr) })
    }

    /// Like [`Self::alloc_uninit_box`] but the value bytes are zeroed.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_zeroed_box`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_zeroed_box<T>(&self) -> Box<MaybeUninit<T>, A> {
        self.try_alloc_zeroed_box::<T>().unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_zeroed_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_zeroed_box<T>(&self) -> Result<Box<MaybeUninit<T>, A>, AllocError> {
        let ptr = self.try_reserve_zeroed::<T>(ChunkSharing::Local)?;
        // SAFETY: ptr lives in a Local chunk with refcount bumped by 1.
        Ok(unsafe { Box::from_raw(ptr) })
    }

    /// Allocate uninitialized space for a `T` and return an
    /// [`Rc<MaybeUninit<T>, A>`](crate::Rc).
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_uninit_rc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_uninit_rc<T>(&self) -> Rc<MaybeUninit<T>, A> {
        self.try_alloc_uninit_rc::<T>().unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_uninit_rc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_uninit_rc<T>(&self) -> Result<Rc<MaybeUninit<T>, A>, AllocError> {
        let ptr = self.try_reserve_uninit::<T>(ChunkSharing::Local)?;
        // SAFETY: ptr lives in a Local chunk with refcount bumped by 1.
        Ok(unsafe { Rc::from_raw(ptr) })
    }

    /// Like [`Self::alloc_uninit_rc`] but the value bytes are zeroed.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_zeroed_rc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_zeroed_rc<T>(&self) -> Rc<MaybeUninit<T>, A> {
        self.try_alloc_zeroed_rc::<T>().unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_zeroed_rc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_zeroed_rc<T>(&self) -> Result<Rc<MaybeUninit<T>, A>, AllocError> {
        let ptr = self.try_reserve_zeroed::<T>(ChunkSharing::Local)?;
        // SAFETY: ptr lives in a Local chunk with refcount bumped by 1.
        Ok(unsafe { Rc::from_raw(ptr) })
    }

    /// Allocate uninitialized space for a `T` and return an
    /// [`Arc<MaybeUninit<T>, A>`](crate::Arc).
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_uninit_arc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_uninit_arc<T>(&self) -> Arc<MaybeUninit<T>, A>
    where
        A: Send + Sync,
    {
        self.try_alloc_uninit_arc::<T>().unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_uninit_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_uninit_arc<T>(&self) -> Result<Arc<MaybeUninit<T>, A>, AllocError>
    where
        A: Send + Sync,
    {
        let ptr = self.try_reserve_uninit::<T>(ChunkSharing::Shared)?;
        // SAFETY: ptr lives in a Shared chunk with refcount bumped by 1.
        Ok(unsafe { Arc::from_raw(ptr) })
    }

    /// Like [`Self::alloc_uninit_arc`] but the value bytes are zeroed.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_zeroed_arc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_zeroed_arc<T>(&self) -> Arc<MaybeUninit<T>, A>
    where
        A: Send + Sync,
    {
        self.try_alloc_zeroed_arc::<T>().unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_zeroed_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_zeroed_arc<T>(&self) -> Result<Arc<MaybeUninit<T>, A>, AllocError>
    where
        A: Send + Sync,
    {
        let ptr = self.try_reserve_zeroed::<T>(ChunkSharing::Shared)?;
        // SAFETY: ptr lives in a Shared chunk with refcount bumped by 1.
        Ok(unsafe { Arc::from_raw(ptr) })
    }

    /// Allocate `len` uninitialized `T` slots and return an
    /// [`Rc<[MaybeUninit<T>], A>`](crate::Rc).
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_uninit_slice_rc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_uninit_slice_rc<T>(&self, len: usize) -> Rc<[MaybeUninit<T>], A> {
        self.try_alloc_uninit_slice_rc::<T>(len).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_uninit_slice_rc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_uninit_slice_rc<T>(&self, len: usize) -> Result<Rc<[MaybeUninit<T>], A>, AllocError> {
        let (ptr, len) = self.try_reserve_uninit_slice::<T>(len, ChunkSharing::Local)?;
        // SAFETY: ptr/len describe a valid slice reservation in a Local chunk.
        Ok(unsafe { Rc::from_raw_slice(ptr, len) })
    }

    /// Like [`Self::alloc_uninit_slice_rc`] but the slice bytes are zeroed.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_zeroed_slice_rc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_zeroed_slice_rc<T>(&self, len: usize) -> Rc<[MaybeUninit<T>], A> {
        self.try_alloc_zeroed_slice_rc::<T>(len).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_zeroed_slice_rc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_zeroed_slice_rc<T>(&self, len: usize) -> Result<Rc<[MaybeUninit<T>], A>, AllocError> {
        let (ptr, len) = self.try_reserve_zeroed_slice::<T>(len, ChunkSharing::Local)?;
        // SAFETY: ptr/len describe a valid slice reservation in a Local chunk.
        Ok(unsafe { Rc::from_raw_slice(ptr, len) })
    }

    /// Allocate `len` uninitialized `T` slots and return an
    /// [`Arc<[MaybeUninit<T>], A>`](crate::Arc).
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_uninit_slice_arc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_uninit_slice_arc<T>(&self, len: usize) -> Arc<[MaybeUninit<T>], A>
    where
        A: Send + Sync,
    {
        self.try_alloc_uninit_slice_arc::<T>(len).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_uninit_slice_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_uninit_slice_arc<T>(&self, len: usize) -> Result<Arc<[MaybeUninit<T>], A>, AllocError>
    where
        A: Send + Sync,
    {
        let (ptr, len) = self.try_reserve_uninit_slice::<T>(len, ChunkSharing::Shared)?;
        // SAFETY: ptr/len describe a valid slice reservation in a Shared chunk.
        Ok(unsafe { Arc::from_raw_slice(ptr, len) })
    }

    /// Like [`Self::alloc_uninit_slice_arc`] but the slice bytes are zeroed.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_zeroed_slice_arc`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_zeroed_slice_arc<T>(&self, len: usize) -> Arc<[MaybeUninit<T>], A>
    where
        A: Send + Sync,
    {
        self.try_alloc_zeroed_slice_arc::<T>(len).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_zeroed_slice_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_zeroed_slice_arc<T>(&self, len: usize) -> Result<Arc<[MaybeUninit<T>], A>, AllocError>
    where
        A: Send + Sync,
    {
        let (ptr, len) = self.try_reserve_zeroed_slice::<T>(len, ChunkSharing::Shared)?;
        // SAFETY: ptr/len describe a valid slice reservation in a Shared chunk.
        Ok(unsafe { Arc::from_raw_slice(ptr, len) })
    }

    /// Allocate `len` uninitialized `T` slots and return an
    /// [`Box<[MaybeUninit<T>], A>`](crate::Box).
    ///
    /// Available only with the `dst` Cargo feature.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_uninit_slice_box`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_uninit_slice_box<T>(&self, len: usize) -> Box<[MaybeUninit<T>], A> {
        self.try_alloc_uninit_slice_box::<T>(len).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_uninit_slice_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_uninit_slice_box<T>(&self, len: usize) -> Result<Box<[MaybeUninit<T>], A>, AllocError> {
        let (ptr, len) = self.try_reserve_uninit_slice::<T>(len, ChunkSharing::Local)?;
        let fat = NonNull::slice_from_raw_parts(ptr, len);
        // SAFETY: ptr/len describe a valid slice reservation in a Local chunk.
        Ok(unsafe { Box::from_raw_unsized(fat) })
    }

    /// Like [`Self::alloc_uninit_slice_box`] but the slice bytes are zeroed.
    ///
    /// Available only with the `dst` Cargo feature.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_alloc_zeroed_slice_box`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_zeroed_slice_box<T>(&self, len: usize) -> Box<[MaybeUninit<T>], A> {
        self.try_alloc_zeroed_slice_box::<T>(len).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_zeroed_slice_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data alignment
    /// is at least 64 KiB.
    #[inline]
    pub fn try_alloc_zeroed_slice_box<T>(&self, len: usize) -> Result<Box<[MaybeUninit<T>], A>, AllocError> {
        let (ptr, len) = self.try_reserve_zeroed_slice::<T>(len, ChunkSharing::Local)?;
        let fat = NonNull::slice_from_raw_parts(ptr, len);
        // SAFETY: ptr/len describe a valid slice reservation in a Local chunk.
        Ok(unsafe { Box::from_raw_unsized(fat) })
    }
}

#[inline]
const fn compute_worst_case_size(layout: Layout, has_drop: bool) -> usize {
    let mut size = layout.size() + layout.align().saturating_sub(1);
    if has_drop {
        size += worst_case_extra_with_entry(layout);
    }
    size
}

#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "shared shim used by smart-pointer assume_init paths in sibling modules"
)]
pub(crate) unsafe fn slice_drop_shim<T>(entry: *mut DropEntry) {
    // SAFETY: entry was constructed by `commit_slice` with `len` stored in slice_len field
    unsafe {
        let len = (*entry).slice_len;
        let value_ptr = value_ptr_after_entry::<T>(entry);
        let slice = core::slice::from_raw_parts_mut(value_ptr, len);
        core::ptr::drop_in_place(slice);
    }
}

struct SliceInitGuard<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Drop for SliceInitGuard<T> {
    fn drop(&mut self) {
        // SAFETY: ptr..ptr+len are initialized by caller before guard's len is incremented
        unsafe {
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(self.ptr, self.len));
        }
    }
}

/// Drop guard for partial-slice init. On a panic during the init
/// loop, drops the already-initialized prefix and releases the
/// protective `+1` taken by `acquire_slice_slot`. On success, the
/// caller `mem::forget`s this guard so the `+1` stays as the smart
/// pointer's refcount.
struct SliceInitFailGuard<T, A: Allocator + Clone> {
    ptr: *mut T,
    len: usize,
    chunk: NonNull<ChunkHeader<A>>,
    sharing: ChunkSharing,
    holds_oversized: bool,
}

/// Drop guard that releases a protective `+1` taken on a chunk
/// before running user code in `try_reserve_and_init_aligned{,_slow}`.
/// Mirrors `SliceInitFailGuard`'s refcount-release logic but without
/// the init-loop tracking.
struct RefcountReleaseGuard<A: Allocator + Clone> {
    chunk: NonNull<ChunkHeader<A>>,
    sharing: ChunkSharing,
    holds_oversized: bool,
}

impl<A: Allocator + Clone> Drop for RefcountReleaseGuard<A> {
    fn drop(&mut self) {
        // Release the protective `+1` taken in
        // `try_reserve_and_init_aligned{,_slow}`.
        //
        // - Oversized chunks: the inc was a real atomic-or-Cell op
        //   bringing ref_count from 0 to 1; release symmetrically.
        // - Normal+Local: the inc was `cell.set(prev+1)` on
        //   `ref_count`; release symmetrically.
        // - Normal+Shared: the inc was non-atomic
        //   `arcs_issued += 1`. Released here via `fetch_sub(1)` on
        //   the atomic `ref_count`, mirroring `Arc::drop`. The
        //   asymmetry (inc on arcs_issued vs. dec on ref_count) is
        //   the deferred-reconciliation scheme: the eviction step
        //   does `fetch_sub(LARGE - arcs_issued)`, which folds
        //   arcs_issued into ref_count and produces the correct
        //   live-Arc count regardless of when our dec landed
        //   relative to eviction.
        //
        // `release_chunk_ref` dispatches on `sharing` and runs
        // `teardown_chunk` if ref_count hits zero. This subsumes all
        // three cases above.
        let _ = self.sharing;
        let _ = self.holds_oversized;
        // SAFETY: we hold the `+1` taken by
        // `try_reserve_and_init_aligned{,_slow}`; release on owner
        // thread.
        unsafe { release_chunk_ref::<A>(self.chunk, true) };
    }
}

impl<T, A: Allocator + Clone> Drop for SliceInitFailGuard<T, A> {
    fn drop(&mut self) {
        // SAFETY: `ptr..ptr+len` are initialized by the caller before
        // `self.len` is incremented.
        unsafe {
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(self.ptr, self.len));
        }
        // Release the protective `+1` taken in
        // `acquire_slice_slot{,_slow}`.
        //
        // See `RefcountReleaseGuard::Drop` for the unified case
        // analysis — `release_chunk_ref` correctly handles all
        // three (Oversized, Normal+Local, Normal+Shared) including
        // the case where the chunk has since been evicted from
        // `current_shared` (in which case the eviction step has
        // already rolled `arcs_issued` into `ref_count` and our
        // protective +1 is now visible directly on `ref_count`).
        let _ = self.sharing;
        let _ = self.holds_oversized;
        // SAFETY: we hold the `+1` taken by
        // `acquire_slice_slot{,_slow}`; release on owner thread.
        unsafe { release_chunk_ref::<A>(self.chunk, true) };
    }
}

#[inline(never)]
#[cold]
#[expect(clippy::panic, reason = "panicking allocation entry points panic on alloc failure by design")]
pub fn panic_alloc() -> ! {
    panic!("multitude: allocator returned AllocError");
}

impl Default for Arena<Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allocator + Clone> Drop for Arena<A> {
    fn drop(&mut self) {
        let last_reclaimer;
        {
            // SAFETY: owner-thread access; inner not freed.
            let inner = unsafe { self.inner.as_ref() };

            // Drain owner-thread chunk holdings BEFORE releasing the
            // handle's `HANDLE_HOLD` contribution. Releasing the
            // handle's contribution is what publishes "arena dropped"
            // to cross-thread teardown (see `ArenaInner::arena_dropped`),
            // so any chunk drops we trigger here still observe an
            // "alive" arena and may push the chunk into the cache —
            // but that's harmless because we drain the cache below.
            let mut made_progress;
            loop {
                made_progress = false;
                if self.current_local.take().is_some() {
                    made_progress = true;
                }
                // For Shared chunks, `OwnedChunk::drop` reconciles the
                // refcount via `LARGE - arcs_issued` — no separate
                // credit release needed.
                if self.current_shared.take().is_some() {
                    made_progress = true;
                }
                while self.pop_pinned().is_some() {
                    made_progress = true;
                }
                while inner.try_pop_cache().is_some() {
                    made_progress = true;
                }
                if !made_progress {
                    break;
                }
            }

            // Release the handle's `HANDLE_HOLD` contribution. If
            // `prev == HANDLE_HOLD` then no chunks remain (the loop
            // above drained everything *and* nothing held cross-thread
            // via Arc), so this caller is the last reclaimer. If
            // `prev > HANDLE_HOLD` then some `Arc<T>` from a `Shared`
            // chunk is still alive on another thread; that thread's
            // eventual `Arc::drop` will see `counter < HANDLE_HOLD`
            // (arena dropped) and run teardown without caching.
            //
            // SAFETY: self.inner is live (we hold the unique parent reference).
            last_reclaimer = unsafe { ArenaInner::register_release(self.inner, crate::arena_inner::HANDLE_HOLD) };
        }
        if last_reclaimer {
            // SAFETY: we observed `prev == HANDLE_HOLD`; we are the unique reclaimer.
            unsafe { ArenaInner::free_storage(self.inner) };
        }
    }
}

impl<A: Allocator + Clone> fmt::Debug for Arena<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Arena");
        #[cfg(feature = "stats")]
        let _ = s.field("stats", &self.stats());
        s.finish_non_exhaustive()
    }
}

#[cfg(feature = "utf16")]
#[cfg_attr(docsrs, doc(cfg(feature = "utf16")))]
impl<A: Allocator + Clone> Arena<A> {
    /// Copy `s` into the arena and return an
    /// [`RcUtf16Str`](crate::RcUtf16Str) smart pointer — a
    /// single-pointer, refcounted, `!Send`/`!Sync` immutable UTF-16
    /// string.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_alloc_utf16_str_rc`] for
    /// a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "utf16")] {
    /// use widestring::utf16str;
    /// let arena = multitude::Arena::new();
    /// let s = arena.alloc_utf16_str_rc(utf16str!("hello"));
    /// assert_eq!(&*s, utf16str!("hello"));
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc_utf16_str_rc(&self, s: impl AsRef<widestring::Utf16Str>) -> crate::RcUtf16Str<A> {
        self.try_alloc_utf16_str_rc(s).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_utf16_str_rc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_utf16_str_rc(&self, s: impl AsRef<widestring::Utf16Str>) -> Result<crate::RcUtf16Str<A>, AllocError> {
        let data = crate::arena_str_helpers::try_reserve_utf16_str_in_chunk(self, s.as_ref(), ChunkSharing::Local)?;
        // SAFETY: helper reserved a length-prefixed buffer and bumped the chunk refcount.
        Ok(unsafe { crate::RcUtf16Str::from_raw_data(data) })
    }

    /// Copy `s` into a `Shared`-flavor chunk and return an
    /// [`ArcUtf16Str`](crate::ArcUtf16Str) smart pointer — a
    /// single-pointer, atomically refcounted, `Send + Sync` immutable
    /// UTF-16 string safe for cross-thread sharing.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_alloc_utf16_str_arc`] for
    /// a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_utf16_str_arc(&self, s: impl AsRef<widestring::Utf16Str>) -> crate::ArcUtf16Str<A>
    where
        A: Send + Sync,
    {
        self.try_alloc_utf16_str_arc(s).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_utf16_str_arc`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_utf16_str_arc(&self, s: impl AsRef<widestring::Utf16Str>) -> Result<crate::ArcUtf16Str<A>, AllocError>
    where
        A: Send + Sync,
    {
        let data = crate::arena_str_helpers::try_reserve_utf16_str_in_chunk(self, s.as_ref(), ChunkSharing::Shared)?;
        // SAFETY: helper reserved a length-prefixed buffer and bumped the chunk refcount.
        Ok(unsafe { crate::ArcUtf16Str::from_raw_data(data) })
    }

    /// Copy `s` into the arena and return an
    /// [`BoxUtf16Str`](crate::BoxUtf16Str) smart pointer — a
    /// single-pointer (8 bytes) owned, mutable UTF-16 string.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_alloc_utf16_str_box`] for
    /// a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_utf16_str_box(&self, s: impl AsRef<widestring::Utf16Str>) -> crate::BoxUtf16Str<A> {
        self.try_alloc_utf16_str_box(s).unwrap_or_else(|_| panic_alloc())
    }

    /// Fallible variant of [`Self::alloc_utf16_str_box`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_utf16_str_box(&self, s: impl AsRef<widestring::Utf16Str>) -> Result<crate::BoxUtf16Str<A>, AllocError> {
        let data = crate::arena_str_helpers::try_reserve_utf16_str_in_chunk(self, s.as_ref(), ChunkSharing::Local)?;
        // SAFETY: helper reserved a length-prefixed buffer and bumped the chunk refcount.
        Ok(unsafe { crate::BoxUtf16Str::from_raw_data(data) })
    }

    #[cfg(feature = "builders")]
    /// Transcode `s` from UTF-8 to UTF-16, copy the result into the
    /// arena, and return an
    /// [`RcUtf16Str`](crate::RcUtf16Str).
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use
    /// [`Self::try_alloc_utf16_str_rc_from_str`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_utf16_str_rc_from_str(&self, s: impl AsRef<str>) -> crate::RcUtf16Str<A> {
        crate::builders::Utf16String::from_str_in(s.as_ref(), self).into_arena_utf16_str()
    }

    #[cfg(feature = "builders")]
    /// Fallible variant of [`Self::alloc_utf16_str_rc_from_str`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_utf16_str_rc_from_str(&self, s: impl AsRef<str>) -> Result<crate::RcUtf16Str<A>, AllocError> {
        let s = s.as_ref();
        let mut buf = crate::builders::Utf16String::try_with_capacity_in(s.len(), self)?;
        buf.try_push_from_str(s)?;
        Ok(buf.into_arena_utf16_str())
    }

    #[cfg(feature = "builders")]
    /// Transcode `s` from UTF-8 to UTF-16, copy the result into a
    /// `Shared`-flavor chunk, and return an
    /// [`ArcUtf16Str`](crate::ArcUtf16Str).
    ///
    /// Internally builds a `Local`-flavor `Utf16String` for the
    /// transcode, then re-allocates the bytes into a `Shared` chunk
    /// before returning. The intermediate `Local` allocation's chunk
    /// refcount is released when the temporary buffer drops.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use
    /// [`Self::try_alloc_utf16_str_arc_from_str`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_utf16_str_arc_from_str(&self, s: impl AsRef<str>) -> crate::ArcUtf16Str<A>
    where
        A: Send + Sync,
    {
        self.try_alloc_utf16_str_arc_from_str(s).unwrap_or_else(|_| panic_alloc())
    }

    #[cfg(feature = "builders")]
    /// Fallible variant of [`Self::alloc_utf16_str_arc_from_str`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_utf16_str_arc_from_str(&self, s: impl AsRef<str>) -> Result<crate::ArcUtf16Str<A>, AllocError>
    where
        A: Send + Sync,
    {
        let s = s.as_ref();
        let mut buf = crate::builders::Utf16String::try_with_capacity_in(s.len(), self)?;
        buf.try_push_from_str(s)?;
        self.try_alloc_utf16_str_arc(buf.as_utf16_str())
    }

    #[cfg(feature = "builders")]
    /// Transcode `s` from UTF-8 to UTF-16, copy the result into the
    /// arena, and return an
    /// [`BoxUtf16Str`](crate::BoxUtf16Str).
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use
    /// [`Self::try_alloc_utf16_str_box_from_str`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_utf16_str_box_from_str(&self, s: impl AsRef<str>) -> crate::BoxUtf16Str<A> {
        self.try_alloc_utf16_str_box_from_str(s).unwrap_or_else(|_| panic_alloc())
    }

    #[cfg(feature = "builders")]
    /// Fallible variant of [`Self::alloc_utf16_str_box_from_str`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_utf16_str_box_from_str(&self, s: impl AsRef<str>) -> Result<crate::BoxUtf16Str<A>, AllocError> {
        let s = s.as_ref();
        let mut buf = crate::builders::Utf16String::try_with_capacity_in(s.len(), self)?;
        buf.try_push_from_str(s)?;
        self.try_alloc_utf16_str_box(buf.as_utf16_str())
    }

    #[cfg(feature = "builders")]
    /// Create a new, empty growable
    /// [`Utf16String`](crate::builders::Utf16String) backed by this
    /// arena. No allocation is performed until the first push.
    #[must_use]
    #[inline]
    pub const fn alloc_utf16_string(&self) -> crate::builders::Utf16String<'_, A> {
        crate::builders::Utf16String::new_in(self)
    }

    #[cfg(feature = "builders")]
    /// Create a new growable
    /// [`Utf16String`](crate::builders::Utf16String) backed by this
    /// arena, with at least `cap` `u16` elements of pre-allocated
    /// capacity.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use
    /// [`Self::try_alloc_utf16_string_with_capacity`] for a fallible variant.
    #[must_use]
    #[inline]
    pub fn alloc_utf16_string_with_capacity(&self, cap: usize) -> crate::builders::Utf16String<'_, A> {
        crate::builders::Utf16String::with_capacity_in(cap, self)
    }

    #[cfg(feature = "builders")]
    /// Fallible variant of [`Self::alloc_utf16_string_with_capacity`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails.
    #[inline]
    pub fn try_alloc_utf16_string_with_capacity(&self, cap: usize) -> Result<crate::builders::Utf16String<'_, A>, AllocError> {
        crate::builders::Utf16String::try_with_capacity_in(cap, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_reservation_debug_format() {
        let arena: Arena = Arena::new();
        let reservation = arena.reserve_slice::<u32>(4, ChunkSharing::Local, false, false).unwrap();
        let s = alloc::format!("{reservation:?}");
        assert!(s.contains("SliceReservation"), "Debug output: {s}");
        assert!(s.contains("layout"), "Debug output: {s}");
        assert!(s.contains("has_drop_entry"), "Debug output: {s}");
        let with_drop = arena.reserve_slice::<u32>(4, ChunkSharing::Local, true, false).unwrap();
        let s2 = alloc::format!("{with_drop:?}");
        assert!(s2.contains("has_drop_entry: true"), "Debug output: {s2}");
    }

    #[test]
    fn reserve_slice_rejects_excessive_alignment() {
        #[repr(align(131072))]
        struct HugeAlign(#[expect(dead_code, reason = "field present to give the type a non-zero size")] u8);

        let arena: Arena = Arena::new();
        let result = arena.reserve_slice::<HugeAlign>(0, ChunkSharing::Local, false, false);
        let _ = result.unwrap_err();
    }
}
