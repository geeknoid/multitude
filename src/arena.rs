//! [`Arena`] — the main entry point.

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::ptr::NonNull;

use alloc::boxed::Box;
use allocator_api2::alloc::{AllocError, Allocator, Global};

use crate::arena_arc::ArenaArc;
use crate::arena_box::ArenaBox;
use crate::arena_inner::ArenaInner;
use crate::arena_rc::ArenaRc;
use crate::chunk_header::{ChunkHeader, init_chunk, revive_cached_chunk};
use crate::chunk_sharing::ChunkSharing;
use crate::chunk_size_class::ChunkSizeClass;
use crate::constants::CHUNK_ALIGN;
use crate::drop_entry::{DropEntry, drop_shim};
use crate::pending_arena_arc::PendingArenaArc;
use crate::pending_arena_rc::PendingArenaRc;
use crate::stats::ArenaStats;

/// A single-threaded arena allocator, backed by `A`.
///
/// Hands out reference-counted handles ([`ArenaRc`], [`ArenaArc`],
/// [`ArenaRcStr`](crate::ArenaRcStr), [`ArenaArcStr`](crate::ArenaArcStr))
/// and an owned single handle ([`ArenaBox`]) backed by 64 KiB-aligned
/// chunks of memory obtained from `A`. Chunks are reclaimed individually
/// as their refcounts drop to zero.
///
/// `Arena<A>` is `!Send` and `!Sync` regardless of `A`. Cross-thread
/// sharing is opt-in at the *handle* level via [`Arena::alloc_shared`].
///
/// # Configuration
///
/// [`Arena::new`] uses sensible defaults (64 KiB chunks, oversized
/// cutover at 16 KiB, cache up to 8 chunks, no byte budget). For
/// non-default configuration, use [`ArenaBuilder`](crate::ArenaBuilder) (or
/// [`Arena::builder`]).
///
/// # Example
///
/// ```
/// use harena::Arena;
///
/// let arena = Arena::new();
/// let value = arena.alloc(42_u32);
/// assert_eq!(*value, 42);
/// ```
pub struct Arena<A: Allocator + Clone = Global> {
    inner: NonNull<ArenaInner<A>>,
    /// `Arena` is single-threaded only.
    _not_sync: PhantomData<*mut ()>,
}

impl Arena<Global> {
    /// Create a new, empty arena backed by [`Global`] with default
    /// configuration. No chunks are allocated until the first
    /// allocation call.
    ///
    /// For non-default configuration, use [`Self::builder`].
    ///
    /// # Example
    ///
    /// ```
    /// let arena = harena::Arena::new();
    /// assert_eq!(arena.stats().chunks_allocated, 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        crate::ArenaBuilder::new().build_unwrap()
    }

    /// Begin an [`ArenaBuilder`](crate::ArenaBuilder) using [`Global`]
    /// as the backing allocator.
    #[must_use]
    pub fn builder() -> crate::ArenaBuilder<Global> {
        crate::ArenaBuilder::new()
    }
}

impl<A: Allocator + Clone> Arena<A> {
    /// Begin an [`ArenaBuilder`](crate::ArenaBuilder) backed by a custom
    /// `allocator`.
    #[must_use]
    pub fn builder_in(allocator: A) -> crate::ArenaBuilder<A> {
        crate::ArenaBuilder::new_in(allocator)
    }

    /// Create a new, empty arena backed by `allocator` with default
    /// configuration.
    ///
    /// For non-default configuration, use [`Self::builder_in`].
    ///
    /// The allocator is stored by value and cloned into each chunk so
    /// chunks can free themselves back to it even after the arena is
    /// dropped. For ZST allocators (e.g., [`Global`]) this clone is free.
    #[must_use]
    pub fn new_in(allocator: A) -> Self {
        crate::ArenaBuilder::new_in(allocator).build_unwrap()
    }

    /// Construct an arena from a fully-initialized [`ArenaInner`]. Used
    /// by [`ArenaBuilder`](crate::ArenaBuilder).
    pub(crate) fn from_inner(inner: ArenaInner<A>) -> Self {
        let boxed = Box::new(inner);
        let inner_ptr = NonNull::from(Box::leak(boxed));
        Self {
            inner: inner_ptr,
            _not_sync: PhantomData,
        }
    }

    /// Borrow the inner state. Owner-thread only.
    ///
    /// # Safety
    ///
    /// The caller must be on the arena's owner thread (which is true
    /// for any method on `Arena<A>`, since `Arena` is `!Send`/`!Sync`).
    #[inline]
    pub(crate) unsafe fn inner_ref(&self) -> &ArenaInner<A> {
        // SAFETY: caller's contract — owner-thread only.
        unsafe { self.inner.as_ref() }
    }

    /// Borrow the backing allocator.
    #[must_use]
    pub fn allocator(&self) -> &A {
        // SAFETY: owner-thread access.
        unsafe { &self.inner_ref().allocator }
    }

    /// Snapshot of the arena's lifetime statistics. See [`ArenaStats`].
    #[must_use]
    pub fn stats(&self) -> ArenaStats {
        // SAFETY: owner-thread access.
        unsafe { self.inner_ref().stats.snapshot() }
    }

    // ---- internal: chunk acquisition -------------------------------------

    /// Get a chunk that can satisfy `request_layout`, of the given
    /// sharing flavor. May return the current chunk, pull from cache, or
    /// allocate a new chunk via `A`.
    fn try_get_chunk_for(
        &self,
        sharing: ChunkSharing,
        request_layout: Layout,
        has_drop: bool,
    ) -> Result<NonNull<ChunkHeader<A>>, AllocError> {
        // SAFETY: owner-thread.
        let inner = unsafe { self.inner_ref() };

        // Compute the worst-case bytes needed for this allocation
        // (alignment overhead + drop entry + value).
        let worst_case_size = compute_worst_case_size(request_layout, has_drop);

        // Decide if this is an oversized request — based on worst-case
        // bytes (per PLAN.md §8.3 / rubber-duck blocking #2). This avoids
        // misclassifying a small-but-high-aligned request as normal and
        // then having it not fit in a normal chunk.
        let usable_in_normal = inner.chunk_size - ChunkHeader::<A>::header_padded_size();
        if worst_case_size > inner.max_normal_alloc || worst_case_size > usable_in_normal {
            return self.try_create_oversized_chunk(sharing, request_layout, has_drop);
        }

        // Fast path: try the current chunk.
        let current_cell = match sharing {
            ChunkSharing::Local => &inner.current_local,
            ChunkSharing::Shared => &inner.current_shared,
        };
        if let Some(chunk) = current_cell.get() {
            // SAFETY: chunk is valid; we hold the arena's refcount on it.
            let h = unsafe { chunk.as_ref() };
            let cur = h.bump.get();
            if h.total_size.saturating_sub(cur) >= worst_case_size {
                return Ok(chunk);
            }
            // Doesn't fit; retire current chunk. Account the slack at
            // retirement (per PLAN.md §8.4 wasted_tail_bytes contract).
            let wasted = h.total_size.saturating_sub(cur);
            crate::stats::StatsStorage::add(&inner.stats.wasted_tail_bytes, wasted as u64);
            current_cell.set(None);
            // SAFETY: arena releases its hold; teardown is sound when
            // refcount reaches zero.
            unsafe {
                if h.dec_ref() {
                    crate::chunk_header::teardown_chunk(chunk, true);
                }
            }
        }

        // Try to pull from cache.
        if let Some(chunk) = inner.try_pop_cache() {
            // SAFETY: just popped, exclusive access.
            unsafe { revive_cached_chunk(chunk, sharing) };
            current_cell.set(Some(chunk));
            return Ok(chunk);
        }

        // Allocate a fresh normal chunk.
        let chunk = self.try_alloc_fresh_chunk_normal()?;
        // The fresh chunk was initialized with sharing = Local for
        // bookkeeping convenience; flip to the caller-requested flavor.
        // SAFETY: chunk is alive; we own it (refcount=1, no other refs).
        unsafe { chunk.as_ref().sharing.set(sharing) };
        current_cell.set(Some(chunk));
        Ok(chunk)
    }

    /// Allocate a fresh normal chunk and return it with refcount=1
    /// (the arena's transient hold). Increments `chunks_allocated` and
    /// charges the byte budget. Used by both `try_get_chunk_for` and
    /// the [`ArenaBuilder`](crate::ArenaBuilder)'s preallocate path.
    pub(crate) fn try_alloc_fresh_chunk_normal(&self) -> Result<NonNull<ChunkHeader<A>>, AllocError> {
        // SAFETY: owner-thread.
        let inner = unsafe { self.inner_ref() };
        let layout = ChunkHeader::<A>::normal_layout(inner.chunk_size);
        self.try_alloc_fresh_chunk(ChunkSharing::Local, ChunkSizeClass::Normal, layout)
    }

    /// Allocate a fresh oversized chunk for one allocation. Doesn't touch
    /// `current_*`; the chunk is returned with refcount=1 (the arena's
    /// transient hold, transferred to the resulting handle).
    fn try_create_oversized_chunk(
        &self,
        sharing: ChunkSharing,
        request_layout: Layout,
        has_drop: bool,
    ) -> Result<NonNull<ChunkHeader<A>>, AllocError> {
        let chunk_layout = ChunkHeader::<A>::oversized_layout(request_layout, has_drop).ok_or(AllocError)?;
        self.try_alloc_fresh_chunk(sharing, ChunkSizeClass::Oversized, chunk_layout)
    }

    fn try_alloc_fresh_chunk(
        &self,
        sharing: ChunkSharing,
        size_class: ChunkSizeClass,
        chunk_layout: Layout,
    ) -> Result<NonNull<ChunkHeader<A>>, AllocError> {
        // SAFETY: owner-thread.
        let inner = unsafe { self.inner_ref() };

        // Charge the byte budget *before* asking the allocator.
        let chunk_bytes = chunk_layout.size();
        let prev = inner.bytes_used_against_budget.get();
        let next = prev.checked_add(chunk_bytes).ok_or(AllocError)?;
        if let Some(budget) = inner.byte_budget
            && next > budget
        {
            return Err(AllocError);
        }

        let raw = inner.allocator.allocate(chunk_layout)?;
        inner.bytes_used_against_budget.set(next);

        let total_size = chunk_layout.size();
        // Initial refcount: 1 for normal chunks (arena's transient hold,
        // released on chunk rotation or arena drop), 0 for oversized
        // chunks (no arena hold; the handle's inc_ref will be the only
        // refcount). Without the 0 here, oversized chunks would always
        // leak (refcount stuck at 1 even after the handle drops).
        let initial_refcount = match size_class {
            ChunkSizeClass::Normal => 1,
            ChunkSizeClass::Oversized => 0,
        };
        // SAFETY: just allocated, properly aligned per Allocator contract.
        let chunk = unsafe {
            init_chunk(
                raw.cast::<u8>(),
                total_size,
                sharing,
                size_class,
                self.inner,
                inner.allocator.clone(),
                initial_refcount,
            )
        };

        // Stat: count chunk by class.
        match size_class {
            ChunkSizeClass::Normal => crate::stats::StatsStorage::add(&inner.stats.chunks_allocated, 1),
            ChunkSizeClass::Oversized => crate::stats::StatsStorage::add(&inner.stats.oversized_chunks_allocated, 1),
        }
        // One more outstanding chunk; ArenaInner can't be freed until
        // this is decremented in `free_chunk`.
        let _ = inner.outstanding_chunks.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(chunk)
    }

    // ---- alloc<T> --------------------------------------------------------

    /// Allocate `value` and return a thread-local reference-counted
    /// handle to it.
    ///
    /// If `T` needs drop, a tiny entry is added to the owning chunk's
    /// drop list so `T::drop` runs exactly once when the chunk is
    /// reclaimed.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails or if
    /// `align_of::<T>() > CHUNK_ALIGN`. Use [`Self::try_alloc`] for a
    /// fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = harena::Arena::new();
    /// let v = arena.alloc(vec![1, 2, 3]);
    /// assert_eq!(v.len(), 3);
    /// ```
    pub fn alloc<T>(&self, value: T) -> ArenaRc<T, A> {
        match self.try_alloc(value) {
            Ok(rc) => rc,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate `value` and return a thread-local reference-counted handle to
    /// it, returning Err([`AllocError`]) instead of panicking if the
    /// backing allocator cannot satisfy the request. The supplied `value` is
    /// dropped on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc<T>(&self, value: T) -> Result<ArenaRc<T, A>, AllocError> {
        self.try_alloc_with(move || value)
    }

    /// Allocate the result of `f`, constructing it directly inside the
    /// arena (no stack copy of `T`).
    ///
    /// The closure may freely allocate from this arena. If it panics,
    /// the reservation is leaked (no drop is registered, no double-drop).
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails. Use [`Self::try_alloc_with`] for
    /// a fallible variant.
    pub fn alloc_with<T, F: FnOnce() -> T>(&self, f: F) -> ArenaRc<T, A> {
        match self.try_alloc_with(f) {
            Ok(rc) => rc,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate the result of `f`, constructing it directly inside the arena,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator cannot satisfy the request. The closure is not called on
    /// allocator failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_with<T, F: FnOnce() -> T>(&self, f: F) -> Result<ArenaRc<T, A>, AllocError> {
        // SAFETY: the helper writes a `T` in place and registers a drop
        // shim if needed.
        unsafe {
            let value_ptr = self.try_reserve_and_init::<T, _>(ChunkSharing::Local, f)?;
            Ok(ArenaRc::from_raw(value_ptr))
        }
    }

    // ---- alloc_shared<T> -------------------------------------------------

    /// Allocate `value` and return a `Send + Sync` reference-counted
    /// handle to it. Costs an atomic RMW per clone/drop.
    ///
    /// `T: Send + Sync` is required for the handle to be sound to share;
    /// `A: Send + Sync` is required because the chunk's `Drop` may run
    /// on a non-owner thread.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails or if `align_of::<T>() > CHUNK_ALIGN`.
    pub fn alloc_shared<T: Send + Sync>(&self, value: T) -> ArenaArc<T, A>
    where
        A: Send + Sync,
    {
        match self.try_alloc_shared(value) {
            Ok(arc) => arc,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate `value` and return a `Send + Sync` [`ArenaArc`] handle to it,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator cannot satisfy the request. The supplied `value` is dropped
    /// on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_shared<T: Send + Sync>(&self, value: T) -> Result<ArenaArc<T, A>, AllocError>
    where
        A: Send + Sync,
    {
        self.try_alloc_with_shared(move || value)
    }

    /// Allocate the result of `f` directly inside a `Shared`-flavor chunk and
    /// return an [`ArenaArc`] handle safe for cross-thread sharing. The closure
    /// constructs the value in place — no stack copy of `T`. Panics if the
    /// allocator fails.
    pub fn alloc_with_shared<T: Send + Sync, F: FnOnce() -> T>(&self, f: F) -> ArenaArc<T, A>
    where
        A: Send + Sync,
    {
        match self.try_alloc_with_shared(f) {
            Ok(arc) => arc,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate the result of `f` directly inside a `Shared`-flavor chunk and
    /// return an [`ArenaArc`] handle, returning Err([`AllocError`]) instead
    /// of panicking if the backing allocator cannot satisfy the request. The
    /// closure is not called on allocator failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_with_shared<T: Send + Sync, F: FnOnce() -> T>(&self, f: F) -> Result<ArenaArc<T, A>, AllocError>
    where
        A: Send + Sync,
    {
        // SAFETY: same as Local path, but Shared chunk.
        unsafe {
            let value_ptr = self.try_reserve_and_init::<T, _>(ChunkSharing::Shared, f)?;
            Ok(ArenaArc::from_raw(value_ptr))
        }
    }

    // ---- alloc_box<T> ----------------------------------------------------

    /// Allocate `value` and return an owned, mutable handle whose `Drop`
    /// runs `T::drop` immediately on handle drop.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails.
    pub fn alloc_box<T>(&self, value: T) -> ArenaBox<'_, T, A> {
        match self.try_alloc_box(value) {
            Ok(b) => b,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate `value` and return an owned, mutable [`ArenaBox`] handle whose
    /// `Drop` runs `T::drop` immediately on handle drop, returning
    /// Err([`AllocError`]) instead of panicking if the backing allocator
    /// cannot satisfy the request. The supplied `value` is dropped on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_box<T>(&self, value: T) -> Result<ArenaBox<'_, T, A>, AllocError> {
        self.try_alloc_box_with(move || value)
    }

    /// Allocate the result of `f` directly inside the arena and return an
    /// owned, mutable [`ArenaBox`] handle whose `Drop` runs `T::drop`
    /// immediately on handle drop. Panics if the allocator fails.
    pub fn alloc_box_with<T, F: FnOnce() -> T>(&self, f: F) -> ArenaBox<'_, T, A> {
        match self.try_alloc_box_with(f) {
            Ok(b) => b,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate the result of `f` directly inside the arena and return an
    /// owned, mutable [`ArenaBox`] handle, returning Err([`AllocError`])
    /// instead of panicking if the backing allocator cannot satisfy the
    /// request. The closure is not called on allocator failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_box_with<T, F: FnOnce() -> T>(&self, f: F) -> Result<ArenaBox<'_, T, A>, AllocError> {
        // SAFETY: same path as alloc_with, but produces an ArenaBox.
        unsafe {
            let value_ptr = self.try_reserve_and_init::<T, _>(ChunkSharing::Local, f)?;
            Ok(ArenaBox::from_raw(value_ptr))
        }
    }

    // ---- alloc_uninit_dst ------------------------------------------------

    /// Reserve uninitialized space for a value with the given `Layout` in
    /// a `Local`-flavor chunk.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails or if `layout.align() > CHUNK_ALIGN`.
    #[must_use]
    pub fn alloc_uninit_dst(&self, layout: Layout) -> PendingArenaRc<'_, A> {
        match self.try_alloc_uninit_dst(layout) {
            Ok(pa) => pa,
            Err(_) => panic_alloc(),
        }
    }

    /// Reserve uninitialized space for a value with the given `Layout` in a
    /// `Local`-flavor chunk, returning Err([`AllocError`]) instead of
    /// panicking if the backing allocator cannot satisfy the request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_uninit_dst(&self, layout: Layout) -> Result<PendingArenaRc<'_, A>, AllocError> {
        // For DSTs, drop registration happens in `finalize`. Reserve a
        // DropEntry slot conservatively only if the user signals via
        // `finalize(_, Some(drop_fn))` — but we don't know yet. To avoid
        // a two-mode interface, we always reserve the entry slot for
        // DST allocations (cost: 24 bytes per DST).
        // SAFETY: we reserve space; the caller initializes via `finalize`.
        unsafe {
            let (entry, value, chunk) = self.try_reserve_dst_with_entry(ChunkSharing::Local, layout)?;
            Ok(PendingArenaRc::new(self, chunk, entry, value, layout))
        }
    }

    /// Reserve uninitialized space for a value with the given `Layout` in a
    /// `Shared`-flavor chunk, returning a [`PendingArenaArc`]. Panics if the
    /// allocator fails or if `layout.align() > CHUNK_ALIGN`.
    #[must_use]
    pub fn alloc_uninit_dst_shared(&self, layout: Layout) -> PendingArenaArc<'_, A>
    where
        A: Send + Sync,
    {
        match self.try_alloc_uninit_dst_shared(layout) {
            Ok(pa) => pa,
            Err(_) => panic_alloc(),
        }
    }

    /// Reserve uninitialized space for a value with the given `Layout` in a
    /// `Shared`-flavor chunk, returning Err([`AllocError`]) instead of
    /// panicking if the backing allocator cannot satisfy the request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_uninit_dst_shared(&self, layout: Layout) -> Result<PendingArenaArc<'_, A>, AllocError>
    where
        A: Send + Sync,
    {
        // SAFETY: see Local variant.
        unsafe {
            let (entry, value, chunk) = self.try_reserve_dst_with_entry(ChunkSharing::Shared, layout)?;
            Ok(PendingArenaArc::new(self, chunk, entry, value, layout))
        }
    }

    // ---- internal: reservation helpers -----------------------------------

    /// Reserve space for one `T` value (with optional `DropEntry` for
    /// `T: needs_drop`), invoke `f` to construct the value, write it,
    /// link the drop entry, and return the value pointer. The chunk's
    /// refcount is bumped by 1 (the new handle's hold).
    ///
    /// # Safety
    ///
    /// On success, the returned pointer references a fully-initialized
    /// `T` inside a chunk whose refcount has been incremented for the
    /// caller. The caller is responsible for wrapping the pointer in a
    /// handle that will eventually decrement the refcount.
    unsafe fn try_reserve_and_init<T, F: FnOnce() -> T>(&self, sharing: ChunkSharing, f: F) -> Result<NonNull<T>, AllocError> {
        let needs_drop = core::mem::needs_drop::<T>();
        let layout = Layout::new::<T>();
        if layout.align() > CHUNK_ALIGN {
            return Err(AllocError);
        }

        let chunk = self.try_get_chunk_for(sharing, layout, needs_drop)?;
        // SAFETY: chunk has at least worst-case-size bytes free.
        let h = unsafe { chunk.as_ref() };
        if needs_drop {
            let (entry, value) = ChunkHeader::<A>::try_alloc_with_drop_entry::<T>(chunk).expect("worst-case sizing was wrong");
            // Step 1: write the value (in place — no stack copy).
            //         If f() panics here, no drop is registered, no
            //         refcount bump done — the bump cursor leaks the
            //         reservation but is otherwise sound.
            // SAFETY: value points to writable, properly-aligned memory
            // with the chunk allocation's full provenance.
            unsafe { value.as_ptr().write(f()) };
            // Step 2 & 3: initialize and link the entry.
            // SAFETY: entry slot is writable.
            unsafe { h.link_drop_entry(entry, drop_shim::<T>, 0) };
            // Step 4: bump refcount for the new handle.
            h.inc_ref();
            self.charge_alloc_stats(layout.size());
            Ok(value)
        } else {
            let value_ptr_u8 = ChunkHeader::<A>::try_alloc(chunk, layout).expect("worst-case sizing was wrong");
            let value = value_ptr_u8.cast::<T>();
            // SAFETY: well-aligned writable slot with full provenance.
            unsafe { value.as_ptr().write(f()) };
            h.inc_ref();
            self.charge_alloc_stats(layout.size());
            Ok(value)
        }
    }

    /// Charge a successful user allocation against `total_bytes_allocated`.
    #[inline]
    pub(crate) fn charge_alloc_stats(&self, bytes: usize) {
        // SAFETY: owner-thread.
        let inner = unsafe { self.inner_ref() };
        crate::stats::StatsStorage::add(&inner.stats.total_bytes_allocated, bytes as u64);
    }

    /// Bump `allocator_relocations` by 1. Called by the
    /// `Allocator::grow` override when the in-place fast path fails.
    #[inline]
    pub(crate) fn bump_allocator_relocation(&self) {
        // SAFETY: owner-thread.
        let inner = unsafe { self.inner_ref() };
        crate::stats::StatsStorage::add(&inner.stats.allocator_relocations, 1);
    }

    /// Bump `string_relocations` by 1. Called by `ArenaString` when its
    /// internal grow helper relocates rather than extending in place.
    #[inline]
    pub(crate) fn bump_string_relocation(&self) {
        // SAFETY: owner-thread.
        let inner = unsafe { self.inner_ref() };
        crate::stats::StatsStorage::add(&inner.stats.string_relocations, 1);
    }
}

/// Triple returned by [`Arena::try_reserve_dst_with_entry`].
type DstReservation<A> = (NonNull<DropEntry>, NonNull<u8>, NonNull<ChunkHeader<A>>);

/// A pending slice reservation: bump-allocated space (and an unlinked
/// `DropEntry` slot when needed) but no committed refcount yet.
/// Returned by `Arena::reserve_slice`; finalized by `Arena::commit_slice`
/// once the caller has fully initialized the elements.
///
/// The reserve/commit split exists so that, if the caller's element-init
/// code panics partway, neither the drop entry is linked (avoiding
/// `drop_in_place` on uninit bytes at chunk teardown) nor the chunk
/// refcount bumped (avoiding chunk + `ArenaInner` leaks). The reserved
/// bump bytes are leaked in-chunk, but the chunk itself reclaims
/// normally.
struct SliceReservation<A: Allocator + Clone> {
    chunk: NonNull<ChunkHeader<A>>,
    ptr: NonNull<u8>,
    /// `Some(entry)` iff the caller asked us to reserve a `DropEntry`
    /// slot; the entry is NOT linked into the drop list yet.
    entry: Option<NonNull<DropEntry>>,
    layout: Layout,
}

impl<A: Allocator + Clone> Arena<A> {
    /// Reserve space for a DST + `DropEntry`. The bytes are uninitialized;
    /// the caller (a `PendingArenaRc`) is responsible for writing the value
    /// before finalizing.
    unsafe fn try_reserve_dst_with_entry(&self, sharing: ChunkSharing, layout: Layout) -> Result<DstReservation<A>, AllocError> {
        if layout.align() > CHUNK_ALIGN {
            return Err(AllocError);
        }
        let chunk = self.try_get_chunk_for(sharing, layout, true)?;
        // SAFETY: chunk has space.
        let h = unsafe { chunk.as_ref() };
        let cur = h.bump.get();
        let entry_addr = crate::constants::checked_align_up(cur, align_of::<DropEntry>()).ok_or(AllocError)?;
        let after_entry = entry_addr.checked_add(size_of::<DropEntry>()).ok_or(AllocError)?;
        // chunk has space; layout fits.
        let value_addr = crate::constants::checked_align_up(after_entry, layout.align()).ok_or(AllocError)?;
        let end = value_addr.checked_add(layout.size()).ok_or(AllocError)?;
        if end > h.total_size {
            return Err(AllocError);
        }
        h.bump.set(end);
        let base = chunk.as_ptr().cast::<u8>();
        // SAFETY: byte offsets are valid within the chunk.
        let entry = unsafe { NonNull::new_unchecked(base.add(entry_addr).cast::<DropEntry>()) };
        // SAFETY: byte offset is valid within the chunk.
        let value = unsafe { NonNull::new_unchecked(base.add(value_addr)) };
        // Bump refcount for the upcoming handle (released by PendingArenaRc::Drop
        // or transferred on finalize).
        h.inc_ref();
        self.charge_alloc_stats(layout.size());
        Ok((entry, value, chunk))
    }

    // ---- slice constructors ---------------------------------------------

    /// Copy `slice` into the arena, returning an immutable handle.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails.
    pub fn alloc_slice_copy<T: Copy>(&self, slice: &[T]) -> ArenaRc<[T], A> {
        match self.try_alloc_slice_copy(slice) {
            Ok(r) => r,
            Err(_) => panic_alloc(),
        }
    }

    /// Copy `slice` into the arena and return an immutable [`ArenaRc`] handle,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator cannot satisfy the request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_slice_copy<T: Copy>(&self, slice: &[T]) -> Result<ArenaRc<[T], A>, AllocError> {
        // SAFETY: T: Copy means no Drop, no drop guard needed.
        unsafe {
            let reservation = self.reserve_slice::<T>(slice.len(), ChunkSharing::Local, false)?;
            let ptr = reservation.ptr.cast::<T>();
            core::ptr::copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), slice.len());
            self.commit_slice::<T>(reservation, slice.len());
            Ok(ArenaRc::from_raw_slice(ptr, slice.len()))
        }
    }

    /// Clone every element of `slice` into the arena, returning an
    /// immutable handle.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails.
    pub fn alloc_slice_clone<T: Clone>(&self, slice: &[T]) -> ArenaRc<[T], A> {
        match self.try_alloc_slice_clone(slice) {
            Ok(r) => r,
            Err(_) => panic_alloc(),
        }
    }

    /// Clone every element of `slice` into the arena and return an immutable
    /// [`ArenaRc`] handle, returning Err([`AllocError`]) instead of
    /// panicking if the backing allocator cannot satisfy the request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_slice_clone<T: Clone>(&self, slice: &[T]) -> Result<ArenaRc<[T], A>, AllocError> {
        let mut idx = 0;
        self.try_alloc_slice_fill_with(slice.len(), |_| {
            let v = slice[idx].clone();
            idx += 1;
            v
        })
    }

    /// Allocate a slice of `len` elements, with element `i` produced by
    /// `f(i)`.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails. If `f` panics, already-initialized
    /// elements are dropped (drop guard) and the panic propagates.
    pub fn alloc_slice_fill_with<T, F: FnMut(usize) -> T>(&self, len: usize, f: F) -> ArenaRc<[T], A> {
        match self.try_alloc_slice_fill_with(len, f) {
            Ok(r) => r,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate a slice of `len` elements with element `i` produced by `f(i)`,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator cannot satisfy the request. If `f` panics, already-initialized
    /// elements are dropped and the panic propagates.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    pub fn try_alloc_slice_fill_with<T, F: FnMut(usize) -> T>(&self, len: usize, mut f: F) -> Result<ArenaRc<[T], A>, AllocError> {
        let needs_drop = core::mem::needs_drop::<T>();
        // SAFETY: we use a drop guard for partial init AND we defer
        // committing the reservation until init succeeds. If `f` panics,
        // SliceInitGuard drops the initialized prefix; the bump-cursor
        // slot (and unlinked DropEntry slot) stay leaked in the chunk
        // but the chunk's refcount is NOT bumped, so the chunk reclaims
        // normally — no leaked chunk + no leaked ArenaInner.
        unsafe {
            let reservation = self.reserve_slice::<T>(len, ChunkSharing::Local, needs_drop)?;
            let ptr = reservation.ptr.cast::<T>();
            let mut guard = SliceInitGuard::<T> { ptr: ptr.as_ptr(), len: 0 };
            for i in 0..len {
                ptr.as_ptr().add(i).write(f(i));
                guard.len += 1;
            }
            // Success — defuse guard, then commit.
            core::mem::forget(guard);
            self.commit_slice::<T>(reservation, len);
            Ok(ArenaRc::from_raw_slice(ptr, len))
        }
    }

    /// Allocate a slice and fill it with values pulled from `iter`.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails or if the iterator is shorter than
    /// its reported `ExactSizeIterator::len()`.
    pub fn alloc_slice_fill_iter<T, I>(&self, iter: I) -> ArenaRc<[T], A>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        match self.try_alloc_slice_fill_iter(iter) {
            Ok(r) => r,
            Err(_) => panic_alloc(),
        }
    }

    /// Allocate a slice and fill it with values pulled from `iter`,
    /// returning Err([`AllocError`]) instead of panicking if the backing
    /// allocator cannot satisfy the request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator cannot satisfy the
    /// request.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields fewer elements than its
    /// `ExactSizeIterator::len()` reported.
    pub fn try_alloc_slice_fill_iter<T, I>(&self, iter: I) -> Result<ArenaRc<[T], A>, AllocError>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut iter = iter.into_iter();
        let len = iter.len();
        self.try_alloc_slice_fill_with(len, |_| iter.next().expect("iterator shorter than ExactSizeIterator len"))
    }

    /// Copy `slice` into a `Shared`-flavor chunk and return an [`ArenaArc`]
    /// handle safe for cross-thread sharing. Panics if the allocator fails.
    pub fn alloc_slice_copy_shared<T: Copy + Send + Sync>(&self, slice: &[T]) -> ArenaArc<[T], A>
    where
        A: Send + Sync,
    {
        match self.try_alloc_slice_copy_shared(slice) {
            Ok(r) => r,
            Err(_) => panic_alloc(),
        }
    }

    /// Copy `slice` into a `Shared`-flavor chunk and return an [`ArenaArc`]
    /// handle, returning Err([`AllocError`]) instead of panicking if the
    /// backing allocator cannot satisfy the request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] on allocator failure.
    pub fn try_alloc_slice_copy_shared<T: Copy + Send + Sync>(&self, slice: &[T]) -> Result<ArenaArc<[T], A>, AllocError>
    where
        A: Send + Sync,
    {
        // SAFETY: T: Copy, no drop.
        unsafe {
            let reservation = self.reserve_slice::<T>(slice.len(), ChunkSharing::Shared, false)?;
            let ptr = reservation.ptr.cast::<T>();
            core::ptr::copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), slice.len());
            self.commit_slice::<T>(reservation, slice.len());
            Ok(ArenaArc::from_raw_slice(ptr, slice.len()))
        }
    }

    /// Reserve `layout`-sized space for a slice in a chunk of the given
    /// `sharing` flavor, optionally also reserving a `DropEntry` slot
    /// just before the slice. Returns a [`SliceReservation`] that the
    /// caller must finalize via [`Self::commit_slice`] AFTER successful
    /// element initialization.
    ///
    /// The chunk refcount is NOT incremented here; the slot+entry are
    /// "leaked" in-chunk if the caller never commits.
    unsafe fn reserve_slice<T>(&self, len: usize, sharing: ChunkSharing, register_drop: bool) -> Result<SliceReservation<A>, AllocError> {
        let elem_size = size_of::<T>();
        let total = elem_size.checked_mul(len).ok_or(AllocError)?;
        let layout = Layout::from_size_align(total, align_of::<T>()).map_err(|_layout_err| AllocError)?;
        if layout.align() > CHUNK_ALIGN {
            return Err(AllocError);
        }
        let chunk = self.try_get_chunk_for(sharing, layout, register_drop)?;
        // SAFETY: chunk has space (worst-case sized).
        let h = unsafe { chunk.as_ref() };
        if register_drop {
            // Reserve DropEntry + slice in one bump-cursor transaction.
            let cur = h.bump.get();
            let entry_addr = crate::constants::checked_align_up(cur, align_of::<DropEntry>()).ok_or(AllocError)?;
            let after_entry = entry_addr.checked_add(size_of::<DropEntry>()).ok_or(AllocError)?;
            let value_addr = crate::constants::checked_align_up(after_entry, layout.align()).ok_or(AllocError)?;
            let end = value_addr.checked_add(layout.size()).ok_or(AllocError)?;
            if end > h.total_size {
                return Err(AllocError);
            }
            h.bump.set(end);
            let base = chunk.as_ptr().cast::<u8>();
            // SAFETY: addresses are within the chunk and well-aligned.
            let entry = unsafe { NonNull::new_unchecked(base.add(entry_addr).cast::<DropEntry>()) };
            // SAFETY: same.
            let ptr = unsafe { NonNull::new_unchecked(base.add(value_addr)) };
            Ok(SliceReservation {
                chunk,
                ptr,
                entry: Some(entry),
                layout,
            })
        } else {
            let ptr = ChunkHeader::<A>::try_alloc(chunk, layout).expect("worst-case sizing");
            Ok(SliceReservation {
                chunk,
                ptr,
                entry: None,
                layout,
            })
        }
    }

    /// Commit a previously-reserved slice once initialization has
    /// succeeded: link the drop entry (if reserved) with the actual
    /// element count, bump the chunk refcount for the upcoming handle,
    /// and account the allocation in stats.
    ///
    /// # Safety
    ///
    /// Caller must have written `len` valid `T` values starting at
    /// `reservation.ptr` (typed as `T`). `len` must be the same length
    /// originally requested in [`Self::reserve_slice`].
    #[expect(
        clippy::needless_pass_by_value,
        reason = "by-value consumption ensures the reservation can't be used twice"
    )]
    unsafe fn commit_slice<T>(&self, reservation: SliceReservation<A>, len: usize) {
        // SAFETY: chunk is alive (we hold the arena's hold on it via current_*).
        let h = unsafe { reservation.chunk.as_ref() };
        if let Some(entry) = reservation.entry {
            // SAFETY: entry slot is writable; chunk is alive.
            unsafe { h.link_drop_entry(entry, slice_drop_shim::<T>, len) };
        }
        h.inc_ref();
        self.charge_alloc_stats(reservation.layout.size());
    }

    /// Bump-allocate `layout` from a chunk of the requested sharing flavor
    /// and bump the chunk refcount by 1 for the upcoming string handle.
    /// Used by [`ArenaRcStr::from_str`](crate::ArenaRcStr) /
    /// [`ArenaArcStr::from_str`](crate::ArenaArcStr) and the `Allocator`
    /// impl on `&Arena<A>`.
    ///
    /// # Safety
    ///
    /// Caller must wrap the returned pointer in a handle (or deallocate
    /// path) that decrements the refcount.
    pub(crate) unsafe fn bump_alloc_for_str(&self, layout: Layout, sharing: ChunkSharing) -> NonNull<u8> {
        match self.try_get_chunk_for(sharing, layout, false) {
            Ok(chunk) => {
                // SAFETY: chunk has space.
                let h = unsafe { chunk.as_ref() };
                let ptr = ChunkHeader::<A>::try_alloc(chunk, layout).expect("worst-case sizing");
                h.inc_ref();
                self.charge_alloc_stats(layout.size());
                ptr
            }
            Err(_) => panic_alloc(),
        }
    }

    // ---- collection factories -------------------------------------------

    /// Create a new, empty growable [`ArenaString`](crate::ArenaString) backed by this
    /// arena. No allocation is performed until the first push.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = harena::Arena::new();
    /// let mut s = arena.new_string();
    /// s.push_str("hello");
    /// assert_eq!(&*s, "hello");
    /// ```
    #[must_use]
    pub fn new_string(&self) -> crate::ArenaString<'_, A> {
        crate::ArenaString::new_in(self)
    }

    /// Create a new growable [`ArenaString`](crate::ArenaString) backed by this arena, with
    /// at least `cap` bytes of pre-allocated capacity.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails on the initial
    /// allocation.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = harena::Arena::new();
    /// let mut s = arena.string_with_capacity(64);
    /// s.push_str("preallocated");
    /// assert!(s.capacity() >= 64);
    /// ```
    #[must_use]
    pub fn string_with_capacity(&self, cap: usize) -> crate::ArenaString<'_, A> {
        crate::ArenaString::with_capacity_in(cap, self)
    }

    /// Create a new, empty growable [`ArenaVec`](crate::ArenaVec) backed by this arena.
    /// No allocation is performed until the first push.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = harena::Arena::new();
    /// let mut v = arena.new_vec::<u32>();
    /// v.push(1);
    /// v.push(2);
    /// assert_eq!(v.as_slice(), &[1, 2]);
    /// ```
    #[must_use]
    pub fn new_vec<T>(&self) -> crate::ArenaVec<'_, T, A> {
        crate::ArenaVec::new_in(self)
    }

    /// Create a new growable [`ArenaVec`](crate::ArenaVec) backed by this arena, with
    /// capacity for at least `cap` elements pre-allocated.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails on the initial
    /// allocation.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = harena::Arena::new();
    /// let mut v = arena.vec_with_capacity::<u32>(100);
    /// for i in 0..50 { v.push(i); }
    /// assert!(v.capacity() >= 100);
    /// ```
    #[must_use]
    pub fn vec_with_capacity<T>(&self, cap: usize) -> crate::ArenaVec<'_, T, A> {
        crate::ArenaVec::with_capacity_in(cap, self)
    }

    /// Internal grow helper used by [`ArenaString`](crate::ArenaString).
    ///
    /// Mirrors the logic of `<&Arena<A> as Allocator>::grow` but
    /// increments [`ArenaStats::string_relocations`] (not
    /// [`ArenaStats::allocator_relocations`]) when a relocation occurs.
    ///
    /// # Safety
    ///
    /// Caller must follow the same rules as `Allocator::grow`: `ptr`
    /// must have come from a previous allocation through this arena
    /// with `old_layout`; `new_layout.size() >= old_layout.size()` and
    /// `new_layout.align() == old_layout.align()`.
    pub(crate) unsafe fn grow_for_string(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>, AllocError> {
        // SAFETY: caller's contract — ptr is one of ours.
        let chunk: NonNull<ChunkHeader<A>> = unsafe { crate::chunk_header::header_for(ptr) };
        // SAFETY: chunk is alive (caller holds the allocation).
        let header = unsafe { chunk.as_ref() };
        let chunk_base = chunk.as_ptr() as usize;
        let buffer_start = ptr.as_ptr() as usize;
        let buffer_end_offset = (buffer_start - chunk_base) + old_layout.size();
        let cur = header.bump.get();
        let extra = new_layout.size() - old_layout.size();
        let new_end = cur.checked_add(extra).ok_or(AllocError)?;
        if buffer_end_offset == cur && new_end <= header.total_size {
            header.bump.set(new_end);
            self.charge_alloc_stats(extra);
            return Ok(ptr);
        }

        // Slow path: allocate a new block, copy, dec_ref the old chunk.
        // SAFETY: bump_alloc_for_str charges stats for the new buffer
        // and bumps the new chunk's refcount.
        let new_ptr = unsafe { self.bump_alloc_for_str(new_layout, ChunkSharing::Local) };
        // SAFETY: source initialized for old size; destinations don't overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), old_layout.size());
            // Release the old block's refcount.
            if header.dec_ref() {
                crate::chunk_header::teardown_chunk(chunk, true);
            }
        }
        self.bump_string_relocation();
        Ok(new_ptr)
    }
}

/// Compute the worst-case byte requirement for an allocation of `layout`,
/// optionally including a `DropEntry`.
#[inline]
fn compute_worst_case_size(layout: Layout, has_drop: bool) -> usize {
    let mut size = layout.size();
    // Up to align-1 bytes of leading padding.
    size += layout.align().saturating_sub(1);
    if has_drop {
        size += size_of::<DropEntry>();
        size += align_of::<DropEntry>().saturating_sub(1);
    }
    size
}

/// Drop shim for a slice value `[T]`. The slice's length is stored in
/// the entry's dedicated `slice_len` field.
unsafe fn slice_drop_shim<T>(entry: *mut DropEntry) {
    // SAFETY: entry was constructed by `commit_slice` with `len`
    // stored in the dedicated `slice_len` field.
    unsafe {
        let len = (*entry).slice_len;
        let after_entry = entry.byte_add(size_of::<DropEntry>());
        let align = align_of::<T>();
        let misalign = (after_entry.cast::<u8>() as usize) & (align - 1);
        let padding = if misalign == 0 { 0 } else { align - misalign };
        let value_ptr = after_entry.byte_add(padding).cast::<T>();
        let slice = core::slice::from_raw_parts_mut(value_ptr, len);
        core::ptr::drop_in_place(slice);
    }
}

/// Drop guard used by `alloc_slice_fill_with` and friends to clean up
/// partially-initialized slices on panic.
struct SliceInitGuard<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Drop for SliceInitGuard<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr..ptr+len` are initialized by the caller before the
        // guard's `len` is incremented.
        unsafe {
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(self.ptr, self.len));
        }
    }
}

#[inline(never)]
#[cold]
#[expect(clippy::panic, reason = "panicking allocation entry points panic on alloc failure by design")]
fn panic_alloc() -> ! {
    panic!("harena2: allocator returned AllocError");
}

impl Default for Arena<Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allocator + Clone> Drop for Arena<A> {
    fn drop(&mut self) {
        // We must NOT call `Box::from_raw(inner)` directly here: doing
        // so would create a `Unique` retag of the entire `ArenaInner`
        // for the duration of `ArenaInner::drop`, conflicting with the
        // chunks' back-pointer accesses (which still use the original
        // `SharedReadWrite` tag from `Box::leak`).
        //
        // Instead, perform cleanup via `&self` (no `Unique` retag),
        // mark the arena dropped, and conditionally free `ArenaInner`'s
        // storage either now (if no chunks survive) or later (when the
        // last surviving chunk frees itself — see
        // `chunk_header::free_chunk`).
        // SAFETY: owner-thread access; inner has not been freed.
        unsafe {
            let inner = self.inner.as_ref();

            // Release the arena's hold on its current chunks (if any).
            // Each chunk's refcount falls by 1; if no handles outstand,
            // the chunk tears down (and is added to the cache or freed
            // back to A).
            if let Some(chunk) = inner.current_local.take()
                && chunk.as_ref().dec_ref()
            {
                crate::chunk_header::teardown_chunk(chunk, true);
            }
            if let Some(chunk) = inner.current_shared.take()
                && chunk.as_ref().dec_ref()
            {
                crate::chunk_header::teardown_chunk(chunk, true);
            }
            // Drain the cache (these chunks have refcount=0).
            while let Some(chunk) = inner.try_pop_cache() {
                crate::chunk_header::free_chunk(chunk);
            }

            // Publish the "arena dropped" flag with Release ordering so
            // that any later Acquire load on a non-owner thread (in
            // `register_chunk_freed`) sees this store and the cleanup
            // it summarizes.
            inner.arena_dropped.store(true, core::sync::atomic::Ordering::Release);

            // If no chunks survived the cleanup above, free `ArenaInner`
            // now. Otherwise the last surviving chunk's free path will
            // do it.
            if inner.outstanding_chunks.load(core::sync::atomic::Ordering::Acquire) == 0 {
                ArenaInner::free_storage(self.inner);
            }
        }
    }
}

impl<A: Allocator + Clone> fmt::Debug for Arena<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena").field("stats", &self.stats()).finish_non_exhaustive()
    }
}
