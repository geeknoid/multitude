use core::cell::{Cell, UnsafeCell};
#[cfg(test)]
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::Allocator;

use crate::chunk_header::{ChunkHeader, init_sentinel_header};
use crate::chunk_ref::pop_from_intrusive_list;
use crate::chunk_sharing::ChunkSharing;
use crate::owned_chunk::RetiredChunk;
use crate::sync::{AtomicUsize, Ordering, fence};

/// The contribution that the live [`Arena`](crate::Arena) handle makes
/// to [`ArenaInner::outstanding_chunks`]. Chosen to be vastly larger
/// than any plausible live-chunk count, so the counter encodes both
/// "is the arena still alive?" and "how many chunks are still alive?"
/// in a single atomic word.
///
/// Counter invariants (see [`ArenaInner::outstanding_chunks`] doc):
/// - Arena alive: `counter >= HANDLE_HOLD` (handle's contribution + chunks).
/// - Arena dropped: `counter < HANDLE_HOLD` (handle's contribution removed).
/// - The "is the arena dropped?" check used by cross-thread teardown is
///   simply `counter.load() < HANDLE_HOLD`, on the same cache line that
///   teardown is about to `fetch_sub` against — no separate atomic field.
///
/// On 64-bit targets this is `1 << 32`, leaving 4 billion live-chunk
/// addresses; on 32-bit targets this is `1 << 16`, capping at 64 K
/// concurrently-live chunks (each chunk is at minimum 8 KiB, so 64 K
/// chunks is 512 MiB of arena memory — well past any plausible
/// allocation pattern on a 32-bit system).
pub const HANDLE_HOLD: usize = if usize::BITS >= 64 { 1 << 32 } else { 1 << 16 };

/// Wraps a value to live on its own cache line, preventing false
/// sharing when the value is touched cross-thread while neighboring
/// fields are touched only by the owner thread. The 128-byte alignment
/// covers `x86_64`, ARM64 (Apple Silicon, with adjacent-line prefetch),
/// and POWER coherence boundaries.
#[repr(C, align(128))]
pub struct CachePadded<T>(pub T);

impl<T> core::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// The cross-thread-accessible slice of arena state. Lives on the heap
/// (via `Box::leak`) so that smart pointers (which find it via the
/// chunk back-pointer) can coordinate with the owner thread even after
/// the [`Arena`](crate::Arena) handle has been dropped. Holds only the
/// fields touched during chunk teardown — including teardown running
/// on a non-owner thread when the last `Arc<T>` from a `Shared` chunk
/// drops there. Owner-only state (current chunks, pinned list,
/// allocator, configuration) lives directly on the `Arena<A>` handle.
pub struct ArenaInner<A: Allocator + Clone> {
    /// Cross-thread liveness counter encoding TWO pieces of state:
    ///
    /// 1. **"How many chunks are still alive?"** — incremented (`+1`)
    ///    on every chunk creation, decremented (`-1`) on every chunk
    ///    free. When the counter reaches 0 the inner storage is freed.
    ///
    /// 2. **"Is the arena handle still alive?"** — encoded as whether
    ///    the counter is `>= HANDLE_HOLD`. The arena handle's
    ///    contribution to the counter is `HANDLE_HOLD` (a constant
    ///    much larger than any plausible live-chunk count), released
    ///    on `Arena::Drop` via `fetch_sub(HANDLE_HOLD)`.
    ///
    /// Cross-thread teardown reads this counter to decide whether the
    /// chunk cache is still useful: if `counter < HANDLE_HOLD`, the
    /// arena handle is gone and there's no point caching the chunk.
    /// The check is a single `Acquire` load on a line the teardown
    /// thread is about to `fetch_sub` against anyway — same cache
    /// traffic as a separate `arena_dropped` flag would have caused,
    /// minus one cache line of struct overhead.
    pub outstanding_chunks: CachePadded<AtomicUsize>,

    /// Cache of empty normal chunks awaiting reuse. Accessed from the
    /// owner thread on alloc-path cache pop, and from any thread that
    /// runs the last teardown on a `Normal` chunk. Cross-thread
    /// teardowns of `Shared` chunks bypass the cache (`can_cache_class`
    /// blocks them) so the `Cell`-based representation is sound.
    pub chunk_cache_head: Cell<Option<NonNull<ChunkHeader<A>>>>,
    pub chunk_cache_len: Cell<usize>,
    pub chunk_cache_capacity: usize,

    /// Per-arena `Local`-flavored sentinel header. Initialized by
    /// [`Self::init_sentinels`] once the parent `ArenaInner` is at its
    /// final heap address (so the embedded `arena` back-pointer can be
    /// set to `NonNull::from(self)`). Sentinels carry `bump = 0`,
    /// `total_size = 0`, and `pinned = false`, so the alloc fast path's
    /// natural fit-check handles "slot is empty" without an extra
    /// branch. They are never reachable from any smart pointer (the
    /// fit-check fails before any allocation can succeed against them),
    /// so their refcount and teardown paths are never exercised.
    pub sentinel_local: UnsafeCell<MaybeUninit<ChunkHeader<A>>>,
    pub sentinel_shared: UnsafeCell<MaybeUninit<ChunkHeader<A>>>,

    _phantom: PhantomData<A>,
}

impl<A: Allocator + Clone> ArenaInner<A> {
    /// Construct a fresh inner with the given chunk-cache capacity.
    /// `outstanding_chunks` starts at [`HANDLE_HOLD`] to account for
    /// the [`Arena`](crate::Arena) handle's own contribution; the
    /// arena handle's `Drop` later releases this via
    /// [`Self::register_release`] with `n = HANDLE_HOLD`.
    ///
    /// The sentinel slots are left uninitialized; the caller must
    /// invoke [`Self::init_sentinels`] before exposing the
    /// `ArenaInner` to the alloc paths.
    #[inline]
    #[cfg_attr(not(loom), expect(clippy::missing_const_for_fn, reason = "loom's AtomicUsize::new is not const"))]
    pub fn new(chunk_cache_capacity: usize) -> Self {
        Self {
            outstanding_chunks: CachePadded(AtomicUsize::new(HANDLE_HOLD)),
            chunk_cache_head: Cell::new(None),
            chunk_cache_len: Cell::new(0),
            chunk_cache_capacity,
            sentinel_local: UnsafeCell::new(MaybeUninit::uninit()),
            sentinel_shared: UnsafeCell::new(MaybeUninit::uninit()),
            _phantom: PhantomData,
        }
    }

    /// Initialize the embedded sentinels.
    ///
    /// # Safety
    ///
    /// Must be called exactly once, after the `ArenaInner` is at its
    /// final heap address (typically the address returned by
    /// `Box::leak`) and before any other use of the sentinels.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw-cell projection + NonNull::new_unchecked + init_sentinel_header all share one freshly-leaked-and-unique invariant"
    )]
    pub unsafe fn init_sentinels(this: NonNull<Self>, allocator: &A) {
        // SAFETY: caller holds exclusive access to a freshly-leaked
        // ArenaInner; the sentinel cells are still uninit.
        unsafe {
            let inner_ref = this.as_ref();
            let local_ptr = inner_ref.sentinel_local.get();
            init_sentinel_header(
                NonNull::new_unchecked((*local_ptr).as_mut_ptr()),
                this,
                ChunkSharing::Local,
                allocator.clone(),
            );
            let shared_ptr = inner_ref.sentinel_shared.get();
            init_sentinel_header(
                NonNull::new_unchecked((*shared_ptr).as_mut_ptr()),
                this,
                ChunkSharing::Shared,
                allocator.clone(),
            );
        }
    }

    /// Pointer to the `Local`-flavored sentinel header. Only valid
    /// after [`Self::init_sentinels`] has run.
    #[inline]
    #[must_use]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "cell projection + NonNull::new_unchecked share one initialized-by-init_sentinels invariant"
    )]
    pub fn sentinel_local_ptr(&self) -> NonNull<ChunkHeader<A>> {
        // SAFETY: init_sentinels initialized the cell.
        unsafe { NonNull::new_unchecked((*self.sentinel_local.get()).as_mut_ptr()) }
    }

    /// Pointer to the `Shared`-flavored sentinel header. Only valid
    /// after [`Self::init_sentinels`] has run.
    #[inline]
    #[must_use]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "cell projection + NonNull::new_unchecked share one initialized-by-init_sentinels invariant"
    )]
    pub fn sentinel_shared_ptr(&self) -> NonNull<ChunkHeader<A>> {
        // SAFETY: init_sentinels initialized the cell.
        unsafe { NonNull::new_unchecked((*self.sentinel_shared.get()).as_mut_ptr()) }
    }

    /// Push `chunk` into the cache, transferring ownership of the
    /// retired chunk to the cache list. On capacity exhaustion the
    /// chunk is returned in `Err` so the caller can drop it (which
    /// runs `free_chunk`).
    ///
    /// Caller is the owner thread.
    #[inline]
    pub fn try_push_to_cache(&self, chunk: RetiredChunk<A>) -> Result<(), RetiredChunk<A>> {
        let len = self.chunk_cache_len.get();
        if len >= self.chunk_cache_capacity {
            return Err(chunk);
        }
        chunk.as_ref().push_into_intrusive_list(&self.chunk_cache_head);
        let _ = chunk.into_raw();
        self.chunk_cache_len.set(len + 1);
        Ok(())
    }

    /// Pop a retired chunk from the cache. Returns `None` if empty.
    /// Dropping the returned `RetiredChunk` runs `free_chunk`; calling
    /// [`RetiredChunk::revive`] reinitializes it for reuse.
    #[inline]
    pub fn try_pop_cache(&self) -> Option<RetiredChunk<A>> {
        let head = pop_from_intrusive_list(&self.chunk_cache_head)?;
        self.chunk_cache_len.set(self.chunk_cache_len.get() - 1);
        // SAFETY: cache entries are exclusively-owned chunks at refcount 0.
        Some(unsafe { RetiredChunk::from_raw(head) })
    }

    /// Note that a contribution of `n` to `outstanding_chunks` has
    /// been released. Used in two places:
    ///
    /// - Per-chunk free: `n = 1` (a chunk's backing memory has been
    ///   reclaimed; the chunk's `+1` contribution goes away).
    /// - [`Arena::Drop`](crate::Arena): `n = HANDLE_HOLD` (the arena
    ///   handle's contribution goes away; this also flips the encoded
    ///   "arena dropped" predicate observable to cross-thread teardown).
    ///
    /// Returns `true` iff this decrement observed `prev == n`, meaning
    /// **this caller is the unique last reclaimer** and is then
    /// responsible for freeing `ArenaInner`'s own storage via
    /// [`Self::free_storage`].
    ///
    /// Takes `NonNull<Self>` (not `&self`) to avoid a function-scoped
    /// strong protector while another thread that won the same race
    /// may try to acquire Unique access in `free_storage` — Stacked /
    /// Tree Borrows reject that interleaving.
    ///
    /// # Safety
    ///
    /// `inner` must point to a live `ArenaInner<A>` whose
    /// `outstanding_chunks` count includes the caller's contribution
    /// of at least `n`.
    #[cold]
    #[inline(never)]
    pub unsafe fn register_release(inner: NonNull<Self>, n: usize) -> bool {
        // SAFETY: caller's contract — `inner` is live. `CachePadded` is
        // `#[repr(C)]` with the inner `AtomicUsize` as its sole field at
        // offset 0, so the cast is sound.
        let counter_ptr: *const AtomicUsize = unsafe { &raw const (*inner.as_ptr()).outstanding_chunks }.cast::<AtomicUsize>();
        // SAFETY: AtomicUsize is interior-mutable.
        let counter = unsafe { &*counter_ptr };
        let prev = counter.fetch_sub(n, Ordering::Release);
        debug_assert!(prev >= n, "outstanding_chunks underflow");
        if prev == n {
            fence(Ordering::Acquire);
            true
        } else {
            false
        }
    }

    /// Per-chunk-free convenience wrapper; equivalent to
    /// [`Self::register_release`] with `n = 1`.
    ///
    /// # Safety
    ///
    /// Same as [`Self::register_release`].
    #[cold]
    #[inline(never)]
    pub unsafe fn register_chunk_freed(inner: NonNull<Self>) -> bool {
        // SAFETY: caller's contract.
        unsafe { Self::register_release(inner, 1) }
    }

    /// Returns `true` iff the arena handle has dropped (i.e. its
    /// `HANDLE_HOLD` contribution to `outstanding_chunks` has been
    /// released). Used by cross-thread chunk teardown to skip the
    /// chunk cache when caching is no longer useful.
    #[inline]
    pub fn arena_dropped(&self) -> bool {
        self.outstanding_chunks.load(Ordering::Acquire) < HANDLE_HOLD
    }

    /// Free this `ArenaInner` storage via `Box::from_raw`.
    ///
    /// # Safety
    ///
    /// `inner` must be a pointer originally returned by
    /// `Box::leak(Box::new(ArenaInner::new(...)))` and must not have
    /// been freed previously. There must be no outstanding references
    /// to the `ArenaInner` (no chunks holding back-pointers).
    #[cold]
    #[inline(never)]
    pub unsafe fn free_storage(inner: NonNull<Self>) {
        // SAFETY: caller's contract.
        unsafe {
            let _ = alloc::boxed::Box::from_raw(inner.as_ptr());
        }
    }
}

// `ArenaInner` has NO `Drop` impl for chunk lifecycles. Cleanup of
// chunks is owned by `Arena::drop`. The `Drop` below only handles the
// embedded sentinel headers (their allocator clones); no other field
// of `ArenaInner` requires custom drop logic.
impl<A: Allocator + Clone> Drop for ArenaInner<A> {
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "both sentinel drops share one initialized-by-init_sentinels invariant"
    )]
    fn drop(&mut self) {
        // SAFETY: `init_sentinels` is invoked unconditionally during
        // arena construction (the only path that produces a heap-
        // allocated `ArenaInner`), so both sentinel cells are
        // initialized for the lifetime of `self`.
        unsafe {
            (*self.sentinel_local.get()).assume_init_drop();
            (*self.sentinel_shared.get()).assume_init_drop();
        }
    }
}

#[cfg(test)]
impl<A: Allocator + Clone> fmt::Debug for ArenaInner<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let counter = self.outstanding_chunks.load(Ordering::Relaxed);
        let arena_alive = counter >= HANDLE_HOLD;
        let live_chunks = if arena_alive { counter - HANDLE_HOLD } else { counter };
        f.debug_struct("ArenaInner")
            .field("chunk_cache_capacity", &self.chunk_cache_capacity)
            .field("chunk_cache_len", &self.chunk_cache_len.get())
            .field("live_chunks", &live_chunks)
            .field("arena_alive", &arena_alive)
            .finish_non_exhaustive()
    }
}
