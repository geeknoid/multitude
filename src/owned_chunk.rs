use core::cell::Cell;
use core::marker::PhantomData;
use core::ptr::NonNull;

use allocator_api2::alloc::Allocator;

use crate::chunk_header::{ChunkHeader, free_chunk, release_chunk_ref, release_chunk_ref_n};
use crate::chunk_ref::ChunkRef;
use crate::chunk_sharing::ChunkSharing;
use crate::constants::LARGE_INITIAL_SHARED_REFCOUNT;

/// RAII handle for a chunk +1 refcount on the owner thread.
///
/// Drop releases the ref; [`Self::into_raw`] suppresses Drop and hands
/// the pointer back. Used for arena-internal +1 holds (current chunks,
/// pinned list).
pub struct OwnedChunk<A: Allocator + Clone> {
    ptr: NonNull<ChunkHeader<A>>,
    _not_sync: PhantomData<*mut ()>,
    _allocator: PhantomData<A>,
}

impl<A: Allocator + Clone> OwnedChunk<A> {
    /// # Safety
    ///
    /// `ptr` must point to a live chunk and the caller must own
    /// exactly one refcount that they are transferring to this wrapper.
    /// The chunk's eventual release will happen on the owner thread.
    #[inline]
    pub const unsafe fn from_raw(ptr: NonNull<ChunkHeader<A>>) -> Self {
        Self {
            ptr,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Release ownership without dropping the refcount.
    #[inline]
    #[must_use]
    pub const fn into_raw(self) -> NonNull<ChunkHeader<A>> {
        let p = self.ptr;
        core::mem::forget(self);
        p
    }

    #[inline]
    #[must_use]
    pub const fn as_ref(&self) -> ChunkRef<'_, A> {
        // SAFETY: we own a +1 refcount, so the chunk is live for `'_`.
        unsafe { ChunkRef::new(self.ptr) }
    }

    #[inline]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<ChunkHeader<A>> {
        self.ptr
    }
}

impl<A: Allocator + Clone> Drop for OwnedChunk<A> {
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "sharing read + arcs_issued read + release share the live-chunk safety invariant"
    )]
    fn drop(&mut self) {
        // SAFETY: we own a hold on a live chunk; release on the owner thread.
        // For Local chunks the hold is the slot's transient +1; for Shared
        // chunks (deferred-reconciliation refcount scheme) the hold is the
        // unused portion of the LARGE pre-payment, computed as
        // `LARGE - arcs_issued`.
        unsafe {
            let sharing = (*self.ptr.as_ptr()).sharing.get();
            match sharing {
                ChunkSharing::Local => release_chunk_ref(self.ptr, true),
                ChunkSharing::Shared => {
                    // Reconcile the chunk's atomic refcount with the
                    // non-atomic `arcs_issued` accumulator. After this
                    // `fetch_sub`, the refcount equals the count of
                    // outstanding live `Arc`s (clones minus drops). If
                    // that's zero, teardown.
                    let m = (*self.ptr.as_ptr()).arcs_issued.replace(0);
                    let release_n = LARGE_INITIAL_SHARED_REFCOUNT - m;
                    release_chunk_ref_n(self.ptr, release_n, true);
                }
            }
        }
    }
}

/// RAII handle for an exclusively-owned chunk at refcount **zero**
/// (cache entries). Drop calls `free_chunk`; [`Self::revive`]
/// reinitializes for reuse and produces an [`OwnedChunk`].
pub struct RetiredChunk<A: Allocator + Clone> {
    ptr: NonNull<ChunkHeader<A>>,
    _not_sync: PhantomData<*mut ()>,
    _allocator: PhantomData<A>,
}

impl<A: Allocator + Clone> RetiredChunk<A> {
    /// # Safety
    ///
    /// `ptr` must point to a chunk whose refcount is zero, with no
    /// outstanding references and no queued drop entries.
    #[inline]
    pub const unsafe fn from_raw(ptr: NonNull<ChunkHeader<A>>) -> Self {
        Self {
            ptr,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub const fn into_raw(self) -> NonNull<ChunkHeader<A>> {
        let p = self.ptr;
        core::mem::forget(self);
        p
    }

    #[inline]
    #[must_use]
    pub const fn as_ref(&self) -> ChunkRef<'_, A> {
        // SAFETY: we exclusively own a chunk whose backing memory is live.
        unsafe { ChunkRef::new(self.ptr) }
    }

    /// Reinitialize the chunk (refcount = `initial_refcount`, requested
    /// sharing flavor) and transfer ownership of the resulting hold.
    #[inline]
    #[must_use]
    pub fn revive(self, sharing: crate::chunk_sharing::ChunkSharing, initial_refcount: usize) -> OwnedChunk<A> {
        let ptr = self.into_raw();
        // SAFETY: we exclusively owned the retired chunk.
        unsafe { crate::chunk_header::revive_cached_chunk(ptr, sharing, initial_refcount) };
        // SAFETY: revive set refcount to `initial_refcount`; transfer that hold to OwnedChunk.
        unsafe { OwnedChunk::from_raw(ptr) }
    }
}

impl<A: Allocator + Clone> Drop for RetiredChunk<A> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: refcount is zero and we exclusively own the chunk.
        unsafe { free_chunk(self.ptr) };
    }
}

/// Owner-thread slot that ALWAYS points to a chunk header — either a
/// real (smart-pointer-bearing) chunk or a per-arena sentinel header.
///
/// The sentinel is an inert [`ChunkHeader<A>`] embedded in the
/// owning [`ArenaInner`](crate::arena_inner::ArenaInner) with
/// `bump = total_size = 0`, so the bump fast-path's natural fit-check
/// (`end > total_size`) returns "doesn't fit" and falls through to the
/// slow path without any explicit "is the slot empty?" branch. This
/// removes the `Option` discriminant from the hot path entirely.
///
/// Smart pointers are never allocated out of a sentinel (the fit-check
/// always fails first), so the refcount/teardown machinery never
/// touches sentinel headers. The slot's [`Self::take`] returns `None`
/// when the slot still holds the sentinel, ensuring an [`OwnedChunk`]
/// can never wrap a sentinel and run its `Drop`.
pub struct ChunkSlot<A: Allocator + Clone> {
    cell: Cell<NonNull<ChunkHeader<A>>>,
    sentinel: NonNull<ChunkHeader<A>>,
    _allocator: PhantomData<A>,
}

impl<A: Allocator + Clone> ChunkSlot<A> {
    /// Construct a slot pointing at the supplied sentinel header.
    ///
    /// # Safety
    ///
    /// `sentinel` must point at an initialized [`ChunkHeader<A>`]
    /// configured as a sentinel (`bump == 0`, `total_size == 0`,
    /// `pinned == false`) and must outlive this `ChunkSlot`.
    #[inline]
    #[must_use]
    pub const unsafe fn new(sentinel: NonNull<ChunkHeader<A>>) -> Self {
        Self {
            cell: Cell::new(sentinel),
            sentinel,
            _allocator: PhantomData,
        }
    }

    /// Return a [`ChunkRef`] for whatever the slot currently points at
    /// (sentinel or real chunk). Callers do *not* need to special-case
    /// the sentinel: it has `total_size == 0`, so any subsequent
    /// fit-check will fail and the slow path takes over.
    #[inline]
    #[must_use]
    pub const fn peek(&self) -> ChunkRef<'_, A> {
        // SAFETY: the slot's pointer is always either the sentinel
        // (alive for the lifetime of the parent ArenaInner, which
        // strictly outlives `self`) or a real chunk on which the slot
        // owns a +1 refcount.
        unsafe { ChunkRef::new(self.cell.get()) }
    }

    /// Replace the held chunk with the sentinel and return any *real*
    /// chunk that was there. Returns `None` when the slot was already
    /// holding the sentinel.
    #[inline]
    #[must_use]
    pub fn take(&self) -> Option<OwnedChunk<A>> {
        let prior = self.cell.replace(self.sentinel);
        if prior == self.sentinel {
            None
        } else {
            // SAFETY: a non-sentinel pointer in the slot is a real
            // chunk on which the slot held a +1 refcount; transfer it.
            Some(unsafe { OwnedChunk::from_raw(prior) })
        }
    }

    /// Install a real chunk into the slot.
    ///
    /// Every call site reaches `set` only after a `take` (or from
    /// arena construction with the slot still pointing at the
    /// sentinel). To stay safe under a future refactor that breaks
    /// that discipline, we `replace` first and drop any prior real
    /// chunk *after* the new one is installed — that way a re-entrant
    /// `Drop` running off the prior chunk observes a populated slot
    /// (matching the invariant relied on in
    /// `Arena::try_get_chunk_for_*`).
    #[inline]
    pub fn set(&self, chunk: OwnedChunk<A>) {
        let prior = self.cell.replace(chunk.into_raw());
        debug_assert!(prior == self.sentinel, "ChunkSlot::set called on a slot holding a real chunk");
        if prior != self.sentinel {
            // SAFETY: prior was a real chunk previously installed via
            // `set`; we own its +1.
            drop(unsafe { OwnedChunk::from_raw(prior) });
        }
    }

    /// Drop any real chunk currently held, leaving the sentinel.
    #[inline]
    pub fn clear(&self) {
        let _ = self.take();
    }
}

impl<A: Allocator + Clone> Drop for ChunkSlot<A> {
    #[inline]
    fn drop(&mut self) {
        self.clear();
    }
}

// SAFETY: the slot is only accessed on the owner thread; the `Send`
// impl exists only to permit shipping the parent `ArenaInner`. `Sync`
// is intentionally omitted (`Cell` is `!Sync`).
unsafe impl<A: Allocator + Clone + Send> Send for ChunkSlot<A> {}
