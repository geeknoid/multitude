use core::marker::PhantomData;
use core::ptr::NonNull;

use allocator_api2::alloc::Allocator;

use crate::arena_inner::ArenaInner;
use crate::chunk_header::ChunkHeader;
use crate::chunk_sharing::ChunkSharing;
use crate::chunk_size_class::ChunkSizeClass;
use crate::drop_entry::DropEntry;
use crate::entry_layout::entry_value_offsets_unchecked;

/// Safe accessor over a live [`ChunkHeader<A>`]. Construction is
/// `unsafe` (caller asserts liveness for `'a`); every field accessor
/// is safe.
///
/// `Copy` so it can be threaded through call chains without re-asserting
/// the invariant. The wrapper carries no runtime data beyond the
/// `NonNull<ChunkHeader<A>>` it wraps.
pub struct ChunkRef<'a, A: Allocator + Clone> {
    ptr: NonNull<ChunkHeader<A>>,
    _marker: PhantomData<&'a ChunkHeader<A>>,
}

// Manual `Copy` / `Clone` impls: the `derive`-generated versions would
// require `A: Copy`/`A: Clone`, but the wrapper's data is just a
// pointer + marker, neither of which depends on `A`.
impl<A: Allocator + Clone> Copy for ChunkRef<'_, A> {}
#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "manual impl avoids the `A: Clone` bound that #[derive(Clone)] would require"
)]
impl<A: Allocator + Clone> Clone for ChunkRef<'_, A> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: Allocator + Clone> ChunkRef<'_, A> {
    /// Wrap a raw chunk pointer. The wrapper deliberately does NOT form
    /// an `&ChunkHeader` retag (Stacked / Tree Borrows hazard against
    /// chunk deallocation) — every accessor below uses raw projection.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a fully-initialized [`ChunkHeader<A>`] that
    /// remains live for the duration of `'a`. The caller MUST NOT
    /// deallocate or repurpose the chunk while any `ChunkRef<'a, A>`
    /// borrow derived from this constructor is alive.
    #[inline]
    pub const unsafe fn new(ptr: NonNull<ChunkHeader<A>>) -> Self {
        Self { ptr, _marker: PhantomData }
    }

    /// Chunk base address (i.e., the same address as the header).
    #[inline]
    pub const fn base(self) -> NonNull<u8> {
        self.ptr.cast()
    }

    /// Raw pointer to the chunk header for cases that must roundtrip
    /// through `NonNull<ChunkHeader<A>>` (e.g., handing the pointer to
    /// a function that internally calls `ChunkRef::new`).
    #[inline]
    pub const fn as_non_null(self) -> NonNull<ChunkHeader<A>> {
        self.ptr
    }

    /// Current bump cursor offset, measured from the chunk base.
    #[inline]
    pub fn bump_get(self) -> usize {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).bump.get() }
    }

    /// Set the bump cursor offset.
    #[inline]
    pub fn bump_set(self, value: usize) {
        // SAFETY: see `bump_get`.
        unsafe { (*self.ptr.as_ptr()).bump.set(value) };
    }

    /// Total chunk size in bytes (including the padded header).
    #[inline]
    pub fn total_size(self) -> usize {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).total_size }
    }

    /// Sharing flavor (`Local`/`Shared`).
    #[inline]
    pub fn sharing(self) -> ChunkSharing {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).sharing.get() }
    }

    /// Drop list head (intrusive linked list of values needing Drop).
    #[inline]
    pub fn drop_head(self) -> Option<NonNull<DropEntry>> {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).drop_head.get() }
    }

    /// Back-pointer to the owning arena.
    #[inline]
    pub fn arena(self) -> NonNull<ArenaInner<A>> {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).arena }
    }

    /// Size class (`Normal`/`Oversized`).
    #[inline]
    pub fn size_class(self) -> ChunkSizeClass {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).size_class }
    }

    /// Mark the chunk as pinned (set the `pinned` flag to `true`).
    #[inline]
    pub fn set_pinned(self) {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).pinned.set(true) };
    }

    /// Read the pinned flag.
    #[inline]
    pub fn pinned(self) -> bool {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { (*self.ptr.as_ptr()).pinned.get() }
    }

    /// Borrow the chunk header.
    ///
    /// # Safety
    ///
    /// Caller must ensure the borrow is consistent with all other
    /// access to the chunk for the duration of the returned lifetime.
    /// In practice this means single-threaded access on the owner
    /// thread (the `ChunkHeader`'s hot fields are owner-only `Cell`s).
    #[inline]
    pub const unsafe fn header_ref<'h>(self) -> &'h ChunkHeader<A> {
        // SAFETY: chunk is live (constructor invariant); caller's lifetime contract.
        unsafe { self.ptr.as_ref() }
    }

    /// `Local`-flavor specialization of [`Self::inc_ref`].
    ///
    /// # Safety
    ///
    /// The chunk must be `Local`-flavored.
    #[inline]
    pub unsafe fn inc_ref_local(self) {
        // SAFETY: caller's contract.
        unsafe { ChunkHeader::inc_ref_local(self.ptr) }
    }

    /// `Shared`-flavor specialization of [`Self::inc_ref`].
    ///
    /// # Safety
    ///
    /// The chunk must be `Shared`-flavored.
    #[inline]
    pub unsafe fn inc_ref_shared(self) {
        // SAFETY: caller's contract.
        unsafe { ChunkHeader::inc_ref_shared(self.ptr) }
    }

    /// Add `n` to the atomic refcount of a `Shared`-flavored chunk.
    /// Used by the deferred-reconciliation overflow guard.
    ///
    /// # Safety
    ///
    /// The chunk must be `Shared`-flavored.
    #[inline]
    pub unsafe fn fetch_add_ref_shared(self, n: usize) {
        // SAFETY: caller's contract.
        unsafe { ChunkHeader::add_ref_shared(self.ptr, n) }
    }

    /// Decrement the chunk refcount by 1, returning `true` iff the
    /// count reached zero (and the caller should run teardown).
    #[cfg(feature = "builders")]
    #[inline]
    pub fn dec_ref(self) -> bool {
        // SAFETY: chunk is live (constructor invariant).
        unsafe { ChunkHeader::dec_ref(self.ptr) }
    }

    /// Push `entry` onto the front of the doubly-linked drop list.
    ///
    /// # Safety
    ///
    /// `entry` must be a freshly-allocated, uninitialized `DropEntry`
    /// inside this chunk; see [`ChunkHeader::link_drop_entry`].
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw projection + unsafe link_drop_entry call share one safety invariant"
    )]
    pub unsafe fn link_drop_entry(self, entry: NonNull<DropEntry>, drop_fn: unsafe fn(*mut DropEntry), slice_len: usize) {
        // SAFETY: caller's contract; chunk is live.
        unsafe { (*self.ptr.as_ptr()).link_drop_entry(entry, drop_fn, slice_len) };
    }

    /// Unlink an `entry` from the doubly-linked drop list.
    ///
    /// # Safety
    ///
    /// `entry` must be a `DropEntry` previously linked into this chunk's
    /// drop list and not yet unlinked. See [`ChunkHeader::unlink_drop_entry`].
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "raw projection + unsafe unlink_drop_entry call share one safety invariant"
    )]
    pub unsafe fn unlink_drop_entry(self, entry: NonNull<DropEntry>) {
        // SAFETY: caller's contract; chunk is live.
        unsafe { (*self.ptr.as_ptr()).unlink_drop_entry(entry) };
    }

    /// If `[buffer_start, buffer_start + buffer_size)` ends exactly at
    /// the bump cursor, lower the cursor to `buffer_start + live_size`,
    /// reclaiming the unused tail. Returns whether reclamation
    /// happened.
    ///
    /// Used by collection types that shrink in place (`String::into_arena_str`,
    /// `Vec::into_arena_rc`) and by `Allocator::shrink`.
    #[inline]
    pub fn try_reclaim_tail(self, buffer_start: NonNull<u8>, buffer_size: usize, live_size: usize) -> bool {
        debug_assert!(live_size <= buffer_size);
        let chunk_base = self.base().as_ptr() as usize;
        let buffer_end_offset = (buffer_start.as_ptr() as usize - chunk_base) + buffer_size;
        if buffer_end_offset == self.bump_get() {
            let live_end_offset = (buffer_start.as_ptr() as usize - chunk_base) + live_size;
            self.bump_set(live_end_offset);
            true
        } else {
            false
        }
    }

    /// Try to grow `[buffer_start, buffer_start + old_size)` in place
    /// by advancing the bump cursor by `extra` bytes. Returns `true`
    /// iff the buffer ends at the bump cursor and the extension fits in
    /// the chunk.
    ///
    /// Used by `Allocator::grow` and `Arena::grow_for_string`.
    #[inline]
    pub fn try_grow_in_place(self, buffer_start: NonNull<u8>, old_size: usize, extra: usize) -> bool {
        let chunk_base = self.base().as_ptr() as usize;
        let buffer_end_offset = (buffer_start.as_ptr() as usize - chunk_base) + old_size;
        let cur = self.bump_get();
        let total = self.total_size();
        if buffer_end_offset == cur
            && let Some(new_end) = cur.checked_add(extra)
            && new_end <= total
        {
            self.bump_set(new_end);
            return true;
        }
        false
    }

    /// Bump-allocate a `DropEntry` slot followed by a value of the given
    /// layout, skipping the overflow checks. Used on the hot allocation
    /// paths after `try_get_chunk_for` has already proved (via
    /// worst-case sizing) that the request fits.
    ///
    /// # Safety
    ///
    /// The chunk must already have enough room (post-alignment) for a
    /// `DropEntry` header followed by a value of `value` layout. The
    /// caller must have established this via
    /// `try_get_chunk_for(.., has_drop=true)`.
    #[inline]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "DropEntry pointer is bump-aligned to align_of::<DropEntry>() before the cast"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "byte_add + NonNull::new_unchecked share the bump-cursor + chunk-bounds invariant"
    )]
    pub unsafe fn alloc_entry_value_slot_unchecked(self, value: core::alloc::Layout) -> (NonNull<DropEntry>, NonNull<u8>) {
        let cur = self.bump_get();
        // SAFETY: caller established that the request fits.
        let (entry_addr, value_addr, end) = unsafe { entry_value_offsets_unchecked(cur, value) };
        debug_assert!(end <= self.total_size(), "worst-case sizing must bound `end`");
        self.bump_set(end);
        let base = self.base();
        // SAFETY: offsets are valid within the chunk allocation.
        let entry = unsafe { NonNull::new_unchecked(base.as_ptr().add(entry_addr).cast::<DropEntry>()) };
        // SAFETY: offsets are valid within the chunk allocation.
        let value_ptr = unsafe { NonNull::new_unchecked(base.as_ptr().add(value_addr)) };
        (entry, value_ptr)
    }

    /// Push this chunk onto the front of an intrusive linked list
    /// whose head is stored in `head`. The chunk's own `next_in_list`
    /// cell stores the previous head (the chunk has been retired so
    /// the bump cursor is no longer needed by the active chunk path).
    ///
    /// Used by both the chunk cache and the pinned-chunks list. Caller
    /// must own the chunk exclusively.
    #[inline]
    pub fn push_into_intrusive_list(self, head: &core::cell::Cell<Option<NonNull<ChunkHeader<A>>>>) {
        let next = head.get();
        // SAFETY: chunk is live (constructor invariant); next_in_list is
        // interior-mutable.
        unsafe { (*self.ptr.as_ptr()).next_in_list.set(next) };
        head.set(Some(self.ptr));
    }
}

/// Pop a chunk from an intrusive linked list maintained by
/// [`ChunkRef::push_into_intrusive_list`]. Safe (the invariant —
/// "every node is a chunk previously pushed via `push_into_intrusive_list`" —
/// is established at every push site).
#[inline]
pub fn pop_from_intrusive_list<A: Allocator + Clone>(
    head: &core::cell::Cell<Option<NonNull<ChunkHeader<A>>>>,
) -> Option<NonNull<ChunkHeader<A>>> {
    let popped = head.get()?;
    // SAFETY: every entry was pushed via push_into_intrusive_list, so
    // `next_in_list` holds a valid `Option<NonNull<ChunkHeader<A>>>`.
    let next = unsafe { (*popped.as_ptr()).next_in_list.replace(None) };
    head.set(next);
    Some(popped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arena;
    use crate::chunk_header::header_for;

    #[test]
    fn chunk_ref_clone_returns_same_chunk_pointer() {
        let arena = Arena::new();
        let value = arena.alloc_rc(0_u32);
        // SAFETY: `value` keeps the underlying allocation (and thus its
        // containing chunk) alive for the duration of this test.
        let value_ptr = unsafe { NonNull::new_unchecked(crate::Rc::as_ptr(&value).cast_mut()) };
        // SAFETY: `value_ptr` came from an arena allocation; its chunk
        // header sits at the canonical offset before the value.
        let chunk = unsafe { header_for(value_ptr) };
        // SAFETY: chunk is live for the lifetime of `value`.
        let original = unsafe { ChunkRef::<allocator_api2::alloc::Global>::new(chunk) };
        // Exercise `Clone::clone` explicitly (silenced because `ChunkRef` is `Copy`).
        #[expect(clippy::clone_on_copy, reason = "the test specifically exercises the Clone impl")]
        let cloned = original.clone();
        assert_eq!(original.base().as_ptr(), cloned.base().as_ptr());
        assert_eq!(original.bump_get(), cloned.bump_get());
    }
}
