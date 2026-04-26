use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::{AllocError, Allocator};

use crate::Arena;
use crate::chunk_header::{ChunkHeader, header_for, release_chunk_ref_local};
use crate::chunk_ref::ChunkRef;
use crate::chunk_sharing::ChunkSharing;
use crate::constants::CHUNK_ALIGN;

/// `&Arena<A>` is the allocator smart pointer (`Copy`, satisfying the trait's
/// "cheap to clone" requirement). Each `allocate` bumps the chunk
/// refcount; each `deallocate` decrements it.
///
/// `Local`-flavor chunks are used for these allocations.
///
/// # Safety
///
/// The standard `Allocator` contract: `deallocate` must receive a pointer
/// previously returned by `allocate` from the **same** `Arena<A>`
/// instance. The masking trick that recovers the chunk header would
/// silently corrupt memory if a foreign pointer happened to land on a
/// 64 KiB boundary.
// SAFETY: allocations remain valid until the matching `deallocate`,
// because the chunk's refcount keeps the chunk alive across that
// interval.
unsafe impl<A: Allocator + Clone> Allocator for &Arena<A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.align() >= CHUNK_ALIGN {
            return Err(AllocError);
        }
        let ptr = self.try_bump_alloc_for_str(layout, ChunkSharing::Local)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let _ = layout;
        // SAFETY: caller's contract — ptr came from this arena's
        // `allocate`, which always uses Local-flavor chunks.
        unsafe { release_chunk_ref_local::<u8, A>(ptr) };
    }

    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    unsafe fn grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());
        debug_assert_eq!(new_layout.align(), old_layout.align());

        // SAFETY: ptr from caller; ChunkRef avoids forming &ChunkHeader retag before deallocation.
        let chunk: NonNull<ChunkHeader<A>> = unsafe { header_for(ptr) };
        // SAFETY: header_for returned live chunk pointer.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        let extra = new_layout.size() - old_layout.size();
        if chunk_ref.try_grow_in_place(ptr, old_layout.size(), extra) {
            self.charge_alloc_stats(extra);
            return Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()));
        }

        let new = self.allocate(new_layout)?;
        // SAFETY: source and destination don't overlap; source initialized for old size.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new.cast::<u8>().as_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }
        self.bump_relocation();
        Ok(new)
    }

    #[inline]
    unsafe fn shrink(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() <= old_layout.size());
        debug_assert_eq!(new_layout.align(), old_layout.align());

        // SAFETY: ptr from caller.
        let chunk: NonNull<ChunkHeader<A>> = unsafe { header_for(ptr) };
        // SAFETY: header_for returned live chunk pointer.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        let _reclaimed = chunk_ref.try_reclaim_tail(ptr, old_layout.size(), new_layout.size());
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}
