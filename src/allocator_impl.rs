//! Implementation of `allocator_api2::alloc::Allocator` for `&Arena<A>`.

use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::{AllocError, Allocator};

use crate::chunk_header::{header_for, teardown_chunk};
use crate::chunk_sharing::ChunkSharing;

/// `&Arena<A>` is the allocator handle (`Copy`, satisfying the trait's
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
unsafe impl<A: Allocator + Clone> Allocator for &crate::Arena<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: this is the public allocator entry point; we don't
        // construct a handle, just hand back the bytes.
        unsafe {
            let ptr = self.bump_alloc_for_str(layout, ChunkSharing::Local);
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let _ = layout;
        // SAFETY: caller's contract — ptr came from this arena's allocate.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(ptr);
            if chunk.as_ref().dec_ref() {
                teardown_chunk(chunk, true);
            }
        }
    }

    unsafe fn grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());
        debug_assert_eq!(new_layout.align(), old_layout.align());

        // Try in-place extension when the buffer sits at the chunk's bump
        // cursor and the growth fits in the chunk. Per PLAN.md §4.19.10.
        // SAFETY: caller's contract — ptr is one of ours.
        let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = unsafe { header_for(ptr) };
        // SAFETY: chunk is alive (caller holds the allocation).
        let header = unsafe { chunk.as_ref() };
        let chunk_base = chunk.as_ptr() as usize;
        let buffer_start = ptr.as_ptr() as usize;
        let buffer_end = buffer_start.saturating_add(old_layout.size());
        let buffer_end_offset = buffer_end - chunk_base;
        let cur = header.bump.get();
        let extra = new_layout.size() - old_layout.size();
        let new_end = cur.checked_add(extra).ok_or(AllocError)?;
        if buffer_end_offset == cur && new_end <= header.total_size {
            header.bump.set(new_end);
            self.charge_alloc_stats(extra);
            return Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()));
        }

        // Fall back to allocate-copy-deallocate. The default impl would do
        // this for us, but we inline it to keep the safety reasoning local
        // and to bump the relocation counter (per PLAN.md §8.4).
        let new = self.allocate(new_layout)?;
        // SAFETY: source and destination don't overlap (different chunks
        // or non-overlapping regions), source is initialized for old size.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new.cast::<u8>().as_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }
        self.bump_allocator_relocation();
        Ok(new)
    }

    unsafe fn shrink(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() <= old_layout.size());
        debug_assert_eq!(new_layout.align(), old_layout.align());

        // Always succeeds in place. If the buffer is at the cursor,
        // reclaim the slack by lowering bump; otherwise, leave the slack
        // until chunk teardown. Per PLAN.md §4.19.10.
        // SAFETY: caller's contract — ptr is one of ours.
        let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = unsafe { header_for(ptr) };
        // SAFETY: chunk is alive.
        let header = unsafe { chunk.as_ref() };
        let chunk_base = chunk.as_ptr() as usize;
        let buffer_start = ptr.as_ptr() as usize;
        let buffer_end = buffer_start.saturating_add(old_layout.size());
        let buffer_end_offset = buffer_end - chunk_base;
        if buffer_end_offset == header.bump.get() {
            let saved = old_layout.size() - new_layout.size();
            header.bump.set(header.bump.get() - saved);
        }
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}
