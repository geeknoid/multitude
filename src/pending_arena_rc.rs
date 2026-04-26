//! [`PendingArenaRc`] — in-progress reservation for a `Local`-flavor DST.

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::arena_rc::ArenaRc;
use crate::chunk_header::ChunkHeader;
use crate::drop_entry::DropEntry;

/// In-progress reservation returned by
/// [`Arena::alloc_uninit_dst`](crate::Arena::alloc_uninit_dst).
///
/// The caller initializes the reserved bytes through [`Self::as_mut_ptr`],
/// then calls [`Self::finalize`] (with a fat-pointer template for DSTs)
/// to obtain an [`ArenaRc<T, A>`]. Forgetting to call `finalize`
/// decrements the chunk refcount; the reserved bytes are leaked until
/// chunk teardown but never dropped.
pub struct PendingArenaRc<'a, A: Allocator + Clone = Global> {
    arena: &'a crate::Arena<A>,
    chunk: NonNull<ChunkHeader<A>>,
    entry: NonNull<DropEntry>,
    bytes: NonNull<MaybeUninit<u8>>,
    layout: Layout,
}

impl<'a, A: Allocator + Clone> PendingArenaRc<'a, A> {
    pub(crate) fn new(
        arena: &'a crate::Arena<A>,
        chunk: NonNull<ChunkHeader<A>>,
        entry: NonNull<DropEntry>,
        bytes: NonNull<u8>,
        layout: Layout,
    ) -> Self {
        Self {
            arena,
            chunk,
            entry,
            bytes: bytes.cast::<MaybeUninit<u8>>(),
            layout,
        }
    }

    /// The reserved layout.
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Mutable pointer to the start of the reserved bytes.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the &mut self enforces exclusive access to the reservation cursor"
    )]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_ptr().cast::<u8>()
    }

    /// Finalize into an [`ArenaRc<T, A>`].
    ///
    /// `fat_template` supplies the metadata half of a fat pointer (length
    /// / vtable). Its data pointer is overwritten with this reservation's
    /// data pointer.
    ///
    /// `drop_fn`, if `Some`, is added to the chunk's drop list and
    /// invoked once when the chunk is reclaimed.
    ///
    /// # Safety
    ///
    /// - All bytes covered by [`Self::layout`] must have been initialized
    ///   to a valid `T`.
    /// - `fat_template`'s metadata must be valid for the value just
    ///   written.
    /// - `drop_fn`, if provided, must be safe to call exactly once and
    ///   must correctly locate the value relative to the entry pointer
    ///   it receives (per PLAN.md §4.5).
    pub unsafe fn finalize<T>(self, fat_template: *const T, drop_fn: Option<unsafe fn(*mut DropEntry)>) -> ArenaRc<T, A> {
        // Suppress the Drop impl on PendingArenaRc.
        let pa = core::mem::ManuallyDrop::new(self);

        // Link drop entry into the chunk's drop list, or leave it unlinked
        // (effectively a no-op slot) if no drop is needed.
        if let Some(drop_fn) = drop_fn {
            // SAFETY: entry slot is writable, chunk is alive.
            unsafe {
                pa.chunk.as_ref().link_drop_entry(pa.entry, drop_fn, 0);
            }
        }

        // Construct fat pointer with the reservation's data address.
        let data_ptr = pa.bytes.as_ptr().cast::<u8>();
        // SAFETY: caller asserts metadata validity.
        let fat = unsafe { crate::dst_helpers::reconstruct_fat::<T>(fat_template, data_ptr) };

        ArenaRc {
            ptr: fat,
            _not_sync: PhantomData,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<A: Allocator + Clone> Drop for PendingArenaRc<'_, A> {
    fn drop(&mut self) {
        // Not finalized — release the chunk refcount and leave the reserved
        // bytes uninitialized in the chunk (no drop fn registered).
        // SAFETY: chunk is alive (we hold a ref).
        unsafe {
            if self.chunk.as_ref().dec_ref() {
                crate::chunk_header::teardown_chunk(self.chunk, true);
            }
        }
        let _ = self.arena;
    }
}

impl<A: Allocator + Clone> fmt::Debug for PendingArenaRc<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingArenaRc")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
