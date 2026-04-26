//! [`PendingArenaArc`] — in-progress reservation for a `Shared`-flavor DST.

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::arena_arc::ArenaArc;
use crate::chunk_header::ChunkHeader;
use crate::drop_entry::DropEntry;

/// In-progress reservation in a `Shared`-flavor chunk. Mirrors
/// [`PendingArenaRc`](crate::PendingArenaRc) but finalizes into an
/// [`ArenaArc<T, A>`].
pub struct PendingArenaArc<'a, A: Allocator + Clone = Global> {
    arena: &'a crate::Arena<A>,
    chunk: NonNull<ChunkHeader<A>>,
    entry: NonNull<DropEntry>,
    bytes: NonNull<MaybeUninit<u8>>,
    layout: Layout,
}

impl<'a, A: Allocator + Clone> PendingArenaArc<'a, A> {
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

    /// Finalize into an [`ArenaArc<T, A>`].
    ///
    /// # Safety
    ///
    /// Same requirements as [`PendingArenaRc::finalize`](crate::PendingArenaRc::finalize),
    /// plus `T: Send + Sync` and `A: Send + Sync` for the resulting
    /// handle to be sound to share across threads.
    pub unsafe fn finalize<T>(self, fat_template: *const T, drop_fn: Option<unsafe fn(*mut DropEntry)>) -> ArenaArc<T, A> {
        let pa = core::mem::ManuallyDrop::new(self);

        if let Some(drop_fn) = drop_fn {
            // SAFETY: entry slot is writable, chunk is alive.
            unsafe {
                pa.chunk.as_ref().link_drop_entry(pa.entry, drop_fn, 0);
            }
        }

        let data_ptr = pa.bytes.as_ptr().cast::<u8>();
        // SAFETY: caller asserts metadata validity.
        let fat = unsafe { crate::dst_helpers::reconstruct_fat::<T>(fat_template, data_ptr) };

        ArenaArc {
            ptr: fat,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<A: Allocator + Clone> Drop for PendingArenaArc<'_, A> {
    fn drop(&mut self) {
        // SAFETY: chunk is alive (we hold a ref).
        unsafe {
            if self.chunk.as_ref().dec_ref() {
                crate::chunk_header::teardown_chunk(self.chunk, false);
            }
        }
        let _ = self.arena;
    }
}

impl<A: Allocator + Clone> fmt::Debug for PendingArenaArc<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingArenaArc")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
