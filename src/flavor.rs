use core::ptr::NonNull;

use allocator_api2::alloc::Allocator;

use crate::chunk_header::{inc_chunk_ref_local, inc_chunk_ref_shared, release_chunk_ref_local, release_chunk_ref_shared};

/// Sealed marker trait selecting a chunk-flavor refcount discipline.
/// [`Local`] uses non-atomic `Cell<usize>` ops; [`Shared`] uses
/// `AtomicUsize`.
pub trait Flavor: sealed::Sealed {
    /// Increment the refcount of the chunk owning `value`.
    ///
    /// # Safety
    /// `value` must point at a value in a live chunk of this flavor and
    /// the caller must already hold a +1 refcount on it.
    unsafe fn inc_ref<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>);

    /// Release a +1 refcount on the chunk owning `value`.
    ///
    /// # Safety
    /// `value` must point at a value in a live chunk of this flavor and
    /// the caller must transfer a +1 refcount to this call.
    unsafe fn release<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>);
}

mod sealed {
    pub trait Sealed {}
}

/// `Local`-flavor smart-pointer marker (non-atomic refcount).
///
/// Smart pointers parameterized by `Local` are `!Send + !Sync`.
pub struct Local;

impl sealed::Sealed for Local {}

impl Flavor for Local {
    #[inline]
    unsafe fn inc_ref<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
        // SAFETY: caller's contract.
        unsafe { inc_chunk_ref_local::<T, A>(value) };
    }

    #[inline]
    unsafe fn release<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
        // SAFETY: caller's contract.
        unsafe { release_chunk_ref_local::<T, A>(value) };
    }
}

/// `Shared`-flavor smart-pointer marker (atomic refcount).
///
/// Smart pointers parameterized by `Shared` may be `Send + Sync`.
pub struct Shared;

impl sealed::Sealed for Shared {}

impl Flavor for Shared {
    #[inline]
    unsafe fn inc_ref<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
        // SAFETY: caller's contract.
        unsafe { inc_chunk_ref_shared::<T, A>(value) };
    }

    #[inline]
    unsafe fn release<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
        // SAFETY: caller's contract.
        unsafe { release_chunk_ref_shared::<T, A>(value) };
    }
}
