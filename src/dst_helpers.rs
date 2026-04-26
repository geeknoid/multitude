//! Helpers for DST construction (used by `PendingArenaRc::finalize` and the
//! `dst-factory` `arena2` mode).

use core::ptr::NonNull;

/// Reconstruct a pointer to `T` whose data half is `data_ptr` and
/// metadata half is taken from `fat_template`.
///
/// For the initial release, we restrict to `T: Sized` so we don't need
/// the unstable `ptr_metadata` feature. DST construction via the
/// `dst-factory` `arena2` mode will require either `ptr_metadata`
/// stabilization or a different code path generated at the macro level.
///
/// # Safety
///
/// Caller must have written a valid `T` at `data_ptr`.
#[inline]
pub(crate) unsafe fn reconstruct_fat<T>(fat_template: *const T, data_ptr: *mut u8) -> NonNull<T> {
    let _ = fat_template;
    // SAFETY: data_ptr is non-null and properly aligned for T.
    unsafe { NonNull::new_unchecked(data_ptr.cast::<T>()) }
}
