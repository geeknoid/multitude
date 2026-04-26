use core::ptr::NonNull;

/// # Safety
///
/// Caller must have written a valid `T` at `data_ptr`; the metadata in
/// `fat_template` must be valid for the value just written.
#[inline]
#[expect(
    clippy::missing_const_for_fn,
    reason = "ptr_meta::metadata / from_raw_parts_mut are not const fn on stable"
)]
pub unsafe fn reconstruct_fat<T: ptr_meta::Pointee + ?Sized>(fat_template: *const T, data_ptr: *mut u8) -> NonNull<T> {
    let metadata = ptr_meta::metadata(fat_template);
    let raw: *mut T = ptr_meta::from_raw_parts_mut(data_ptr.cast::<()>(), metadata);
    // SAFETY: raw pointer is non-null (data_ptr is non-null).
    unsafe { NonNull::new_unchecked(raw) }
}
