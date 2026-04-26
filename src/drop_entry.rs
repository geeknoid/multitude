use core::fmt;
use core::ptr::NonNull;

/// Opaque drop-list node.
///
/// Three pointers — the value pointer is *not* stored, because the value
/// is co-allocated immediately after the entry (at the value's
/// alignment). The `drop_fn` shim recovers the value's address from the
/// entry's own address.
///
/// `slice_len` is used only by slice-drop entries (`alloc_slice_*` paths)
/// — for single-value entries it's `0` and ignored.
///
/// Doubly-linked so [`Box`](crate::Box) can unlink an entry in
/// O(1) when the smart pointer is dropped.
///
/// This type is `pub` only because it appears in the
/// [`PendingRc::finalize`](crate::dst::PendingRc::finalize)
/// signature (gated on the `dst` feature) as the parameter type of the
/// caller-supplied `drop_fn` callback. It is re-exported as
/// [`crate::dst::DropEntry`]. Its fields are crate-private and its
/// layout is an implementation detail.
#[repr(C)]
pub struct DropEntry {
    pub(crate) drop_fn: unsafe fn(*mut Self),
    pub(crate) prev: Option<NonNull<Self>>,
    pub(crate) next: Option<NonNull<Self>>,
    pub(crate) slice_len: usize,
}

impl fmt::Debug for DropEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DropEntry").finish_non_exhaustive()
    }
}

/// # Safety
///
/// `entry` must have been constructed with a valid `T` placed immediately
/// after it, and `entry`'s provenance must cover the chunk's bytes.
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
pub unsafe fn drop_shim<T>(entry: *mut DropEntry) {
    // SAFETY: see function-level safety comment.
    unsafe {
        let value_ptr = value_ptr_after_entry::<T>(entry);
        core::ptr::drop_in_place(value_ptr);
    }
}

/// # Safety
///
/// `entry` must point at a [`DropEntry`] whose co-allocated value was
/// laid out for type `T`, and whose provenance covers the value bytes.
#[inline]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "byte_add chain forms a single layout calculation with one safety invariant"
)]
pub unsafe fn value_ptr_after_entry<T>(entry: *mut DropEntry) -> *mut T {
    // SAFETY: caller guarantees provenance.
    unsafe {
        let after_entry = entry.byte_add(size_of::<DropEntry>());
        let align = align_of::<T>();
        let misalign = (after_entry.cast::<u8>() as usize) & (align - 1);
        let padding = if misalign == 0 { 0 } else { align - misalign };
        after_entry.byte_add(padding).cast::<T>()
    }
}

/// Compute the byte offset from a `DropEntry`'s start to the start of the
/// co-allocated value of alignment `align`.
#[inline]
#[must_use]
pub const fn value_offset_after_entry_for_align(align: usize) -> usize {
    let entry_size = size_of::<DropEntry>();
    (entry_size + align - 1) & !(align - 1)
}

/// Recover the `DropEntry` co-allocated immediately before a value pointer.
///
/// # Safety
///
/// `value` must point at a value that was co-allocated with a `DropEntry`
/// at the canonical `[DropEntry | padding | value]` offset (i.e., from
/// `alloc_with_drop_entry_unchecked` or `alloc_entry_value_slot_unchecked`).
/// `value_align` must match the alignment used to size the padding.
#[inline]
#[expect(
    clippy::cast_ptr_alignment,
    reason = "entry pointer sits at value_offset_after_entry-back from value, which is align_of::<DropEntry>()-aligned by construction"
)]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "byte_sub + NonNull::new_unchecked share the canonical entry/value layout invariant"
)]
pub const unsafe fn entry_for_value(value: NonNull<u8>, value_align: usize) -> NonNull<DropEntry> {
    let offset = value_offset_after_entry_for_align(value_align);
    // SAFETY: caller's contract — entry sits exactly `offset` bytes before value.
    unsafe { NonNull::new_unchecked(value.as_ptr().byte_sub(offset).cast::<DropEntry>()) }
}

/// Overwrite the `drop_fn` of a live `DropEntry`. Used by the
/// `MaybeUninit`-style `assume_init` paths to retarget a placeholder
/// no-op shim to the real `T::drop` shim once the value is initialized.
///
/// # Safety
///
/// `entry` must point at a live, linked `DropEntry` in a chunk owned by
/// the caller; the caller must have exclusive write access to the entry.
#[inline]
pub unsafe fn rewrite_drop_fn(entry: NonNull<DropEntry>, drop_fn: unsafe fn(*mut DropEntry)) {
    // SAFETY: caller's contract.
    unsafe { (*entry.as_ptr()).drop_fn = drop_fn };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_entry_debug() {
        unsafe fn shim(_e: *mut DropEntry) {}
        let e = DropEntry {
            drop_fn: shim,
            prev: None,
            next: None,
            slice_len: 0,
        };
        // SAFETY: shim is a no-op.
        unsafe { (e.drop_fn)(core::ptr::from_ref(&e).cast_mut()) };
        let s = alloc::format!("{e:?}");
        assert!(s.contains("DropEntry"));
    }

    #[test]
    fn value_offset_after_entry_for_align_low() {
        assert_eq!(value_offset_after_entry_for_align(1), size_of::<DropEntry>());
        assert_eq!(value_offset_after_entry_for_align(8), size_of::<DropEntry>());
    }

    #[test]
    fn value_offset_after_entry_for_align_high() {
        assert_eq!(value_offset_after_entry_for_align(64), 64);
        assert_eq!(value_offset_after_entry_for_align(128), 128);
    }
}
