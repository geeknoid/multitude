//! `DropEntry` — node in the per-chunk doubly-linked drop list.

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
/// Doubly-linked so [`ArenaBox`](crate::ArenaBox) can unlink an entry in
/// O(1) on handle drop.
///
/// This type is `pub` only because it appears in
/// [`PendingArenaRc::finalize`](crate::PendingArenaRc::finalize)'s signature;
/// its fields are private and its layout is an implementation detail.
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

/// Drop shim for a sized value `T` that lives immediately after the entry
/// in the chunk (at the next address ≥ `entry + sizeof::<DropEntry>()`
/// satisfying `align_of::<T>()`).
///
/// # Safety
///
/// The caller must ensure `entry` was constructed by the matching
/// allocator path that places a valid `T` immediately after the entry.
/// Drop shim for a sized value `T` that lives immediately after the entry
/// in the chunk (at the next address ≥ `entry + sizeof::<DropEntry>()`
/// satisfying `align_of::<T>()`).
///
/// Preserves provenance: derives the value pointer from `entry` via
/// `byte_add`, not via integer round-trip.
///
/// # Safety
///
/// The caller must ensure `entry` was constructed by the matching
/// allocator path that places a valid `T` immediately after the entry,
/// AND that `entry`'s provenance covers the chunk's bytes.
pub(crate) unsafe fn drop_shim<T>(entry: *mut DropEntry) {
    // SAFETY: bump allocator placed a valid `T` immediately after the
    // entry at the next properly-aligned address; both lie within the
    // same chunk allocation (so `entry`'s provenance covers them).
    unsafe {
        let after_entry = entry.byte_add(size_of::<DropEntry>());
        let align = align_of::<T>();
        let misalign = (after_entry.cast::<u8>() as usize) & (align - 1);
        let padding = if misalign == 0 { 0 } else { align - misalign };
        let value_ptr = after_entry.byte_add(padding).cast::<T>();
        core::ptr::drop_in_place(value_ptr);
    }
}

/// Compute the byte offset from a `DropEntry`'s start to the start of the
/// co-allocated value of type `T`.
#[inline]
#[must_use]
pub(crate) const fn value_offset_after_entry<T>() -> usize {
    let entry_size = size_of::<DropEntry>();
    let align = align_of::<T>();
    (entry_size + align - 1) & !(align - 1)
}
