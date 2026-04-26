use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::{AllocError, Allocator};

use crate::Arena;
use crate::chunk_sharing::ChunkSharing;

/// Size of the inline length prefix for arena-resident string smart
/// pointers and [`String`](crate::builders::String).
pub const PREFIX_SIZE: usize = size_of::<usize>();

pub const PREFIX_ALIGN: usize = align_of::<usize>();

/// Bump-allocate `[ usize | bytes ]` for a string smart pointer and
/// write the prefix length. The returned pointer references the byte
/// immediately after the length prefix; the chunk refcount has been
/// bumped by 1 for the smart pointer that will wrap the result.
///
/// # Errors
///
/// Returns [`AllocError`] if the layout overflows or if the backing
/// allocator cannot satisfy the request.
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
#[expect(
    clippy::inline_always,
    reason = "callers supply constant `sharing`; const-folding the dispatcher chain requires inlining"
)]
#[inline(always)]
pub fn try_reserve_str_in_chunk<A: Allocator + Clone>(arena: &Arena<A>, s: &str, sharing: ChunkSharing) -> Result<NonNull<u8>, AllocError> {
    let total = PREFIX_SIZE.checked_add(s.len()).ok_or(AllocError)?;
    let layout = Layout::from_size_align(total, PREFIX_ALIGN).map_err(|_e| AllocError)?;
    let prefix_ptr = arena.try_bump_alloc_for_str(layout, sharing)?;
    // SAFETY: prefix_ptr is `align_of::<usize>()`-aligned and points to
    // a fresh, exclusively-owned `total`-byte allocation; we write the
    // length prefix followed by the string bytes within those bounds.
    Ok(unsafe {
        prefix_ptr.cast::<usize>().as_ptr().write(s.len());
        let data_ptr = prefix_ptr.as_ptr().add(PREFIX_SIZE);
        core::ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, s.len());
        NonNull::new_unchecked(data_ptr)
    })
}

/// Read the inline length prefix that precedes a string.
///
/// # Safety
///
/// `data` must point at the byte immediately following a `usize` length
/// prefix.
#[inline]
#[expect(
    clippy::cast_ptr_alignment,
    reason = "the prefix is bump-aligned to align_of::<usize>() at allocation time"
)]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
pub const unsafe fn read_str_len(data: NonNull<u8>) -> usize {
    // SAFETY: caller guarantees the prefix is there.
    unsafe { data.as_ptr().cast::<usize>().sub(1).read() }
}

/// Bump-allocate `[ usize | u16 elements ]` for a UTF-16 string smart
/// pointer and write the prefix (number of `u16` elements). The returned
/// pointer references the first `u16` element immediately after the
/// length prefix; the chunk refcount has been bumped by 1 for the smart
/// pointer that will wrap the result.
///
/// # Errors
///
/// Returns [`AllocError`] if the layout overflows or if the backing
/// allocator cannot satisfy the request.
#[cfg(feature = "utf16")]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
#[expect(
    clippy::inline_always,
    reason = "callers supply constant `sharing`; const-folding the dispatcher chain requires inlining"
)]
#[inline(always)]
pub fn try_reserve_utf16_str_in_chunk<A: Allocator + Clone>(
    arena: &Arena<A>,
    s: &widestring::Utf16Str,
    sharing: ChunkSharing,
) -> Result<NonNull<u16>, AllocError> {
    let elems = s.len();
    let payload_bytes = elems.checked_mul(size_of::<u16>()).ok_or(AllocError)?;
    let total = PREFIX_SIZE.checked_add(payload_bytes).ok_or(AllocError)?;
    let layout = Layout::from_size_align(total, PREFIX_ALIGN).map_err(|_e| AllocError)?;
    let prefix_ptr = arena.try_bump_alloc_for_str(layout, sharing)?;
    // SAFETY: prefix_ptr is `align_of::<usize>()`-aligned and points to a
    // fresh, exclusively-owned `total`-byte allocation; we write the
    // element-count prefix followed by `payload_bytes` of valid UTF-16
    // within those bounds.
    Ok(unsafe {
        prefix_ptr.cast::<usize>().as_ptr().write(elems);
        // The bump allocator returns an `align_of::<usize>()`-aligned
        // pointer (≥ 2 bytes), so adding PREFIX_SIZE keeps `u16` alignment.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "prefix_ptr is align_of::<usize>()-aligned (8 bytes); offset + PREFIX_SIZE remains u16-aligned"
        )]
        let data_ptr = prefix_ptr.as_ptr().add(PREFIX_SIZE).cast::<u16>();
        core::ptr::copy_nonoverlapping(s.as_slice().as_ptr(), data_ptr, elems);
        NonNull::new_unchecked(data_ptr)
    })
}

/// Read the inline length prefix (number of `u16` elements) that precedes
/// a UTF-16 string buffer.
///
/// # Safety
///
/// `data` must point at the first `u16` element immediately following a
/// `usize` element-count prefix.
#[cfg(feature = "utf16")]
#[inline]
#[expect(
    clippy::cast_ptr_alignment,
    reason = "the prefix is bump-aligned to align_of::<usize>() at allocation time and sits 8 bytes before `data`"
)]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
pub const unsafe fn read_utf16_str_len(data: NonNull<u16>) -> usize {
    // SAFETY: `data.cast::<usize>().sub(1)` lands on the prefix slot the
    // allocator placed `size_of::<usize>()` bytes before the data.
    unsafe { data.as_ptr().cast::<usize>().sub(1).read() }
}

/// `Display` adapter that streams UTF-8 chars without ever materializing
/// a heap `String`. Each `Utf16Str` is by definition valid UTF-16 (no
/// lone surrogates), so `decode_utf16` always yields `Ok` for our payloads.
#[cfg(all(feature = "utf16", feature = "serde"))]
struct DisplayUtf16<'a>(&'a widestring::Utf16Str);

#[cfg(all(feature = "utf16", feature = "serde"))]
impl core::fmt::Display for DisplayUtf16<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use core::char::decode_utf16;
        use core::fmt::Write as _;
        for ch in decode_utf16(self.0.as_slice().iter().copied()) {
            f.write_char(ch.unwrap_or(core::char::REPLACEMENT_CHARACTER))?;
        }
        Ok(())
    }
}

/// Stream-transcode a `&Utf16Str` to UTF-8 for serde output. Serializers
/// that build the string in place avoid the intermediate `alloc::String`
/// allocation that `s.to_string()` would impose.
#[cfg(all(feature = "utf16", feature = "serde"))]
pub fn serialize_utf16<S: serde::ser::Serializer>(s: &widestring::Utf16Str, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&DisplayUtf16(s))
}
