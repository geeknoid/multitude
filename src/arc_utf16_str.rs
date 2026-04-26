use allocator_api2::alloc::{Allocator, Global};

use crate::arena_str_macros::{impl_utf16_str_accessors, impl_utf16_str_handle_core, impl_utf16_str_read_traits};
use crate::flavor::Shared;
use crate::raw_handle::RawHandle;

/// An immutable, single-pointer reference-counted UTF-16 string stored
/// in an [`Arena`](crate::Arena), safe to share across threads.
///
/// 8 bytes on 64-bit (one pointer); contrast with `&'arena Utf16Str`'s
/// 16 bytes. Cloning is **O(1)** (one atomic refcount bump). For
/// single-threaded code, prefer [`RcUtf16Str`](crate::RcUtf16Str) — it
/// has the same layout and same cost model with a non-atomic refcount.
///
/// Build via either:
///
/// - [`Arena::alloc_utf16_str_arc`](crate::Arena::alloc_utf16_str_arc) —
///   copy an `&Utf16Str` directly into the arena.
/// - [`Utf16String`](crate::builders::Utf16String) +
///   [`into_arena_utf16_str`](crate::builders::Utf16String::into_arena_utf16_str),
///   then convert with `.into()` — build incrementally, then freeze.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "utf16")] {
/// use multitude::Arena;
/// use std::thread;
/// use widestring::utf16str;
///
/// let arena = Arena::new();
/// let s = arena.alloc_utf16_str_arc(utf16str!("shared"));
/// let s2 = s.clone();
/// let h = thread::spawn(move || s2.len());
/// assert_eq!(s.len(), h.join().unwrap());
/// # }
/// ```
pub struct ArcUtf16Str<A: Allocator + Clone = Global> {
    inner: RawHandle<u16, Shared, A>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "RawHandle holds a NonNull which is structurally !Send; the Shared flavor's atomic refcount makes cross-thread sharing sound"
)]
// SAFETY: backed by a Shared chunk with atomic refcount.
unsafe impl<A: Allocator + Clone + Send + Sync> Send for ArcUtf16Str<A> {}
// SAFETY: see above.
unsafe impl<A: Allocator + Clone + Send + Sync> Sync for ArcUtf16Str<A> {}

impl_utf16_str_handle_core!(ArcUtf16Str, Shared);
impl_utf16_str_accessors!([<A: Allocator + Clone>], ArcUtf16Str<A>);
impl_utf16_str_read_traits!([<A: Allocator + Clone>], ArcUtf16Str<A>);

impl<A: Allocator + Clone> ::core::convert::From<ArcUtf16Str<A>> for crate::Arc<[u16], A> {
    /// Convert an [`ArcUtf16Str<A>`] into an [`Arc<[u16], A>`](crate::Arc).
    /// O(1) — the chunk's +1 refcount transfers; the new handle's slice
    /// covers the underlying `u16` elements (excluding the inline length
    /// prefix).
    #[inline]
    fn from(s: ArcUtf16Str<A>) -> Self {
        let len = s.len();
        let data = s.data_ptr();
        core::mem::forget(s);
        // SAFETY: `data` is `len` u16 elements of valid UTF-16 in a Shared chunk;
        // refcount transferred.
        unsafe { Self::from_raw_slice(data, len) }
    }
}
