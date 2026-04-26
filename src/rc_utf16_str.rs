use allocator_api2::alloc::{Allocator, Global};

use crate::arena_str_macros::{impl_utf16_str_accessors, impl_utf16_str_handle_core, impl_utf16_str_read_traits};
use crate::flavor::Local;
use crate::raw_handle::RawHandle;

/// An immutable, single-pointer reference-counted UTF-16 string stored
/// in an [`Arena`](crate::Arena).
///
/// 8 bytes on 64-bit (one pointer); contrast with `&'arena Utf16Str`'s
/// 16 bytes. Cloning is **O(1)** (a non-atomic refcount bump). For
/// cross-thread sharing, use [`ArcUtf16Str`](crate::ArcUtf16Str)
/// instead.
///
/// `RcUtf16Str` is the recommended long-term storage type for arena
/// UTF-16 strings. Build via either:
///
/// - [`Arena::alloc_utf16_str_rc`](crate::Arena::alloc_utf16_str_rc) —
///   copy an `&Utf16Str` directly into the arena.
/// - [`Utf16String`](crate::builders::Utf16String) +
///   [`into_arena_utf16_str`](crate::builders::Utf16String::into_arena_utf16_str)
///   — build incrementally, then freeze.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "utf16")] {
/// use multitude::Arena;
/// use widestring::utf16str;
///
/// let arena = Arena::new();
/// let s = arena.alloc_utf16_str_rc(utf16str!("hello"));
/// assert_eq!(&*s, utf16str!("hello"));
/// assert_eq!(s.len(), 5);
/// # }
/// ```
pub struct RcUtf16Str<A: Allocator + Clone = Global> {
    /// Points at the first `u16` element. The `usize` element-count
    /// prefix lives at `inner.as_ptr().cast::<usize>().sub(1)`.
    inner: RawHandle<u16, Local, A>,
}

impl_utf16_str_handle_core!(RcUtf16Str, Local);
impl_utf16_str_accessors!([<A: Allocator + Clone>], RcUtf16Str<A>);
impl_utf16_str_read_traits!([<A: Allocator + Clone>], RcUtf16Str<A>);

#[cfg(feature = "builders")]
impl<'a, A: Allocator + Clone> ::core::convert::From<crate::builders::Utf16String<'a, A>> for RcUtf16Str<A> {
    /// Freeze an [`Utf16String`](crate::builders::Utf16String) into an
    /// immutable [`RcUtf16Str<A>`]. See
    /// [`Utf16String::into_arena_utf16_str`](crate::builders::Utf16String::into_arena_utf16_str).
    #[inline]
    fn from(s: crate::builders::Utf16String<'a, A>) -> Self {
        s.into_arena_utf16_str()
    }
}

impl<A: Allocator + Clone> ::core::convert::From<RcUtf16Str<A>> for crate::Rc<[u16], A> {
    /// Convert an [`RcUtf16Str<A>`] into an [`Rc<[u16], A>`](crate::Rc).
    /// O(1) — the chunk's +1 refcount transfers; the new handle's slice
    /// covers the underlying `u16` elements (excluding the inline length
    /// prefix).
    #[inline]
    fn from(s: RcUtf16Str<A>) -> Self {
        let len = s.len();
        let data = s.data_ptr();
        core::mem::forget(s);
        // SAFETY: `data` is `len` u16 elements of valid UTF-16 in a Local chunk;
        // refcount transferred.
        unsafe { Self::from_raw_slice(data, len) }
    }
}
