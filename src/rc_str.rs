use allocator_api2::alloc::{Allocator, Global};

use crate::arena_str_macros::{impl_str_accessors, impl_str_handle_core, impl_str_read_traits};
use crate::flavor::Local;
use crate::raw_handle::RawHandle;

/// An immutable, single-pointer reference-counted UTF-8 string stored in
/// an [`Arena`](crate::Arena).
///
/// 8 bytes on 64-bit (one pointer); contrast with `&'arena str`'s 16
/// bytes. Cloning is **O(1)** (a non-atomic refcount bump). For
/// cross-thread sharing, use [`ArcStr`](crate::ArcStr) instead.
///
/// `RcStr` is the recommended long-term storage type for arena strings.
/// Build via either:
///
/// - [`Arena::alloc_str_rc`](crate::Arena::alloc_str_rc) — copy a `&str`
///   directly into the arena.
/// - [`String`](crate::builders::String) +
///   [`into_arena_str`](crate::builders::String::into_arena_str) — build
///   incrementally, then freeze.
///
/// # Example
///
/// ```
/// use multitude::{Arena, RcStr};
///
/// let arena = Arena::new();
/// let s = arena.alloc_str_rc("hello");
/// assert_eq!(&*s, "hello");
/// assert_eq!(s.len(), 5);
/// ```
pub struct RcStr<A: Allocator + Clone = Global> {
    /// Points at the first data byte. The `usize` length prefix lives at
    /// `inner.as_ptr().sub(size_of::<usize>())`.
    inner: RawHandle<u8, Local, A>,
}

impl_str_handle_core!(RcStr, Local);
impl_str_accessors!([<A: Allocator + Clone>], RcStr<A>);
impl_str_read_traits!([<A: Allocator + Clone>], RcStr<A>);

#[cfg(feature = "builders")]
impl<'a, A: Allocator + Clone> ::core::convert::From<crate::builders::String<'a, A>> for RcStr<A> {
    /// Freeze an [`String`](crate::builders::String) into an immutable
    /// [`RcStr<A>`]. See [`String::into_arena_str`](crate::builders::String::into_arena_str).
    #[inline]
    fn from(s: crate::builders::String<'a, A>) -> Self {
        s.into_arena_str()
    }
}

impl<A: Allocator + Clone> ::core::convert::From<RcStr<A>> for crate::Rc<[u8], A> {
    /// Convert an [`RcStr<A>`] into an [`Rc<[u8], A>`](crate::Rc).
    /// O(1) — the chunk's +1 refcount transfers; the new handle's slice
    /// covers the underlying UTF-8 bytes (excluding the inline length
    /// prefix). Mirrors `std::rc::Rc::<[u8]>::from(Rc<str>)`.
    #[inline]
    fn from(s: RcStr<A>) -> Self {
        let len = s.len();
        let data = s.data_ptr();
        core::mem::forget(s);
        // SAFETY: `data` is `len` bytes of UTF-8 in a Local chunk; refcount transferred.
        unsafe { Self::from_raw_slice(data, len) }
    }
}
