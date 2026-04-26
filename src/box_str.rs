use core::borrow::BorrowMut;
use core::marker::PhantomData;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::RcStr;
use crate::arena_str_helpers::read_str_len;
use crate::arena_str_macros::{impl_str_accessors, impl_str_read_traits};
use crate::chunk_header::release_chunk_ref_local;

/// An owned, mutable, single-pointer UTF-8 string stored in an
/// [`Arena`](crate::Arena).
///
/// 8 bytes on 64-bit (one pointer). Unlike [`RcStr`](crate::RcStr) /
/// [`ArcStr`](crate::ArcStr):
///
/// - Provides `&mut str` through `DerefMut`.
/// - **Not** [`Clone`] — single owner.
///
/// `BoxStr` keeps its containing chunk alive by holding a +1 refcount on
/// it, so the smart pointer can outlive the arena it came from and
/// survives [`Arena::reset`](crate::Arena::reset). For atomically-
/// sharable storage after the build phase, freeze into an
/// [`RcStr`](crate::RcStr) via [`Self::into_rc_str`].
///
/// # Example
///
/// ```
/// use multitude::Arena;
///
/// let arena = Arena::new();
/// let mut s = arena.alloc_str_box("hello");
/// // Mutable in place:
/// s.make_ascii_uppercase();
/// assert_eq!(&*s, "HELLO");
/// ```
pub struct BoxStr<A: Allocator + Clone = Global> {
    /// Points at the first data byte. The `usize` length prefix lives at
    /// `data.sub(size_of::<usize>())`.
    data: NonNull<u8>,
    _not_sync: PhantomData<*mut ()>,
    _allocator: PhantomData<A>,
}

impl<A: Allocator + Clone> BoxStr<A> {
    /// # Safety
    ///
    /// `data` must point at the first byte of a valid UTF-8 string laid
    /// out by [`try_reserve_str_in_chunk`] in `arena`'s chunk pool, and
    /// the chunk's refcount must have been bumped by 1 for this smart pointer's
    /// hold.
    #[inline]
    pub(crate) const unsafe fn from_raw_data(data: NonNull<u8>) -> Self {
        Self {
            data,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Pointer to the first data byte; the inline length prefix lives at
    /// `data_ptr().sub(size_of::<usize>())`.
    #[inline]
    pub(crate) const fn data_ptr(&self) -> NonNull<u8> {
        self.data
    }

    /// Borrow as `&mut str`.
    #[must_use]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "length read + slice construction + utf8 reinterpretation share one safety invariant"
    )]
    #[inline]
    pub const fn as_mut_str(&mut self) -> &mut str {
        // SAFETY: unique owner; data points at valid UTF-8.
        unsafe {
            let len = read_str_len(self.data);
            let bytes = core::slice::from_raw_parts_mut(self.data.as_ptr(), len);
            core::str::from_utf8_unchecked_mut(bytes)
        }
    }

    /// Convert this owned, mutable string into a shared, immutable
    /// [`RcStr<A>`](crate::RcStr). O(1) — no copy, no
    /// allocation.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let mut b = arena.alloc_str_box("hello");
    /// b.make_ascii_uppercase();
    /// let s = b.into_rc_str();
    /// let s2 = s.clone();
    /// assert_eq!(&*s, "HELLO");
    /// assert_eq!(&*s, &*s2);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_rc_str(self) -> RcStr<A> {
        let data = self.data;
        core::mem::forget(self);
        // SAFETY: data points at valid UTF-8 with prefix-length layout; +1 refcount preserved.
        unsafe { RcStr::from_raw_data(data) }
    }
}

impl_str_accessors!([<A: Allocator + Clone>], BoxStr<A>);
impl_str_read_traits!([<A: Allocator + Clone>], BoxStr<A>);

impl<A: Allocator + Clone> Drop for BoxStr<A> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: chunk alive via our refcount; BoxStr is Local-flavored.
        unsafe { release_chunk_ref_local::<_, A>(self.data) };
    }
}

impl<A: Allocator + Clone> ::core::ops::DerefMut for BoxStr<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A: Allocator + Clone> AsMut<str> for BoxStr<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A: Allocator + Clone> BorrowMut<str> for BoxStr<A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A: Allocator + Clone> From<BoxStr<A>> for RcStr<A> {
    /// Convert an [`BoxStr<A>`] into an [`RcStr<A>`]. O(1) — see
    /// [`BoxStr::into_rc_str`].
    #[inline]
    fn from(b: BoxStr<A>) -> Self {
        b.into_rc_str()
    }
}

impl<A: Allocator + Clone> ::core::cmp::PartialEq<str> for BoxStr<A> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl<A: Allocator + Clone> ::core::cmp::PartialEq<&str> for BoxStr<A> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<A: Allocator + Clone> serde::ser::Serialize for BoxStr<A> {
    fn serialize<S: serde::ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}
