use core::borrow::BorrowMut;
use core::marker::PhantomData;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};
use widestring::Utf16Str;

use crate::RcUtf16Str;
use crate::arena_str_helpers::read_utf16_str_len;
use crate::arena_str_macros::{impl_utf16_str_accessors, impl_utf16_str_read_traits};
use crate::chunk_header::release_chunk_ref_local;

/// An owned, mutable, single-pointer UTF-16 string stored in an
/// [`Arena`](crate::Arena).
///
/// 8 bytes on 64-bit (one pointer). Unlike
/// [`RcUtf16Str`](crate::RcUtf16Str) /
/// [`ArcUtf16Str`](crate::ArcUtf16Str):
///
/// - Provides `&mut Utf16Str` through `DerefMut`.
/// - **Not** [`Clone`] — single owner.
///
/// `BoxUtf16Str` keeps its containing chunk alive by holding a +1
/// refcount on it, so the smart pointer can outlive the arena it came
/// from and survives [`Arena::reset`](crate::Arena::reset). For shared
/// storage after the build phase, freeze into an
/// [`RcUtf16Str`](crate::RcUtf16Str) via [`Self::into_rc_utf16_str`].
///
/// # Example
///
/// ```
/// # #[cfg(feature = "utf16")] {
/// use multitude::Arena;
/// use widestring::utf16str;
///
/// let arena = Arena::new();
/// let mut s = arena.alloc_utf16_str_box(utf16str!("hello"));
/// assert_eq!(s.len(), 5);
/// assert_eq!(&*s, utf16str!("hello"));
/// // `&mut Utf16Str` access is available via DerefMut / `as_mut_utf16_str`.
/// let _: &mut widestring::Utf16Str = &mut *s;
/// # }
/// ```
pub struct BoxUtf16Str<A: Allocator + Clone = Global> {
    /// Points at the first `u16` element. The `usize` element-count
    /// prefix lives at `data.cast::<usize>().sub(1)`.
    data: NonNull<u16>,
    _not_sync: PhantomData<*mut ()>,
    _allocator: PhantomData<A>,
}

impl<A: Allocator + Clone> BoxUtf16Str<A> {
    /// # Safety
    ///
    /// `data` must point at the first `u16` element of a valid UTF-16
    /// string laid out by `try_reserve_utf16_str_in_chunk` in `arena`'s
    /// chunk pool, and the chunk's refcount must have been bumped by 1
    /// for this smart pointer's hold.
    #[inline]
    pub(crate) const unsafe fn from_raw_data(data: NonNull<u16>) -> Self {
        Self {
            data,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Pointer to the first `u16` element.
    #[inline]
    pub(crate) const fn data_ptr(&self) -> NonNull<u16> {
        self.data
    }

    /// Borrow as `&mut Utf16Str`.
    #[must_use]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "length read + slice construction + utf16 reinterpretation share one safety invariant"
    )]
    #[inline]
    pub fn as_mut_utf16_str(&mut self) -> &mut Utf16Str {
        // SAFETY: unique owner; data points at valid UTF-16.
        unsafe {
            let len = read_utf16_str_len(self.data);
            let slice = core::slice::from_raw_parts_mut(self.data.as_ptr(), len);
            Utf16Str::from_slice_unchecked_mut(slice)
        }
    }

    /// Convert this owned, mutable string into a shared, immutable
    /// [`RcUtf16Str<A>`](crate::RcUtf16Str). O(1) — no copy,
    /// no allocation.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "utf16")] {
    /// use widestring::utf16str;
    /// let arena = multitude::Arena::new();
    /// let b = arena.alloc_utf16_str_box(utf16str!("hello"));
    /// let s = b.into_rc_utf16_str();
    /// let s2 = s.clone();
    /// assert_eq!(&*s, utf16str!("hello"));
    /// assert_eq!(&*s, &*s2);
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_rc_utf16_str(self) -> RcUtf16Str<A> {
        let data = self.data;
        core::mem::forget(self);
        // SAFETY: data points at valid UTF-16 with prefix-length layout; +1 refcount preserved.
        unsafe { RcUtf16Str::from_raw_data(data) }
    }
}

impl_utf16_str_accessors!([<A: Allocator + Clone>], BoxUtf16Str<A>);
impl_utf16_str_read_traits!([<A: Allocator + Clone>], BoxUtf16Str<A>);

impl<A: Allocator + Clone> Drop for BoxUtf16Str<A> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: chunk alive via our refcount; BoxUtf16Str is Local-flavored.
        unsafe { release_chunk_ref_local::<u16, A>(self.data) };
    }
}

impl<A: Allocator + Clone> ::core::ops::DerefMut for BoxUtf16Str<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Utf16Str {
        self.as_mut_utf16_str()
    }
}

impl<A: Allocator + Clone> AsMut<Utf16Str> for BoxUtf16Str<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut Utf16Str {
        self.as_mut_utf16_str()
    }
}

impl<A: Allocator + Clone> BorrowMut<Utf16Str> for BoxUtf16Str<A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut Utf16Str {
        self.as_mut_utf16_str()
    }
}

impl<A: Allocator + Clone> From<BoxUtf16Str<A>> for RcUtf16Str<A> {
    /// Convert an [`BoxUtf16Str<A>`] into an
    /// [`RcUtf16Str<A>`]. O(1) — see
    /// [`BoxUtf16Str::into_rc_utf16_str`].
    #[inline]
    fn from(b: BoxUtf16Str<A>) -> Self {
        b.into_rc_utf16_str()
    }
}

impl<A: Allocator + Clone> ::core::cmp::PartialEq<Utf16Str> for BoxUtf16Str<A> {
    #[inline]
    fn eq(&self, other: &Utf16Str) -> bool {
        self.as_utf16_str() == other
    }
}

impl<A: Allocator + Clone> ::core::cmp::PartialEq<&Utf16Str> for BoxUtf16Str<A> {
    #[inline]
    fn eq(&self, other: &&Utf16Str) -> bool {
        self.as_utf16_str() == *other
    }
}
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<A: Allocator + Clone> serde::ser::Serialize for BoxUtf16Str<A> {
    fn serialize<S: serde::ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        crate::arena_str_helpers::serialize_utf16(self.as_utf16_str(), serializer)
    }
}
