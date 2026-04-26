/// Emit the shared `as_str` / `len` / `is_empty` accessor impls for an
/// arena-string smart pointer whose `data: NonNull<u8>` field points at the
/// first data byte (with a `usize` length prefix at `data.sub(8)`).
///
/// `$ty` is the smart pointer type with all of its generics (e.g.
/// `RcStr<A>`); `$($g:tt)+` is the generic-parameter list that
/// goes after `impl` (e.g. `<A: Allocator + Clone>`).
macro_rules! impl_str_accessors {
    ([$($g:tt)+], $ty:ty) => {
        impl $($g)+ $ty {
            /// Borrow as `&str`.
            #[must_use]
            #[inline]
            pub const fn as_str(&self) -> &str {
                // SAFETY: `data_ptr` points at a valid UTF-8 string of the
                // length recorded in its inline prefix.
                unsafe {
                    let data = self.data_ptr();
                    let len = $crate::arena_str_helpers::read_str_len(data);
                    let bytes = ::core::slice::from_raw_parts(data.as_ptr(), len);
                    ::core::str::from_utf8_unchecked(bytes)
                }
            }

            /// String length in bytes.
            #[must_use]
            #[inline]
            pub const fn len(&self) -> usize {
                // SAFETY: see `as_str`.
                unsafe { $crate::arena_str_helpers::read_str_len(self.data_ptr()) }
            }

            /// True iff the string is empty.
            #[must_use]
            #[inline]
            pub const fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }
    };
}

/// Emit the shared read-only trait impls (`Deref`, `AsRef<str>`,
/// `Borrow<str>`, `Debug`, `Display`, `PartialEq`, `Eq`, `PartialOrd`,
/// `Ord`, `Hash`) that delegate to `as_str`.
macro_rules! impl_str_read_traits {
    ([$($g:tt)+], $ty:ty) => {
        impl $($g)+ ::core::ops::Deref for $ty {
            type Target = str;
            #[inline]
            fn deref(&self) -> &str {
                self.as_str()
            }
        }

        impl $($g)+ ::core::convert::AsRef<str> for $ty {
            #[inline]
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl $($g)+ ::core::borrow::Borrow<str> for $ty {
            #[inline]
            fn borrow(&self) -> &str {
                self.as_str()
            }
        }

        impl $($g)+ ::core::fmt::Debug for $ty {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(self.as_str(), f)
            }
        }

        impl $($g)+ ::core::fmt::Display for $ty {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(self.as_str(), f)
            }
        }

        impl $($g)+ ::core::cmp::PartialEq for $ty {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_str() == other.as_str()
            }
        }
        impl $($g)+ ::core::cmp::Eq for $ty {}

        impl $($g)+ ::core::cmp::PartialOrd for $ty {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> ::core::option::Option<::core::cmp::Ordering> {
                ::core::option::Option::Some(self.cmp(other))
            }
        }

        impl $($g)+ ::core::cmp::Ord for $ty {
            #[inline]
            fn cmp(&self, other: &Self) -> ::core::cmp::Ordering {
                self.as_str().cmp(other.as_str())
            }
        }

        impl $($g)+ ::core::hash::Hash for $ty {
            fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                self.as_str().hash(state);
            }
        }

        impl $($g)+ ::core::fmt::Pointer for $ty {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let ptr: *const u8 = self.data_ptr().as_ptr();
                ::core::fmt::Pointer::fmt(&ptr, f)
            }
        }
    };
}

pub(crate) use impl_str_accessors;
pub(crate) use impl_str_read_traits;

/// Emit the boilerplate handle scaffolding (`Clone`, `from_raw_data`,
/// `data_ptr`, `Unpin`, `PartialEq<str>` / `PartialEq<&str>`, optional
/// `serde::Serialize`) for a UTF-8 arena-string smart pointer whose
/// `inner: RawHandle<u8, $flavor, A>` field holds the chunk-refcounted
/// pointer to the first data byte.
///
/// This is the cross-flavor twin: `RcStr` and `ArcStr` both expand to
/// the same body, parameterized only by the flavor type.
macro_rules! impl_str_handle_core {
    ($ty:ident, $flavor:ty) => {
        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::clone::Clone for $ty<A> {
            #[inline]
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> $ty<A> {
            /// # Safety
            ///
            /// `data` must point at the first byte of a properly-formatted
            /// arena-resident string buffer (length-prefixed at
            /// `data - size_of::<usize>()`), and the chunk's refcount must
            /// already have been incremented for this smart pointer.
            #[inline]
            #[must_use]
            pub(crate) const unsafe fn from_raw_data(data: ::core::ptr::NonNull<u8>) -> Self {
                Self {
                    // SAFETY: caller's contract — live in-arena alloc with +1 refcount transferred.
                    inner: unsafe { $crate::raw_handle::RawHandle::from_raw(data) },
                }
            }

            /// Pointer to the first data byte; the inline length prefix lives at
            /// `data_ptr().sub(size_of::<usize>())`.
            #[inline]
            pub(crate) const fn data_ptr(&self) -> ::core::ptr::NonNull<u8> {
                self.inner.as_non_null()
            }
        }

        // Mirrors the underlying smart-pointer flavor's `Unpin` impl.
        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::marker::Unpin for $ty<A> {}

        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::cmp::PartialEq<str> for $ty<A> {
            #[inline]
            fn eq(&self, other: &str) -> bool {
                self.as_str() == other
            }
        }

        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::cmp::PartialEq<&str> for $ty<A> {
            #[inline]
            fn eq(&self, other: &&str) -> bool {
                self.as_str() == *other
            }
        }

        #[cfg(feature = "serde")]
        #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::serde::ser::Serialize for $ty<A> {
            fn serialize<S: ::serde::ser::Serializer>(&self, serializer: S) -> ::core::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }
    };
}

pub(crate) use impl_str_handle_core;

/// UTF-16 sibling of [`impl_str_handle_core`] — emits `Clone`,
/// `from_raw_data`, `data_ptr`, `Unpin`, `PartialEq<Utf16Str>` /
/// `PartialEq<&Utf16Str>`, and (when the `serde` feature is enabled)
/// `Serialize` for a UTF-16 string smart pointer whose
/// `inner: RawHandle<u16, $flavor, A>` holds the chunk-refcounted
/// pointer to the first `u16` element.
#[cfg(feature = "utf16")]
macro_rules! impl_utf16_str_handle_core {
    ($ty:ident, $flavor:ty) => {
        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::clone::Clone for $ty<A> {
            #[inline]
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> $ty<A> {
            /// # Safety
            ///
            /// `data` must point at the first `u16` element of a properly-formatted
            /// arena-resident UTF-16 buffer (element-count prefix at
            /// `data.cast::<usize>().sub(1)`), and the chunk's refcount must
            /// already have been incremented for this smart pointer.
            #[inline]
            #[must_use]
            pub(crate) const unsafe fn from_raw_data(data: ::core::ptr::NonNull<u16>) -> Self {
                Self {
                    // SAFETY: caller's contract — live in-arena alloc with +1 refcount transferred.
                    inner: unsafe { $crate::raw_handle::RawHandle::from_raw(data) },
                }
            }

            /// Pointer to the first `u16` element; the inline length prefix lives at
            /// `data_ptr().cast::<usize>().sub(1)`.
            #[inline]
            pub(crate) const fn data_ptr(&self) -> ::core::ptr::NonNull<u16> {
                self.inner.as_non_null()
            }
        }

        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::marker::Unpin for $ty<A> {}

        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::cmp::PartialEq<::widestring::Utf16Str> for $ty<A> {
            #[inline]
            fn eq(&self, other: &::widestring::Utf16Str) -> bool {
                self.as_utf16_str() == other
            }
        }

        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::core::cmp::PartialEq<&::widestring::Utf16Str> for $ty<A> {
            #[inline]
            fn eq(&self, other: &&::widestring::Utf16Str) -> bool {
                self.as_utf16_str() == *other
            }
        }

        #[cfg(feature = "serde")]
        #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
        impl<A: ::allocator_api2::alloc::Allocator + ::core::clone::Clone> ::serde::ser::Serialize for $ty<A> {
            fn serialize<S: ::serde::ser::Serializer>(&self, serializer: S) -> ::core::result::Result<S::Ok, S::Error> {
                $crate::arena_str_helpers::serialize_utf16(self.as_utf16_str(), serializer)
            }
        }
    };
}

#[cfg(feature = "utf16")]
pub(crate) use impl_utf16_str_handle_core;

/// Emit the shared `as_utf16_str` / `len` / `is_empty` accessor impls
/// for an arena UTF-16 string smart pointer whose `data: NonNull<u16>`
/// field points at the first `u16` element (with a `usize` element-count
/// prefix at `data.cast::<usize>().sub(1)`).
#[cfg(feature = "utf16")]
macro_rules! impl_utf16_str_accessors {
    ([$($g:tt)+], $ty:ty) => {
        impl $($g)+ $ty {
            /// Borrow as `&Utf16Str`.
            #[must_use]
            #[inline]
            pub const fn as_utf16_str(&self) -> &::widestring::Utf16Str {
                // SAFETY: `data_ptr` points at a valid UTF-16 string of the
                // length recorded in its inline prefix.
                unsafe {
                    let data = self.data_ptr();
                    let len = $crate::arena_str_helpers::read_utf16_str_len(data);
                    let slice = ::core::slice::from_raw_parts(data.as_ptr(), len);
                    ::widestring::Utf16Str::from_slice_unchecked(slice)
                }
            }

            /// String length in `u16` elements.
            #[must_use]
            #[inline]
            pub const fn len(&self) -> usize {
                // SAFETY: see `as_utf16_str`.
                unsafe { $crate::arena_str_helpers::read_utf16_str_len(self.data_ptr()) }
            }

            /// True iff the string is empty.
            #[must_use]
            #[inline]
            pub const fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }
    };
}

/// Emit the shared read-only trait impls (`Deref`, `AsRef<Utf16Str>`,
/// `Borrow<Utf16Str>`, `Debug`, `Display`, `PartialEq`, `Eq`, `PartialOrd`,
/// `Ord`, `Hash`, `Pointer`) that delegate to `as_utf16_str`.
#[cfg(feature = "utf16")]
macro_rules! impl_utf16_str_read_traits {
    ([$($g:tt)+], $ty:ty) => {
        impl $($g)+ ::core::ops::Deref for $ty {
            type Target = ::widestring::Utf16Str;
            #[inline]
            fn deref(&self) -> &::widestring::Utf16Str {
                self.as_utf16_str()
            }
        }

        impl $($g)+ ::core::convert::AsRef<::widestring::Utf16Str> for $ty {
            #[inline]
            fn as_ref(&self) -> &::widestring::Utf16Str {
                self.as_utf16_str()
            }
        }

        impl $($g)+ ::core::borrow::Borrow<::widestring::Utf16Str> for $ty {
            #[inline]
            fn borrow(&self) -> &::widestring::Utf16Str {
                self.as_utf16_str()
            }
        }

        impl $($g)+ ::core::fmt::Debug for $ty {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(self.as_utf16_str(), f)
            }
        }

        impl $($g)+ ::core::fmt::Display for $ty {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(self.as_utf16_str(), f)
            }
        }

        impl $($g)+ ::core::cmp::PartialEq for $ty {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_utf16_str() == other.as_utf16_str()
            }
        }
        impl $($g)+ ::core::cmp::Eq for $ty {}

        impl $($g)+ ::core::cmp::PartialOrd for $ty {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> ::core::option::Option<::core::cmp::Ordering> {
                ::core::option::Option::Some(self.cmp(other))
            }
        }

        impl $($g)+ ::core::cmp::Ord for $ty {
            #[inline]
            fn cmp(&self, other: &Self) -> ::core::cmp::Ordering {
                self.as_utf16_str().cmp(other.as_utf16_str())
            }
        }

        impl $($g)+ ::core::hash::Hash for $ty {
            fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                self.as_utf16_str().hash(state);
            }
        }

        impl $($g)+ ::core::fmt::Pointer for $ty {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let ptr: *const u16 = self.data_ptr().as_ptr();
                ::core::fmt::Pointer::fmt(&ptr, f)
            }
        }
    };
}

#[cfg(feature = "utf16")]
pub(crate) use impl_utf16_str_accessors;
#[cfg(feature = "utf16")]
pub(crate) use impl_utf16_str_read_traits;
