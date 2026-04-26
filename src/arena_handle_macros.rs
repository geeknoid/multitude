/// Emit the shared read-only trait impls (`Deref`, `Debug`, `Display`,
/// `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`, `AsRef`, `Borrow`,
/// `Pointer`) for a refcounted smart pointer whose value pointer is
/// reachable through `self.$field`.
///
/// `$field` may name either a `NonNull<$target>` field directly (as on
/// [`Box`](crate::Box)) or a
/// [`RawHandle`](crate::raw_handle::RawHandle) field (as on
/// [`Rc`](crate::Rc) / [`Arc`](crate::Arc)) — both
/// expose the same `as_ref` accessor used here.
macro_rules! impl_handle_read_traits {
    (
        generics = [$($g:tt)+],
        type = $ty:ty,
        deref_target = $target:ty,
        ptr_field = $field:ident
        $(,)?
    ) => {
        impl<$($g)+> ::core::ops::Deref for $ty {
            type Target = $target;
            #[inline]
            fn deref(&self) -> &$target {
                // SAFETY: `self.$field` references a live T inside a
                // chunk whose refcount is held by self.
                unsafe { self.$field.as_ref() }
            }
        }

        impl<$($g)+> ::core::fmt::Debug for $ty
        where
            $target: ::core::fmt::Debug,
        {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&**self, f)
            }
        }

        impl<$($g)+> ::core::fmt::Display for $ty
        where
            $target: ::core::fmt::Display,
        {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(&**self, f)
            }
        }

        impl<$($g)+> ::core::cmp::PartialEq for $ty
        where
            $target: ::core::cmp::PartialEq,
        {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                **self == **other
            }
        }

        impl<$($g)+> ::core::cmp::Eq for $ty
        where
            $target: ::core::cmp::Eq,
        {}

        impl<$($g)+> ::core::cmp::PartialOrd for $ty
        where
            $target: ::core::cmp::PartialOrd,
        {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> ::core::option::Option<::core::cmp::Ordering> {
                (**self).partial_cmp(&**other)
            }
        }

        impl<$($g)+> ::core::cmp::Ord for $ty
        where
            $target: ::core::cmp::Ord,
        {
            #[inline]
            fn cmp(&self, other: &Self) -> ::core::cmp::Ordering {
                (**self).cmp(&**other)
            }
        }

        impl<$($g)+> ::core::hash::Hash for $ty
        where
            $target: ::core::hash::Hash,
        {
            #[inline]
            fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                (**self).hash(state);
            }
        }

        impl<$($g)+> ::core::convert::AsRef<$target> for $ty {
            #[inline]
            fn as_ref(&self) -> &$target {
                self
            }
        }

        impl<$($g)+> ::core::borrow::Borrow<$target> for $ty {
            #[inline]
            fn borrow(&self) -> &$target {
                self
            }
        }

        impl<$($g)+> ::core::fmt::Pointer for $ty {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let ptr: *const $target = &**self;
                ::core::fmt::Pointer::fmt(&ptr, f)
            }
        }
    };
}

pub(crate) use impl_handle_read_traits;
