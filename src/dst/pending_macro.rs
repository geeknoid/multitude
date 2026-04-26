macro_rules! define_pending_arena {
    (
        $(#[$meta:meta])*
        $name:ident,
        handle = [$($handle:tt)+],
        sharing = $sharing:ident,
        on_owner_thread = $owner:literal,
        debug_name = $debug:literal,
        kind = $kind:ident
        $(,)?
    ) => {
        $(#[$meta])*
        #[cfg_attr(docsrs, doc(cfg(feature = "dst")))]
        pub struct $name<'a, A: ::allocator_api2::alloc::Allocator + Clone = ::allocator_api2::alloc::Global> {
            arena: &'a $crate::Arena<A>,
            chunk: ::core::ptr::NonNull<$crate::chunk_header::ChunkHeader<A>>,
            entry: ::core::ptr::NonNull<$crate::drop_entry::DropEntry>,
            bytes: ::core::ptr::NonNull<::core::mem::MaybeUninit<u8>>,
            layout: ::core::alloc::Layout,
        }

        impl<'a, A: ::allocator_api2::alloc::Allocator + Clone> $name<'a, A> {
            /// Reserve uninitialized space for a value with the given
            /// `Layout` in a chunk owned by `arena`.
            ///
            /// # Panics
            ///
            /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
            #[must_use]
            pub fn new(arena: &'a $crate::Arena<A>, layout: ::core::alloc::Layout) -> Self {
                Self::try_new(arena, layout).unwrap_or_else(|_| $crate::arena::panic_alloc())
            }

            /// Fallible variant of [`Self::new`].
            ///
            /// # Errors
            ///
            /// Returns [`AllocError`](::allocator_api2::alloc::AllocError) if the backing allocator fails or if the data alignment is at least 64 KiB.
            pub fn try_new(
                arena: &'a $crate::Arena<A>,
                layout: ::core::alloc::Layout,
            ) -> ::core::result::Result<Self, ::allocator_api2::alloc::AllocError> {
                let (entry, value, chunk) =
                    arena.try_reserve_dst_with_entry($crate::chunk_sharing::ChunkSharing::$sharing, layout)?;
                Ok(Self::from_reservation(arena, chunk, entry, value, layout))
            }

            #[must_use]
            pub(crate) const fn from_reservation(
                arena: &'a $crate::Arena<A>,
                chunk: ::core::ptr::NonNull<$crate::chunk_header::ChunkHeader<A>>,
                entry: ::core::ptr::NonNull<$crate::drop_entry::DropEntry>,
                bytes: ::core::ptr::NonNull<u8>,
                layout: ::core::alloc::Layout,
            ) -> Self {
                Self {
                    arena,
                    chunk,
                    entry,
                    bytes: bytes.cast::<::core::mem::MaybeUninit<u8>>(),
                    layout,
                }
            }

            /// The reserved layout.
            #[must_use]
            pub const fn layout(&self) -> ::core::alloc::Layout {
                self.layout
            }

            /// Mutable pointer to the start of the reserved bytes.
            #[must_use]
            pub const fn as_mut_ptr(&mut self) -> *mut u8 {
                self.bytes.as_ptr().cast::<u8>()
            }

            $crate::dst::pending_macro::define_pending_arena!(
                @finalize_methods $kind, [$($handle)+]
            );
        }

        impl<A: ::allocator_api2::alloc::Allocator + Clone> ::core::ops::Drop for $name<'_, A> {
            fn drop(&mut self) {
                // SAFETY: chunk alive via our ref.
                unsafe { $crate::chunk_header::release_chunk_ref(self.chunk, $owner) };
                let _ = self.arena;
            }
        }

        impl<A: ::allocator_api2::alloc::Allocator + Clone> ::core::fmt::Debug for $name<'_, A> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.debug_struct($debug)
                    .field("layout", &self.layout)
                    .finish_non_exhaustive()
            }
        }
    };

    (@finalize_methods sized_and_dst, [$($handle:tt)+]) => {
        /// # Safety
        ///
        /// - All bytes covered by [`Self::layout`] must have been
        ///   initialized to a valid `T`.
        /// - `fat_template`'s metadata must be valid for the value
        ///   just written.
        /// - `drop_fn`, if provided, must be safe to call exactly once
        ///   and must correctly locate the value relative to the entry
        ///   pointer it receives.
        pub unsafe fn finalize<T>(
            self,
            fat_template: *const T,
            drop_fn: ::core::option::Option<unsafe fn(*mut $crate::drop_entry::DropEntry)>,
        ) -> $($handle)+<T, A> {
            // SAFETY: caller's contract; sized T is trivially Pointee.
            unsafe { Self::finalize_inner::<T>(self, fat_template, drop_fn) }
        }

        /// # Safety
        ///
        /// Same contract as [`Self::finalize`].
        pub unsafe fn finalize_dst<T: ::ptr_meta::Pointee + ?Sized>(
            self,
            fat_template: *const T,
            drop_fn: ::core::option::Option<unsafe fn(*mut $crate::drop_entry::DropEntry)>,
        ) -> $($handle)+<T, A> {
            // SAFETY: caller's contract.
            unsafe { Self::finalize_inner::<T>(self, fat_template, drop_fn) }
        }

        $crate::dst::pending_macro::define_pending_arena!(
            @finalize_inner [$($handle)+], handle_args = [<T, A>]
        );
    };

    (@finalize_methods dst_only, [$($handle:tt)+]) => {
        /// # Safety
        ///
        /// - All bytes covered by [`Self::layout`] must have been
        ///   initialized to a valid `T`.
        /// - `fat_template`'s metadata must be valid for the value just
        ///   written.
        /// - If `T` has a non-trivial destructor, `drop_fn` must be `Some(_)`;
        ///   otherwise dropping the resulting `Box<T, A>` leaks `T::drop`,
        ///   which is undefined behavior whenever `T`'s destructor is
        ///   required for soundness.
        /// - If `T` has no destructor, pass `None`.
        pub unsafe fn finalize<T: ::ptr_meta::Pointee + ?Sized>(
            self,
            fat_template: *const T,
            drop_fn: ::core::option::Option<unsafe fn(*mut $crate::drop_entry::DropEntry)>,
        ) -> $($handle)+<T, A> {
            // SAFETY: caller's contract.
            unsafe { Self::finalize_inner::<T>(self, fat_template, drop_fn) }
        }

        $crate::dst::pending_macro::define_pending_arena!(
            @finalize_inner [$($handle)+], handle_args = [<T, A>]
        );
    };

    (@finalize_inner [$($handle:tt)+], handle_args = [$($handle_args:tt)+]) => {
        /// # Safety
        ///
        /// Same contract as the public callers.
        unsafe fn finalize_inner<T: ::ptr_meta::Pointee + ?Sized>(
            this: Self,
            fat_template: *const T,
            drop_fn: ::core::option::Option<unsafe fn(*mut $crate::drop_entry::DropEntry)>,
        ) -> $($handle)+ $($handle_args)+ {
            let pa = ::core::mem::ManuallyDrop::new(this);

            if let Some(drop_fn) = drop_fn {
                // SAFETY: entry slot writable, chunk alive.
                unsafe {
                    pa.chunk.as_ref().link_drop_entry(pa.entry, drop_fn, 0);
                }
            }

            let data_ptr = pa.bytes.as_ptr().cast::<u8>();
            // SAFETY: caller guarantees metadata validity.
            let fat = unsafe { $crate::dst::helpers::reconstruct_fat::<T>(fat_template, data_ptr) };

            // SAFETY: refcount bumped; data initialized per caller's contract.
            unsafe { <$($handle)+ $($handle_args)+>::from_raw_unsized(fat) }
        }
    };
}

pub(crate) use define_pending_arena;
