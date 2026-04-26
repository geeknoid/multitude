use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::Allocator;

use crate::arena::slice_drop_shim;
use crate::drop_entry::{drop_shim, entry_for_value, rewrite_drop_fn};
use crate::flavor::Flavor;

/// Refcounted-handle plumbing shared by [`Rc`](crate::Rc),
/// [`Arc`](crate::Arc), [`RcStr`](crate::RcStr), and
/// [`ArcStr`](crate::ArcStr).
///
/// `Clone` bumps the chunk refcount; `Drop` releases it. The flavor
/// parameter selects the inc/release primitive (Local = `Cell`,
/// Shared = `AtomicUsize`).
pub struct RawHandle<T: ?Sized, F: Flavor, A: Allocator + Clone> {
    ptr: NonNull<T>,
    _owns: PhantomData<T>,
    _flavor: PhantomData<F>,
    _allocator: PhantomData<A>,
}

impl<T: ?Sized, F: Flavor, A: Allocator + Clone> RawHandle<T, F, A> {
    /// # Safety
    ///
    /// `ptr` must point to a value in a live chunk of flavor `F` whose
    /// refcount has already been incremented by 1 for this handle.
    #[inline]
    pub const unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _owns: PhantomData,
            _flavor: PhantomData,
            _allocator: PhantomData,
        }
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub const fn as_non_null(&self) -> NonNull<T> {
        self.ptr
    }

    /// True iff both handles point at the same address.
    #[inline]
    #[must_use]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        core::ptr::addr_eq(a.ptr.as_ptr(), b.ptr.as_ptr())
    }

    /// # Safety
    ///
    /// The pointee must currently be a fully-initialized, valid `T` and
    /// the usual aliasing rules for `&T` must hold for the duration of
    /// the returned borrow.
    #[inline]
    pub const unsafe fn as_ref(&self) -> &T {
        // SAFETY: caller's contract.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T, F: Flavor, A: Allocator + Clone> RawHandle<[T], F, A> {
    /// Construct a slice-typed handle from an element pointer + length.
    ///
    /// # Safety
    ///
    /// `ptr..ptr+len` must point to fully-initialized `T`s inside a
    /// live chunk of flavor `F` with the chunk's refcount already
    /// incremented by 1 for this handle.
    #[inline]
    pub const unsafe fn from_raw_slice(ptr: NonNull<T>, len: usize) -> Self {
        let slice_ptr = NonNull::slice_from_raw_parts(ptr, len);
        // SAFETY: caller's contract.
        unsafe { Self::from_raw(slice_ptr) }
    }
}

impl<T, F: Flavor, A: Allocator + Clone> RawHandle<MaybeUninit<T>, F, A> {
    /// Convert a handle to `MaybeUninit<T>` whose value is now
    /// initialized into a handle to `T`. O(1) — no copy or alloc.
    ///
    /// If `T: Drop`, this rewrites the chunk's drop-list entry from
    /// the `MaybeUninit<T>`-shim placeholder to the real `T`-shim.
    ///
    /// # Safety
    ///
    /// The `MaybeUninit<T>` must contain a fully-initialized, valid
    /// `T`.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "rewrite block forms one canonical-layout invariant; final from_raw block forms the post-init transfer invariant"
    )]
    pub unsafe fn assume_init(self) -> RawHandle<T, F, A> {
        let ptr = self.ptr;
        core::mem::forget(self);
        if core::mem::needs_drop::<T>() {
            // SAFETY: alloc_uninit_* linked a no-op shim; rewrite to the T-shim.
            unsafe {
                let value = NonNull::new_unchecked(ptr.as_ptr().cast::<u8>());
                let entry = entry_for_value(value, align_of::<T>());
                rewrite_drop_fn(entry, drop_shim::<T>);
            }
        }
        // SAFETY: caller guarantees init; refcount transfers verbatim.
        unsafe { RawHandle::from_raw(ptr.cast::<T>()) }
    }
}

impl<T, F: Flavor, A: Allocator + Clone> RawHandle<[MaybeUninit<T>], F, A> {
    /// Convert a slice handle of `MaybeUninit<T>` whose elements are
    /// now initialized into a slice handle of `T`. O(1).
    ///
    /// If `T: Drop`, rewrites the chunk's drop-list entry from the
    /// `MaybeUninit<T>`-slice-shim placeholder to the real `T`-slice-shim.
    ///
    /// # Safety
    ///
    /// Every element of the slice must contain a fully-initialized,
    /// valid `T`.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "rewrite block forms one canonical-layout invariant; final from_raw block forms the post-init transfer invariant"
    )]
    pub unsafe fn assume_init_slice(self) -> RawHandle<[T], F, A> {
        let ptr = self.as_non_null();
        let len = ptr.len();
        let data = ptr.cast::<u8>();
        core::mem::forget(self);
        if core::mem::needs_drop::<T>() {
            // SAFETY: alloc_uninit_slice_* linked a no-op shim; rewrite to the T-slice-shim.
            unsafe {
                let entry = entry_for_value(data, align_of::<T>());
                rewrite_drop_fn(entry, slice_drop_shim::<T>);
            }
        }
        // SAFETY: caller guarantees init; refcount transfers verbatim.
        unsafe { RawHandle::from_raw_slice(data.cast::<T>(), len) }
    }
}

impl<T: ?Sized, F: Flavor, A: Allocator + Clone> Clone for RawHandle<T, F, A> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: we hold a +1 refcount on a live chunk of flavor `F`.
        unsafe { F::inc_ref::<T, A>(self.ptr) };
        Self {
            ptr: self.ptr,
            _owns: PhantomData,
            _flavor: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<T: ?Sized, F: Flavor, A: Allocator + Clone> Drop for RawHandle<T, F, A> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: we hold a +1 refcount on a live chunk of flavor `F`.
        unsafe { F::release::<T, A>(self.ptr) };
    }
}
