use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::arena_handle_macros::impl_handle_read_traits;
use crate::flavor::Local;
use crate::raw_handle::RawHandle;

/// A single-threaded reference-counted smart pointer to a `T` stored in
/// an [`Arena`](crate::Arena).
///
/// Created via [`Arena::alloc_rc`](crate::Arena::alloc_rc). Cloning is
/// **O(1)** (a non-atomic refcount bump). For cross-thread sharing, use
/// [`Arc`](crate::Arc) instead.
///
/// `Rc` keeps its containing chunk alive by holding a +1 refcount on
/// it, so the smart pointer can outlive the arena it came from and
/// survives [`Arena::reset`](crate::Arena::reset). `T::drop` runs when
/// the chunk is reclaimed (i.e. when its last live allocation is
/// released).
///
/// # Example
///
/// ```
/// use multitude::Arena;
///
/// struct Point { x: f64, y: f64 }
///
/// let arena = Arena::new();
/// let a = arena.alloc_rc(Point { x: 3.0, y: 4.0 });
/// let b = a.clone();
/// assert_eq!(a.x, b.x);
/// ```
pub struct Rc<T: ?Sized, A: Allocator + Clone = Global> {
    inner: RawHandle<T, Local, A>,
}

impl<T: ?Sized, A: Allocator + Clone> Clone for Rc<T, A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Rc<T, A> {
    /// # Safety
    ///
    /// `ptr` must point to a fully-initialized `T` inside a `Local`-flavor
    /// chunk allocated by an `Arena<A>`, with the chunk's refcount already
    /// incremented for this smart pointer.
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        T: Sized,
    {
        Self {
            // SAFETY: caller's contract.
            inner: unsafe { RawHandle::from_raw(ptr) },
        }
    }

    /// # Safety
    ///
    /// Same contract as [`Self::from_raw`] but accepts a possibly-unsized
    /// `T` (e.g., a fat pointer reconstructed from a layout + metadata).
    #[cfg(feature = "dst")]
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn from_raw_unsized(ptr: NonNull<T>) -> Self {
        Self {
            // SAFETY: caller's contract.
            inner: unsafe { RawHandle::from_raw(ptr) },
        }
    }

    /// Returns a raw pointer to the value inside the arena.
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// True iff both smart pointers point to the same value.
    #[must_use]
    #[inline]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        RawHandle::ptr_eq(&a.inner, &b.inner)
    }
}

impl<T, A: Allocator + Clone> Rc<[T], A> {
    /// # Safety
    ///
    /// `ptr..ptr+len` must point to fully-initialized `T`s inside a
    /// `Local`-flavor chunk with the chunk's refcount already incremented.
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn from_raw_slice(ptr: NonNull<T>, len: usize) -> Self {
        Self {
            // SAFETY: caller's contract.
            inner: unsafe { RawHandle::from_raw_slice(ptr, len) },
        }
    }
}

impl<T, A: Allocator + Clone> Rc<MaybeUninit<T>, A> {
    /// Convert an [`Rc<MaybeUninit<T>, A>`] whose value has been
    /// fully initialized into an [`Rc<T, A>`]. O(1) — no copy,
    /// no allocation.
    ///
    /// # Safety
    ///
    /// The `MaybeUninit<T>` must contain a fully-initialized, valid `T`.
    /// Any other [`Rc<MaybeUninit<T>, A>`] clones outstanding for
    /// the same value will, after this call, observe the freshly-typed
    /// `T` through their (still `MaybeUninit<T>`-typed) view.
    #[must_use]
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<T, A> {
        // Move out of `self.inner` without running `Self`'s drop.
        // SAFETY: we own `self`; `inner` is read once and then we forget `self`.
        let inner = unsafe { core::ptr::read(&raw const self.inner) };
        core::mem::forget(self);
        Rc {
            // SAFETY: caller guarantees init; the rewrite + transfer happens inside.
            inner: unsafe { inner.assume_init() },
        }
    }
}

impl<T, A: Allocator + Clone> Rc<[MaybeUninit<T>], A> {
    /// Convert an [`Rc<[MaybeUninit<T>], A>`](crate::Rc) whose elements have
    /// all been fully initialized into an [`Rc<[T], A>`](crate::Rc). O(1).
    ///
    /// # Safety
    ///
    /// Every element of the slice must contain a fully-initialized,
    /// valid `T`.
    #[must_use]
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<[T], A> {
        // SAFETY: we own `self`; `inner` is read once and then we forget `self`.
        let inner = unsafe { core::ptr::read(&raw const self.inner) };
        core::mem::forget(self);
        Rc {
            // SAFETY: caller guarantees init.
            inner: unsafe { inner.assume_init_slice() },
        }
    }
}

impl_handle_read_traits!(
    generics = [T: ?Sized, A: Allocator + Clone],
    type = Rc<T, A>,
    deref_target = T,
    ptr_field = inner,
);

// `Unpin` mirrors `std::rc::Rc<T>: Unpin`: the pointee lives in its chunk.
impl<T: ?Sized, A: Allocator + Clone> Unpin for Rc<T, A> {}

#[cfg(feature = "builders")]
impl<'a, T, A: Allocator + Clone> ::core::convert::From<crate::builders::Vec<'a, T, A>> for Rc<[T], A> {
    /// Freeze an [`Vec`](crate::builders::Vec) into an immutable
    /// [`Rc<[T], A>`](crate::Rc). See [`Vec::into_arena_rc`](crate::builders::Vec::into_arena_rc).
    #[inline]
    fn from(v: crate::builders::Vec<'a, T, A>) -> Self {
        v.into_arena_rc()
    }
}
