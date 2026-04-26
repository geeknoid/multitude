use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::arena_handle_macros::impl_handle_read_traits;
use crate::flavor::Shared;
use crate::raw_handle::RawHandle;

/// A reference-counted smart pointer to a `T` stored in an
/// [`Arena`](crate::Arena), safe to share across threads when
/// `T: Send + Sync`.
///
/// Created via [`Arena::alloc_arc`](crate::Arena::alloc_arc). Cloning is
/// **O(1)** but uses an atomic refcount (one Acquire-RMW). For
/// single-threaded code, prefer [`Rc`](crate::Rc) — it has the same
/// cost model with a non-atomic refcount.
///
/// `Arc` keeps its containing chunk alive by holding a +1 refcount on
/// it, so the smart pointer can outlive the arena it came from and
/// survives [`Arena::reset`](crate::Arena::reset). `T::drop` runs when
/// the chunk is reclaimed (i.e. when its last live allocation is
/// released).
///
/// # Example
///
/// ```
/// use multitude::Arena;
/// use std::thread;
///
/// let arena = Arena::new();
/// let a = arena.alloc_arc(42_u32);
/// let b = a.clone();
/// let h = thread::spawn(move || *b);
/// assert_eq!(*a, h.join().unwrap());
/// ```
pub struct Arc<T: ?Sized, A: Allocator + Clone = Global> {
    inner: RawHandle<T, Shared, A>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "RawHandle holds a NonNull which is structurally !Send; the Shared flavor's atomic refcount makes cross-thread sharing sound"
)]
// SAFETY: backed by a `Shared`-flavor chunk with atomic refcount; same
// reasoning as `Arc<T>`.
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Clone + Send + Sync> Send for Arc<T, A> {}
// SAFETY: see above.
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Clone + Send + Sync> Sync for Arc<T, A> {}

impl<T: ?Sized, A: Allocator + Clone> Clone for Arc<T, A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Arc<T, A> {
    /// # Safety
    ///
    /// `ptr` must point to a fully-initialized `T` inside a `Shared`-
    /// flavor chunk allocated by an `Arena<A>`, with the chunk's refcount
    /// already incremented for this smart pointer.
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

impl<T, A: Allocator + Clone> Arc<[T], A> {
    /// # Safety
    ///
    /// See [`Arc::from_raw`].
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn from_raw_slice(ptr: NonNull<T>, len: usize) -> Self {
        Self {
            // SAFETY: caller's contract.
            inner: unsafe { RawHandle::from_raw_slice(ptr, len) },
        }
    }
}

impl<T, A: Allocator + Clone + Send + Sync> Arc<MaybeUninit<T>, A> {
    /// Convert an [`Arc<MaybeUninit<T>, A>`] whose value has been
    /// fully initialized into an [`Arc<T, A>`]. O(1) — no copy,
    /// no allocation.
    ///
    /// # Safety
    ///
    /// The `MaybeUninit<T>` must contain a fully-initialized, valid `T`.
    /// Any other [`Arc<MaybeUninit<T>, A>`] clones outstanding for
    /// the same value will, after this call, observe the freshly-typed
    /// `T` through their (still `MaybeUninit<T>`-typed) view.
    #[must_use]
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<T, A> {
        // SAFETY: we own `self`; `inner` is read once and then we forget `self`.
        let inner = unsafe { core::ptr::read(&raw const self.inner) };
        core::mem::forget(self);
        Arc {
            // SAFETY: caller guarantees init.
            inner: unsafe { inner.assume_init() },
        }
    }
}

impl<T, A: Allocator + Clone + Send + Sync> Arc<[MaybeUninit<T>], A> {
    /// Convert an [`Arc<[MaybeUninit<T>], A>`](crate::Arc) whose elements have
    /// all been fully initialized into an [`Arc<[T], A>`](crate::Arc). O(1).
    ///
    /// # Safety
    ///
    /// Every element of the slice must contain a fully-initialized,
    /// valid `T`.
    #[must_use]
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<[T], A> {
        // SAFETY: we own `self`; `inner` is read once and then we forget `self`.
        let inner = unsafe { core::ptr::read(&raw const self.inner) };
        core::mem::forget(self);
        Arc {
            // SAFETY: caller guarantees init.
            inner: unsafe { inner.assume_init_slice() },
        }
    }
}

impl_handle_read_traits!(
    generics = [T: ?Sized, A: Allocator + Clone],
    type = Arc<T, A>,
    deref_target = T,
    ptr_field = inner,
);

// `Unpin` mirrors `std::sync::Arc<T>: Unpin`.
impl<T: ?Sized, A: Allocator + Clone> Unpin for Arc<T, A> {}
