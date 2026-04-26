use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

#[cfg(feature = "dst")]
use crate::arena::slice_drop_shim;
use crate::arena_handle_macros::impl_handle_read_traits;
use crate::chunk_header::{ChunkHeader, header_for, release_chunk_ref_local};
use crate::chunk_ref::ChunkRef;
use crate::drop_entry::{drop_shim, entry_for_value, rewrite_drop_fn};
use crate::rc::Rc;

/// An owned, mutable smart pointer to a `T` stored in an
/// [`Arena`](crate::Arena).
///
/// Created via [`Arena::alloc_box`](crate::Arena::alloc_box).
///
/// Unlike [`Rc`](crate::Rc) / [`Arc`](crate::Arc):
///
/// - **Drop runs when the smart pointer is dropped**, not at chunk teardown. Useful for
///   `T`s that hold OS resources which must be released promptly.
/// - Provides `&mut T` through `DerefMut`.
/// - **Not** [`Clone`] — single owner.
///
/// Like [`Rc`](crate::Rc), `Box` keeps its containing
/// chunk alive by holding a +1 refcount, so it can outlive the arena it
/// came from and survives [`Arena::reset`](crate::Arena::reset).
///
/// # Example
///
/// ```
/// use multitude::Arena;
///
/// let arena = Arena::new();
/// let mut b = arena.alloc_box(vec![1, 2, 3]);
/// b.push(4);
/// assert_eq!(*b, vec![1, 2, 3, 4]);
/// ```
pub struct Box<T: ?Sized, A: Allocator + Clone = Global> {
    ptr: NonNull<T>,
    _owns: PhantomData<T>,
    _not_sync: PhantomData<*mut ()>,
    _allocator: PhantomData<A>,
}

impl<T, A: Allocator + Clone> Box<T, A> {
    /// # Safety
    ///
    /// `ptr` must point to a fully-initialized `T` inside a `Local`-flavor
    /// chunk allocated by an `Arena<A>`. The chunk's drop list must
    /// contain a `DropEntry` for this value (with `drop_fn = drop_shim::<T>`)
    /// linked at the position computed by `value_offset_after_entry::<T>()`,
    /// and the chunk's refcount must have been incremented for this
    /// smart pointer.
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _owns: PhantomData,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Convert this owned, mutable box into a shared, immutable
    /// [`Rc<T, A>`](crate::Rc). O(1) — no copy, no allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use multitude::{Arena, Rc};
    ///
    /// let arena = Arena::new();
    /// let mut b = arena.alloc_box(vec![1, 2, 3]);
    /// b.push(4);
    /// // Done mutating — freeze and share.
    /// let rc: Rc<Vec<i32>> = b.into_rc();
    /// let rc2 = rc.clone();
    /// assert_eq!(*rc, *rc2);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_rc(self) -> Rc<T, A> {
        let ptr = self.ptr;
        core::mem::forget(self);
        // SAFETY: chunk's +1 refcount preserved; linked DropEntry stays linked.
        unsafe { Rc::from_raw(ptr) }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Box<T, A> {
    /// # Safety
    ///
    /// Same contract as [`Self::from_raw`] but accepts a possibly-unsized
    /// `T`. The fat pointer must already encode valid metadata for the
    /// value at `ptr`.
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn from_raw_unsized(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _owns: PhantomData,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Returns a raw pointer to the value.
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the value.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "associated-fn convention (like alloc::rc::Rc::as_ptr); &mut self conveys exclusive access"
    )]
    #[must_use]
    #[inline]
    pub const fn as_mut_ptr(this: &mut Self) -> *mut T {
        this.ptr.as_ptr()
    }
}

// No `leak` method: releasing the refcount risks UAF, keeping it leaks the chunk.

#[cfg(feature = "dst")]
impl<T, A: Allocator + Clone> Box<[T], A> {
    /// Convert this owned, mutable
    /// slice box into a shared, immutable [`Rc<[T], A>`](crate::Rc).
    /// O(1) — no copy, no allocation.
    ///
    /// # Example
    ///
    /// ```
    /// let arena = multitude::Arena::new();
    /// let mut b = arena.alloc_slice_copy_box([1_u32, 2, 3]);
    /// b[1] = 99;
    /// let rc = b.into_rc();
    /// assert_eq!(&*rc, &[1, 99, 3]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_rc(self) -> Rc<[T], A> {
        let ptr = self.ptr;
        core::mem::forget(self);
        // SAFETY: refcount + DropEntry transfer; slice_drop_shim matches.
        unsafe { Rc::from_raw_unsized(ptr) }
    }
}

impl<T, A: Allocator + Clone> Box<MaybeUninit<T>, A> {
    /// Convert an [`Box<MaybeUninit<T>, A>`] whose value has been
    /// fully initialized into an [`Box<T, A>`]. O(1) — no copy,
    /// no allocation.
    ///
    /// # Safety
    ///
    /// The `MaybeUninit<T>` must contain a fully-initialized, valid `T`.
    #[must_use]
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "rewrite block forms one canonical-layout invariant; final from_raw block forms the post-init transfer invariant"
    )]
    pub unsafe fn assume_init(self) -> Box<T, A> {
        let ptr = self.ptr;
        core::mem::forget(self);
        if core::mem::needs_drop::<T>() {
            // SAFETY: alloc_uninit_box reserved a DropEntry with a no-op shim;
            // rewrite to the real T-shim.
            unsafe {
                let value = NonNull::new_unchecked(ptr.as_ptr().cast::<u8>());
                let entry = entry_for_value(value, align_of::<T>());
                rewrite_drop_fn(entry, drop_shim::<T>);
            }
        }
        // SAFETY: caller guarantees init; refcount + DropEntry transfer.
        unsafe { Box::from_raw(ptr.cast::<T>()) }
    }
}

#[cfg(feature = "dst")]
impl<T, A: Allocator + Clone> Box<[MaybeUninit<T>], A> {
    /// Convert an [`Box<[MaybeUninit<T>], A>`](crate::Box) whose elements have
    /// all been fully initialized into an [`Box<[T], A>`](crate::Box). O(1) —
    /// no copy, no allocation.
    ///
    /// # Safety
    ///
    /// Every element of the slice must contain a fully-initialized,
    /// valid `T`.
    #[must_use]
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "rewrite block forms one canonical-layout invariant; final from_raw block forms the post-init transfer invariant"
    )]
    pub unsafe fn assume_init(self) -> Box<[T], A> {
        let ptr = self.ptr;
        let len = ptr.len();
        let data = ptr.cast::<u8>();
        core::mem::forget(self);
        if core::mem::needs_drop::<T>() {
            // SAFETY: alloc_uninit_slice_box linked a slice DropEntry with a
            // no-op shim and slice_len = len; rewrite to the real per-element shim.
            unsafe {
                let entry = entry_for_value(data, align_of::<T>());
                rewrite_drop_fn(entry, slice_drop_shim::<T>);
            }
        }
        let fat = NonNull::slice_from_raw_parts(data.cast::<T>(), len);
        // SAFETY: caller guarantees init; refcount + DropEntry transfer.
        unsafe { Box::from_raw_unsized(fat) }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Drop for Box<T, A> {
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    fn drop(&mut self) {
        // SAFETY: ptr in a live chunk with +1 refcount. Raw projection
        // avoids a SharedReadOnly retag when the chunk teardown
        // deallocates.
        unsafe {
            if core::mem::needs_drop::<T>() {
                let chunk: NonNull<ChunkHeader<A>> = header_for(self.ptr);
                let chunk_ref = ChunkRef::<A>::new(chunk);
                let align = align_of_val(self.ptr.as_ref());
                let value_thin = NonNull::new_unchecked(self.ptr.as_ptr().cast::<u8>());
                let entry = entry_for_value(value_thin, align);
                chunk_ref.unlink_drop_entry(entry);
                core::ptr::drop_in_place(self.ptr.as_ptr());
            }
            // `Box` is always `Local`-flavored.
            release_chunk_ref_local::<T, A>(self.ptr);
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> ::core::ops::DerefMut for Box<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: unique owner.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: ?Sized, A: Allocator + Clone> ::core::convert::AsMut<T> for Box<T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: ?Sized, A: Allocator + Clone> ::core::borrow::BorrowMut<T> for Box<T, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

impl<T, A: Allocator + Clone> ::core::convert::From<Box<T, A>> for Rc<T, A> {
    /// Convert an [`Box<T, A>`] into an [`Rc<T, A>`]. O(1) — see
    /// [`Box::into_rc`].
    #[inline]
    fn from(b: Box<T, A>) -> Self {
        b.into_rc()
    }
}

#[cfg(feature = "dst")]
impl<T, A: Allocator + Clone> ::core::convert::From<Box<[T], A>> for Rc<[T], A> {
    /// Convert an [`Box<[T], A>`](crate::Box) into an [`Rc<[T], A>`](crate::Rc). O(1) — see
    /// [`Box::into_rc`].
    #[inline]
    fn from(b: Box<[T], A>) -> Self {
        b.into_rc()
    }
}

impl_handle_read_traits!(
    generics = [T: ?Sized, A: Allocator + Clone],
    type = Box<T, A>,
    deref_target = T,
    ptr_field = ptr,
);

// Iterator-family trait forwarding, mirroring `std::boxed::Box`.

impl<I: Iterator + ?Sized, A: Allocator + Clone> Iterator for Box<I, A> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (**self).next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (**self).nth(n)
    }
}

impl<I: DoubleEndedIterator + ?Sized, A: Allocator + Clone> DoubleEndedIterator for Box<I, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        (**self).next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        (**self).nth_back(n)
    }
}

impl<I: ExactSizeIterator + ?Sized, A: Allocator + Clone> ExactSizeIterator for Box<I, A> {
    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<I: core::iter::FusedIterator + ?Sized, A: Allocator + Clone> core::iter::FusedIterator for Box<I, A> {}

// `Unpin` mirrors `std::boxed::Box`: the pointee lives in its chunk.
impl<T: ?Sized, A: Allocator + Clone> Unpin for Box<T, A> {}
