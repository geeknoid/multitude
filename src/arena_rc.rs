//! [`ArenaRc`] — single-threaded reference-counted handle.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::chunk_header::{header_for, teardown_chunk};

/// A single-threaded reference-counted handle to a `T` (possibly unsized)
/// allocated inside an [`Arena`](crate::Arena).
///
/// Cloning increments the owning chunk's reference count; dropping
/// decrements it. When the chunk's reference count reaches zero, its drop
/// list runs and the chunk's backing memory is released.
///
/// `ArenaRc<T, A>` is **not** [`Send`] / [`Sync`].
///
/// # Example
///
/// ```
/// use harena::Arena;
///
/// let arena = Arena::new();
/// let a = arena.alloc(String::from("hello"));
/// let b = a.clone();
/// assert_eq!(*a, *b);
/// ```
pub struct ArenaRc<T: ?Sized, A: Allocator + Clone = Global> {
    pub(crate) ptr: NonNull<T>,
    /// `ArenaRc` is single-threaded only.
    pub(crate) _not_sync: PhantomData<*mut ()>,
    /// Conceptually owns a `T` for variance/dropck purposes.
    pub(crate) _owns: PhantomData<T>,
    /// Ties the handle to its arena's allocator type so the chunk header
    /// can be reified at the right type for refcount/teardown.
    pub(crate) _allocator: PhantomData<A>,
}

impl<T: ?Sized, A: Allocator + Clone> ArenaRc<T, A> {
    /// Construct from a raw value pointer (sized `T`).
    ///
    /// # Safety
    ///
    /// `ptr` must point to a fully-initialized `T` inside a `Local`-flavor
    /// chunk allocated by an `Arena<A>`, with the chunk's refcount already
    /// incremented for this handle.
    pub(crate) unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        T: Sized,
    {
        Self {
            ptr,
            _not_sync: PhantomData,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Returns a raw pointer to the value inside the arena.
    #[must_use]
    pub fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr()
    }

    /// Returns the current reference count of the owning chunk.
    ///
    /// Note: this is the *chunk* refcount — multiple distinct values
    /// allocated from the same chunk share a single count.
    #[must_use]
    pub fn chunk_ref_count(this: &Self) -> usize {
        // SAFETY: ptr points into a valid chunk.
        unsafe {
            let header: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(this.ptr);
            header.as_ref().current_ref_count()
        }
    }

    /// True iff both handles point to the same value.
    #[must_use]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        core::ptr::addr_eq(a.ptr.as_ptr(), b.ptr.as_ptr())
    }
}

impl<T, A: Allocator + Clone> ArenaRc<[T], A> {
    /// Construct from a raw element pointer + length (slice).
    ///
    /// # Safety
    ///
    /// `ptr..ptr+len` must point to fully-initialized `T`s inside a
    /// `Local`-flavor chunk with the chunk's refcount already incremented.
    pub(crate) unsafe fn from_raw_slice(ptr: NonNull<T>, len: usize) -> Self {
        // SAFETY: caller guarantees the slice is valid.
        let slice_ptr = unsafe { NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), len)) };
        Self {
            ptr: slice_ptr,
            _not_sync: PhantomData,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Clone for ArenaRc<T, A> {
    fn clone(&self) -> Self {
        // SAFETY: ptr is in a valid chunk; bump its refcount.
        unsafe {
            let header: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.ptr);
            header.as_ref().inc_ref();
        }
        Self {
            ptr: self.ptr,
            _not_sync: PhantomData,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Drop for ArenaRc<T, A> {
    fn drop(&mut self) {
        // SAFETY: ptr in a valid chunk; dec refcount and tear down if
        // we were the last.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.ptr);
            if chunk.as_ref().dec_ref() {
                // Local chunk → owner thread (we're !Send).
                teardown_chunk(chunk, true);
            }
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Deref for ArenaRc<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: ptr references a live T inside a chunk whose refcount
        // is held by self.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized + fmt::Debug, A: Allocator + Clone> fmt::Debug for ArenaRc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Display, A: Allocator + Clone> fmt::Display for ArenaRc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + PartialEq, A: Allocator + Clone> PartialEq for ArenaRc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl<T: ?Sized + Eq, A: Allocator + Clone> Eq for ArenaRc<T, A> {}
impl<T: ?Sized + PartialOrd, A: Allocator + Clone> PartialOrd for ArenaRc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}
impl<T: ?Sized + Ord, A: Allocator + Clone> Ord for ArenaRc<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: ?Sized + Hash, A: Allocator + Clone> Hash for ArenaRc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}
