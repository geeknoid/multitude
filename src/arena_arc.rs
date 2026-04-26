//! [`ArenaArc`] — `Send + Sync` reference-counted handle.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::chunk_header::{header_for, teardown_chunk};

/// A reference-counted handle to a `T` (possibly unsized) allocated inside
/// an [`Arena`](crate::Arena).
///
/// Uses atomic operations on the owning chunk's refcount, making the handle
/// safe to send and share across threads. `ArenaArc<T, A>` is `Send + Sync`
/// whenever `T: Send + Sync` and `A: Send + Sync`. For single-threaded use
/// without atomic overhead, see [`ArenaRc`](crate::ArenaRc).
///
/// Created via [`Arena::alloc_shared`](crate::Arena::alloc_shared).
///
/// # Example
///
/// ```
/// use harena::Arena;
/// use std::thread;
///
/// let arena = Arena::new();
/// let a = arena.alloc_shared(42_u32);
/// let b = a.clone();
/// let h = thread::spawn(move || *b);
/// assert_eq!(*a, h.join().unwrap());
/// ```
pub struct ArenaArc<T: ?Sized, A: Allocator + Clone = Global> {
    pub(crate) ptr: NonNull<T>,
    pub(crate) _owns: PhantomData<T>,
    pub(crate) _allocator: PhantomData<A>,
}

// SAFETY: backed by a `Shared`-flavor chunk with atomic refcount; same
// reasoning as `Arc<T>`.
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Clone + Send + Sync> Send for ArenaArc<T, A> {}
// SAFETY: see above.
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Clone + Send + Sync> Sync for ArenaArc<T, A> {}

impl<T: ?Sized, A: Allocator + Clone> ArenaArc<T, A> {
    /// Construct from a raw value pointer (sized).
    ///
    /// # Safety
    ///
    /// `ptr` must point to a fully-initialized `T` inside a `Shared`-
    /// flavor chunk allocated by an `Arena<A>`, with the chunk's refcount
    /// already incremented for this handle.
    pub(crate) unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        T: Sized,
    {
        Self {
            ptr,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Returns a raw pointer to the value inside the arena.
    #[must_use]
    pub fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr()
    }

    /// Returns the current reference count (Acquire load).
    #[must_use]
    pub fn chunk_ref_count(this: &Self) -> usize {
        // SAFETY: ptr is in a valid chunk.
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

impl<T, A: Allocator + Clone> ArenaArc<[T], A> {
    /// Construct from a raw element pointer + length (slice).
    ///
    /// # Safety
    ///
    /// See [`ArenaArc::from_raw`].
    pub(crate) unsafe fn from_raw_slice(ptr: NonNull<T>, len: usize) -> Self {
        // SAFETY: caller guarantees validity.
        let slice_ptr = unsafe { NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), len)) };
        Self {
            ptr: slice_ptr,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Clone for ArenaArc<T, A> {
    fn clone(&self) -> Self {
        // SAFETY: chunk is alive (we hold a refcount).
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.ptr);
            chunk.as_ref().inc_ref();
        }
        Self {
            ptr: self.ptr,
            _owns: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Drop for ArenaArc<T, A> {
    fn drop(&mut self) {
        // SAFETY: chunk is alive; dec refcount.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.ptr);
            if chunk.as_ref().dec_ref() {
                // We may be on the owner thread or another thread. We
                // pass `false` (not on owner thread) conservatively
                // because the cache is owner-only anyway, and `Shared`
                // chunks generally shouldn't be cached. The teardown
                // code respects this.
                teardown_chunk(chunk, false);
            }
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> Deref for ArenaArc<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: ptr references a live T whose chunk we hold a ref on.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized + fmt::Debug, A: Allocator + Clone> fmt::Debug for ArenaArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Display, A: Allocator + Clone> fmt::Display for ArenaArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + PartialEq, A: Allocator + Clone> PartialEq for ArenaArc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl<T: ?Sized + Eq, A: Allocator + Clone> Eq for ArenaArc<T, A> {}
impl<T: ?Sized + PartialOrd, A: Allocator + Clone> PartialOrd for ArenaArc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}
impl<T: ?Sized + Ord, A: Allocator + Clone> Ord for ArenaArc<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: ?Sized + Hash, A: Allocator + Clone> Hash for ArenaArc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}
