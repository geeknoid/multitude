//! [`ArenaBox`] — owned, mutable single handle whose `Drop` runs `T::drop`
//! immediately.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::chunk_header::{header_for, teardown_chunk};
use crate::drop_entry::{DropEntry, value_offset_after_entry};

/// An owned, mutable handle to a `T` allocated inside an
/// [`Arena`](crate::Arena).
///
/// Unlike [`ArenaRc`](crate::ArenaRc) / [`ArenaArc`](crate::ArenaArc):
///
/// - **Drop runs on handle drop**, not at chunk teardown. Useful for
///   `T`s that hold OS resources which must be released promptly.
/// - Provides `&mut T` through `DerefMut`.
/// - Borrows `&'a Arena<A>` (not strictly required for soundness, but
///   prevents accidentally long-lived `ArenaBox` leaks).
///
/// Currently restricted to `T: Sized` so the `Drop` impl can statically
/// compute the value-to-entry offset. DST support may be added later.
///
/// # Example
///
/// ```
/// use harena::Arena;
///
/// let arena = Arena::new();
/// let mut b = arena.alloc_box(vec![1, 2, 3]);
/// b.push(4);
/// assert_eq!(*b, vec![1, 2, 3, 4]);
/// ```
pub struct ArenaBox<'a, T, A: Allocator + Clone = Global> {
    ptr: NonNull<T>,
    _arena: PhantomData<&'a crate::Arena<A>>,
    _owns: PhantomData<T>,
    _not_sync: PhantomData<*mut ()>,
}

impl<T, A: Allocator + Clone> ArenaBox<'_, T, A> {
    /// Construct from a raw value pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a fully-initialized `T` inside a `Local`-flavor
    /// chunk allocated by an `Arena<A>`. The chunk's drop list must
    /// contain a `DropEntry` for this value (with `drop_fn = drop_shim::<T>`)
    /// linked at the position computed by `value_offset_after_entry::<T>()`,
    /// and the chunk's refcount must have been incremented for this
    /// handle.
    pub(crate) unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _arena: PhantomData,
            _owns: PhantomData,
            _not_sync: PhantomData,
        }
    }

    /// Returns a raw pointer to the value.
    #[must_use]
    pub fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the value.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "associated-fn convention (like Rc::as_ptr); &mut self conveys exclusive access"
    )]
    #[must_use]
    pub fn as_mut_ptr(this: &mut Self) -> *mut T {
        this.ptr.as_ptr()
    }
}

impl<T, A: Allocator + Clone> Drop for ArenaBox<'_, T, A> {
    fn drop(&mut self) {
        // SAFETY: ptr is in a valid chunk; drop entry is linked iff
        // T needs drop.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.ptr);
            if core::mem::needs_drop::<T>() {
                // Step 1: locate and unlink the DropEntry. Derive the
                // entry pointer from the value pointer via byte_sub
                // (preserves provenance — both lie within the same chunk
                // allocation).
                let value_raw = self.ptr.as_ptr().cast::<u8>();
                let entry_raw = value_raw.byte_sub(value_offset_after_entry::<T>()).cast::<DropEntry>();
                let entry = NonNull::new_unchecked(entry_raw);
                chunk.as_ref().unlink_drop_entry(entry);
                // Step 2: drop the value in place.
                core::ptr::drop_in_place(self.ptr.as_ptr());
            }
            // Step 3: dec refcount, possibly tear down.
            if chunk.as_ref().dec_ref() {
                teardown_chunk(chunk, true);
            }
        }
    }
}

impl<T, A: Allocator + Clone> Deref for ArenaBox<'_, T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: unique owner.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T, A: Allocator + Clone> DerefMut for ArenaBox<'_, T, A> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: unique owner.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: fmt::Debug, A: Allocator + Clone> fmt::Debug for ArenaBox<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display, A: Allocator + Clone> fmt::Display for ArenaBox<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: PartialEq, A: Allocator + Clone> PartialEq for ArenaBox<'_, T, A> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl<T: Eq, A: Allocator + Clone> Eq for ArenaBox<'_, T, A> {}
impl<T: PartialOrd, A: Allocator + Clone> PartialOrd for ArenaBox<'_, T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}
impl<T: Ord, A: Allocator + Clone> Ord for ArenaBox<'_, T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: Hash, A: Allocator + Clone> Hash for ArenaBox<'_, T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}
