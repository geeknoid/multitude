//! [`ArenaVec`] — a growable vector that allocates its buffer inside an arena.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};
use allocator_api2::vec::Vec as ApiVec;

use crate::arena_rc::ArenaRc;

/// A growable, bump-allocated vector that lives inside an
/// [`Arena`](crate::Arena).
///
/// Wraps an `allocator_api2::vec::Vec<T, &'a Arena<A>>`, so `push`,
/// `pop`, `extend`, and friends are inherited.
///
/// `ArenaVec` is a **transient builder**: 32 bytes on 64-bit (data
/// pointer + length + capacity + arena reference) because every `push`
/// may need to grow. Once you're done building, call
/// [`Self::into_arena_rc`] to freeze into a 16-byte
/// [`ArenaRc<[T], A>`](crate::ArenaRc) — immutable, cloneable,
/// refcount-based. See the crate-level docs ("build-then-freeze") for
/// why and when this matters.
///
/// # Example
///
/// ```
/// use harena::Arena;
///
/// let arena = Arena::new();
/// let mut v = arena.new_vec::<i32>();
/// v.push(1);
/// v.push(2);
/// v.push(3);
/// let frozen = v.into_arena_rc();    // 32 bytes → 16-byte ArenaRc<[i32]>
/// assert_eq!(&*frozen, &[1, 2, 3]);
/// ```
pub struct ArenaVec<'a, T, A: Allocator + Clone = Global> {
    inner: ApiVec<T, &'a crate::Arena<A>>,
}

impl<'a, T, A: Allocator + Clone> ArenaVec<'a, T, A> {
    /// Create an empty vector backed by `arena`. No allocation until the
    /// first push.
    #[must_use]
    pub(crate) fn new_in(arena: &'a crate::Arena<A>) -> Self {
        Self {
            inner: ApiVec::new_in(arena),
        }
    }

    /// Create an empty vector with capacity for at least `cap` elements.
    #[must_use]
    pub(crate) fn with_capacity_in(cap: usize, arena: &'a crate::Arena<A>) -> Self {
        Self {
            inner: ApiVec::with_capacity_in(cap, arena),
        }
    }

    /// Push a value.
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    /// Pop a value.
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Reserve capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Drop all elements but keep the capacity.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Number of live elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True iff empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Allocated capacity in elements.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Borrow as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Borrow as a mutable slice.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    /// Extend from a slice (`T: Copy`).
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Copy,
    {
        self.inner.extend_from_slice(other);
    }

    /// Freeze into an [`ArenaRc<[T], A>`]. The buffer becomes immutable;
    /// the arena's drop list is updated to drop the live elements at
    /// chunk teardown (if `T: Drop`).
    ///
    /// Fast path: when `T` doesn't need `Drop` and the buffer sits at
    /// the chunk's bump cursor, the buffer is reused in place (no copy)
    /// and the cursor is reset to just past the live elements (slack
    /// reclaimed). Otherwise falls back to copy-into-fresh-allocation,
    /// which also registers the per-element drop.
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails on the copy fallback.
    #[must_use]
    pub fn into_arena_rc(self) -> ArenaRc<[T], A> {
        // Fast path: T doesn't need Drop and the buffer is at the chunk's
        // bump cursor → reuse the buffer in place.
        if !core::mem::needs_drop::<T>() && !self.inner.is_empty() {
            let buffer_ptr = self.inner.as_ptr();
            let len = self.inner.len();
            let cap = self.inner.capacity();
            // SAFETY: the Vec's allocator is `&Arena<A>`, and our
            // `Allocator::allocate` returns pointers into 64 KiB-aligned
            // chunks, so the masking trick recovers a real header.
            let chunk = unsafe { crate::chunk_header::header_for::<T, A>(NonNull::new_unchecked(buffer_ptr.cast_mut())) };
            // SAFETY: chunk is alive (Vec holds a refcount on it).
            let header = unsafe { chunk.as_ref() };
            let chunk_base = chunk.as_ptr() as usize;
            let buffer_start = buffer_ptr as usize;
            let buffer_end = buffer_start + cap * size_of::<T>();
            let buffer_end_offset = buffer_end - chunk_base;
            if buffer_end_offset == header.bump.get() {
                // Reset bump to just past the live elements — reclaims
                // unused capacity at the buffer's tail.
                let live_end_offset = (buffer_start - chunk_base) + len * size_of::<T>();
                header.bump.set(live_end_offset);
                // Suppress Vec::Drop (which would deallocate and dec
                // the chunk refcount). The refcount transfers to the
                // resulting ArenaRc.
                let me = core::mem::ManuallyDrop::new(self);
                let ptr = me.inner.as_ptr().cast_mut();
                // SAFETY: ptr..ptr+len is initialized; refcount has been
                // transferred from the suppressed Vec to this handle.
                return unsafe { ArenaRc::from_raw_slice(NonNull::new_unchecked(ptr), len) };
            }
        }

        // Slow path: copy into a fresh allocation. This is required when
        // T needs Drop (so we can register a drop entry covering the
        // slice), and used as a fallback when the buffer isn't at the
        // bump cursor.
        let arena = *self.inner.allocator();
        let len = self.inner.len();
        let mut iter = self.inner.into_iter();
        arena.alloc_slice_fill_with(len, |_| iter.next().expect("len matches"))
    }
}

impl<T, A: Allocator + Clone> Deref for ArenaVec<'_, T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.inner.as_slice()
    }
}

impl<T, A: Allocator + Clone> DerefMut for ArenaVec<'_, T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }
}

impl<T, A: Allocator + Clone> AsRef<[T]> for ArenaVec<'_, T, A> {
    fn as_ref(&self) -> &[T] {
        self.inner.as_slice()
    }
}

impl<T, A: Allocator + Clone> AsMut<[T]> for ArenaVec<'_, T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }
}

impl<T, A: Allocator + Clone> core::borrow::Borrow<[T]> for ArenaVec<'_, T, A> {
    fn borrow(&self) -> &[T] {
        self.inner.as_slice()
    }
}

impl<T, A: Allocator + Clone> Extend<T> for ArenaVec<'_, T, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter);
    }
}

impl<T: fmt::Debug, A: Allocator + Clone> fmt::Debug for ArenaVec<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.as_slice()).finish()
    }
}

impl<T: PartialEq, A: Allocator + Clone> PartialEq for ArenaVec<'_, T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Eq, A: Allocator + Clone> Eq for ArenaVec<'_, T, A> {}
impl<T: PartialOrd, A: Allocator + Clone> PartialOrd for ArenaVec<'_, T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}
impl<T: Ord, A: Allocator + Clone> Ord for ArenaVec<'_, T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
impl<T: Hash, A: Allocator + Clone> Hash for ArenaVec<'_, T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}
