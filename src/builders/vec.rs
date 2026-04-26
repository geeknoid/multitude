// File-level lint relaxations: the unsafe blocks in this module are
// short, tightly-coupled pointer-arithmetic sequences whose safety
// invariants are documented at the block level rather than per
// primitive op (mirrors `builders/string.rs`'s per-function pattern).
#![allow(
    clippy::multiple_unsafe_ops_per_block,
    reason = "this module's unsafe blocks each document a single tightly-coupled invariant"
)]

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::ManuallyDrop;
use core::ops::{Bound, Deref, DerefMut, RangeBounds};
use core::ptr::{self, NonNull};

use allocator_api2::alloc::{AllocError, Allocator, Global, Layout};
use allocator_api2::vec::{IntoIter as ApiIntoIter, Vec as ApiVec};

use crate::Arena;
use crate::chunk_header::header_for;
use crate::chunk_ref::ChunkRef;
use crate::chunk_sharing::ChunkSharing;
use crate::rc::Rc;

/// A growable, mutable vector that lives in an [`Arena`].
///
/// `Vec` is a **transient builder**: 32 bytes on 64-bit (data pointer +
/// length + capacity + arena reference). Its purpose is to be filled and
/// then frozen via [`Self::into_arena_rc`] into a 16-byte
/// [`Rc<[T], A>`](crate::Rc) — immutable, cloneable, refcounted. For
/// `T: !Drop`, the freeze is **O(1)**.
///
/// `push`, `pop`, `extend`, `iter`, and other standard vector methods
/// behave the same as on `std::vec::Vec`.
///
/// # Example
///
/// ```
/// use multitude::Arena;
///
/// let arena = Arena::new();
/// let mut v = arena.alloc_vec::<i32>();
/// v.push(1);
/// v.push(2);
/// v.push(3);
/// let frozen = v.into_arena_rc();    // 32 bytes → 16-byte Rc<[i32]>
/// assert_eq!(&*frozen, &[1, 2, 3]);
/// ```
pub struct Vec<'a, T, A: Allocator + Clone = Global> {
    // Hand-rolled `(ptr, len, cap)` storage. The `&Arena` reference
    // lives in a separate field so it does not pollute register
    // allocation in the `push` hot loop. Methods that operate on the
    // arena (grow / shrink / freeze / iterators) read it from `arena`
    // out of the loop.
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    arena: &'a Arena<A>,
}

// SAFETY: `Vec<'a, T, A>` owns its `T` values plus an `&Arena<A>`. As
// with `std::vec::Vec`, sending it across threads is sound when the
// element type and the borrowed allocator can also be sent.
unsafe impl<T: Send, A: Allocator + Clone + Sync> Send for Vec<'_, T, A> {}
// SAFETY: shared access to the storage is sound when both the element
// type and the borrowed allocator allow shared access across threads.
unsafe impl<T: Sync, A: Allocator + Clone + Sync> Sync for Vec<'_, T, A> {}

impl<'a, T, A: Allocator + Clone> Vec<'a, T, A> {
    /// Create an empty vector backed by `arena`. No allocation until the first push.
    #[inline]
    #[must_use]
    pub const fn new_in(arena: &'a Arena<A>) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            arena,
        }
    }

    /// Create an empty vector with capacity for at least `cap` elements.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_with_capacity_in`] for a fallible variant.
    #[must_use]
    pub fn with_capacity_in(cap: usize, arena: &'a Arena<A>) -> Self {
        Self::try_with_capacity_in(cap, arena).unwrap_or_else(|_| crate::arena::panic_alloc())
    }

    /// Fallible variant of [`Self::with_capacity_in`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data
    /// alignment is at least 64 KiB.
    pub fn try_with_capacity_in(cap: usize, arena: &'a Arena<A>) -> Result<Self, AllocError> {
        let mut v = Self::new_in(arena);
        if cap > 0 {
            v.try_reserve_exact(cap)?;
        }
        Ok(v)
    }

    /// Push a value.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_push`] for a fallible variant.
    #[expect(
        clippy::inline_always,
        reason = "hot path; force-inlining keeps the cap-check + bump + write tight in callers' loops"
    )]
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            self.grow_one();
        }
        // SAFETY: `cap > len` after the (cold) `grow_one`; the slot
        // at `ptr + len` is uninitialized but in-bounds for write.
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }

    #[cold]
    #[inline(never)]
    fn grow_one(&mut self) {
        if self.try_grow_amortized(1).is_err() {
            crate::arena::panic_alloc();
        }
    }

    /// Fallible variant of [`Self::push`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data
    /// alignment is at least 64 KiB.
    #[inline]
    pub fn try_push(&mut self, value: T) -> Result<(), AllocError> {
        if self.len == self.cap {
            self.try_grow_amortized(1)?;
        }
        // SAFETY: `cap > len` after the (cold) grow.
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop a value.
    pub const fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: `len` was > 0; the element at `len - 1` is initialized.
        unsafe { Some(self.ptr.as_ptr().add(self.len).read()) }
    }

    /// Reserve capacity for at least `additional` more elements.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_reserve`] for a fallible variant.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        if self.cap.wrapping_sub(self.len) < additional && self.try_grow_amortized(additional).is_err() {
            crate::arena::panic_alloc();
        }
    }

    /// Fallible variant of [`Self::reserve`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data
    /// alignment is at least 64 KiB.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocError> {
        if self.cap.wrapping_sub(self.len) < additional {
            self.try_grow_amortized(additional)?;
        }
        Ok(())
    }

    /// Drop all elements but keep the capacity.
    pub fn clear(&mut self) {
        let len = self.len;
        self.len = 0;
        // SAFETY: `ptr..ptr+len` are initialized; we set `len = 0` first
        // so a panic in `drop_in_place` does not leave a double-drop hazard.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), len));
        }
    }

    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr..ptr+len` are initialized.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `ptr..ptr+len` are initialized; exclusive borrow.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn extend_from_slice(&mut self, other: impl AsRef<[T]>)
    where
        T: Copy,
    {
        let s = other.as_ref();
        self.reserve(s.len());
        // SAFETY: `cap >= len + s.len()` after `reserve`; source and
        // destination are disjoint regions.
        unsafe {
            ptr::copy_nonoverlapping(s.as_ptr(), self.ptr.as_ptr().add(self.len), s.len());
            self.len += s.len();
        }
    }

    /// Build a `Vec` by collecting `iter` into a fresh vector backed by `arena`.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails on growth.
    pub fn from_iter_in<I: IntoIterator<Item = T>>(iter: I, arena: &'a Arena<A>) -> Self {
        let it = iter.into_iter();
        let (lower, _) = it.size_hint();
        let mut v = if lower == 0 {
            Self::new_in(arena)
        } else {
            Self::with_capacity_in(lower, arena)
        };
        for item in it {
            v.push(item);
        }
        v
    }

    /// Returns a raw pointer to the vector's buffer.
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns an unsafe mutable pointer to the vector's buffer.
    #[must_use]
    #[inline]
    #[expect(clippy::needless_pass_by_ref_mut, reason = "API shape mirrors std::Vec::as_mut_ptr")]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Insert `value` at position `idx`, shifting subsequent elements right.
    ///
    /// # Panics
    ///
    /// Panics if `idx > len`, or if the backing allocator fails on growth.
    pub fn insert(&mut self, idx: usize, value: T) {
        assert!(idx <= self.len, "insertion index (is {idx}) should be <= len (is {})", self.len);
        if self.len == self.cap {
            self.grow_one();
        }
        // SAFETY: `idx <= len < cap`; both `idx` and `idx + 1` plus the
        // shifted suffix lie within the allocated buffer.
        unsafe {
            let p = self.ptr.as_ptr().add(idx);
            ptr::copy(p, p.add(1), self.len - idx);
            ptr::write(p, value);
        }
        self.len += 1;
    }

    /// Remove and return the element at position `idx`, shifting subsequent
    /// elements to the left.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= len`.
    pub fn remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "removal index (is {idx}) should be < len (is {})", self.len);
        // SAFETY: `idx < len`; reading `[idx]` and shifting `[idx+1..len]`
        // are both in-bounds.
        unsafe {
            let p = self.ptr.as_ptr().add(idx);
            let value = ptr::read(p);
            ptr::copy(p.add(1), p, self.len - idx - 1);
            self.len -= 1;
            value
        }
    }

    /// Swap-remove: O(1) but does not preserve order.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= len`.
    pub fn swap_remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "swap_remove index (is {idx}) should be < len (is {})", self.len);
        let last = self.len - 1;
        self.len = last;
        // SAFETY: `idx <= last < cap`; both indices are in-bounds for the
        // allocated buffer, and we just shrank `len`, so the temporary
        // bit-copy (when `idx != last`) does not alias with any logically
        // live element.
        unsafe {
            let p = self.ptr.as_ptr();
            let value = ptr::read(p.add(idx));
            if idx != last {
                ptr::copy_nonoverlapping(p.add(last), p.add(idx), 1);
            }
            value
        }
    }

    /// Shorten the vector to `new_len`, dropping the excess elements.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        let drop_count = self.len - new_len;
        self.len = new_len;
        // SAFETY: elements `[new_len, old_len)` are initialized; we set
        // `len = new_len` first so a panic in `drop_in_place` cannot
        // double-drop later.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr().add(new_len), drop_count));
        }
    }

    /// Force the length of the vector to `new_len`.
    ///
    /// # Safety
    ///
    /// `new_len` must be `<= self.capacity()` and the elements at
    /// `old_len..new_len` must be initialized.
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    /// Shrink the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self) {
        if self.cap == self.len || size_of::<T>() == 0 {
            return;
        }
        let new_cap = self.len;
        let _ = self.realloc(new_cap);
    }

    /// Retain only elements for which the predicate returns `true`.
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut f: F) {
        self.retain_mut(move |v| f(v));
    }

    /// Retain (mutable predicate variant).
    pub fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, mut f: F) {
        self.with_apivec(|v| v.retain_mut(&mut f));
    }

    /// Remove consecutive duplicates by `PartialEq`.
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.with_apivec(allocator_api2::vec::Vec::dedup);
    }

    /// Remove consecutive duplicates by `same_bucket`.
    pub fn dedup_by<F: FnMut(&mut T, &mut T) -> bool>(&mut self, mut same_bucket: F) {
        self.with_apivec(|v| v.dedup_by(&mut same_bucket));
    }

    /// Remove consecutive duplicates by key.
    pub fn dedup_by_key<K, F>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.with_apivec(|v| v.dedup_by_key(&mut key));
    }

    /// Move all elements of `other` into `self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails on growth.
    pub fn append(&mut self, other: &mut Self) {
        let other_len = other.len;
        self.reserve(other_len);
        // SAFETY: `cap >= self.len + other_len` after `reserve`; the
        // source and destination ranges live in distinct allocations
        // (each `Vec` owns its own buffer), so they cannot overlap.
        unsafe {
            ptr::copy_nonoverlapping(other.ptr.as_ptr(), self.ptr.as_ptr().add(self.len), other_len);
        }
        self.len += other_len;
        other.len = 0;
    }

    /// Reserve the minimum capacity for at least `additional` more elements.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails or if the data alignment is at least 64 KiB.
    /// Use [`Self::try_reserve_exact`] for a fallible variant.
    pub fn reserve_exact(&mut self, additional: usize) {
        if self.try_reserve_exact(additional).is_err() {
            crate::arena::panic_alloc();
        }
    }

    /// Fallible variant of [`Self::reserve_exact`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the backing allocator fails or if the data
    /// alignment is at least 64 KiB.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocError> {
        let needed = self.len.checked_add(additional).ok_or(AllocError)?;
        if needed > self.cap {
            self.realloc(needed)?;
        }
        Ok(())
    }

    /// Resize the vector to `new_len`, cloning `value` to fill new slots.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails on growth.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.len {
            let extra = new_len - self.len;
            self.reserve(extra);
            for _ in 0..extra - 1 {
                self.push(value.clone());
            }
            self.push(value);
        } else {
            self.truncate(new_len);
        }
    }

    /// Resize the vector to `new_len`, calling `f` for new elements.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails on growth.
    pub fn resize_with<F: FnMut() -> T>(&mut self, new_len: usize, mut f: F) {
        if new_len > self.len {
            let extra = new_len - self.len;
            self.reserve(extra);
            for _ in 0..extra {
                self.push(f());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Split the vector at `at`, returning a new vector containing `[at, len)`.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    #[must_use]
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len, "split_off at (is {at}) should be <= len (is {})", self.len);
        let tail_len = self.len - at;
        let mut other = Self::with_capacity_in(tail_len, self.arena);
        // SAFETY: `at..len` is initialized; the source and destination
        // live in distinct buffers and so cannot overlap.
        unsafe {
            ptr::copy_nonoverlapping(self.ptr.as_ptr().add(at), other.ptr.as_ptr(), tail_len);
            other.len = tail_len;
            self.len = at;
        }
        other
    }

    /// Pop the last element if the predicate returns `true`.
    pub fn pop_if<F: FnOnce(&mut T) -> bool>(&mut self, predicate: F) -> Option<T> {
        let last = self.as_mut_slice().last_mut()?;
        if predicate(last) { self.pop() } else { None }
    }

    /// Drain a range of elements.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end is greater than `len`.
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> Drain<'_, 'a, T, A> {
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).expect("drain start overflow"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("drain end overflow"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.len,
        };
        assert!(start <= end && end <= self.len, "drain range out of bounds");
        let tail_start = end;
        let tail_len = self.len - end;
        // Force the source vector's len to `start` so a leak of the
        // Drain (mem::forget) leaves the head intact.
        self.len = start;
        Drain {
            vec: NonNull::from(self),
            start,
            tail_start,
            tail_len,
            cur: start,
            end,
            _marker: core::marker::PhantomData,
        }
    }

    /// Freeze into an [`Rc<[T], A>`](crate::Rc).
    ///
    /// # Panics
    ///
    /// Panics if the underlying allocator fails on the copy fallback.
    #[must_use]
    pub fn into_arena_rc(self) -> Rc<[T], A> {
        // ZSTs use NonNull::dangling() for their backing buffer.
        if size_of::<T>() != 0 && !core::mem::needs_drop::<T>() && self.len > 0 {
            // Try to freeze the buffer in place. When `cap > len`, the
            // tail-reclaim path returns the unused capacity to the chunk's
            // bump cursor; when `cap == len`, the buffer is already a tight
            // fit and the reclaim is a no-op write — but in BOTH cases the
            // buffer ends at the chunk's bump cursor only if no other
            // allocation has happened against this chunk since our last
            // grow, which is the common case for a `with_capacity` + push
            // loop. Either way we transfer the +1 chunk refcount that the
            // buffer held to the resulting `Rc<[T]>` and skip the slow
            // copy fallback.
            let buffer_nn = self.ptr.cast::<u8>();
            // SAFETY: ptr came from this arena's bump alloc (Local-flavored).
            let chunk = unsafe { header_for::<u8, A>(buffer_nn) };
            // SAFETY: chunk is alive (we hold a refcount via the buffer).
            let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
            if chunk_ref.try_reclaim_tail(buffer_nn, self.cap * size_of::<T>(), self.len * size_of::<T>()) {
                let me = ManuallyDrop::new(self);
                // SAFETY: `ptr..ptr+len` are initialized; the chunk's
                // refcount taken at allocate time transfers to the Rc.
                return unsafe { Rc::from_raw_slice(me.ptr, me.len) };
            }
        }
        // ZST, has Drop, empty, or the in-place reclaim failed (some
        // other alloc landed past our buffer). Fall back to copying.
        let arena = self.arena;
        let len = self.len;
        if len == 0 {
            return arena.alloc_slice_fill_with_rc::<T, _>(0, |_| unreachable!());
        }
        let needs_drop = core::mem::needs_drop::<T>();
        let reservation = arena
            .reserve_slice::<T>(len, ChunkSharing::Local, needs_drop, false)
            .unwrap_or_else(|_| crate::arena::panic_alloc());
        let dst = reservation.ptr.cast::<T>().as_ptr();
        // SAFETY: `self.ptr..self.ptr+len` is initialized; `dst..dst+len`
        // is freshly reserved; the two ranges are in distinct chunks
        // (the reservation is a brand-new bump allocation), so they
        // cannot overlap.
        unsafe {
            ptr::copy_nonoverlapping(self.ptr.as_ptr(), dst, len);
            let mut me = ManuallyDrop::new(self);
            // We've already moved every element bit-wise; clear the
            // length so `me`'s `Drop` does not double-drop.
            me.len = 0;
            // Run `me`'s `Drop` to deallocate the original buffer
            // (releasing the +1 chunk refcount it held).
            ManuallyDrop::drop(&mut me);
            arena.commit_slice::<T>(reservation, len);
            Rc::from_raw_slice(NonNull::new_unchecked(dst), len)
        }
    }

    // ---- Internal helpers ----

    #[cold]
    fn try_grow_amortized(&mut self, additional: usize) -> Result<(), AllocError> {
        let needed = self.len.checked_add(additional).ok_or(AllocError)?;
        if needed <= self.cap {
            return Ok(());
        }
        // Standard `Vec` growth: max(2*cap, needed, 4).
        let new_cap = needed.max(self.cap.saturating_mul(2)).max(4);
        self.realloc(new_cap)
    }

    fn realloc(&mut self, new_cap: usize) -> Result<(), AllocError> {
        if size_of::<T>() == 0 {
            self.cap = new_cap;
            return Ok(());
        }
        let new_layout = Layout::array::<T>(new_cap).map_err(|_e| AllocError)?;
        let new_nn = if self.cap == 0 {
            self.arena.allocate(new_layout)?
        } else {
            let old_layout = Layout::array::<T>(self.cap).map_err(|_e| AllocError)?;
            // SAFETY: `self.ptr` was returned by this arena's allocate
            // and currently holds `self.cap` `T`s.
            unsafe {
                if new_cap >= self.cap {
                    self.arena.grow(self.ptr.cast::<u8>(), old_layout, new_layout)?
                } else {
                    self.arena.shrink(self.ptr.cast::<u8>(), old_layout, new_layout)?
                }
            }
        };
        self.ptr = new_nn.cast::<T>();
        self.cap = new_cap;
        Ok(())
    }

    /// Delegate a complex operation to a temporary `ApiVec` built from
    /// our raw parts. Used for slow-path methods (`retain`, `dedup`, …)
    /// to avoid reimplementing the standard semantics.
    #[inline]
    fn with_apivec<R, F: FnOnce(&mut ApiVec<T, &'a Arena<A>>) -> R>(&mut self, f: F) -> R {
        let arena = self.arena;
        // Hand the buffer ownership to a fresh ApiVec, clearing our
        // own state so a panic in `f` doesn't lead to a double-free.
        // SAFETY: `(ptr, len, cap)` describe an `ApiVec`-compatible
        // allocation backed by the same `arena` allocator.
        let mut v: ApiVec<T, &'a Arena<A>> = unsafe { ApiVec::from_raw_parts_in(self.ptr.as_ptr(), self.len, self.cap, arena) };
        self.cap = 0;
        self.len = 0;
        let r = f(&mut v);
        let (ptr, len, cap, _alloc) = v.into_raw_parts_with_alloc();
        debug_assert!(cap >= len);
        // Restore our raw fields. `ApiVec::into_raw_parts_with_alloc`
        // returns `dangling()` for `cap == 0`; preserve that.
        self.ptr = NonNull::new(ptr).unwrap_or_else(NonNull::dangling);
        self.len = len;
        self.cap = cap;
        r
    }
}

impl<T, A: Allocator + Clone> Drop for Vec<'_, T, A> {
    fn drop(&mut self) {
        // SAFETY: `ptr..ptr+len` are initialized.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
        }
        if self.cap > 0 && size_of::<T>() != 0 {
            let layout = Layout::array::<T>(self.cap).expect("layout was validated at allocation");
            // SAFETY: `self.ptr` was returned by this arena's allocate
            // for an allocation of `self.cap` `T`s; we still hold the
            // associated +1 chunk refcount.
            unsafe {
                self.arena.deallocate(self.ptr.cast::<u8>(), layout);
            }
        }
    }
}

impl<T, A: Allocator + Clone> Deref for Vec<'_, T, A> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator + Clone> DerefMut for Vec<'_, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator + Clone> AsRef<[T]> for Vec<'_, T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator + Clone> AsMut<[T]> for Vec<'_, T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator + Clone> core::borrow::Borrow<[T]> for Vec<'_, T, A> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator + Clone> core::borrow::BorrowMut<[T]> for Vec<'_, T, A> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator + Clone> Extend<T> for Vec<'_, T, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let it = iter.into_iter();
        let (lower, _) = it.size_hint();
        if lower > 0 {
            self.reserve(lower);
        }
        for item in it {
            self.push(item);
        }
    }
}

impl<'a, 'b, T: Copy + 'b, A: Allocator + Clone> Extend<&'b T> for Vec<'a, T, A>
where
    'a: 'b,
{
    fn extend<I: IntoIterator<Item = &'b T>>(&mut self, iter: I) {
        let it = iter.into_iter();
        let (lower, _) = it.size_hint();
        if lower > 0 {
            self.reserve(lower);
        }
        for &item in it {
            self.push(item);
        }
    }
}

impl<T: fmt::Debug, A: Allocator + Clone> fmt::Debug for Vec<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: PartialEq, A: Allocator + Clone> PartialEq for Vec<'_, T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Eq, A: Allocator + Clone> Eq for Vec<'_, T, A> {}
impl<T: PartialOrd, A: Allocator + Clone> PartialOrd for Vec<'_, T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}
impl<T: Ord, A: Allocator + Clone> Ord for Vec<'_, T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
impl<T: Hash, A: Allocator + Clone> Hash for Vec<'_, T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for Vec<'_, T, A> {
    fn clone(&self) -> Self {
        let mut v = Self::with_capacity_in(self.len, self.arena);
        for item in self.as_slice() {
            v.push(item.clone());
        }
        v
    }
}

impl<'a, T, A: Allocator + Clone> IntoIterator for Vec<'a, T, A> {
    type Item = T;
    type IntoIter = IntoIter<'a, T, A>;
    fn into_iter(self) -> Self::IntoIter {
        // Move into an `ApiVec` to reuse its `into_iter` machinery; the
        // resulting iterator owns the buffer and frees it on drop via
        // the same arena allocator.
        let me = ManuallyDrop::new(self);
        // SAFETY: `(ptr, len, cap)` describe an `ApiVec`-compatible
        // allocation backed by the same `arena` allocator.
        unsafe { ApiVec::from_raw_parts_in(me.ptr.as_ptr(), me.len, me.cap, me.arena).into_iter() }
    }
}

impl<'b, T, A: Allocator + Clone> IntoIterator for &'b Vec<'_, T, A> {
    type Item = &'b T;
    type IntoIter = core::slice::Iter<'b, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'b, T, A: Allocator + Clone> IntoIterator for &'b mut Vec<'_, T, A> {
    type Item = &'b mut T;
    type IntoIter = core::slice::IterMut<'b, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

/// Owning iterator returned by [`Vec::into_iter`].
pub type IntoIter<'a, T, A> = ApiIntoIter<T, &'a Arena<A>>;

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T: serde::ser::Serialize, A: Allocator + Clone> serde::ser::Serialize for Vec<'_, T, A> {
    fn serialize<S: serde::ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq as _;
        let slice = self.as_slice();
        let mut seq = serializer.serialize_seq(Some(slice.len()))?;
        for elem in slice {
            seq.serialize_element(elem)?;
        }
        seq.end()
    }
}

impl<'a, T, A: Allocator + Clone> crate::builders::FromIteratorIn<T> for Vec<'a, T, A> {
    type Allocator = &'a Arena<A>;

    fn from_iter_in<I: IntoIterator<Item = T>>(iter: I, allocator: &'a Arena<A>) -> Self {
        Self::from_iter_in(iter, allocator)
    }
}

// ---- Drain iterator ----

/// Draining iterator returned from [`Vec::drain`].
pub struct Drain<'d, 'a, T, A: Allocator + Clone> {
    vec: NonNull<Vec<'a, T, A>>,
    /// Range origin (the original drain `start`).
    start: usize,
    /// Index where the tail begins (original `end`).
    tail_start: usize,
    /// Number of elements still in the tail.
    tail_len: usize,
    /// Front cursor (advances on `next`).
    cur: usize,
    /// Back cursor (retreats on `next_back`); also marks the end of
    /// the undrained range.
    end: usize,
    _marker: core::marker::PhantomData<&'d mut Vec<'a, T, A>>,
}

impl<T: fmt::Debug, A: Allocator + Clone> fmt::Debug for Drain<'_, '_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Drain").field("remaining", &(self.end - self.cur)).finish()
    }
}

// SAFETY: `Drain` borrows the source vector for `'d`; like `std::vec::Drain`,
// shipping it across threads is sound when the element type and the borrowed
// allocator can also be shared/sent.
unsafe impl<T: Send, A: Allocator + Clone + Sync> Send for Drain<'_, '_, T, A> {}
// SAFETY: see `Send` impl above.
unsafe impl<T: Sync, A: Allocator + Clone + Sync> Sync for Drain<'_, '_, T, A> {}

impl<T, A: Allocator + Clone> Iterator for Drain<'_, '_, T, A> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.cur >= self.end {
            return None;
        }
        // SAFETY: `cur..end` lies within `[start, tail_start)`, the
        // initialized drain range; the source `Vec`'s `len` was
        // shrunk to `start` so this read does not race with any
        // outstanding owner.
        let item = unsafe {
            let p = self.vec.as_ref().ptr.as_ptr().add(self.cur);
            ptr::read(p)
        };
        self.cur += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.cur;
        (n, Some(n))
    }
}

impl<T, A: Allocator + Clone> DoubleEndedIterator for Drain<'_, '_, T, A> {
    fn next_back(&mut self) -> Option<T> {
        if self.cur >= self.end {
            return None;
        }
        self.end -= 1;
        // SAFETY: see `next`.
        unsafe {
            let p = self.vec.as_ref().ptr.as_ptr().add(self.end);
            Some(ptr::read(p))
        }
    }
}

impl<T, A: Allocator + Clone> ExactSizeIterator for Drain<'_, '_, T, A> {}
impl<T, A: Allocator + Clone> core::iter::FusedIterator for Drain<'_, '_, T, A> {}

impl<T, A: Allocator + Clone> Drop for Drain<'_, '_, T, A> {
    fn drop(&mut self) {
        // Drop any remaining un-yielded elements.
        while self.next().is_some() {}
        // Move the tail into place.
        // SAFETY: the source `Vec` is alive for `'d`; we are the only
        // mutator while the `Drain` exists.
        unsafe {
            let v = self.vec.as_mut();
            if self.tail_len > 0 {
                let src = v.ptr.as_ptr().add(self.tail_start);
                let dst = v.ptr.as_ptr().add(self.start);
                ptr::copy(src, dst, self.tail_len);
            }
            v.len = self.start + self.tail_len;
        }
    }
}
