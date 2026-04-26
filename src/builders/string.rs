use core::alloc::Layout;
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::{Bound, Deref, DerefMut, RangeBounds};
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::Arena;
use crate::arena::panic_alloc;
use crate::arena_str_helpers::PREFIX_SIZE;
use crate::chunk_header::{header_for, release_chunk_ref_local};
use crate::chunk_ref::ChunkRef;
use crate::chunk_sharing::ChunkSharing;
use crate::rc_str::RcStr;

/// Initial capacity granted on the first growth (when none was specified).
const MIN_INITIAL_CAP: usize = 16;

/// A growable, mutable UTF-8 string that lives in an [`Arena`].
///
/// `String` is a **transient builder**: 32 bytes on 64-bit (data pointer +
/// length + capacity + arena reference). Its purpose is to be filled and
/// then frozen via [`Self::into_arena_str`] into a compact, immutable
/// [`RcStr`] (8 bytes). The freeze is **O(1)** — no copy, no new
/// allocation.
///
/// # Example
///
/// ```
/// use multitude::Arena;
///
/// let arena = Arena::new();
/// let mut s = arena.alloc_string();
/// s.push_str("hello, ");
/// s.push_str("world!");
/// assert_eq!(s.as_str(), "hello, world!");
/// let frozen = s.into_arena_str();   // O(1), no copy
/// assert_eq!(&*frozen, "hello, world!");
/// ```
pub struct String<'a, A: Allocator + Clone = Global> {
    data: NonNull<u8>,
    len: usize,
    cap: usize,
    arena: &'a Arena<A>,
    _not_sync: PhantomData<*mut ()>,
}

impl<'a, A: Allocator + Clone> String<'a, A> {
    /// Create a new, empty arena-backed string.
    ///
    /// No allocation is performed until the first push.
    #[must_use]
    pub const fn new_in(arena: &'a Arena<A>) -> Self {
        Self {
            data: NonNull::dangling(),
            len: 0,
            cap: 0,
            arena,
            _not_sync: PhantomData,
        }
    }

    /// Create a new arena-backed string with at least `cap` bytes of
    /// pre-allocated capacity.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_with_capacity_in`] for a
    /// fallible variant.
    #[must_use]
    pub fn with_capacity_in(cap: usize, arena: &'a Arena<A>) -> Self {
        let mut s = Self::new_in(arena);
        if cap > 0 {
            s.allocate_initial(cap);
        }
        s
    }

    /// Fallible variant of [`Self::with_capacity_in`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing
    /// allocator fails.
    pub fn try_with_capacity_in(cap: usize, arena: &'a Arena<A>) -> Result<Self, allocator_api2::alloc::AllocError> {
        let mut s = Self::new_in(arena);
        if cap > 0 {
            s.try_allocate_initial(cap)?;
        }
        Ok(s)
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
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    #[must_use]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub const fn as_str(&self) -> &str {
        if self.cap == 0 {
            return "";
        }
        // SAFETY: valid UTF-8 of length `self.len` at `self.data`.
        unsafe {
            let bytes = core::slice::from_raw_parts(self.data.as_ptr(), self.len);
            core::str::from_utf8_unchecked(bytes)
        }
    }

    /// Return the bytes view of this string.
    #[must_use]
    #[inline]
    pub const fn as_bytes(&self) -> &[u8] {
        if self.cap == 0 {
            return &[];
        }
        // SAFETY: valid initialized bytes of length `self.len` at `self.data`.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Return a mutable `str` view of this string.
    ///
    /// Callers must preserve UTF-8 well-formedness; mutating the bytes
    /// in a way that produces invalid UTF-8 is undefined behavior, but
    /// only via the unsafe `str` APIs that allow byte-level edits.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub const fn as_mut_str(&mut self) -> &mut str {
        if self.cap == 0 {
            // SAFETY: empty slice is valid UTF-8.
            return unsafe { core::str::from_utf8_unchecked_mut(&mut []) };
        }
        // SAFETY: valid UTF-8 of length `self.len` at `self.data`; `&mut self` excludes aliasing.
        unsafe {
            let bytes = core::slice::from_raw_parts_mut(self.data.as_ptr(), self.len);
            core::str::from_utf8_unchecked_mut(bytes)
        }
    }

    /// Return a raw const pointer to the string's bytes.
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr().cast_const()
    }

    /// Return a raw mutable pointer to the string's bytes.
    #[expect(clippy::needless_pass_by_ref_mut, reason = "API shape mirrors std::String::as_mut_ptr")]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_ptr()
    }

    /// Construct an `String` containing `s`, copied into `arena`.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails.
    #[must_use]
    pub fn from_str_in(s: &str, arena: &'a Arena<A>) -> Self {
        let mut new = Self::with_capacity_in(s.len(), arena);
        new.push_str(s);
        new
    }

    /// Remove the last character from the string and return it.
    ///
    /// Returns `None` if the string is empty.
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.as_str().chars().next_back()?;
        self.len -= ch.len_utf8();
        Some(ch)
    }

    /// Shorten the string to `new_len` bytes.
    ///
    /// If `new_len >= self.len()`, this has no effect. Capacity is
    /// unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `new_len` is not on a UTF-8 character boundary.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        assert!(
            self.as_str().is_char_boundary(new_len),
            "String::truncate: index {new_len} is not on a UTF-8 char boundary"
        );
        self.len = new_len;
    }

    /// Try to release any unused capacity back to the chunk's bump
    /// cursor.
    ///
    /// Only succeeds if this string's buffer ends exactly at the
    /// chunk's bump cursor (no later allocations have been made into
    /// the same chunk). On failure this is a no-op.
    pub fn shrink_to_fit(&mut self) {
        if self.cap == 0 || self.cap == self.len {
            return;
        }
        // SAFETY: data is in a chunk we hold a refcount on.
        let chunk = unsafe { header_for::<u8, A>(self.data) };
        // SAFETY: chunk is alive.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        if chunk_ref.try_reclaim_tail(self.data, self.cap, self.len) {
            self.cap = self.len;
        }
    }

    /// Insert a character at byte index `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is greater than `self.len()` or not on a UTF-8
    /// character boundary, or if the backing allocator fails on growth.
    pub fn insert(&mut self, idx: usize, ch: char) {
        let mut buf = [0_u8; 4];
        let s = ch.encode_utf8(&mut buf);
        self.insert_str(idx, s);
    }

    /// Insert a string slice at byte index `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is greater than `self.len()` or not on a UTF-8
    /// character boundary, if the resulting length would overflow
    /// `usize`, or if the backing allocator fails on growth.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn insert_str(&mut self, idx: usize, s: &str) {
        assert!(
            self.as_str().is_char_boundary(idx),
            "String::insert_str: index {idx} is out of bounds or not on a UTF-8 char boundary"
        );
        let amt = s.len();
        if amt == 0 {
            return;
        }
        let new_len = self.len.checked_add(amt).expect("String overflow");
        if new_len > self.cap {
            self.grow_to_at_least(new_len);
        }
        // SAFETY: capacity is sufficient, idx <= self.len, regions are inside the buffer.
        unsafe {
            let p = self.data.as_ptr();
            core::ptr::copy(p.add(idx), p.add(idx + amt), self.len - idx);
            core::ptr::copy_nonoverlapping(s.as_ptr(), p.add(idx), amt);
        }
        self.len = new_len;
    }

    /// Remove the character at byte index `idx` and return it.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len()` or `idx` is not on a UTF-8
    /// character boundary.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn remove(&mut self, idx: usize) -> char {
        let s = self.as_str();
        assert!(
            idx < s.len() && s.is_char_boundary(idx),
            "String::remove: index {idx} is out of bounds or not on a UTF-8 char boundary"
        );
        // SAFETY: idx is on a char boundary and < len, so a char starts here.
        let ch = unsafe { s.get_unchecked(idx..).chars().next().unwrap_unchecked() };
        let ch_len = ch.len_utf8();
        let next = idx + ch_len;
        // SAFETY: next <= len, copy stays inside the buffer.
        unsafe {
            let p = self.data.as_ptr();
            core::ptr::copy(p.add(next), p.add(idx), self.len - next);
        }
        self.len -= ch_len;
        ch
    }

    /// Retain only the characters for which `f` returns `true`, in
    /// order.
    ///
    /// If `f` panics, any not-yet-processed characters are dropped from
    /// the string (matching `std::string::String::retain`).
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn retain<F: FnMut(char) -> bool>(&mut self, mut f: F) {
        struct Guard<'s, 'a, A: Allocator + Clone> {
            s: &'s mut String<'a, A>,
            idx: usize,
            del_bytes: usize,
        }

        impl<A: Allocator + Clone> Drop for Guard<'_, '_, A> {
            fn drop(&mut self) {
                self.s.len = self.idx - self.del_bytes;
            }
        }

        let len = self.len;
        let mut g = Guard {
            s: self,
            idx: 0,
            del_bytes: 0,
        };

        while g.idx < len {
            // SAFETY: g.idx is on a char boundary and < len within the buffer.
            let ch = unsafe {
                let p = g.s.data.as_ptr().add(g.idx);
                let bytes = core::slice::from_raw_parts(p, len - g.idx);
                core::str::from_utf8_unchecked(bytes).chars().next().unwrap_unchecked()
            };
            let ch_len = ch.len_utf8();

            if !f(ch) {
                g.del_bytes += ch_len;
            } else if g.del_bytes > 0 {
                // SAFETY: source and destination both lie within the buffer; lengths are valid.
                unsafe {
                    core::ptr::copy(g.s.data.as_ptr().add(g.idx), g.s.data.as_ptr().add(g.idx - g.del_bytes), ch_len);
                }
            }
            g.idx += ch_len;
        }
    }

    /// Replace the bytes in `range` with the contents of `replace_with`.
    ///
    /// # Panics
    ///
    /// Panics if either bound is out of range, the bounds are not on
    /// UTF-8 character boundaries, the resulting length would overflow
    /// `usize`, or the backing allocator fails on growth.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn replace_range<R: RangeBounds<usize>>(&mut self, range: R, replace_with: &str) {
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).expect("String overflow"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("String overflow"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.len,
        };
        let cur = self.as_str();
        assert!(start <= end, "String::replace_range: start ({start}) > end ({end})");
        assert!(end <= self.len, "String::replace_range: end ({end}) > len ({len})", len = self.len);
        assert!(
            cur.is_char_boundary(start),
            "String::replace_range: start ({start}) is not on a UTF-8 char boundary"
        );
        assert!(
            cur.is_char_boundary(end),
            "String::replace_range: end ({end}) is not on a UTF-8 char boundary"
        );

        let removed = end - start;
        let added = replace_with.len();

        if added > removed {
            let extra = added - removed;
            let new_len = self.len.checked_add(extra).expect("String overflow");
            if new_len > self.cap {
                self.grow_to_at_least(new_len);
            }
            // SAFETY: capacity is sufficient; both regions lie inside the buffer.
            unsafe {
                let p = self.data.as_ptr();
                core::ptr::copy(p.add(end), p.add(end + extra), self.len - end);
                core::ptr::copy_nonoverlapping(replace_with.as_ptr(), p.add(start), added);
            }
            self.len = new_len;
        } else {
            let shrink = removed - added;
            // SAFETY: replace_with cannot alias self; tail shift stays in-buffer.
            unsafe {
                let p = self.data.as_ptr();
                core::ptr::copy_nonoverlapping(replace_with.as_ptr(), p.add(start), added);
                if shrink > 0 {
                    core::ptr::copy(p.add(end), p.add(end - shrink), self.len - end);
                }
            }
            self.len -= shrink;
        }
    }

    /// Append a single character.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_push`] for a fallible
    /// variant.
    #[inline]
    pub fn push(&mut self, ch: char) {
        let mut buf = [0_u8; 4];
        let s = ch.encode_utf8(&mut buf);
        self.push_str(s);
    }

    /// Fallible variant of [`Self::push`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing
    /// allocator fails.
    #[inline]
    pub fn try_push(&mut self, ch: char) -> Result<(), allocator_api2::alloc::AllocError> {
        let mut buf = [0_u8; 4];
        let s = ch.encode_utf8(&mut buf);
        self.try_push_str(s)
    }

    /// Append a string slice.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_push_str`] for a fallible
    /// variant.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    #[inline]
    pub fn push_str(&mut self, s: impl AsRef<str>) {
        let s = s.as_ref();
        if s.is_empty() {
            return;
        }
        let cur_len = self.len;
        let needed = cur_len.checked_add(s.len()).expect("String overflow");
        if needed > self.cap {
            self.grow_to_at_least(needed);
        }
        // SAFETY: capacity is sufficient.
        unsafe {
            let dst = self.data.as_ptr().add(cur_len);
            core::ptr::copy_nonoverlapping(s.as_ptr(), dst, s.len());
        }
        self.len = needed;
    }

    /// Fallible variant of [`Self::push_str`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing
    /// allocator fails.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    #[inline]
    pub fn try_push_str(&mut self, s: impl AsRef<str>) -> Result<(), allocator_api2::alloc::AllocError> {
        let s = s.as_ref();
        if s.is_empty() {
            return Ok(());
        }
        let cur_len = self.len;
        let needed = cur_len.checked_add(s.len()).ok_or(allocator_api2::alloc::AllocError)?;
        if needed > self.cap {
            self.try_grow_to_at_least(needed)?;
        }
        // SAFETY: capacity is sufficient.
        unsafe {
            let dst = self.data.as_ptr().add(cur_len);
            core::ptr::copy_nonoverlapping(s.as_ptr(), dst, s.len());
        }
        self.len = needed;
        Ok(())
    }

    /// Reserve capacity for at least `additional` more bytes.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_reserve`] for a fallible
    /// variant.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len.checked_add(additional).expect("String overflow");
        if needed > self.cap {
            self.grow_to_at_least(needed);
        }
    }

    /// Fallible variant of [`Self::reserve`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing
    /// allocator fails.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), allocator_api2::alloc::AllocError> {
        let needed = self.len.checked_add(additional).ok_or(allocator_api2::alloc::AllocError)?;
        if needed > self.cap {
            self.try_grow_to_at_least(needed)?;
        }
        Ok(())
    }

    pub const fn clear(&mut self) {
        self.len = 0;
    }

    /// Freeze into an immutable [`RcStr<A>`].
    ///
    /// **O(1)** — the length is already stored inline at the right
    /// position, so this just transfers the data pointer.
    ///
    /// If the buffer sits at the chunk's bump cursor (the common case
    /// when no other allocations happened during the build), any unused
    /// capacity is reclaimed back to the cursor so subsequent
    /// allocations can reuse those bytes.
    #[must_use]
    pub fn into_arena_str(self) -> RcStr<A> {
        if self.cap == 0 {
            return self.arena.alloc_str_rc("");
        }

        let len = self.len;
        // SAFETY: cap > 0 means prefix is allocated; len <= cap.
        unsafe { self.flush_len_to_prefix() };

        // SAFETY: data is in a chunk we hold a refcount on.
        let chunk = unsafe { header_for::<u8, A>(self.data) };
        // SAFETY: chunk is alive.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        let _ = chunk_ref.try_reclaim_tail(self.data, self.cap, len);

        let me = ManuallyDrop::new(self);
        // SAFETY: data points at proper RcStr layout; refcount transferred.
        unsafe { RcStr::from_raw_data(me.data) }
    }

    /// Write `self.len` to the inline prefix slot in the chunk.
    ///
    /// # Safety
    ///
    /// Requires `self.cap > 0` and `self.len <= self.cap`.
    #[inline]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the prefix slot is bump-aligned to align_of::<usize>() at allocation time"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    unsafe fn flush_len_to_prefix(&self) {
        debug_assert!(self.cap > 0);
        debug_assert!(self.len <= self.cap);
        // SAFETY: slot at data - PREFIX_SIZE is the inline len prefix.
        unsafe {
            self.data.as_ptr().cast::<usize>().sub(1).write(self.len);
        }
    }

    fn allocate_initial(&mut self, cap: usize) {
        self.try_allocate_initial(cap).unwrap_or_else(|_| panic_alloc());
    }
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the prefix slot is bump-aligned to align_of::<usize>() at allocation time"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    fn try_allocate_initial(&mut self, cap: usize) -> Result<(), allocator_api2::alloc::AllocError> {
        debug_assert_eq!(self.cap, 0);
        debug_assert!(cap > 0);
        let total = PREFIX_SIZE.checked_add(cap).ok_or(allocator_api2::alloc::AllocError)?;
        if isize::try_from(total).is_err() {
            return Err(allocator_api2::alloc::AllocError);
        }
        // SAFETY: align_of::<usize>() is power of two; total <= isize::MAX.
        let layout = unsafe { Layout::from_size_align_unchecked(total, align_of::<usize>()) };
        let prefix_ptr = self.arena.try_bump_alloc_for_str(layout, ChunkSharing::Local)?;
        // SAFETY: writing 0 to prefix.
        unsafe {
            prefix_ptr.as_ptr().cast::<usize>().write(0);
        }
        // SAFETY: data starts after prefix.
        self.data = unsafe { NonNull::new_unchecked(prefix_ptr.as_ptr().add(PREFIX_SIZE)) };
        self.cap = cap;
        Ok(())
    }

    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    fn try_grow_to_at_least(&mut self, min_cap: usize) -> Result<(), allocator_api2::alloc::AllocError> {
        if self.cap == 0 {
            return self.try_allocate_initial(min_cap.max(MIN_INITIAL_CAP));
        }
        let new_cap = self.cap.checked_mul(2).map_or(min_cap, |doubled| doubled.max(min_cap));
        let old_total = PREFIX_SIZE + self.cap;
        let new_total = PREFIX_SIZE.checked_add(new_cap).ok_or(allocator_api2::alloc::AllocError)?;
        if isize::try_from(new_total).is_err() {
            return Err(allocator_api2::alloc::AllocError);
        }
        // SAFETY: cap > 0 and len <= cap.
        unsafe { self.flush_len_to_prefix() };
        // SAFETY: prefix_ptr was issued by arena.
        let prefix_ptr = unsafe { NonNull::new_unchecked(self.data.as_ptr().sub(PREFIX_SIZE)) };
        // SAFETY: prefix_ptr is ours.
        let new_ptr = unsafe { self.arena.grow_for_string(prefix_ptr, old_total, new_total) }?;

        // SAFETY: data starts at offset PREFIX_SIZE.
        self.data = unsafe { NonNull::new_unchecked(new_ptr.as_ptr().add(PREFIX_SIZE)) };
        self.cap = new_cap;
        Ok(())
    }

    fn grow_to_at_least(&mut self, min_cap: usize) {
        self.try_grow_to_at_least(min_cap).unwrap_or_else(|_| panic_alloc());
    }
}

impl<A: Allocator + Clone> Drop for String<'_, A> {
    #[inline]
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        // SAFETY: data is in Local-flavor chunk we hold refcount on.
        unsafe { release_chunk_ref_local::<u8, A>(self.data) };
    }
}

impl<A: Allocator + Clone> Deref for String<'_, A> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> DerefMut for String<'_, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A: Allocator + Clone> AsRef<str> for String<'_, A> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> AsMut<str> for String<'_, A> {
    fn as_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A: Allocator + Clone> Borrow<str> for String<'_, A> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> BorrowMut<str> for String<'_, A> {
    fn borrow_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A: Allocator + Clone> Clone for String<'_, A> {
    fn clone(&self) -> Self {
        let mut new = Self::with_capacity_in(self.len, self.arena);
        new.push_str(self.as_str());
        new
    }
}

impl<A: Allocator + Clone> fmt::Debug for String<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> fmt::Display for String<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> PartialEq for String<'_, A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<A: Allocator + Clone> PartialEq<str> for String<'_, A> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl<A: Allocator + Clone> PartialEq<&str> for String<'_, A> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
impl<A: Allocator + Clone> Eq for String<'_, A> {}
impl<A: Allocator + Clone> PartialOrd for String<'_, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<A: Allocator + Clone> Ord for String<'_, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl<A: Allocator + Clone> Hash for String<'_, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl<A: Allocator + Clone> Extend<char> for String<'_, A> {
    fn extend<I: IntoIterator<Item = char>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.reserve(lower);
        }
        for ch in iter {
            self.push(ch);
        }
    }
}

impl<'a, A: Allocator + Clone> Extend<&'a str> for String<'_, A> {
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        for s in iter {
            self.push_str(s);
        }
    }
}
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<A: Allocator + Clone> serde::ser::Serialize for String<'_, A> {
    fn serialize<S: serde::ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}
impl<'a, A: Allocator + Clone> crate::builders::FromIteratorIn<char> for String<'a, A> {
    type Allocator = &'a Arena<A>;

    fn from_iter_in<I: IntoIterator<Item = char>>(iter: I, allocator: &'a Arena<A>) -> Self {
        let arena = allocator;
        let mut s = Self::new_in(arena);
        for ch in iter {
            s.push(ch);
        }
        s
    }
}
