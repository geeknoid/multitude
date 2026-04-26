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
use widestring::Utf16Str;

use crate::Arena;
use crate::arena::panic_alloc;
use crate::arena_str_helpers::PREFIX_SIZE;
use crate::chunk_header::{header_for, release_chunk_ref_local};
use crate::chunk_ref::ChunkRef;
use crate::chunk_sharing::ChunkSharing;
use crate::rc_utf16_str::RcUtf16Str;

/// Initial capacity (in `u16` elements) granted on the first growth
/// when none was specified.
const MIN_INITIAL_CAP: usize = 16;

/// A growable, mutable UTF-16 string that lives in an [`Arena`].
///
/// `Utf16String` is a **transient builder**: 32 bytes on 64-bit (data
/// pointer + length + capacity + arena reference). All length and capacity
/// values are counted in `u16` elements (matching
/// `widestring::Utf16Str::len()`), not bytes. Its purpose is to be filled
/// and then frozen via [`Self::into_arena_utf16_str`] into a compact,
/// immutable [`RcUtf16Str`] (8 bytes). The freeze is **O(1)** — no copy,
/// no new allocation.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "utf16")] {
/// use multitude::Arena;
/// use widestring::utf16str;
///
/// let arena = Arena::new();
/// let mut s = arena.alloc_utf16_string();
/// s.push_str(utf16str!("hello, "));
/// s.push_str(utf16str!("world!"));
/// assert_eq!(s.as_utf16_str(), utf16str!("hello, world!"));
/// let frozen = s.into_arena_utf16_str();   // O(1), no copy
/// assert_eq!(&*frozen, utf16str!("hello, world!"));
/// # }
/// ```
pub struct Utf16String<'a, A: Allocator + Clone = Global> {
    data: NonNull<u16>,
    /// Length in `u16` elements.
    len: usize,
    /// Capacity in `u16` elements.
    cap: usize,
    arena: &'a Arena<A>,
    _not_sync: PhantomData<*mut ()>,
}

impl<'a, A: Allocator + Clone> Utf16String<'a, A> {
    /// Create a new, empty arena-backed UTF-16 string.
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

    /// Create a new arena-backed UTF-16 string with at least `cap` `u16`
    /// elements of pre-allocated capacity.
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

    /// String length in `u16` elements.
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

    /// Capacity in `u16` elements.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Borrow as `&Utf16Str`.
    #[must_use]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub const fn as_utf16_str(&self) -> &Utf16Str {
        if self.cap == 0 {
            // SAFETY: empty slice is valid UTF-16.
            return unsafe { Utf16Str::from_slice_unchecked(&[]) };
        }
        // SAFETY: valid UTF-16 of length `self.len` at `self.data`.
        unsafe {
            let slice = core::slice::from_raw_parts(self.data.as_ptr(), self.len);
            Utf16Str::from_slice_unchecked(slice)
        }
    }

    /// Return the `u16` slice view of this string.
    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[u16] {
        if self.cap == 0 {
            return &[];
        }
        // SAFETY: valid initialized elements of length `self.len` at `self.data`.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Return a mutable `Utf16Str` view of this string.
    ///
    /// Callers must preserve UTF-16 well-formedness; mutating the bytes
    /// in a way that produces invalid UTF-16 is undefined behavior, but
    /// only via the unsafe `Utf16Str` APIs that allow element-level edits.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn as_mut_utf16_str(&mut self) -> &mut Utf16Str {
        if self.cap == 0 {
            // SAFETY: empty slice is valid UTF-16.
            return unsafe { Utf16Str::from_slice_unchecked_mut(&mut []) };
        }
        // SAFETY: valid UTF-16 of length `self.len` at `self.data`; `&mut self` excludes aliasing.
        unsafe {
            let slice = core::slice::from_raw_parts_mut(self.data.as_ptr(), self.len);
            Utf16Str::from_slice_unchecked_mut(slice)
        }
    }

    /// Return a raw const pointer to the string's `u16` elements.
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const u16 {
        self.data.as_ptr().cast_const()
    }

    /// Return a raw mutable pointer to the string's `u16` elements.
    #[expect(clippy::needless_pass_by_ref_mut, reason = "API shape mirrors std::String::as_mut_ptr")]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut u16 {
        self.data.as_ptr()
    }

    /// Construct an `Utf16String` containing `s`, copied into `arena`.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails.
    #[must_use]
    pub fn from_utf16_str_in(s: &Utf16Str, arena: &'a Arena<A>) -> Self {
        let mut new = Self::with_capacity_in(s.len(), arena);
        new.push_str(s);
        new
    }

    /// Construct an `Utf16String` by transcoding a `&str` into
    /// UTF-16, copied into `arena`.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails.
    #[must_use]
    pub fn from_str_in(s: &str, arena: &'a Arena<A>) -> Self {
        // ASCII is the worst case at 1 u16 per byte; non-ASCII codepoints
        // use ≥ 2 bytes per ≤ 2 u16, so `s.len()` (UTF-8 byte count) is
        // always a safe upper bound on the resulting u16 element count.
        let mut new = Self::with_capacity_in(s.len(), arena);
        new.push_from_str(s);
        new
    }

    /// Remove the last character from the string and return it.
    ///
    /// Returns `None` if the string is empty.
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.as_utf16_str().chars().next_back()?;
        self.len -= ch.len_utf16();
        Some(ch)
    }

    /// Shorten the string to `new_len` `u16` elements.
    ///
    /// If `new_len >= self.len()`, this has no effect. Capacity is
    /// unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `new_len` is not on a UTF-16 character boundary
    /// (i.e. would split a surrogate pair).
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        assert!(
            self.as_utf16_str().is_char_boundary(new_len),
            "Utf16String::truncate: index {new_len} is not on a UTF-16 char boundary"
        );
        self.len = new_len;
    }

    /// Try to release any unused capacity back to the chunk's bump
    /// cursor. Only succeeds if this string's buffer ends exactly at
    /// the chunk's bump cursor (no later allocations have been made
    /// into the same chunk). On failure this is a no-op.
    pub fn shrink_to_fit(&mut self) {
        if self.cap == 0 || self.cap == self.len {
            return;
        }
        // SAFETY: data is in a chunk we hold a refcount on.
        let chunk = unsafe { header_for::<u16, A>(self.data) };
        // SAFETY: chunk is alive.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        if chunk_ref.try_reclaim_tail(self.data.cast::<u8>(), self.cap * size_of::<u16>(), self.len * size_of::<u16>()) {
            self.cap = self.len;
        }
    }

    /// Insert a character at element index `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is greater than `self.len()` or not on a UTF-16
    /// character boundary, or if the backing allocator fails on growth.
    pub fn insert(&mut self, idx: usize, ch: char) {
        let mut buf = [0_u16; 2];
        let s = ch.encode_utf16(&mut buf);
        // SAFETY: encode_utf16 always returns a complete, valid UTF-16 sequence.
        let utf16 = unsafe { Utf16Str::from_slice_unchecked(s) };
        self.insert_utf16_str(idx, utf16);
    }

    /// Insert a `Utf16Str` at element index `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is greater than `self.len()` or not on a UTF-16
    /// character boundary, if the resulting length would overflow
    /// `usize`, or if the backing allocator fails on growth.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn insert_utf16_str(&mut self, idx: usize, s: &Utf16Str) {
        assert!(
            self.as_utf16_str().is_char_boundary(idx),
            "Utf16String::insert_utf16_str: index {idx} is out of bounds or not on a UTF-16 char boundary"
        );
        let amt = s.len();
        if amt == 0 {
            return;
        }
        let new_len = self.len.checked_add(amt).expect("Utf16String overflow");
        if new_len > self.cap {
            self.grow_to_at_least(new_len);
        }
        // SAFETY: capacity is sufficient, idx <= self.len, regions are inside the buffer.
        unsafe {
            let p = self.data.as_ptr();
            core::ptr::copy(p.add(idx), p.add(idx + amt), self.len - idx);
            core::ptr::copy_nonoverlapping(s.as_slice().as_ptr(), p.add(idx), amt);
        }
        self.len = new_len;
    }

    /// Remove the character at element index `idx` and return it.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len()` or `idx` is not on a UTF-16
    /// character boundary.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn remove(&mut self, idx: usize) -> char {
        let s = self.as_utf16_str();
        assert!(
            idx < s.len() && s.is_char_boundary(idx),
            "Utf16String::remove: index {idx} is out of bounds or not on a UTF-16 char boundary"
        );
        // SAFETY: idx is on a char boundary and < len, so a char starts here.
        let ch = unsafe { s.get_unchecked(idx..).chars().next().unwrap_unchecked() };
        let ch_len = ch.len_utf16();
        let next = idx + ch_len;
        // SAFETY: next <= len, copy stays inside the buffer.
        unsafe {
            let p = self.data.as_ptr();
            core::ptr::copy(p.add(next), p.add(idx), self.len - next);
        }
        self.len -= ch_len;
        ch
    }

    /// Retain only the characters for which `f` returns `true`, in order.
    ///
    /// If `f` panics, any not-yet-processed characters are dropped from
    /// the string (matching `std::string::String::retain`).
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn retain<F: FnMut(char) -> bool>(&mut self, mut f: F) {
        struct Guard<'s, 'a, A: Allocator + Clone> {
            s: &'s mut Utf16String<'a, A>,
            idx: usize,
            del_units: usize,
        }

        impl<A: Allocator + Clone> Drop for Guard<'_, '_, A> {
            fn drop(&mut self) {
                self.s.len = self.idx - self.del_units;
            }
        }

        let len = self.len;
        let mut g = Guard {
            s: self,
            idx: 0,
            del_units: 0,
        };

        while g.idx < len {
            // SAFETY: g.idx is on a char boundary and < len within the buffer.
            let ch = unsafe {
                let p = g.s.data.as_ptr().add(g.idx);
                let slice = core::slice::from_raw_parts(p, len - g.idx);
                Utf16Str::from_slice_unchecked(slice).chars().next().unwrap_unchecked()
            };
            let ch_len = ch.len_utf16();

            if !f(ch) {
                g.del_units += ch_len;
            } else if g.del_units > 0 {
                // SAFETY: source and destination both lie within the buffer; lengths are valid.
                unsafe {
                    core::ptr::copy(g.s.data.as_ptr().add(g.idx), g.s.data.as_ptr().add(g.idx - g.del_units), ch_len);
                }
            }
            g.idx += ch_len;
        }
    }

    /// Replace the elements in `range` with the contents of `replace_with`.
    ///
    /// # Panics
    ///
    /// Panics if either bound is out of range, the bounds are not on
    /// UTF-16 character boundaries, the resulting length would overflow
    /// `usize`, or the backing allocator fails on growth.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn replace_range<R: RangeBounds<usize>>(&mut self, range: R, replace_with: &Utf16Str) {
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).expect("Utf16String overflow"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("Utf16String overflow"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.len,
        };
        let cur = self.as_utf16_str();
        assert!(start <= end, "Utf16String::replace_range: start ({start}) > end ({end})");
        assert!(
            end <= self.len,
            "Utf16String::replace_range: end ({end}) > len ({len})",
            len = self.len
        );
        assert!(
            cur.is_char_boundary(start),
            "Utf16String::replace_range: start ({start}) is not on a UTF-16 char boundary"
        );
        assert!(
            cur.is_char_boundary(end),
            "Utf16String::replace_range: end ({end}) is not on a UTF-16 char boundary"
        );

        let removed = end - start;
        let added = replace_with.len();

        if added > removed {
            let extra = added - removed;
            let new_len = self.len.checked_add(extra).expect("Utf16String overflow");
            if new_len > self.cap {
                self.grow_to_at_least(new_len);
            }
            // SAFETY: capacity is sufficient; both regions lie inside the buffer.
            unsafe {
                let p = self.data.as_ptr();
                core::ptr::copy(p.add(end), p.add(end + extra), self.len - end);
                core::ptr::copy_nonoverlapping(replace_with.as_slice().as_ptr(), p.add(start), added);
            }
            self.len = new_len;
        } else {
            let shrink = removed - added;
            // SAFETY: replace_with cannot alias self; tail shift stays in-buffer.
            unsafe {
                let p = self.data.as_ptr();
                core::ptr::copy_nonoverlapping(replace_with.as_slice().as_ptr(), p.add(start), added);
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
        let mut buf = [0_u16; 2];
        let s = ch.encode_utf16(&mut buf);
        self.push_slice(s);
    }

    /// Fallible variant of [`Self::push`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing
    /// allocator fails.
    #[inline]
    pub fn try_push(&mut self, ch: char) -> Result<(), allocator_api2::alloc::AllocError> {
        let mut buf = [0_u16; 2];
        let s = ch.encode_utf16(&mut buf);
        self.try_push_slice(s)
    }

    /// Append a `Utf16Str`.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_push_str`] for a fallible
    /// variant.
    pub fn push_str(&mut self, s: &Utf16Str) {
        self.push_slice(s.as_slice());
    }

    /// Fallible variant of [`Self::push_str`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing
    /// allocator fails.
    pub fn try_push_str(&mut self, s: &Utf16Str) -> Result<(), allocator_api2::alloc::AllocError> {
        self.try_push_slice(s.as_slice())
    }

    /// Append a `&str`, transcoding it to UTF-16.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_push_from_str`] for a
    /// fallible variant.
    pub fn push_from_str(&mut self, s: &str) {
        self.try_push_from_str(s).unwrap_or_else(|_| panic_alloc());
    }

    /// Fallible variant of [`Self::push_from_str`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing
    /// allocator fails.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub fn try_push_from_str(&mut self, s: &str) -> Result<(), allocator_api2::alloc::AllocError> {
        if s.is_empty() {
            return Ok(());
        }
        // `s.len()` is an upper bound on the resulting `u16` count: ASCII
        // (the worst case) costs 1 u16 per byte, and any non-ASCII codepoint
        // uses ≥ 2 bytes to produce ≤ 2 u16 elements. So a single reserve
        // covers every input — no growth happens inside the loop.
        let new_len = self.len.checked_add(s.len()).ok_or(allocator_api2::alloc::AllocError)?;
        if new_len > self.cap {
            self.try_grow_to_at_least(new_len)?;
        }
        let mut written = self.len;
        // SAFETY: `written + 2 <= self.cap` because `self.len + s.len()
        // <= self.cap` and each char produces ≤ 2 u16; the loop body
        // writes at most 2 u16s past `written` per iteration.
        unsafe {
            let base = self.data.as_ptr();
            for ch in s.chars() {
                let n = ch.encode_utf16(&mut [0_u16; 2]).len();
                if n == 1 {
                    base.add(written).write(ch as u16);
                } else {
                    let mut buf = [0_u16; 2];
                    let _ = ch.encode_utf16(&mut buf);
                    base.add(written).write(buf[0]);
                    base.add(written + 1).write(buf[1]);
                }
                written += n;
            }
        }
        self.len = written;
        Ok(())
    }

    fn push_slice(&mut self, s: &[u16]) {
        self.try_push_slice(s).unwrap_or_else(|_| panic_alloc());
    }

    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    fn try_push_slice(&mut self, s: &[u16]) -> Result<(), allocator_api2::alloc::AllocError> {
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

    /// Reserve capacity for at least `additional` more `u16` elements.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails. Use [`Self::try_reserve`] for a fallible
    /// variant.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len.checked_add(additional).expect("Utf16String overflow");
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

    /// Freeze into an immutable [`RcUtf16Str<A>`].
    ///
    /// **O(1)** — the length is already stored inline at the right
    /// position, so this just transfers the data pointer.
    ///
    /// If the buffer sits at the chunk's bump cursor (the common case
    /// when no other allocations happened during the build), any unused
    /// capacity is reclaimed back to the cursor so subsequent
    /// allocations can reuse those bytes.
    #[must_use]
    pub fn into_arena_utf16_str(self) -> RcUtf16Str<A> {
        if self.cap == 0 {
            // SAFETY: empty slice is valid UTF-16.
            let empty = unsafe { Utf16Str::from_slice_unchecked(&[]) };
            return self.arena.alloc_utf16_str_rc(empty);
        }

        let len = self.len;
        // SAFETY: cap > 0 means prefix is allocated; len <= cap.
        unsafe { self.flush_len_to_prefix() };

        // SAFETY: data is in a chunk we hold a refcount on.
        let chunk = unsafe { header_for::<u16, A>(self.data) };
        // SAFETY: chunk is alive.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        let _ = chunk_ref.try_reclaim_tail(self.data.cast::<u8>(), self.cap * size_of::<u16>(), len * size_of::<u16>());

        let me = ManuallyDrop::new(self);
        // SAFETY: data points at proper RcUtf16Str layout; refcount transferred.
        unsafe { RcUtf16Str::from_raw_data(me.data) }
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
        // SAFETY: slot at data.cast::<usize>().sub(1) is the inline len prefix.
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
        let payload_bytes = cap.checked_mul(size_of::<u16>()).ok_or(allocator_api2::alloc::AllocError)?;
        let total = PREFIX_SIZE.checked_add(payload_bytes).ok_or(allocator_api2::alloc::AllocError)?;
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
        self.data = unsafe { NonNull::new_unchecked(prefix_ptr.as_ptr().add(PREFIX_SIZE).cast::<u16>()) };
        self.cap = cap;
        Ok(())
    }

    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the prefix slot is bump-aligned to align_of::<usize>() at allocation time"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    fn try_grow_to_at_least(&mut self, min_cap: usize) -> Result<(), allocator_api2::alloc::AllocError> {
        if self.cap == 0 {
            return self.try_allocate_initial(min_cap.max(MIN_INITIAL_CAP));
        }
        let new_cap = self.cap.checked_mul(2).map_or(min_cap, |doubled| doubled.max(min_cap));
        let old_payload_bytes = self.cap * size_of::<u16>();
        let old_total = PREFIX_SIZE + old_payload_bytes;
        let new_payload_bytes = new_cap.checked_mul(size_of::<u16>()).ok_or(allocator_api2::alloc::AllocError)?;
        let new_total = PREFIX_SIZE
            .checked_add(new_payload_bytes)
            .ok_or(allocator_api2::alloc::AllocError)?;
        if isize::try_from(new_total).is_err() {
            return Err(allocator_api2::alloc::AllocError);
        }
        // SAFETY: cap > 0 and len <= cap.
        unsafe { self.flush_len_to_prefix() };
        // SAFETY: prefix_ptr was issued by arena.
        let prefix_ptr = unsafe { NonNull::new_unchecked(self.data.as_ptr().cast::<u8>().sub(PREFIX_SIZE)) };
        // SAFETY: prefix_ptr is ours.
        let new_ptr = unsafe { self.arena.grow_for_string(prefix_ptr, old_total, new_total) }?;

        // SAFETY: data starts at offset PREFIX_SIZE.
        self.data = unsafe { NonNull::new_unchecked(new_ptr.as_ptr().add(PREFIX_SIZE).cast::<u16>()) };
        self.cap = new_cap;
        Ok(())
    }

    fn grow_to_at_least(&mut self, min_cap: usize) {
        self.try_grow_to_at_least(min_cap).unwrap_or_else(|_| panic_alloc());
    }
}

impl<A: Allocator + Clone> Drop for Utf16String<'_, A> {
    #[inline]
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        // SAFETY: data is in Local-flavor chunk we hold refcount on.
        unsafe { release_chunk_ref_local::<u16, A>(self.data) };
    }
}

impl<A: Allocator + Clone> Deref for Utf16String<'_, A> {
    type Target = Utf16Str;
    #[inline]
    fn deref(&self) -> &Utf16Str {
        self.as_utf16_str()
    }
}

impl<A: Allocator + Clone> DerefMut for Utf16String<'_, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Utf16Str {
        self.as_mut_utf16_str()
    }
}

impl<A: Allocator + Clone> AsRef<Utf16Str> for Utf16String<'_, A> {
    fn as_ref(&self) -> &Utf16Str {
        self.as_utf16_str()
    }
}

impl<A: Allocator + Clone> AsMut<Utf16Str> for Utf16String<'_, A> {
    fn as_mut(&mut self) -> &mut Utf16Str {
        self.as_mut_utf16_str()
    }
}

impl<A: Allocator + Clone> Borrow<Utf16Str> for Utf16String<'_, A> {
    fn borrow(&self) -> &Utf16Str {
        self.as_utf16_str()
    }
}

impl<A: Allocator + Clone> BorrowMut<Utf16Str> for Utf16String<'_, A> {
    fn borrow_mut(&mut self) -> &mut Utf16Str {
        self.as_mut_utf16_str()
    }
}

impl<A: Allocator + Clone> Clone for Utf16String<'_, A> {
    fn clone(&self) -> Self {
        let mut new = Self::with_capacity_in(self.len, self.arena);
        new.push_str(self.as_utf16_str());
        new
    }
}

impl<A: Allocator + Clone> fmt::Debug for Utf16String<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_utf16_str(), f)
    }
}

impl<A: Allocator + Clone> fmt::Display for Utf16String<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_utf16_str(), f)
    }
}

impl<A: Allocator + Clone> PartialEq for Utf16String<'_, A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_utf16_str() == other.as_utf16_str()
    }
}
impl<A: Allocator + Clone> Eq for Utf16String<'_, A> {}
impl<A: Allocator + Clone> PartialOrd for Utf16String<'_, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<A: Allocator + Clone> Ord for Utf16String<'_, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_utf16_str().cmp(other.as_utf16_str())
    }
}
impl<A: Allocator + Clone> Hash for Utf16String<'_, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_utf16_str().hash(state);
    }
}

impl<A: Allocator + Clone> PartialEq<Utf16Str> for Utf16String<'_, A> {
    fn eq(&self, other: &Utf16Str) -> bool {
        self.as_utf16_str() == other
    }
}

impl<A: Allocator + Clone> PartialEq<&Utf16Str> for Utf16String<'_, A> {
    fn eq(&self, other: &&Utf16Str) -> bool {
        self.as_utf16_str() == *other
    }
}

impl<A: Allocator + Clone> Extend<char> for Utf16String<'_, A> {
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

impl<'a, A: Allocator + Clone> Extend<&'a Utf16Str> for Utf16String<'_, A> {
    fn extend<I: IntoIterator<Item = &'a Utf16Str>>(&mut self, iter: I) {
        for s in iter {
            self.push_str(s);
        }
    }
}

impl<'a, A: Allocator + Clone> Extend<&'a str> for Utf16String<'_, A> {
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        for s in iter {
            self.push_from_str(s);
        }
    }
}
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<A: Allocator + Clone> serde::ser::Serialize for Utf16String<'_, A> {
    fn serialize<S: serde::ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        crate::arena_str_helpers::serialize_utf16(self.as_utf16_str(), serializer)
    }
}
