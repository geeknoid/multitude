//! [`ArenaString`] — a growable UTF-8 string that allocates its buffer inside an arena.
//!
//! Custom implementation (not a `Vec` wrapper) so that the length is
//! stored inline in the chunk, **immediately before** the data buffer —
//! exactly where [`ArenaRcStr`] expects it. This makes the freeze step into
//! [`ArenaRcStr`] **O(1)**: no allocation, no copy, just transfer the data
//! pointer.

use core::alloc::Layout;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Deref;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::arena_rc_str::ArenaRcStr;
use crate::chunk_header::{header_for, teardown_chunk};
use crate::chunk_sharing::ChunkSharing;

/// Number of bytes occupied by the inline length prefix that precedes
/// the data buffer in the chunk.
const PREFIX_SIZE: usize = size_of::<usize>();

/// Initial capacity granted on the first growth (when none was specified).
const MIN_INITIAL_CAP: usize = 16;

/// A growable, bump-allocated UTF-8 string that lives inside an
/// [`Arena`](crate::Arena).
///
/// **24 bytes on 64-bit** (data pointer + capacity + arena reference).
/// The length is stored inline in the chunk immediately before the data
/// buffer (matching [`ArenaRcStr`]'s layout), so [`Self::into_arena_str`]
/// is **O(1)** — it transfers the data pointer with no copy and no new
/// allocation.
///
/// # Example
///
/// ```
/// use harena::Arena;
///
/// let arena = Arena::new();
/// let mut s = arena.new_string();
/// s.push_str("hello, ");
/// s.push_str("world!");
/// assert_eq!(s.as_str(), "hello, world!");
/// let frozen = s.into_arena_str();   // O(1), no copy
/// assert_eq!(&*frozen, "hello, world!");
/// ```
pub struct ArenaString<'a, A: Allocator + Clone = Global> {
    /// Points at the first byte of the data buffer. The `usize` length
    /// prefix lives at `data.sub(PREFIX_SIZE)`. While `cap == 0`, this
    /// is a `dangling` sentinel meaning "no allocation yet".
    data: NonNull<u8>,
    /// Capacity in data bytes (excluding the prefix). Zero iff no
    /// allocation has been made yet.
    cap: usize,
    /// Arena handle for growth-time reallocation.
    arena: &'a crate::Arena<A>,
    /// `ArenaString` is single-threaded only.
    _not_sync: PhantomData<*mut ()>,
}

impl<'a, A: Allocator + Clone> ArenaString<'a, A> {
    /// Create a new, empty arena-backed string.
    ///
    /// No allocation is performed until the first push.
    #[must_use]
    pub(crate) fn new_in(arena: &'a crate::Arena<A>) -> Self {
        Self {
            data: NonNull::dangling(),
            cap: 0,
            arena,
            _not_sync: PhantomData,
        }
    }

    /// Create a new arena-backed string with at least `cap` bytes of
    /// pre-allocated capacity.
    #[must_use]
    pub(crate) fn with_capacity_in(cap: usize, arena: &'a crate::Arena<A>) -> Self {
        let mut s = Self::new_in(arena);
        if cap > 0 {
            s.allocate_initial(cap);
        }
        s
    }

    /// Number of bytes (not chars) currently in the string.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.cap == 0 {
            0
        } else {
            // SAFETY: data points at the first data byte; the inline len
            // prefix lives at data - PREFIX_SIZE.
            unsafe { self.data.as_ptr().cast::<usize>().sub(1).read() }
        }
    }

    /// True iff the string is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Allocated capacity in bytes (not counting the prefix).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Borrow as `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        if self.cap == 0 {
            return "";
        }
        // SAFETY: invariants of `ArenaString` ensure valid UTF-8 of
        // length `self.len()` at `self.data`.
        unsafe {
            let len = self.len();
            let bytes = core::slice::from_raw_parts(self.data.as_ptr(), len);
            core::str::from_utf8_unchecked(bytes)
        }
    }

    /// Append a single character.
    pub fn push(&mut self, ch: char) {
        let mut buf = [0_u8; 4];
        let s = ch.encode_utf8(&mut buf);
        self.push_str(s);
    }

    /// Append a string slice.
    ///
    /// # Panics
    ///
    /// Panics if the resulting length would overflow `usize`, or if the
    /// underlying allocator fails on growth.
    pub fn push_str(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        let cur_len = self.len();
        let needed = cur_len.checked_add(s.len()).expect("ArenaString overflow");
        if needed > self.cap {
            self.grow_to_at_least(needed);
        }
        // SAFETY: capacity is now sufficient; copy bytes and update the
        // inline len prefix.
        unsafe {
            let dst = self.data.as_ptr().add(cur_len);
            core::ptr::copy_nonoverlapping(s.as_ptr(), dst, s.len());
            self.set_len(needed);
        }
    }

    /// Reserve capacity for at least `additional` more bytes.
    ///
    /// # Panics
    ///
    /// Panics if the resulting capacity would overflow `usize`, or if
    /// the underlying allocator fails.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len().checked_add(additional).expect("ArenaString overflow");
        if needed > self.cap {
            self.grow_to_at_least(needed);
        }
    }

    /// Empty the string but keep the allocated capacity.
    pub fn clear(&mut self) {
        if self.cap > 0 {
            // SAFETY: cap > 0 means data is allocated; len = 0 is valid.
            unsafe {
                self.set_len(0);
            }
        }
    }

    /// Freeze into an immutable [`ArenaRcStr<A>`].
    ///
    /// **O(1)** — the length is already stored inline at the right
    /// position, so this just transfers the data pointer.
    ///
    /// If the buffer sits at the chunk's bump cursor (the common case
    /// when no other allocations happened during the build), any unused
    /// capacity is reclaimed back to the cursor so subsequent
    /// allocations can reuse those bytes.
    #[must_use]
    pub fn into_arena_str(self) -> ArenaRcStr<A> {
        if self.cap == 0 {
            // Never allocated. Build an empty ArenaRcStr through the
            // normal path (allocates an 8-byte prefix slot containing 0).
            return ArenaRcStr::from_str(self.arena, "");
        }

        let len = self.len();

        // Slack reclamation: if the buffer is at the chunk's bump cursor,
        // lower the cursor to just past the live bytes so the unused
        // capacity becomes available for subsequent allocations.
        // SAFETY: data is in a chunk we hold a refcount on.
        let chunk = unsafe { header_for::<u8, A>(self.data) };
        // SAFETY: chunk is alive.
        let header = unsafe { chunk.as_ref() };
        let chunk_base = chunk.as_ptr() as usize;
        let buffer_end = self.data.as_ptr() as usize + self.cap;
        let buffer_end_offset = buffer_end - chunk_base;
        if buffer_end_offset == header.bump.get() {
            let live_end_offset = (self.data.as_ptr() as usize - chunk_base) + len;
            header.bump.set(live_end_offset);
        }

        // Transfer the chunk refcount to the resulting ArenaRcStr by
        // suppressing our Drop.
        let me = ManuallyDrop::new(self);
        // SAFETY: data points at a properly-formatted ArenaRcStr buffer
        // (inline prefix + UTF-8 bytes); chunk refcount transferred.
        unsafe { ArenaRcStr::from_raw_data(me.data) }
    }

    // --- internal helpers -------------------------------------------------

    /// Set the inline length prefix.
    ///
    /// # Safety
    ///
    /// Requires `self.cap > 0` and `new_len <= self.cap`.
    unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(self.cap > 0);
        debug_assert!(new_len <= self.cap);
        // SAFETY: data is allocated; the slot at data - PREFIX_SIZE is
        // the inline len prefix.
        unsafe {
            self.data.as_ptr().cast::<usize>().sub(1).write(new_len);
        }
    }

    /// First-time allocation: allocate `[ prefix | cap bytes ]`.
    ///
    /// `cap` must be > 0 and the current state must be unallocated.
    fn allocate_initial(&mut self, cap: usize) {
        debug_assert_eq!(self.cap, 0);
        debug_assert!(cap > 0);
        let total = PREFIX_SIZE.checked_add(cap).expect("ArenaString overflow");
        let layout = Layout::from_size_align(total, align_of::<usize>()).expect("ArenaString layout");
        // SAFETY: Local-flavor allocation; bumps the chunk refcount for
        // this handle's hold.
        let prefix_ptr = unsafe { self.arena.bump_alloc_for_str(layout, ChunkSharing::Local) };
        // SAFETY: writing 0 to the prefix slot.
        unsafe {
            prefix_ptr.as_ptr().cast::<usize>().write(0);
        }
        // SAFETY: data starts immediately after the prefix.
        self.data = unsafe { NonNull::new_unchecked(prefix_ptr.as_ptr().add(PREFIX_SIZE)) };
        self.cap = cap;
    }

    /// Grow to at least `min_cap` bytes of data capacity. May reallocate
    /// (in which case the entire `[prefix|data]` block is moved as a
    /// unit, preserving the inline-prefix-then-data layout).
    fn grow_to_at_least(&mut self, min_cap: usize) {
        if self.cap == 0 {
            self.allocate_initial(min_cap.max(MIN_INITIAL_CAP));
            return;
        }
        // Doubling strategy with a min, like std::Vec.
        let new_cap = self.cap.checked_mul(2).map_or(min_cap, |doubled| doubled.max(min_cap));
        let old_total = PREFIX_SIZE + self.cap;
        let new_total = PREFIX_SIZE.checked_add(new_cap).expect("ArenaString overflow");
        let align = align_of::<usize>();
        let old_layout = Layout::from_size_align(old_total, align).expect("layout");
        let new_layout = Layout::from_size_align(new_total, align).expect("layout");
        // Ask the arena's internal grow helper to grow the [prefix|data]
        // block as one unit. If the buffer sits at the chunk's bump
        // cursor (the common case), it extends in place and returns the
        // same pointer. Otherwise it allocates a new region and copies
        // the bytes — which automatically preserves our inline len
        // prefix because it's the first 8 bytes of the allocation. The
        // helper bumps `string_relocations` (not `allocator_relocations`)
        // when a relocation occurs.
        // SAFETY: prefix_ptr was issued by `self.arena`'s allocator.
        let prefix_ptr = unsafe { NonNull::new_unchecked(self.data.as_ptr().sub(PREFIX_SIZE)) };
        // SAFETY: helper contract — prefix_ptr is one of ours, old_layout
        // matches what we allocated, new_layout >= old_layout with same
        // alignment.
        let new_ptr = unsafe { self.arena.grow_for_string(prefix_ptr, old_layout, new_layout) }.expect("ArenaString grow failed");

        // SAFETY: the returned pointer's first PREFIX_SIZE bytes hold
        // our (preserved) len; data starts at offset PREFIX_SIZE.
        self.data = unsafe { NonNull::new_unchecked(new_ptr.as_ptr().add(PREFIX_SIZE)) };
        self.cap = new_cap;
    }
}

impl<A: Allocator + Clone> Drop for ArenaString<'_, A> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        // SAFETY: data is in a chunk we hold a refcount on.
        unsafe {
            let chunk = header_for::<u8, A>(self.data);
            if chunk.as_ref().dec_ref() {
                teardown_chunk(chunk, true);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Standard trait impls.
// ---------------------------------------------------------------------------

impl<A: Allocator + Clone> Deref for ArenaString<'_, A> {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> AsRef<str> for ArenaString<'_, A> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> Borrow<str> for ArenaString<'_, A> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> fmt::Debug for ArenaString<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> fmt::Display for ArenaString<'_, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> PartialEq for ArenaString<'_, A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl<A: Allocator + Clone> Eq for ArenaString<'_, A> {}
impl<A: Allocator + Clone> PartialOrd for ArenaString<'_, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<A: Allocator + Clone> Ord for ArenaString<'_, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl<A: Allocator + Clone> Hash for ArenaString<'_, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl<A: Allocator + Clone> Extend<char> for ArenaString<'_, A> {
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

impl<'a, A: Allocator + Clone> Extend<&'a str> for ArenaString<'_, A> {
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        for s in iter {
            self.push_str(s);
        }
    }
}
