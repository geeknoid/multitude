//! [`ArenaRcStr`] — single-pointer immutable arena-backed string.

use core::alloc::Layout;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::chunk_header::{header_for, teardown_chunk};
use crate::chunk_sharing::ChunkSharing;
use crate::constants::CHUNK_ALIGN;

/// An immutable, single-pointer reference-counted UTF-8 string stored in
/// an [`Arena`](crate::Arena).
///
/// **8 bytes on 64-bit** (a single `NonNull<u8>` plus zero-sized
/// markers). The string's length is stored inline in the chunk,
/// immediately before the string bytes — three times more compact than
/// `String` (24 bytes) and half the size of `&str` (16 bytes).
///
/// `ArenaRcStr` is the recommended long-term storage type for arena
/// strings. Build them via either:
/// - [`ArenaRcStr::from_str`] — copy a `&str` directly into the arena, or
/// - [`ArenaString`](crate::ArenaString) + [`into_arena_str`](crate::ArenaString::into_arena_str)
///   — build incrementally, then freeze.
///
/// **Not** [`Send`] / [`Sync`] — see [`ArenaArcStr`](crate::ArenaArcStr)
/// for the cross-thread variant.
///
/// # Example
///
/// ```
/// use harena::{Arena, ArenaRcStr};
///
/// let arena = Arena::new();
/// let s = ArenaRcStr::from_str(&arena, "hello");
/// assert_eq!(&*s, "hello");
/// assert_eq!(s.len(), 5);
/// ```
pub struct ArenaRcStr<A: Allocator + Clone = Global> {
    /// Points at the first data byte. The `usize` length prefix lives at
    /// `data.sub(size_of::<usize>())`.
    data: NonNull<u8>,
    _not_sync: PhantomData<*mut ()>,
    _allocator: PhantomData<A>,
}

impl<A: Allocator + Clone> ArenaRcStr<A> {
    /// Copy `s` into the arena (preceded by a `usize` length prefix) and
    /// return an immutable handle to it.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails.
    #[must_use]
    pub fn from_str(arena: &crate::Arena<A>, s: impl AsRef<str>) -> Self {
        // SAFETY: helper allocates and bumps refcount.
        let data = unsafe { reserve_str_in_chunk(arena, s.as_ref(), ChunkSharing::Local) };
        Self {
            data,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }

    /// Borrow as `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // SAFETY: data points at valid UTF-8 of the prefix-stored length.
        unsafe {
            let len = read_str_len(self.data);
            let bytes = core::slice::from_raw_parts(self.data.as_ptr(), len);
            core::str::from_utf8_unchecked(bytes)
        }
    }

    /// String length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        // SAFETY: see `as_str`.
        unsafe { read_str_len(self.data) }
    }

    /// True iff the string is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Construct an `ArenaRcStr` from a raw data pointer.
    ///
    /// # Safety
    ///
    /// `data` must point at the first byte of a properly-formatted
    /// arena-resident string buffer (length-prefixed at
    /// `data - size_of::<usize>()`), and the chunk's refcount must
    /// already have been incremented for this handle.
    #[inline]
    #[must_use]
    pub(crate) unsafe fn from_raw_data(data: NonNull<u8>) -> Self {
        Self {
            data,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }
}

/// Internal: copy `s` into a chunk of the requested sharing flavor,
/// returning the data pointer (just past the inline length prefix). The
/// chunk's refcount is incremented for the upcoming handle.
///
/// # Safety
///
/// Caller must wrap the returned pointer in a handle that decrements the
/// refcount on drop.
pub(crate) unsafe fn reserve_str_in_chunk<A: Allocator + Clone>(arena: &crate::Arena<A>, s: &str, sharing: ChunkSharing) -> NonNull<u8> {
    let prefix_size = size_of::<usize>();
    let prefix_align = align_of::<usize>();
    let total = prefix_size.checked_add(s.len()).expect("overflow");
    let layout = Layout::from_size_align(total, prefix_align).expect("layout");
    assert!(layout.align() <= CHUNK_ALIGN, "harena2: alignment too large");
    // SAFETY: arena helper allocates and bumps refcount.
    let prefix_ptr = unsafe { arena.bump_alloc_for_str(layout, sharing) };
    // SAFETY: write prefix, then string bytes.
    unsafe {
        prefix_ptr.cast::<usize>().as_ptr().write(s.len());
        let data_ptr = prefix_ptr.as_ptr().add(prefix_size);
        core::ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, s.len());
        NonNull::new_unchecked(data_ptr)
    }
}

/// Read the inline length prefix that precedes a string allocated by
/// [`ArenaRcStr::from_str`].
///
/// # Safety
///
/// `data` must point at the byte immediately following a `usize` length
/// prefix laid down by the arena allocator.
#[inline]
pub(crate) unsafe fn read_str_len(data: NonNull<u8>) -> usize {
    // SAFETY: caller guarantees the prefix is there.
    unsafe { data.as_ptr().cast::<usize>().sub(1).read() }
}

impl<A: Allocator + Clone> Clone for ArenaRcStr<A> {
    fn clone(&self) -> Self {
        // SAFETY: chunk is alive via our refcount.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.data);
            chunk.as_ref().inc_ref();
        }
        Self {
            data: self.data,
            _not_sync: PhantomData,
            _allocator: PhantomData,
        }
    }
}

impl<A: Allocator + Clone> Drop for ArenaRcStr<A> {
    fn drop(&mut self) {
        // SAFETY: chunk alive; dec refcount.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.data);
            if chunk.as_ref().dec_ref() {
                teardown_chunk(chunk, true);
            }
        }
    }
}

impl<A: Allocator + Clone> Deref for ArenaRcStr<A> {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> AsRef<str> for ArenaRcStr<A> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> Borrow<str> for ArenaRcStr<A> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> fmt::Debug for ArenaRcStr<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> fmt::Display for ArenaRcStr<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> PartialEq for ArenaRcStr<A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl<A: Allocator + Clone> Eq for ArenaRcStr<A> {}
impl<A: Allocator + Clone> PartialOrd for ArenaRcStr<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<A: Allocator + Clone> Ord for ArenaRcStr<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl<A: Allocator + Clone> Hash for ArenaRcStr<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}
