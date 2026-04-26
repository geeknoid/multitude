//! [`ArenaArcStr`] — `Send + Sync` single-pointer immutable arena-backed
//! string.

use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::arena_rc_str::read_str_len;
use crate::chunk_header::{header_for, teardown_chunk};
use crate::chunk_sharing::ChunkSharing;

/// `Send + Sync` counterpart to [`ArenaRcStr`](crate::ArenaRcStr). Same on-disk
/// layout but backed by a `Shared`-flavor chunk.
///
/// `ArenaArcStr<A>: Send + Sync` requires `A: Send + Sync`.
///
/// # Example
///
/// ```
/// use harena::{Arena, ArenaArcStr};
/// use std::thread;
///
/// let arena = Arena::new();
/// let s = ArenaArcStr::from_str(&arena, "shared");
/// let s2 = s.clone();
/// let h = thread::spawn(move || s2.len());
/// assert_eq!(s.len(), h.join().unwrap());
/// ```
pub struct ArenaArcStr<A: Allocator + Clone = Global> {
    data: NonNull<u8>,
    _allocator: PhantomData<A>,
}

// SAFETY: backed by a Shared chunk with atomic refcount.
unsafe impl<A: Allocator + Clone + Send + Sync> Send for ArenaArcStr<A> {}
// SAFETY: see above.
unsafe impl<A: Allocator + Clone + Send + Sync> Sync for ArenaArcStr<A> {}

impl<A: Allocator + Clone> ArenaArcStr<A> {
    /// Copy `s` into the arena (in a `Shared`-flavor chunk) and return a
    /// handle to it.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails.
    #[must_use]
    pub fn from_str(arena: &crate::Arena<A>, s: impl AsRef<str>) -> Self
    where
        A: Send + Sync,
    {
        // SAFETY: helper writes prefix + bytes; we wrap in ArenaArcStr.
        let data = unsafe { crate::arena_rc_str::reserve_str_in_chunk(arena, s.as_ref(), ChunkSharing::Shared) };
        Self {
            data,
            _allocator: PhantomData,
        }
    }

    /// Borrow as `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // SAFETY: same reasoning as ArenaRcStr::as_str.
        unsafe {
            let len = read_str_len(self.data);
            let bytes = core::slice::from_raw_parts(self.data.as_ptr(), len);
            core::str::from_utf8_unchecked(bytes)
        }
    }

    /// String length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        // SAFETY: see as_str.
        unsafe { read_str_len(self.data) }
    }

    /// True iff the string is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<A: Allocator + Clone> Clone for ArenaArcStr<A> {
    fn clone(&self) -> Self {
        // SAFETY: chunk alive.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.data);
            chunk.as_ref().inc_ref();
        }
        Self {
            data: self.data,
            _allocator: PhantomData,
        }
    }
}

impl<A: Allocator + Clone> Drop for ArenaArcStr<A> {
    fn drop(&mut self) {
        // SAFETY: chunk alive; dec refcount.
        unsafe {
            let chunk: NonNull<crate::chunk_header::ChunkHeader<A>> = header_for(self.data);
            if chunk.as_ref().dec_ref() {
                teardown_chunk(chunk, false);
            }
        }
    }
}

impl<A: Allocator + Clone> Deref for ArenaArcStr<A> {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> AsRef<str> for ArenaArcStr<A> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> Borrow<str> for ArenaArcStr<A> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator + Clone> fmt::Debug for ArenaArcStr<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> fmt::Display for ArenaArcStr<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<A: Allocator + Clone> PartialEq for ArenaArcStr<A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl<A: Allocator + Clone> Eq for ArenaArcStr<A> {}
impl<A: Allocator + Clone> PartialOrd for ArenaArcStr<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<A: Allocator + Clone> Ord for ArenaArcStr<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl<A: Allocator + Clone> Hash for ArenaArcStr<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}
