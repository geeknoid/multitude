//! `CollectIn` and `FromIteratorIn` — arena-aware iterator collection.

use allocator_api2::alloc::Allocator;

use crate::arena_string::ArenaString;
use crate::arena_vec::ArenaVec;

/// Build a collection from an iterator, allocating into a user-supplied
/// allocator handle (for our types, `&'a Arena<A>`).
///
/// The arena-aware counterpart to [`core::iter::FromIterator`]. Implemented
/// for [`ArenaVec`] and [`ArenaString`].
pub trait FromIteratorIn<T>: Sized {
    /// The allocator handle this collection needs in order to be built.
    type Allocator;

    /// Build the collection from `iter`, allocating into `allocator`.
    fn from_iter_in<I: IntoIterator<Item = T>>(iter: I, allocator: Self::Allocator) -> Self;
}

/// Extension trait on iterators that lets you collect directly into an
/// arena-backed collection.
///
/// Blanket-implemented for every `IntoIterator`. Usage typically annotates
/// the result type so the compiler picks the right `C`:
///
/// ```
/// use harena::{Arena, ArenaVec, CollectIn};
///
/// let arena = Arena::new();
/// let v: ArenaVec<u32, _> = (1..=10).collect_in(&arena);
/// assert_eq!(v.len(), 10);
/// ```
pub trait CollectIn: IntoIterator + Sized {
    /// Collect this iterator into `C`, using `allocator` as the backing
    /// allocator handle.
    fn collect_in<C: FromIteratorIn<Self::Item>>(self, allocator: C::Allocator) -> C {
        C::from_iter_in(self, allocator)
    }
}

impl<I: IntoIterator + Sized> CollectIn for I {}

impl<'a, T, A: Allocator + Clone> FromIteratorIn<T> for ArenaVec<'a, T, A> {
    type Allocator = &'a crate::Arena<A>;

    fn from_iter_in<I: IntoIterator<Item = T>>(iter: I, allocator: &'a crate::Arena<A>) -> Self {
        let arena = allocator;
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut v = ArenaVec::with_capacity_in(lower, arena);
        for item in iter {
            v.push(item);
        }
        v
    }
}

impl<'a, A: Allocator + Clone> FromIteratorIn<char> for ArenaString<'a, A> {
    type Allocator = &'a crate::Arena<A>;

    fn from_iter_in<I: IntoIterator<Item = char>>(iter: I, allocator: &'a crate::Arena<A>) -> Self {
        let arena = allocator;
        let mut s = ArenaString::new_in(arena);
        for ch in iter {
            s.push(ch);
        }
        s
    }
}
