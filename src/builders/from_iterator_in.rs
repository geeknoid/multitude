/// Build a collection from an iterator, allocating into a user-supplied
/// allocator smart pointer (for our types, `&'a Arena<A>`).
///
/// The arena-aware counterpart to [`core::iter::FromIterator`]. Implemented
/// for [`Vec`](crate::builders::Vec) and [`String`](crate::builders::String).
pub trait FromIteratorIn<T>: Sized {
    /// The allocator smart pointer this collection needs in order to be built.
    type Allocator;

    /// Build the collection from `iter`, allocating into `allocator`.
    fn from_iter_in<I: IntoIterator<Item = T>>(iter: I, allocator: Self::Allocator) -> Self;
}
