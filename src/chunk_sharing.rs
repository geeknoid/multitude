/// Whether a chunk's refcount is touched only from the arena's owner
/// thread (`Local`, non-atomic-equivalent ops) or potentially from any
/// thread (`Shared`, atomic Acquire/Release ops).
///
/// Fixed at chunk birth and never changes. An [`Rc<T>`](crate::Rc)
/// / [`RcStr`](crate::RcStr) is only ever produced from a `Local`
/// chunk; an [`Arc<T>`](crate::Arc) /
/// [`ArcStr`](crate::ArcStr) only from a `Shared` chunk.
///
/// Internal — users interact with chunk sharing through the typed method
/// distinctions ([`Arena::alloc_rc`](crate::Arena::alloc_rc) vs
/// [`Arena::alloc_arc`](crate::Arena::alloc_arc)) rather than this
/// enum directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkSharing {
    /// Refcount touched only from the arena's owner thread. Refcount ops
    /// compile to plain `MOV`s on x86 — no `LOCK` prefix.
    Local,
    /// Refcount may be touched from any thread. Uses the standard
    /// `Arc<T>` Acquire/Release ordering pattern.
    Shared,
}
