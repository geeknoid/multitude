//! `ChunkSharing` — internal marker for a chunk's refcount flavor.

/// Whether a chunk's refcount is touched only from the arena's owner
/// thread (`Local`, non-atomic-equivalent ops) or potentially from any
/// thread (`Shared`, atomic Acquire/Release ops).
///
/// Fixed at chunk birth and never changes. An [`ArenaRc<T>`](crate::ArenaRc)
/// / [`ArenaRcStr`](crate::ArenaRcStr) is only ever produced from a `Local`
/// chunk; an [`ArenaArc<T>`](crate::ArenaArc) /
/// [`ArenaArcStr`](crate::ArenaArcStr) only from a `Shared` chunk.
///
/// Internal — users interact with chunk sharing through the typed method
/// distinctions ([`Arena::alloc`](crate::Arena::alloc) vs
/// [`Arena::alloc_shared`](crate::Arena::alloc_shared)) rather than this
/// enum directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkSharing {
    /// Refcount touched only from the arena's owner thread. Refcount ops
    /// compile to plain `MOV`s on x86 — no `LOCK` prefix.
    Local,
    /// Refcount may be touched from any thread. Uses the standard
    /// `Arc<T>` Acquire/Release ordering pattern.
    Shared,
}
