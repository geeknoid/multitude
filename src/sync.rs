//! Atomic primitives, conditionally re-exported from either [`core::sync::atomic`]
//! (production builds) or [`loom::sync::atomic`] (under `--cfg loom` for
//! model-checked tests).
//!
//! Multitude's correctness depends on the C++/Rust memory ordering of a small
//! number of `AtomicUsize`/`AtomicBool` operations:
//!
//! - `ChunkHeader::ref_count` (the `Shared` arm of the `RefCount` union):
//!   `fetch_add(Relaxed)` for `Arc::clone`, `fetch_sub(Release) + fence(Acquire)`
//!   for `Arc::drop`, `fetch_sub(Release, n)` for the deferred-reconciliation
//!   release at chunk eviction, and `store(Relaxed)` for chunk re-init.
//! - `ArenaInner::outstanding_chunks`: `fetch_add(Relaxed)` per chunk creation,
//!   `fetch_sub(Release)` per chunk free, paired with an `Acquire` fence at
//!   the last-reclaimer point that frees `ArenaInner`.
//! - `ArenaInner::arena_dropped`: `store(Release)` from the arena handle's
//!   `Drop`, paired with `load(Acquire)` from cross-thread teardown deciding
//!   whether to push to the cache.
//!
//! Routing these through this module lets [`tests/loom_*.rs`] verify those
//! orderings under Loom's permutation tester.
#[cfg(loom)]
pub use loom::sync::atomic::{AtomicUsize, Ordering, fence};

#[cfg(not(loom))]
pub use core::sync::atomic::{AtomicUsize, Ordering, fence};
