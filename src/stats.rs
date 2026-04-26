//! [`ArenaStats`] — runtime instrumentation snapshot for an [`Arena`].
//!
//! Returned by [`Arena::stats`](crate::Arena::stats). All counters are
//! lifetime monotonic — they only ever go up over the life of the arena.

use core::cell::Cell;

/// Runtime statistics for an [`Arena`](crate::Arena).
///
/// All fields are lifetime counters: they accumulate over the life of
/// the arena and never decrease. A zero-cost snapshot is returned by
/// [`Arena::stats`](crate::Arena::stats).
///
/// The fields are `pub` because this is a value-semantic data type; the
/// arena owns the running counters internally and hands you a copy.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct ArenaStats {
    /// Total normal-size chunks ever allocated by this arena.
    pub chunks_allocated: u64,

    /// Total oversized stand-alone chunks ever allocated by this arena.
    /// Oversized chunks hold a single allocation that exceeded
    /// `max_normal_alloc`; they are never cached.
    pub oversized_chunks_allocated: u64,

    /// Sum of bytes requested by user allocations (i.e., the `size`
    /// field of each successful allocation's `Layout`). Excludes chunk
    /// headers, alignment padding, and `DropEntry` overhead.
    pub total_bytes_allocated: u64,

    /// Bytes "wasted" as unused tail space when a chunk was rotated out
    /// of `current_local`/`current_shared` because the next allocation
    /// didn't fit.
    ///
    /// Counted only at retirement. Does **not** include slack still
    /// sitting in the chunks the arena is currently bumping into — that
    /// space remains usable. Does not include slack at chunk teardown
    /// (which is conceptually different — those bytes were used to host
    /// values, the slack was just the leftover at retirement).
    pub wasted_tail_bytes: u64,

    /// Number of times an [`ArenaString`](crate::ArenaString) had to be
    /// relocated to a larger buffer because the in-place grow fast path
    /// (extending into the chunk's bump cursor) didn't apply.
    pub string_relocations: u64,

    /// Number of times any [`Allocator::grow`] caller (e.g.,
    /// [`ArenaVec`](crate::ArenaVec) via `allocator-api2::vec::Vec`,
    /// `hashbrown::HashMap`, etc.) had to be relocated because the
    /// in-place fast path didn't apply.
    ///
    /// Excludes [`ArenaString`](crate::ArenaString), which uses its own
    /// internal grow path and is counted under
    /// [`Self::string_relocations`].
    ///
    /// [`Allocator::grow`]: allocator_api2::alloc::Allocator::grow
    pub allocator_relocations: u64,
}

/// Internal per-field storage for the running stats counters. Lives on
/// `ArenaInner`.
///
/// Stored as separate `Cell<u64>` fields (rather than a `Cell<ArenaStats>`)
/// so that bumping a single counter only touches 8 bytes of memory instead
/// of the full 48-byte `ArenaStats` struct. This matters on the hot
/// allocation path, which bumps `total_bytes_allocated` for every alloc.
#[derive(Debug, Default)]
pub(crate) struct StatsStorage {
    pub(crate) chunks_allocated: Cell<u64>,
    pub(crate) oversized_chunks_allocated: Cell<u64>,
    pub(crate) total_bytes_allocated: Cell<u64>,
    pub(crate) wasted_tail_bytes: Cell<u64>,
    pub(crate) string_relocations: Cell<u64>,
    pub(crate) allocator_relocations: Cell<u64>,
}

impl StatsStorage {
    /// Build a public snapshot from the current counter values.
    #[inline]
    #[must_use]
    pub(crate) fn snapshot(&self) -> ArenaStats {
        ArenaStats {
            chunks_allocated: self.chunks_allocated.get(),
            oversized_chunks_allocated: self.oversized_chunks_allocated.get(),
            total_bytes_allocated: self.total_bytes_allocated.get(),
            wasted_tail_bytes: self.wasted_tail_bytes.get(),
            string_relocations: self.string_relocations.get(),
            allocator_relocations: self.allocator_relocations.get(),
        }
    }

    /// Add `delta` to the field selected by `field`. Inlined; expected
    /// to fold into a single load+add+store at the call site.
    #[inline]
    pub(crate) fn add(field: &Cell<u64>, delta: u64) {
        field.set(field.get() + delta);
    }
}
