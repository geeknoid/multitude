//! [`ArenaBuilder`] — fluent constructor for [`Arena`] with all configurable knobs.
//!
//! Use [`ArenaBuilder::new`] (or [`Arena::builder`](crate::Arena::builder)) to
//! start, chain setters, then [`ArenaBuilder::build`] (fallible) or
//! [`ArenaBuilder::build_unwrap`] (panicking).

use core::fmt;

use allocator_api2::alloc::{Allocator, Global};

use crate::Arena;
use crate::arena_inner::ArenaInner;
use crate::chunk_header::ChunkHeader;
use crate::chunk_size_class::ChunkSizeClass;
use crate::constants::{CHUNK_ALIGN, DEFAULT_CHUNK_CACHE_CAPACITY, DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE};

/// Fluent builder for [`Arena`].
///
/// All knobs have sensible defaults. The defaults reproduce
/// `Arena::new()` exactly: 64 KiB normal chunks, oversized cutover at
/// `chunk_size / 4` (= 16 KiB), no preallocation, cache up to 8 chunks,
/// no byte budget.
///
/// # Example
///
/// ```
/// use harena::Arena;
///
/// // A small, capped arena suitable for a per-request workload.
/// let arena = Arena::builder()
///     .chunk_size(16 * 1024)
///     .max_normal_alloc(4 * 1024)
///     .byte_budget(1 * 1024 * 1024)
///     .preallocate(2)
///     .chunk_cache_capacity(4)
///     .build()
///     .expect("valid configuration");
///
/// let v = arena.alloc(42_u32);
/// assert_eq!(*v, 42);
/// ```
pub struct ArenaBuilder<A: Allocator + Clone = Global> {
    chunk_size: usize,
    max_normal_alloc: Option<usize>,
    byte_budget: Option<usize>,
    preallocate: usize,
    chunk_cache_capacity: usize,
    allocator: A,
}

impl ArenaBuilder<Global> {
    /// Start a new builder with default knobs and the [`Global`] allocator.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::new_in(Global)
    }
}

impl<A: Allocator + Clone> ArenaBuilder<A> {
    /// Start a new builder with default knobs and a custom backing
    /// allocator.
    #[must_use]
    pub(crate) fn new_in(allocator: A) -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_normal_alloc: None,
            byte_budget: None,
            preallocate: 0,
            chunk_cache_capacity: DEFAULT_CHUNK_CACHE_CAPACITY,
            allocator,
        }
    }

    /// Set the per-arena chunk size, in bytes.
    ///
    /// Must be a power of two in
    /// `[MIN_CHUNK_SIZE, CHUNK_ALIGN]` (= `[4 KiB, 64 KiB]`); validated
    /// in [`Self::build`]. Defaults to 64 KiB.
    #[must_use]
    pub fn chunk_size(mut self, bytes: usize) -> Self {
        self.chunk_size = bytes;
        self
    }

    /// Set the size threshold (in *worst-case* bytes — payload size +
    /// alignment padding + optional `DropEntry` overhead) above which an
    /// allocation is routed to its own oversized stand-alone chunk.
    ///
    /// Defaults to `chunk_size / 4`. Must leave room for at least one
    /// such allocation in a normal chunk after the chunk header; the
    /// builder validates this in [`Self::build`].
    #[must_use]
    pub fn max_normal_alloc(mut self, bytes: usize) -> Self {
        self.max_normal_alloc = Some(bytes);
        self
    }

    /// Set the lifetime fuel cap on total chunk bytes ever allocated
    /// through this arena.
    ///
    /// This is **monotonic**: it counts the `total_size` of every chunk
    /// the arena ever asks the underlying allocator for, including
    /// chunks that were later freed. It is **not** a peak-live-bytes
    /// cap — see crate-level docs for why.
    ///
    /// When a new chunk allocation would push the running total past
    /// the budget, allocation fails with [`AllocError`]. Defaults to no
    /// budget (unlimited).
    ///
    /// [`AllocError`]: allocator_api2::alloc::AllocError
    #[must_use]
    pub fn byte_budget(mut self, bytes: usize) -> Self {
        self.byte_budget = Some(bytes);
        self
    }

    /// Allocate `chunks` normal chunks eagerly during [`Self::build`]
    /// and seed the cache with them. Eliminates first-allocation
    /// latency for the first `chunks` chunk-rotations. Must be
    /// `≤ chunk_cache_capacity`.
    #[must_use]
    pub fn preallocate(mut self, chunks: usize) -> Self {
        self.preallocate = chunks;
        self
    }

    /// Cap on the number of empty normal chunks the arena will hold in
    /// its free-list for reuse. `0` disables caching. Defaults to 8.
    #[must_use]
    pub fn chunk_cache_capacity(mut self, chunks: usize) -> Self {
        self.chunk_cache_capacity = chunks;
        self
    }

    /// Replace the backing allocator. Returns a builder over the new
    /// allocator type with all other settings preserved.
    #[must_use]
    pub fn allocator_in<A2: Allocator + Clone>(self, allocator: A2) -> ArenaBuilder<A2> {
        ArenaBuilder {
            chunk_size: self.chunk_size,
            max_normal_alloc: self.max_normal_alloc,
            byte_budget: self.byte_budget,
            preallocate: self.preallocate,
            chunk_cache_capacity: self.chunk_cache_capacity,
            allocator,
        }
    }

    /// Validate the configuration, optionally preallocate, and return
    /// the [`Arena`].
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] for any constraint violation or for
    /// preallocation failure.
    pub fn build(self) -> Result<Arena<A>, BuildError> {
        // ---- validate chunk_size ---------------------------------------
        if !self.chunk_size.is_power_of_two() {
            return Err(BuildError::ChunkSizeNotPowerOfTwo);
        }
        if self.chunk_size < MIN_CHUNK_SIZE || self.chunk_size > CHUNK_ALIGN {
            return Err(BuildError::ChunkSizeOutOfRange);
        }

        // ---- validate max_normal_alloc ---------------------------------
        let header_padded = ChunkHeader::<A>::header_padded_size();
        // The chunk has `chunk_size - header_padded` bytes of usable space
        // after the header is placed. An allocation classified as "normal"
        // must, in the worst case, fit in that space.
        let usable = self.chunk_size.checked_sub(header_padded).ok_or(BuildError::ChunkSizeOutOfRange)?;
        let max_normal_alloc = self.max_normal_alloc.unwrap_or(self.chunk_size / 4);
        if max_normal_alloc == 0 || max_normal_alloc > usable {
            return Err(BuildError::MaxNormalAllocTooLarge);
        }

        // ---- validate preallocate vs cache capacity --------------------
        if self.preallocate > self.chunk_cache_capacity {
            return Err(BuildError::PreallocateExceedsCache);
        }

        // ---- build the arena ------------------------------------------
        let inner = ArenaInner::new_with_config(
            self.allocator,
            self.chunk_size,
            max_normal_alloc,
            self.byte_budget,
            self.chunk_cache_capacity,
        );
        let arena = Arena::from_inner(inner);

        // ---- preallocate ----------------------------------------------
        for _ in 0..self.preallocate {
            // Allocate a fresh chunk via the arena's normal path and
            // immediately push it into the cache (refcount drops to 0,
            // teardown happens via dec_ref → cache).
            match arena.preallocate_one_into_cache() {
                Ok(()) => {}
                Err(_) => return Err(BuildError::AllocFailed),
            }
        }
        Ok(arena)
    }

    /// Validate the configuration, optionally preallocate, and return the
    /// [`Arena`], panicking instead of returning an error on any validation or
    /// preallocation failure.
    ///
    /// # Panics
    ///
    /// Panics with the [`BuildError`] message on any validation or
    /// preallocation failure.
    #[must_use]
    pub fn build_unwrap(self) -> Arena<A> {
        match self.build() {
            Ok(a) => a,
            Err(e) => panic_build(e),
        }
    }
}

impl<A: Allocator + Clone> fmt::Debug for ArenaBuilder<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArenaBuilder")
            .field("chunk_size", &self.chunk_size)
            .field("max_normal_alloc", &self.max_normal_alloc)
            .field("byte_budget", &self.byte_budget)
            .field("preallocate", &self.preallocate)
            .field("chunk_cache_capacity", &self.chunk_cache_capacity)
            .finish_non_exhaustive()
    }
}

/// Reasons [`ArenaBuilder::build`] may reject a configuration.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum BuildError {
    /// `chunk_size` was not a power of two.
    ChunkSizeNotPowerOfTwo,
    /// `chunk_size` was outside `[MIN_CHUNK_SIZE, CHUNK_ALIGN]`
    /// (= `[4 KiB, 64 KiB]`).
    ChunkSizeOutOfRange,
    /// `max_normal_alloc` was zero, or larger than the usable space in
    /// a normal chunk after the chunk header.
    MaxNormalAllocTooLarge,
    /// `preallocate` was larger than `chunk_cache_capacity` (would
    /// allocate chunks that immediately get freed).
    PreallocateExceedsCache,
    /// Preallocation hit the underlying allocator's failure path.
    AllocFailed,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::ChunkSizeNotPowerOfTwo => "chunk_size must be a power of two",
            Self::ChunkSizeOutOfRange => "chunk_size must be in [MIN_CHUNK_SIZE, CHUNK_ALIGN]",
            Self::MaxNormalAllocTooLarge => "max_normal_alloc is zero or larger than the usable bytes in a chunk",
            Self::PreallocateExceedsCache => "preallocate must be ≤ chunk_cache_capacity",
            Self::AllocFailed => "preallocation failed",
        };
        f.write_str(msg)
    }
}

impl core::error::Error for BuildError {}

#[inline(never)]
#[cold]
#[expect(clippy::panic, reason = "build_unwrap is the explicit panicking entry point")]
fn panic_build(e: BuildError) -> ! {
    panic!("harena::ArenaBuilder::build_unwrap: {e}");
}

// Internal hooks used by ArenaBuilder. Implemented on Arena in arena.rs to
// keep the per-type module organization clean.
impl<A: Allocator + Clone> Arena<A> {
    /// Allocate one fresh normal chunk (counted in stats and budget) and
    /// stash it in the cache as an empty, refcount==0 chunk. Used by
    /// [`ArenaBuilder::preallocate`].
    pub(crate) fn preallocate_one_into_cache(&self) -> Result<(), allocator_api2::alloc::AllocError> {
        // SAFETY: Arena state is owner-thread-only.
        let inner = unsafe { self.inner_ref() };
        let chunk = self.try_alloc_fresh_chunk_normal()?;
        // Chunk was just created with refcount=1 (the arena's transient
        // hold). Drop it back to 0 and push into the cache.
        // SAFETY: chunk is alive; we own the only reference.
        unsafe {
            // Drop list is empty (fresh chunk), so no shim runs. We
            // bypass `teardown_chunk`'s drop walk and go straight to the
            // cache push.
            let header = chunk.as_ref();
            // Reset bump+drop_head to fresh state (already are, but be
            // explicit so we match the cache push contract).
            header.reset();
            header.ref_count.store(0, core::sync::atomic::Ordering::Relaxed);
            // Try to cache; if cache full (shouldn't happen because the
            // builder validated preallocate ≤ cap), free instead.
            if !inner.try_push_to_cache(chunk) {
                crate::chunk_header::free_chunk(chunk);
            }
        }
        // The chunk we created was a Normal chunk; the stat was already
        // incremented by `try_alloc_fresh_chunk_normal`. Nothing else to do.
        let _ = ChunkSizeClass::Normal; // silence unused import in some configs
        Ok(())
    }
}
