use core::fmt;

use allocator_api2::alloc::{Allocator, Global};

use crate::Arena;
use crate::chunk_header::ChunkHeader;
use crate::chunk_sharing::ChunkSharing;
use crate::chunk_size_class::ChunkSizeClass;
use crate::constants::{CHUNK_ALIGN, DEFAULT_CHUNK_CACHE_CAPACITY, DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE, MIN_MAX_NORMAL_ALLOC};

/// Fluent builder for [`Arena`].
///
/// All knobs have sensible defaults. The defaults reproduce
/// `Arena::new()` exactly: 64 KiB normal chunks, oversized cutover at
/// 16 KiB, no preallocation, cache up to 8 chunks, no byte budget.
///
/// # Example
///
/// ```
/// use multitude::Arena;
///
/// // A small, capped arena suitable for a per-request workload.
/// let arena = Arena::builder()
///     .chunk_size(16 * 1024)
///     .max_normal_alloc(4 * 1024)
///     .byte_budget(1 * 1024 * 1024)
///     .preallocate(2)
///     .chunk_cache_capacity(4)
///     .build();
///
/// let v = arena.alloc_rc(42_u32);
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
    #[inline]
    pub const fn new() -> Self {
        Self::new_in(Global)
    }
}

impl Default for ArenaBuilder<Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allocator + Clone> ArenaBuilder<A> {
    /// Start a new builder with default knobs and a custom backing
    /// allocator.
    #[must_use]
    #[inline]
    pub const fn new_in(allocator: A) -> Self {
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
    /// Must be in `[4096, 65536]`. Defaults to 65536. Power-of-two values
    /// typically interact best with the system allocator's bin sizes, but
    /// any value in range is accepted.
    #[must_use]
    #[inline]
    pub const fn chunk_size(mut self, bytes: usize) -> Self {
        self.chunk_size = bytes;
        self
    }

    /// Set the size threshold (in worst-case bytes) above which an
    /// allocation is routed to its own oversized chunk.
    ///
    /// Defaults to `chunk_size / 4`. Validated in [`Self::build`].
    #[must_use]
    #[inline]
    pub const fn max_normal_alloc(mut self, bytes: usize) -> Self {
        self.max_normal_alloc = Some(bytes);
        self
    }

    /// Set the lifetime cap on total chunk bytes ever allocated.
    ///
    /// Monotonic: counts every chunk the arena ever asks the allocator
    /// for, including freed chunks. Not a peak-live-bytes cap.
    ///
    /// When a new chunk allocation would push the running total past
    /// the budget, allocation fails with [`AllocError`]. Defaults to no
    /// budget (unlimited).
    ///
    /// [`AllocError`]: allocator_api2::alloc::AllocError
    #[must_use]
    #[inline]
    pub const fn byte_budget(mut self, bytes: usize) -> Self {
        self.byte_budget = Some(bytes);
        self
    }

    /// Allocate `chunks` normal chunks eagerly during [`Self::build`]
    /// and seed the cache. Must be `≤ chunk_cache_capacity`.
    #[must_use]
    #[inline]
    pub const fn preallocate(mut self, chunks: usize) -> Self {
        self.preallocate = chunks;
        self
    }

    /// Cap on the number of empty chunks the arena will hold for
    /// reuse. `0` disables caching. Defaults to 8.
    #[must_use]
    #[inline]
    pub const fn chunk_cache_capacity(mut self, chunks: usize) -> Self {
        self.chunk_cache_capacity = chunks;
        self
    }

    /// Replace the backing allocator. Returns a builder over the new
    /// allocator type with all other settings preserved.
    #[must_use]
    #[inline]
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
    /// # Panics
    ///
    /// Panics with the [`BuildError`] message on any validation or
    /// preallocation failure. Use [`Self::try_build`] for a fallible
    /// variant.
    #[must_use]
    #[cold]
    pub fn build(self) -> Arena<A> {
        self.try_build().unwrap_or_else(|e| panic_build(e))
    }

    /// Fallible variant of [`Self::build`].
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] for any constraint violation or for
    /// preallocation failure.
    #[cold]
    pub fn try_build(self) -> Result<Arena<A>, BuildError> {
        if self.chunk_size < MIN_CHUNK_SIZE || self.chunk_size > CHUNK_ALIGN {
            return Err(BuildError::ChunkSizeOutOfRange);
        }

        let header_padded = ChunkHeader::<A>::header_padded_size();
        debug_assert!(self.chunk_size > header_padded);
        let usable = self.chunk_size - header_padded;
        let max_normal_alloc = self
            .max_normal_alloc
            .unwrap_or_else(|| (self.chunk_size / 4).max(MIN_MAX_NORMAL_ALLOC));
        if max_normal_alloc < MIN_MAX_NORMAL_ALLOC || max_normal_alloc > usable {
            return Err(BuildError::MaxNormalAllocOutOfRange);
        }

        if self.preallocate > self.chunk_cache_capacity {
            return Err(BuildError::PreallocateExceedsCache);
        }

        let arena = Arena::from_config(
            self.allocator,
            self.chunk_size,
            max_normal_alloc,
            self.byte_budget,
            self.chunk_cache_capacity,
        );

        for _ in 0..self.preallocate {
            match arena.preallocate_one_into_cache() {
                Ok(()) => {}
                Err(_) => return Err(BuildError::AllocFailed),
            }
        }
        Ok(arena)
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
    /// `chunk_size` was outside the supported range (4 KiB to 64 KiB).
    ChunkSizeOutOfRange,
    /// `max_normal_alloc` was outside the supported range — below
    /// 4 KiB or larger than the bytes a chunk can hold after the
    /// per-chunk bookkeeping overhead.
    MaxNormalAllocOutOfRange,
    /// `preallocate` was greater than `chunk_cache_capacity`. Prealloc'd
    /// chunks must fit in the cache.
    PreallocateExceedsCache,
    /// The backing allocator returned an error while preallocating.
    AllocFailed,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::ChunkSizeOutOfRange => "chunk_size must be in [4 KiB, 64 KiB]",
            Self::MaxNormalAllocOutOfRange => "max_normal_alloc must be in [4 KiB, chunk_size - header overhead]",
            Self::PreallocateExceedsCache => "preallocate must be ≤ chunk_cache_capacity",
            Self::AllocFailed => "preallocation failed",
        };
        f.write_str(msg)
    }
}

impl core::error::Error for BuildError {}

#[inline(never)]
#[cold]
#[expect(clippy::panic, reason = "build is the explicit panicking entry point")]
fn panic_build(e: BuildError) -> ! {
    panic!("multitude::ArenaBuilder::build: {e}");
}

// Internal hooks used by ArenaBuilder.
impl<A: Allocator + Clone> Arena<A> {
    /// Allocate one fresh normal chunk and stash it in the cache.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`](allocator_api2::alloc::AllocError) if the backing allocator fails.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    #[cold]
    #[inline(never)]
    pub(crate) fn preallocate_one_into_cache(&self) -> Result<(), allocator_api2::alloc::AllocError> {
        let inner = self.inner_ref();
        let chunk = self.try_alloc_fresh_chunk_normal(ChunkSharing::Local)?;
        // SAFETY: chunk is alive; we own the only reference. Reset its
        // bump cursor and refcount, then hand the retired chunk to the
        // cache.
        unsafe {
            let header_ptr = chunk.as_ptr();
            (*header_ptr).reset();
            ChunkHeader::set_ref_count(chunk, 0);
        }
        // SAFETY: we just set refcount=0 and reset state; the chunk is
        // exclusively owned and ready to be cached.
        let retired = unsafe { crate::owned_chunk::RetiredChunk::from_raw(chunk) };
        let pushed = inner.try_push_to_cache(retired).is_ok();
        debug_assert!(pushed, "preallocate exceeded cache capacity");
        let _ = ChunkSizeClass::Normal;
        Ok(())
    }
}
