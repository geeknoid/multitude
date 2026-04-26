//! Crate-wide constants and small helpers.

/// Alignment, in bytes, of every chunk's base address.
///
/// 64 KiB. The chunk header for any pointer in a chunk is recovered by
/// masking off the low 16 bits of the pointer's address.
///
/// This is a compile-time constant because handles store only the value
/// pointer and must be able to find the chunk header without knowing
/// per-arena state. Per-arena `chunk_size` (configured via
/// [`ArenaBuilder`](crate::ArenaBuilder)) can be smaller than this — chunks
/// smaller than `CHUNK_ALIGN` waste virtual address space alignment but
/// no real memory, and the same mask still recovers the header.
pub(crate) const CHUNK_ALIGN: usize = 64 * 1024;

/// Default per-arena chunk size, in bytes.
pub(crate) const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Minimum per-arena chunk size that the [`ArenaBuilder`](crate::ArenaBuilder)
/// will accept.
pub(crate) const MIN_CHUNK_SIZE: usize = 4 * 1024;

/// Default cap on the number of empty normal chunks the arena will
/// hold in its free-list for reuse.
pub(crate) const DEFAULT_CHUNK_CACHE_CAPACITY: usize = 8;

/// The starting alignment of every chunk's bump cursor (post-header).
/// Per-allocation alignment takes care of higher-aligned `T`s.
pub(crate) const MAX_INITIAL_ALIGN: usize = 8;

/// Round `addr` up to the next multiple of `align` (which must be a power
/// of 2). Wraps on overflow — callers should `checked_add` first if that
/// matters.
#[inline]
#[must_use]
pub(crate) const fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

/// Same as [`align_up`] but checked for overflow.
#[inline]
#[must_use]
pub(crate) const fn checked_align_up(addr: usize, align: usize) -> Option<usize> {
    match addr.checked_add(align - 1) {
        Some(v) => Some(v & !(align - 1)),
        None => None,
    }
}
