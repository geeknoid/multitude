/// Alignment, in bytes, of every chunk's base address (64 KiB).
/// The chunk header is recovered by masking off the low 16 bits.
pub const CHUNK_ALIGN: usize = 64 * 1024;

/// Default per-arena chunk size, in bytes.
pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Minimum per-arena chunk size that the [`ArenaBuilder`](crate::ArenaBuilder)
/// will accept. Sized so that the default
/// `max_normal_alloc = chunk_size / 4` (or the floor below) can fit
/// inside a single chunk after subtracting the per-chunk header
/// overhead.
pub const MIN_CHUNK_SIZE: usize = 8 * 1024;

/// Default cap on empty normal chunks cached for reuse.
pub const DEFAULT_CHUNK_CACHE_CAPACITY: usize = 8;

/// Floor on `max_normal_alloc` enforced by the
/// [`ArenaBuilder`](crate::ArenaBuilder). Setting a lower value would
/// leak the oversized-chunk fallback into the hot path; floor of 4 KiB
/// keeps the oversized check elidable at compile time for all `T`
/// whose worst-case footprint fits in a minimum-sized chunk's normal
/// allocation.
pub const MIN_MAX_NORMAL_ALLOC: usize = 4 * 1024;

/// The starting alignment of every chunk's bump cursor (post-header).
/// Per-allocation alignment takes care of higher-aligned `T`s.
pub const MAX_INITIAL_ALIGN: usize = 8;

/// Initial atomic-refcount value for a `Shared`-flavor `Normal` chunk
/// under the deferred-reconciliation refcount scheme. The chunk's
/// atomic `ref_count` is initialized here at chunk creation; each
/// `alloc_arc` then increments `chunk.arcs_issued` non-atomically
/// instead of doing a per-allocation LOCK RMW. On eviction, one
/// `fetch_sub(LARGE - arcs_issued)` reconciles, leaving `ref_count`
/// equal to the count of outstanding live `Arc`s.
///
/// `isize::MAX as usize / 2` (≈ 4.6 × 10¹⁸ on 64-bit) gives plenty of
/// headroom: `Arc::clone` can climb to `MAX_REFCOUNT = isize::MAX`
/// before tripping the overflow guard. A chunk's max plausible
/// `arcs_issued` is `chunk_size / min_alloc_size` (~8 K for a 64 KiB
/// chunk holding `Arc<()>`-sized payloads); LARGE leaves ~14 orders
/// of magnitude of safety.
pub const LARGE_INITIAL_SHARED_REFCOUNT: usize = (isize::MAX as usize) / 2;

/// Round `addr` up to the next multiple of `align` (power of 2).
#[inline]
#[must_use]
pub const fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

/// Same as [`align_up`] but checked for overflow.
#[inline]
#[must_use]
pub const fn checked_align_up(addr: usize, align: usize) -> Option<usize> {
    match addr.checked_add(align - 1) {
        Some(v) => Some(v & !(align - 1)),
        None => None,
    }
}

/// Padded size of the chunk header.
#[inline]
#[must_use]
pub const fn padded_header_size(header_size: usize) -> usize {
    align_up(header_size, MAX_INITIAL_ALIGN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_align_up_some() {
        assert_eq!(checked_align_up(0, 8), Some(0));
        assert_eq!(checked_align_up(1, 8), Some(8));
        assert_eq!(checked_align_up(8, 8), Some(8));
        assert_eq!(checked_align_up(9, 8), Some(16));
    }

    #[test]
    fn checked_align_up_overflow_returns_none() {
        assert_eq!(checked_align_up(usize::MAX, 8), None);
        assert_eq!(checked_align_up(usize::MAX - 6, 8), None);
    }

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(15, 16), 16);
    }

    #[test]
    fn padded_header_size_rounds_up() {
        assert_eq!(padded_header_size(1), MAX_INITIAL_ALIGN);
        assert_eq!(padded_header_size(MAX_INITIAL_ALIGN), MAX_INITIAL_ALIGN);
        assert_eq!(padded_header_size(MAX_INITIAL_ALIGN + 1), 2 * MAX_INITIAL_ALIGN);
    }
}
