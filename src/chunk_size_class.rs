//! `ChunkSizeClass` — internal classification of a chunk's size category.

/// Whether a chunk is a normal 64 KiB shared bump pool or a one-shot
/// oversized chunk holding a single large allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkSizeClass {
    /// 64 KiB chunk used as a shared bump pool for many small/medium
    /// allocations.
    Normal,
    /// A chunk sized to fit one allocation larger than 16 KiB
    /// (`MAX_NORMAL_ALLOC`). Always rounded up to a multiple of
    /// 64 KiB (`CHUNK_ALIGN`).
    Oversized,
}
