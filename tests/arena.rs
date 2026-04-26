//! Tests for the [`Arena`] type itself: constructors, builder, stats,
//! cache behavior, byte budget, preallocation.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std for thread/sync primitives")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::large_stack_arrays, reason = "test allocations are intentional")]
#![allow(clippy::collection_is_never_read, reason = "tests retain handles to keep chunks alive")]
#![allow(clippy::manual_assert, reason = "explicit panic clarifies safety-net intent")]

use harena::Arena;

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

#[test]
fn new_creates_empty_arena() {
    let arena = Arena::new();
    assert_eq!(arena.stats().chunks_allocated, 0);
}

#[test]
fn default_works() {
    let arena: Arena = Arena::default();
    let v = arena.alloc(42_u32);
    assert_eq!(*v, 42);
}

#[test]
fn allocator_accessor() {
    let arena = Arena::new();
    let _: &allocator_api2::alloc::Global = arena.allocator();
}

#[test]
fn debug_format_includes_stats() {
    let arena = Arena::new();
    let _ = arena.alloc(1_u8);
    let s = format!("{arena:?}");
    assert!(s.contains("Arena"));
    assert!(s.contains("stats"));
}

#[test]
fn new_in_with_global() {
    let arena: Arena<allocator_api2::alloc::Global> = Arena::new_in(allocator_api2::alloc::Global);
    let v = arena.alloc(7_i32);
    assert_eq!(*v, 7);
}

#[test]
fn builder_in_with_global() {
    let arena = Arena::builder_in(allocator_api2::alloc::Global).build_unwrap();
    let v = arena.alloc(7_i32);
    assert_eq!(*v, 7);
}

// ---------------------------------------------------------------------------
// Builder validation
// ---------------------------------------------------------------------------

#[test]
fn builder_default_matches_arena_new() {
    let a = Arena::builder().build_unwrap();
    let b = Arena::new();
    assert_eq!(a.stats().chunks_allocated, 0);
    assert_eq!(b.stats().chunks_allocated, 0);
    let _ = a.alloc(0_u32);
    let _ = b.alloc(0_u32);
    assert_eq!(a.stats().chunks_allocated, 1);
    assert_eq!(b.stats().chunks_allocated, 1);
}

#[test]
fn builder_chunk_size_not_power_of_two_rejected() {
    let err = Arena::builder().chunk_size(5000).build().unwrap_err();
    assert_eq!(err, harena::BuildError::ChunkSizeNotPowerOfTwo);
}

#[test]
fn builder_chunk_size_too_small_rejected() {
    let err = Arena::builder().chunk_size(1024).build().unwrap_err();
    assert_eq!(err, harena::BuildError::ChunkSizeOutOfRange);
}

#[test]
fn builder_chunk_size_too_large_rejected() {
    let err = Arena::builder().chunk_size(128 * 1024).build().unwrap_err();
    assert_eq!(err, harena::BuildError::ChunkSizeOutOfRange);
}

#[test]
fn builder_max_normal_alloc_zero_rejected() {
    let err = Arena::builder().max_normal_alloc(0).build().unwrap_err();
    assert_eq!(err, harena::BuildError::MaxNormalAllocTooLarge);
}

#[test]
fn builder_max_normal_alloc_too_large_rejected() {
    let err = Arena::builder().max_normal_alloc(64 * 1024).build().unwrap_err();
    assert_eq!(err, harena::BuildError::MaxNormalAllocTooLarge);
}

#[test]
fn builder_preallocate_exceeds_cache_rejected() {
    let err = Arena::builder().chunk_cache_capacity(2).preallocate(3).build().unwrap_err();
    assert_eq!(err, harena::BuildError::PreallocateExceedsCache);
}

#[test]
fn builder_small_chunk_size_works() {
    // 4 KiB chunks are tiny but legal. The mask trick must still recover
    // the chunk header from a value pointer in a 4 KiB chunk (chunks
    // are still 64 KiB-aligned, just smaller in actual size).
    let arena = Arena::builder().chunk_size(4 * 1024).build_unwrap();
    let v = arena.alloc(42_u32);
    assert_eq!(*v, 42);
    let v2 = arena.alloc(99_u32);
    assert_eq!(*v2, 99);
}

#[test]
fn small_chunk_size_round_trip_many_allocs() {
    let arena = Arena::builder().chunk_size(8 * 1024).build_unwrap();
    let mut handles = Vec::new();
    for i in 0..1000_u32 {
        handles.push(arena.alloc(i));
    }
    drop(handles);
    let h = arena.alloc(99_u32);
    assert_eq!(*h, 99);
}

// ---------------------------------------------------------------------------
// Byte budget
// ---------------------------------------------------------------------------

#[test]
fn byte_budget_caps_total_chunk_bytes() {
    let arena = Arena::builder().chunk_size(4 * 1024).byte_budget(4 * 1024).build_unwrap();
    let mut handles = Vec::new();
    let mut iters = 0_u32;
    loop {
        iters += 1;
        match arena.try_alloc([0_u8; 256]) {
            Ok(h) => handles.push(h),
            Err(_) => break,
        }
        if iters > 1000 {
            panic!("byte_budget did not stop allocations");
        }
    }
    assert_eq!(arena.stats().chunks_allocated, 1);
    assert!(handles.len() < 16);
}

// ---------------------------------------------------------------------------
// Cache + preallocation
// ---------------------------------------------------------------------------

#[test]
fn cache_reuse() {
    let arena = Arena::builder().chunk_cache_capacity(4).build_unwrap();
    let mut handles = Vec::new();
    for i in 0..30_000_u64 {
        handles.push(arena.alloc(i));
    }
    let stats = arena.stats();
    assert!(stats.chunks_allocated >= 2);
    drop(handles);
    let chunks_before = arena.stats().chunks_allocated;
    let _v = arena.alloc(0_u64);
    let chunks_after = arena.stats().chunks_allocated;
    assert_eq!(chunks_after, chunks_before, "expected cache hit");
}

#[test]
fn cache_capacity_zero_disables_caching() {
    let arena = Arena::builder().chunk_cache_capacity(0).build_unwrap();
    let mut handles = Vec::new();
    for i in 0..30_000_u64 {
        handles.push(arena.alloc(i));
    }
    let n_chunks = arena.stats().chunks_allocated;
    assert!(n_chunks >= 2);
    drop(handles);
    let _ = arena.alloc(0_u64);
    let mut handles2 = Vec::new();
    for i in 0..30_000_u64 {
        handles2.push(arena.alloc(i));
    }
    let n_chunks_after = arena.stats().chunks_allocated;
    assert!(n_chunks_after > n_chunks);
}

#[test]
fn preallocate_skips_underlying_allocation_calls() {
    let arena = Arena::builder().preallocate(2).chunk_cache_capacity(4).build_unwrap();
    assert_eq!(arena.stats().chunks_allocated, 2);
    let _ = arena.alloc(0_u32);
    assert_eq!(arena.stats().chunks_allocated, 2);
    let mut handles = Vec::new();
    for i in 0..10_000_u64 {
        handles.push(arena.alloc(i));
    }
    assert!(arena.stats().chunks_allocated >= 2);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

#[test]
fn chunks_allocated_grows_on_first_alloc() {
    let arena = Arena::new();
    assert_eq!(arena.stats().chunks_allocated, 0);
    let _a = arena.alloc(0_u8);
    assert_eq!(arena.stats().chunks_allocated, 1);
}

#[test]
fn stats_total_bytes_allocated() {
    let arena = Arena::new();
    let _a = arena.alloc(0_u64);
    let _b = arena.alloc(0_u32);
    assert!(arena.stats().total_bytes_allocated >= 12);
}

#[test]
fn stats_oversized_chunks_counted() {
    // Default max_normal_alloc is 16 KiB; a 32 KiB allocation goes oversized.
    let arena = Arena::new();
    let _big = arena.alloc_slice_copy(&[0_u8; 32 * 1024]);
    assert!(arena.stats().oversized_chunks_allocated >= 1);
}

#[test]
fn stats_wasted_tail_bytes_at_retirement() {
    // Build an arena, fill a chunk so its slack can't fit the next
    // request, then trigger retirement. Retain handles so the chunk
    // doesn't get cached and reused.
    let arena = Arena::builder().chunk_size(4 * 1024).build_unwrap();
    let mut handles = Vec::new();
    for _ in 0..12 {
        handles.push(arena.alloc([0_u8; 256])); // 12 * 256 = 3072 B
    }
    let _h2 = arena.alloc([0_u8; 1024]);
    assert!(arena.stats().wasted_tail_bytes > 0);
}

#[test]
fn stats_string_relocation_counted() {
    let arena = Arena::builder().chunk_size(4 * 1024).build_unwrap();
    let _other = arena.alloc(0_u32);
    let mut s2 = arena.new_string();
    s2.push_str("first");
    let _another = arena.alloc(1_u32); // breaks cursor adjacency
    s2.push_str(&"x".repeat(100));
    assert!(arena.stats().string_relocations >= 1);
}

#[test]
fn stats_allocator_relocations_counted() {
    let arena = Arena::new();
    let mut v = arena.new_vec::<u32>();
    v.push(1);
    let _other = arena.alloc(0_u8); // breaks cursor adjacency
    for i in 0..1000_u32 {
        v.push(i);
    }
    let stats = arena.stats();
    assert!(stats.allocator_relocations >= 1);
    assert_eq!(stats.string_relocations, 0);
}

// ---------------------------------------------------------------------------
// ZST + edge cases
// ---------------------------------------------------------------------------

#[test]
fn alloc_zst_works() {
    #[derive(Debug, PartialEq)]
    struct Zst;
    let arena = Arena::new();
    let r = arena.alloc(Zst);
    assert_eq!(*r, Zst);
}
