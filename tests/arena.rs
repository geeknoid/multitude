#![cfg(feature = "builders")]
//! Tests for the [`Arena`] type itself: constructors, builder, stats,
//! cache behavior, byte budget, preallocation.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std for thread/sync primitives")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::large_stack_arrays, reason = "test allocations are intentional")]
#![allow(clippy::collection_is_never_read, reason = "tests retain smart pointers to keep chunks alive")]
#![allow(clippy::manual_assert, reason = "explicit panic clarifies safety-net intent")]
#![allow(clippy::cast_possible_truncation, reason = "test code: bounded indices fit in u32")]
#![allow(clippy::needless_borrows_for_generic_args, reason = "explicit borrows clarify intent in tests")]

use multitude::Arena;

mod common;

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

#[cfg(feature = "stats")]
#[test]
fn new_does_not_eagerly_allocate_chunk() {
    // Sentinel-based slots remove the need to pre-allocate a Local
    // chunk at construction; the first allocation lazily pulls one in.
    let arena = Arena::new();
    assert_eq!(arena.stats().chunks_allocated, 0);
    let _a = arena.alloc_rc(0_u8);
    assert_eq!(arena.stats().chunks_allocated, 1);
}

#[test]
fn default_works() {
    let arena: Arena = Arena::default();
    let v = arena.alloc_rc(42_u32);
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
    let _ = arena.alloc_rc(1_u8);
    let s = format!("{arena:?}");
    assert!(s.contains("Arena"));
    #[cfg(feature = "stats")]
    assert!(s.contains("stats"));
}

#[test]
fn new_in_with_global() {
    let arena: Arena<allocator_api2::alloc::Global> = Arena::new_in(allocator_api2::alloc::Global);
    let v = arena.alloc_rc(7_i32);
    assert_eq!(*v, 7);
}

#[test]
fn builder_in_with_global() {
    let arena = Arena::builder_in(allocator_api2::alloc::Global).build();
    let v = arena.alloc_rc(7_i32);
    assert_eq!(*v, 7);
}

// ---------------------------------------------------------------------------
// Builder validation
// ---------------------------------------------------------------------------

#[cfg(feature = "stats")]
#[test]
fn builder_default_matches_arena_new() {
    let a = Arena::builder().build();
    let b = Arena::new();
    // Sentinel-based slots: no chunk is allocated until the first user request.
    assert_eq!(a.stats().chunks_allocated, 0);
    assert_eq!(b.stats().chunks_allocated, 0);
    let _ = a.alloc_rc(0_u32);
    let _ = b.alloc_rc(0_u32);
    assert_eq!(a.stats().chunks_allocated, 1);
    assert_eq!(b.stats().chunks_allocated, 1);
}

#[test]
fn builder_default_impl() {
    // Drives `<ArenaBuilder<Global> as Default>::default()`.
    let builder = multitude::ArenaBuilder::default();
    let arena = builder.build();
    let v = arena.alloc_rc(99_u32);
    assert_eq!(*v, 99);
}

#[test]
fn builder_chunk_size_too_small_rejected() {
    let err = Arena::builder().chunk_size(1024).try_build().unwrap_err();
    assert_eq!(err, multitude::BuildError::ChunkSizeOutOfRange);
}

#[test]
fn builder_chunk_size_too_large_rejected() {
    let err = Arena::builder().chunk_size(128 * 1024).try_build().unwrap_err();
    assert_eq!(err, multitude::BuildError::ChunkSizeOutOfRange);
}

#[test]
fn builder_max_normal_alloc_zero_rejected() {
    let err = Arena::builder().max_normal_alloc(0).try_build().unwrap_err();
    assert_eq!(err, multitude::BuildError::MaxNormalAllocOutOfRange);
}

#[test]
fn builder_max_normal_alloc_below_min_rejected() {
    // Anything below the 4 KiB floor is rejected.
    let err = Arena::builder().max_normal_alloc(2048).try_build().unwrap_err();
    assert_eq!(err, multitude::BuildError::MaxNormalAllocOutOfRange);
}

#[test]
fn builder_max_normal_alloc_too_large_rejected() {
    let err = Arena::builder().max_normal_alloc(64 * 1024).try_build().unwrap_err();
    assert_eq!(err, multitude::BuildError::MaxNormalAllocOutOfRange);
}

#[test]
fn builder_preallocate_exceeds_cache_rejected() {
    let err = Arena::builder().chunk_cache_capacity(2).preallocate(3).try_build().unwrap_err();
    assert_eq!(err, multitude::BuildError::PreallocateExceedsCache);
}

#[test]
fn builder_small_chunk_size_works() {
    // 4 KiB chunks are tiny but legal. The mask trick must still recover
    // the chunk header from a value pointer in a 4 KiB chunk (chunks
    // are still 64 KiB-aligned, just smaller in actual size).
    let arena = Arena::builder().chunk_size(8 * 1024).build();
    let v = arena.alloc_rc(42_u32);
    assert_eq!(*v, 42);
    let v2 = arena.alloc_rc(99_u32);
    assert_eq!(*v2, 99);
}

#[test]
fn small_chunk_size_round_trip_many_allocs() {
    let arena = Arena::builder().chunk_size(8 * 1024).build();
    let mut handles = std::vec::Vec::new();
    for i in 0..1000_u32 {
        handles.push(arena.alloc_rc(i));
    }
    drop(handles);
    let h = arena.alloc_rc(99_u32);
    assert_eq!(*h, 99);
}

// ---------------------------------------------------------------------------
// Byte budget
// ---------------------------------------------------------------------------

#[cfg(feature = "stats")]
#[test]
fn byte_budget_caps_total_chunk_bytes() {
    let arena = Arena::builder().chunk_size(8 * 1024).byte_budget(8 * 1024).build();
    let mut handles = std::vec::Vec::new();
    let mut iters = 0_u32;
    loop {
        iters += 1;
        match arena.try_alloc_rc([0_u8; 256]) {
            Ok(h) => handles.push(h),
            Err(_) => break,
        }
        if iters > 1000 {
            panic!("byte_budget did not stop allocations");
        }
    }
    assert_eq!(arena.stats().chunks_allocated, 1);
    assert!(handles.len() < 32);
}

// ---------------------------------------------------------------------------
// Cache + preallocation
// ---------------------------------------------------------------------------

#[cfg(feature = "stats")]
#[test]
fn cache_reuse() {
    let arena = Arena::builder().chunk_cache_capacity(4).build();
    let mut handles = std::vec::Vec::new();
    for i in 0..30_000_u64 {
        handles.push(arena.alloc_rc(i));
    }
    let stats = arena.stats();
    assert!(stats.chunks_allocated >= 2);
    drop(handles);
    let chunks_before = arena.stats().chunks_allocated;
    let _v = arena.alloc_rc(0_u64);
    let chunks_after = arena.stats().chunks_allocated;
    assert_eq!(chunks_after, chunks_before, "expected cache hit");
}

#[cfg(feature = "stats")]
#[test]
fn cache_capacity_zero_disables_caching() {
    let arena = Arena::builder().chunk_cache_capacity(0).build();
    let mut handles = std::vec::Vec::new();
    for i in 0..30_000_u64 {
        handles.push(arena.alloc_rc(i));
    }
    let n_chunks = arena.stats().chunks_allocated;
    assert!(n_chunks >= 2);
    drop(handles);
    let _ = arena.alloc_rc(0_u64);
    let mut handles2 = std::vec::Vec::new();
    for i in 0..30_000_u64 {
        handles2.push(arena.alloc_rc(i));
    }
    let n_chunks_after = arena.stats().chunks_allocated;
    assert!(n_chunks_after > n_chunks);
}

#[cfg(feature = "stats")]
#[test]
fn preallocate_skips_underlying_allocation_calls() {
    // 2 preallocated cache chunks; no eager Local chunk under
    // sentinel-slot construction.
    let arena = Arena::builder().preallocate(2).chunk_cache_capacity(4).build();
    assert_eq!(arena.stats().chunks_allocated, 2);
    // First user alloc pulls from the cache — no new chunk allocated.
    let _ = arena.alloc_rc(0_u32);
    assert_eq!(arena.stats().chunks_allocated, 2);
    let mut handles = std::vec::Vec::new();
    for i in 0..10_000_u64 {
        handles.push(arena.alloc_rc(i));
    }
    assert!(arena.stats().chunks_allocated >= 2);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

#[cfg(feature = "stats")]
#[test]
fn chunks_allocated_first_user_alloc_creates_chunk() {
    // Sentinel-based slots: no chunk is allocated until first user request.
    let arena = Arena::new();
    assert_eq!(arena.stats().chunks_allocated, 0);
    let _a = arena.alloc_rc(0_u8);
    assert_eq!(arena.stats().chunks_allocated, 1);
}

#[cfg(feature = "stats")]
#[test]
fn stats_total_bytes_allocated() {
    let arena = Arena::new();
    let _a = arena.alloc_rc(0_u64);
    let _b = arena.alloc_rc(0_u32);
    assert!(arena.stats().total_bytes_allocated >= 12);
}

#[cfg(feature = "stats")]
#[test]
fn stats_oversized_chunks_counted() {
    // Default max_normal_alloc is 16 KiB; a 32 KiB allocation goes oversized.
    let arena = Arena::new();
    let _big = arena.alloc_slice_copy_rc([0_u8; 32 * 1024]);
    assert!(arena.stats().oversized_chunks_allocated >= 1);
}

#[cfg(feature = "stats")]
#[test]
fn stats_wasted_tail_bytes_at_retirement() {
    // Build an arena, fill a chunk so its slack can't fit the next
    // request, then trigger retirement. Retain smart pointers so the chunk
    // doesn't get cached and reused.
    let arena = Arena::builder().chunk_size(8 * 1024).build();
    let mut handles = std::vec::Vec::new();
    for _ in 0..28 {
        handles.push(arena.alloc_rc([0_u8; 256])); // 28 * 256 = 7168 B
    }
    let _h2 = arena.alloc_rc([0_u8; 2048]);
    assert!(arena.stats().wasted_tail_bytes > 0);
}

#[cfg(feature = "stats")]
#[test]
fn stats_string_relocation_counted() {
    let arena = Arena::builder().chunk_size(8 * 1024).build();
    let _other = arena.alloc_rc(0_u32);
    let mut s2 = arena.alloc_string();
    s2.push_str("first");
    let _another = arena.alloc_rc(1_u32); // breaks cursor adjacency
    s2.push_str("x".repeat(100));
    assert!(arena.stats().relocations >= 1);
}

#[cfg(feature = "stats")]
#[test]
fn stats_vec_relocation_counted() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
    v.push(1);
    let _other = arena.alloc_rc(0_u8); // breaks cursor adjacency
    for i in 0..1000_u32 {
        v.push(i);
    }
    let stats = arena.stats();
    assert!(stats.relocations >= 1);
}

// ---------------------------------------------------------------------------
// ZST + edge cases
// ---------------------------------------------------------------------------

#[test]
fn alloc_zst_works() {
    #[derive(Debug, PartialEq)]
    struct Zst;
    let arena = Arena::new();
    let r = arena.alloc_rc(Zst);
    assert_eq!(*r, Zst);
}

// ---------------------------------------------------------------------------
// String-smart pointer convenience methods on Arena (try_* variants of alloc_str*)
// ---------------------------------------------------------------------------

#[test]
fn try_alloc_str_returns_mutable_str() {
    let arena = Arena::new();
    let s: &mut str = arena.try_alloc_str("hello").unwrap();
    s.make_ascii_uppercase();
    assert_eq!(s, "HELLO");
}

#[test]
fn try_alloc_str_rc_returns_handle() {
    let arena = Arena::new();
    let s = arena.try_alloc_str_rc("rc").unwrap();
    assert_eq!(&*s, "rc");
}

#[test]
fn try_alloc_str_arc_returns_handle() {
    let arena = Arena::new();
    let s = arena.try_alloc_str_arc("arc").unwrap();
    assert_eq!(&*s, "arc");
}

#[test]
fn try_alloc_str_accepts_string() {
    // impl AsRef<str> covers both &str and String.
    let arena = Arena::new();
    let owned = std::string::String::from("from String");
    let s: &mut str = arena.try_alloc_str(owned).unwrap();
    assert_eq!(s, "from String");
}

// ---------------------------------------------------------------------------
// Collection factories — fallible variants
// ---------------------------------------------------------------------------

#[test]
fn try_alloc_string_with_capacity_succeeds() {
    let arena = Arena::new();
    let mut s = arena.try_alloc_string_with_capacity(64).unwrap();
    s.push_str("preallocated");
    assert!(s.capacity() >= 64);
    assert_eq!(s.as_str(), "preallocated");
}

#[test]
fn try_alloc_string_with_capacity_zero_works() {
    let arena = Arena::new();
    let s = arena.try_alloc_string_with_capacity(0).unwrap();
    assert_eq!(s.capacity(), 0);
    assert_eq!(s.len(), 0);
}

#[test]
fn try_alloc_vec_with_capacity_succeeds() {
    let arena = Arena::new();
    let mut v = arena.try_alloc_vec_with_capacity::<u32>(50).unwrap();
    for i in 0..50 {
        v.push(i);
    }
    assert!(v.capacity() >= 50);
    assert_eq!(v.len(), 50);
}

#[test]
fn try_alloc_vec_with_capacity_zero_works() {
    let arena = Arena::new();
    let v: multitude::builders::Vec<u8, _> = arena.try_alloc_vec_with_capacity(0).unwrap();
    assert_eq!(v.capacity(), 0);
    assert_eq!(v.len(), 0);
}

#[test]
fn oversized_bump_alloc_does_not_leak_on_drop() {
    let alloc = common::TrackingAllocator::new();
    {
        let arena = Arena::builder_in(alloc.clone()).chunk_size(64 * 1024).build();
        // `max_normal_alloc` is `chunk_size / 4` = 16 KiB, so this 32 KiB
        // allocation must take the oversized chunk path and was previously
        // leaked because the chunk was never linked into current_*, the
        // pinned list, or the cache.
        let _slice = arena.alloc_slice_copy(&[0_u8; 32 * 1024]);
        assert!(alloc.live_chunks() >= 1);
    }
    assert_eq!(alloc.live_chunks(), 0, "arena drop must free all chunks");
    assert_eq!(alloc.live_bytes(), 0);
}

#[test]
fn oversized_bump_alloc_does_not_leak_on_reset() {
    let alloc = common::TrackingAllocator::new();
    let mut arena = Arena::builder_in(alloc.clone()).chunk_size(64 * 1024).build();
    let _ = arena.alloc_slice_copy(&[0_u8; 32 * 1024]);
    let after_alloc = alloc.live_chunks();
    arena.reset();
    assert!(
        alloc.live_chunks() < after_alloc,
        "reset must release oversized chunks (had {after_alloc}, now {})",
        alloc.live_chunks()
    );
    drop(arena);
    assert_eq!(alloc.live_chunks(), 0);
    assert_eq!(alloc.live_bytes(), 0);
}

#[test]
fn oversized_alloc_with_does_not_leak() {
    let alloc = common::TrackingAllocator::new();
    {
        let arena = Arena::builder_in(alloc.clone()).chunk_size(64 * 1024).build();
        // Force oversized via a large array.
        let _r: &mut [u32; 8 * 1024] = arena.alloc_with(|| [0_u32; 8 * 1024]);
    }
    assert_eq!(alloc.live_chunks(), 0);
    assert_eq!(alloc.live_bytes(), 0);
}

#[test]
fn oversized_slice_fill_with_does_not_leak() {
    let alloc = common::TrackingAllocator::new();
    {
        let arena = Arena::builder_in(alloc.clone()).chunk_size(64 * 1024).build();
        let _slice = arena.alloc_slice_fill_with::<u32, _>(8 * 1024, |i| i as u32);
    }
    assert_eq!(alloc.live_chunks(), 0);
    assert_eq!(alloc.live_bytes(), 0);
}

// Regression: a panic in the smart-pointer slice-fill closure on an
// oversized chunk used to leak the chunk + its `ArenaInner` because
// `SliceReservation` had no Drop and the chunk's refcount stayed at 0
// until `commit_slice` ran.
#[test]
fn panic_in_oversized_slice_fill_with_rc_does_not_leak() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let alloc = common::TrackingAllocator::new();
    {
        let arena = Arena::builder_in(alloc.clone()).chunk_size(64 * 1024).build();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _r = arena.alloc_slice_fill_with_rc::<u32, _>(8 * 1024, |i| {
                assert!(i < 5, "synthetic panic");
                i as u32
            });
        }));
        assert!(result.is_err());
    }
    assert_eq!(alloc.live_chunks(), 0);
    assert_eq!(alloc.live_bytes(), 0);
}

// Regression: a panic inside the user closure of `alloc_with` on a
// payload large enough to land on an oversized chunk used to leak the
// chunk because the pin/inc-ref happened only after the closure
// returned.
#[test]
fn panic_in_oversized_alloc_with_does_not_leak() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let alloc = common::TrackingAllocator::new();
    {
        let arena = Arena::builder_in(alloc.clone()).chunk_size(64 * 1024).build();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _r: &mut [u32; 8 * 1024] = arena.alloc_with(|| panic!("synthetic panic"));
        }));
        assert!(result.is_err());
    }
    assert_eq!(alloc.live_chunks(), 0);
    assert_eq!(alloc.live_bytes(), 0);
}
