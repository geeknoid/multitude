#![cfg(feature = "builders")]
//! Tests for `&Arena<A>: Allocator` (the `allocator-api2` integration).
//! Exercises `allocate` / `deallocate` / `grow` / `shrink` paths
//! indirectly via `ArenaVec` (which wraps an `allocator-api2` `Vec`).

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::large_stack_arrays, reason = "test allocations are intentional")]

use multitude::Arena;

#[cfg(feature = "stats")]
#[test]
fn allocator_grow_via_arena_vec_records_relocation() {
    // ArenaVec push that can't grow in place must relocate via
    // <&Arena<A> as Allocator>::grow → counted in stats.relocations.
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
    v.push(1);
    let _decoy = arena.alloc_rc(0_u8); // breaks cursor adjacency
    for i in 0..1000_u32 {
        v.push(i);
    }
    assert!(arena.stats().relocations >= 1);
}

#[test]
fn allocator_shrink_in_place_path() {
    // shrink is called internally by Vec when capacity reduces.
    // Exercise that no UB arises from the typical reserve/clear cycle.
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
    v.extend(0..50_u32);
    v.clear();
    v.reserve(10);
    assert!(v.capacity() >= 10);
}

#[cfg(feature = "stats")]
#[test]
fn oversized_chunk_used_when_alloc_too_big() {
    let arena = Arena::new();
    let big = arena.alloc_slice_copy_rc([0_u8; 32 * 1024]);
    assert_eq!(big.len(), 32 * 1024);
    assert!(arena.stats().oversized_chunks_allocated >= 1);
}

#[test]
fn allocator_rejects_excessive_alignment() {
    // `<&Arena>::allocate` must reject layouts whose alignment exceeds
    // CHUNK_ALIGN (64 KiB). Without this guard the oversized chunk's
    // base would only be 64 KiB-aligned, and the data pointer derived
    // from it would be misaligned for the user's request — UB on first
    // typed access.
    use allocator_api2::alloc::Allocator;
    let arena = Arena::new();
    let allocator: &Arena = &arena;
    let layout = core::alloc::Layout::from_size_align(8, 128 * 1024).unwrap();
    let _ = allocator.allocate(layout).unwrap_err();
}

#[test]
fn allocator_rejects_alignment_equal_to_chunk_align() {
    // `<&Arena>::allocate` must also reject layouts whose alignment
    // equals CHUNK_ALIGN (64 KiB). For such allocations the value
    // would land at offset == CHUNK_ALIGN within the chunk, where
    // `header_for`'s `addr & (CHUNK_ALIGN - 1)` mask returns 0 and
    // so reports the value pointer itself as the chunk header
    // address — UB on the next refcount op.
    use allocator_api2::alloc::Allocator;
    let arena = Arena::new();
    let allocator: &Arena = &arena;
    let layout = core::alloc::Layout::from_size_align(8, 64 * 1024).unwrap();
    let _ = allocator.allocate(layout).unwrap_err();
}
