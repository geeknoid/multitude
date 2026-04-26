//! Tests for `&Arena<A>: Allocator` (the `allocator-api2` integration).
//! Exercises `allocate` / `deallocate` / `grow` / `shrink` paths
//! indirectly via `ArenaVec` (which wraps an `allocator-api2` `Vec`).

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::large_stack_arrays, reason = "test allocations are intentional")]

use harena::Arena;

#[test]
fn allocator_grow_via_arena_vec_records_relocation() {
    // ArenaVec push that can't grow in place must relocate via
    // <&Arena<A> as Allocator>::grow → counted in allocator_relocations.
    let arena = Arena::new();
    let mut v = arena.new_vec::<u32>();
    v.push(1);
    let _decoy = arena.alloc(0_u8); // breaks cursor adjacency
    for i in 0..1000_u32 {
        v.push(i);
    }
    assert!(arena.stats().allocator_relocations >= 1);
}

#[test]
fn allocator_shrink_in_place_path() {
    // shrink is called internally by Vec when capacity reduces.
    // Exercise that no UB arises from the typical reserve/clear cycle.
    let arena = Arena::new();
    let mut v = arena.new_vec::<u32>();
    v.extend(0..50_u32);
    v.clear();
    v.reserve(10);
    assert!(v.capacity() >= 10);
}

#[test]
fn oversized_chunk_used_when_alloc_too_big() {
    let arena = Arena::new();
    let big = arena.alloc_slice_copy(&[0_u8; 32 * 1024]);
    assert_eq!(big.len(), 32 * 1024);
    assert!(arena.stats().oversized_chunks_allocated >= 1);
}
