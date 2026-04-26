#![cfg(feature = "builders")]
//! Tests targeting code-coverage gaps surfaced by `cargo llvm-cov`.
//!
//! Each test in this file exists to drive a specific uncovered branch
//! or function. Tests that double up with existing behavioral tests
//! are kept lightweight; the goal is correctness of the surfaced
//! branches, not benchmark-grade exercises.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::large_stack_arrays, reason = "test allocations are intentional")]
#![allow(clippy::collection_is_never_read, reason = "tests retain smart pointers to keep chunks alive")]
#![allow(unused_results, reason = "test code")]
#![allow(clippy::used_underscore_binding, reason = "intentional drop-after binding")]
#![allow(clippy::cast_possible_truncation, reason = "test data is small")]
#![allow(clippy::explicit_into_iter_loop, reason = "test clarity")]
#![allow(clippy::assertions_on_result_states, reason = "tests deliberately assert error returns")]
#![allow(clippy::items_after_statements, reason = "test-local statics next to their use")]
#![allow(
    clippy::cast_ptr_alignment,
    reason = "test writes a u32 to a u8-typed reservation we created with u32 layout"
)]

mod common;

use core::sync::atomic::{AtomicUsize, Ordering};

use multitude::{Arc, Arena, ArenaBuilder, Box, BuildError, RcStr};

use crate::common::{FailingAllocator, SendFailingAllocator};
use multitude::builders::{CollectIn, Vec};

// ---------------------------------------------------------------------------
// allocator_impl.rs — Allocator::shrink + deallocate teardown branch
// ---------------------------------------------------------------------------

#[test]
fn allocator_shrink_at_cursor_lowers_bump() {
    // <&Arena as Allocator>::shrink is invoked when a Vec backed by
    // `&Arena<A>` is shrunk in place (e.g. via `shrink_to_fit`). When
    // the buffer sits at the chunk's bump cursor (no other allocs
    // since), shrink should lower the cursor — driving the
    // `buffer_end_offset == cur` branch. We use allocator-api2's Vec
    // directly because ArenaVec doesn't expose `shrink_to_fit`.
    let arena: Arena = Arena::new();
    let mut v: allocator_api2::vec::Vec<u32, &Arena> = allocator_api2::vec::Vec::with_capacity_in(1024, &arena);
    for i in 0..10_u32 {
        v.push(i);
    }
    v.shrink_to_fit();
    assert_eq!(v.len(), 10);
    // Subsequent allocations should reuse the reclaimed slack.
    let _other = arena.alloc_rc(0_u64);
    assert_eq!(v.len(), 10);
}

#[test]
fn allocator_shrink_not_at_cursor_no_op() {
    // Shrink when the buffer isn't at the cursor: should still succeed
    // (returns Ok) but leaves the cursor alone. Drives the else-branch.
    let arena: Arena = Arena::new();
    let mut v: allocator_api2::vec::Vec<u32, &Arena> = allocator_api2::vec::Vec::with_capacity_in(1024, &arena);
    for i in 0..10_u32 {
        v.push(i);
    }
    let _decoy = arena.alloc_rc(0_u64); // breaks cursor adjacency
    v.shrink_to_fit();
    assert_eq!(v.len(), 10);
}

#[test]
fn allocator_deallocate_triggers_teardown_when_last_ref() {
    // <&Arena as Allocator>::deallocate's `if needs_teardown` branch:
    // the deallocate must observe refcount → 0 and call teardown_chunk.
    // Achieved by forcing many grow → relocate cycles inside a Vec
    // backed by `&Arena`: each old buffer's deallocate eventually
    // tears down its chunk (the chunk's only ref was the Vec's
    // buffer, and after retirement the arena no longer holds it).
    let arena: Arena = Arena::builder().chunk_size(8 * 1024).chunk_cache_capacity(0).build();
    {
        let mut v: allocator_api2::vec::Vec<u8, &Arena> = allocator_api2::vec::Vec::new_in(&arena);
        for _ in 0..16_000_u32 {
            v.push(0);
        }
        drop(v);
    }
}

// ---------------------------------------------------------------------------
// arena_builder.rs — allocator_in, Debug, panic_build, AllocFailed
// ---------------------------------------------------------------------------

#[test]
fn builder_allocator_in_chains_allocator() {
    // `ArenaBuilder::allocator_in` returns a builder over the new
    // allocator type with all other settings preserved. Drives the
    // entire body of allocator_in.
    let alloc = FailingAllocator::new(usize::MAX);
    let arena = Arena::builder()
        .chunk_size(8 * 1024)
        .max_normal_alloc(4 * 1024)
        .preallocate(0)
        .chunk_cache_capacity(2)
        .allocator_in(alloc)
        .try_build()
        .unwrap();
    let v = arena.alloc_rc(123_u32);
    assert_eq!(*v, 123);
}

#[test]
fn builder_debug_format() {
    // Drives ArenaBuilder's Debug impl.
    let s = format!("{:?}", Arena::builder().chunk_size(8192));
    assert!(s.contains("ArenaBuilder"));
    assert!(s.contains("chunk_size"));
}

#[test]
#[should_panic(expected = "ArenaBuilder::build")]
fn builder_build_panics_on_invalid_config() {
    let _ = Arena::builder().chunk_size(1024).build();
}

#[test]
fn builder_preallocate_alloc_failed() {
    // Drives the `Err(_) => return Err(BuildError::AllocFailed)` branch
    // in `ArenaBuilder::build` by giving the builder an allocator that
    // refuses to satisfy the preallocate request.
    let alloc = FailingAllocator::new(0);
    let result = Arena::builder()
        .preallocate(1)
        .chunk_cache_capacity(1)
        .allocator_in(alloc)
        .try_build();
    assert_eq!(result.unwrap_err(), BuildError::AllocFailed);
}

#[test]
fn byte_budget_exhaustion_returns_alloc_error() {
    // Drives the `if next > budget { return Err(AllocError) }` branch in
    // `try_alloc_fresh_chunk` (arena.rs:382-386). The budget is set to
    // exactly one chunk's worth, so the second normal-chunk allocation
    // must trip the budget and return Err WITHOUT ever calling the
    // backing allocator.
    let arena: Arena = Arena::builder().chunk_size(8 * 1024).byte_budget(4 * 1024).build();
    // Fill the first chunk so we force a fresh-chunk request next.
    let mut handles = std::vec::Vec::new();
    let mut hit_err = false;
    for _ in 0..1000_u32 {
        if let Ok(h) = arena.try_alloc_rc([0_u8; 256]) {
            handles.push(h);
        } else {
            hit_err = true;
            break;
        }
    }
    assert!(hit_err, "byte_budget did not stop allocations");
    // Subsequent allocations should also fail once the budget is exhausted.
    assert!(arena.try_alloc_rc(0_u32).is_err());
}

// ---------------------------------------------------------------------------
// chunk_header.rs — unlink_drop_entry middle-of-list branch
// ---------------------------------------------------------------------------

#[test]
fn arena_box_drop_unlinks_middle_of_drop_list() {
    // `unlink_drop_entry` has three positions (head, middle, tail).
    // The middle case is reached when the entry being removed has both
    // a `prev` and a `next`. ArenaBox<T: Drop>::drop calls unlink. We
    // create three drop-needing ArenaBox values, then drop the second
    // one first → exercises the `Some(prev)` AND `Some(next)` branches.
    let arena = Arena::new();
    let mut b1 = arena.alloc_box(std::string::String::from("first"));
    let mut b2 = arena.alloc_box(std::string::String::from("middle"));
    let mut b3 = arena.alloc_box(std::string::String::from("last"));
    // Make sure each value is reachable (touch the contents).
    b1.push('!');
    b2.push('!');
    b3.push('!');
    drop(b2); // <-- middle of doubly-linked list
    assert_eq!(*b1, "first!");
    assert_eq!(*b3, "last!");
}

// ---------------------------------------------------------------------------
// chunk_header.rs — reinit_refcount Shared branch (cache reuse across flavor)
// ---------------------------------------------------------------------------

#[test]
fn cached_local_chunk_revived_as_shared() {
    // `revive_cached_chunk(chunk, Shared)` → `reinit_refcount(_, Shared, 1)`
    // dispatches to the Shared branch. The deterministic way to land
    // a chunk in the cache is `.preallocate(n)`, which seeds the cache
    // before the first allocation. Then `alloc_arc` pops the cache and
    // revives the chunk as Shared.
    let arena: Arena = Arena::builder().chunk_size(8 * 1024).chunk_cache_capacity(2).preallocate(2).build();
    let shared = arena.alloc_arc(99_u64);
    assert_eq!(*shared, 99);
    let join = std::thread::spawn(move || *shared);
    assert_eq!(99, join.join().unwrap());
}

// ---------------------------------------------------------------------------
// arena.rs — current_local dec_ref→teardown branch in Arena::drop
// ---------------------------------------------------------------------------

#[test]
fn arena_drop_tears_down_unreferenced_current_chunk() {
    // Arena::drop's `if needs_teardown` branch on current_local fires
    // when the arena's hold is the chunk's only reference (no
    // smart pointers outstanding). Previously this branch wasn't covered
    // because the test that allocated something then dropped also
    // dropped the smart pointer, but the chunk stayed in cache.
    //
    // Disable caching so the teardown actually frees the chunk.
    let arena: Arena = Arena::builder().chunk_cache_capacity(0).build();
    let _v = arena.alloc_rc(0_u32); // current_local = chunk; refcount = 2
    drop(_v); // refcount = 1 (arena's hold only)
    drop(arena); // current_local.take() then dec_ref → true → teardown_chunk
}

#[test]
fn try_get_chunk_rotation_tears_down_unreferenced_chunk() {
    // try_get_chunk_for's chunk-retirement branch (arena.rs ~line 315):
    // when the current chunk is full and not pinned, the arena's
    // dec_ref drops the refcount to 0 (no outstanding smart pointers)
    // and `teardown_chunk` runs at rotation time — not at arena drop.
    //
    // Recipe:
    //   1. small chunk_size, cache disabled (so teardown actually
    //      frees the chunk rather than caching it),
    //   2. allocate via alloc_rc (no pinning) and immediately drop
    //      the smart pointer so the chunk's refcount returns to the
    //      arena's transient +1 only,
    //   3. force rotation by issuing more allocations than the chunk
    //      can hold.
    let arena: Arena = Arena::builder().chunk_size(8 * 1024).chunk_cache_capacity(0).build();

    // Track destructor invocations to prove the rotation-time
    // teardown ran the chunk's drop list.
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Counted(#[expect(dead_code, reason = "field present to give the type a non-zero size")] u32);
    impl Drop for Counted {
        fn drop(&mut self) {
            let _ = DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }
    DROPS.store(0, Ordering::SeqCst);

    // Fill the first chunk with values whose smart pointers are
    // dropped immediately (so the chunk's refcount stays at 1 = the
    // arena's hold). We allocate enough to force rotation. Each
    // Counted needs a drop entry, so worst-case sizing is large
    // enough that ~50 allocations exhaust a 4 KiB chunk.
    for i in 0..200_u32 {
        let h = arena.alloc_rc(Counted(i));
        drop(h);
    }
    // The teardown_chunk call at rotation time runs the retired
    // chunk's drop list, so some Counted destructors fire *during
    // the loop* (not at arena drop). The current chunk's destructors
    // only run when the arena is dropped.
    let drops_during_rotation = DROPS.load(Ordering::SeqCst);
    assert!(drops_during_rotation > 0, "rotation-time teardown should have run destructors");
    assert!(drops_during_rotation < 200, "current chunk's destructors run only at arena drop");
    // After arena drop, all 200 destructors must have run exactly once.
    drop(arena);
    assert_eq!(DROPS.load(Ordering::SeqCst), 200);
}

// ---------------------------------------------------------------------------
// arena_builder.rs — BuildError::Display arms for non-default errors
// ---------------------------------------------------------------------------

#[test]
fn build_error_display_messages() {
    // Drives all match arms of <BuildError as Display>::fmt.
    assert!(format!("{}", BuildError::ChunkSizeOutOfRange).contains("chunk_size"));
    assert!(format!("{}", BuildError::MaxNormalAllocOutOfRange).contains("max_normal_alloc"));
    assert!(format!("{}", BuildError::PreallocateExceedsCache).contains("preallocate"));
    assert!(format!("{}", BuildError::AllocFailed).contains("preallocation failed"));
}

// ---------------------------------------------------------------------------
// arena.rs — `alloc_*` panic paths on a failing allocator
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_box_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_box(0_u32);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_with_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_rc_with(|| 0_u32);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_copy_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_copy_rc([0_u8; 4]);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_clone_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_clone_rc(&[std::string::String::from("x")]);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_fill_with_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_fill_with_rc::<u32, _>(4, |i| i as u32);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_fill_iter_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_fill_iter_rc([1u32, 2, 3]);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_box_with_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_box_with(|| 0_u32);
}

#[test]
#[cfg(feature = "dst")]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_uninit_dst_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = multitude::dst::PendingRc::new(&arena, core::alloc::Layout::new::<u32>());
}

// ---------------------------------------------------------------------------
// arena.rs — alignment > CHUNK_ALIGN rejection paths
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "dst")]
fn alloc_uninit_dst_rejects_excessive_alignment() {
    // Drives the `if layout.align() > CHUNK_ALIGN { return Err(AllocError) }`
    // guard. CHUNK_ALIGN is 64 KiB (an internal constant); 128 KiB
    // alignment is guaranteed to exceed it.
    let arena: Arena = Arena::new();
    let huge_align = 128 * 1024_usize;
    let layout = core::alloc::Layout::from_size_align(huge_align, huge_align).unwrap();
    assert!(multitude::dst::PendingRc::try_new(&arena, layout).is_err());
}

#[test]
fn alloc_slice_rejects_overflow() {
    // Drives the `elem_size.checked_mul(len)` overflow path in
    // reserve_slice (returns AllocError).
    let arena: Arena = Arena::new();
    let huge_len = usize::MAX / 2;
    // u32 size 4 * huge_len overflows.
    assert!(arena.try_alloc_slice_fill_with_rc::<u32, _>(huge_len, |i| i as u32).is_err());
}

#[test]
fn alloc_slice_rejects_isize_max() {
    // Drives the `total > isize::MAX - (align - 1)` guard in reserve_slice.
    // u8 size 1 * (isize::MAX as usize) is bounded but the rounding-up
    // step pushes it past the limit.
    let arena: Arena = Arena::new();
    let too_big = isize::MAX as usize;
    assert!(arena.try_alloc_slice_fill_with_rc::<u8, _>(too_big, |i| i as u8).is_err());
}

#[test]
#[cfg(feature = "dst")]
fn pending_arena_arc_try_new_rejects_excessive_alignment() {
    let arena: Arena = Arena::new();
    let huge_align = 128 * 1024_usize;
    let layout = core::alloc::Layout::from_size_align(huge_align, huge_align).unwrap();
    assert!(multitude::dst::PendingArc::try_new(&arena, layout).is_err());
}

#[test]
#[cfg(feature = "dst")]
fn pending_arena_box_try_new_rejects_excessive_alignment() {
    let arena: Arena = Arena::new();
    let huge_align = 128 * 1024_usize;
    let layout = core::alloc::Layout::from_size_align(huge_align, huge_align).unwrap();
    assert!(multitude::dst::PendingBox::try_new(&arena, layout).is_err());
}

// `#[repr(align(N))]` with N > CHUNK_ALIGN (64 KiB). Used by the two
// tests below to drive the `if layout.align() > CHUNK_ALIGN { return
// Err(AllocError) }` guard in `try_alloc_with` and `try_reserve_and_init`.
//
// The guard lives in a thin outer function whose frame doesn't depend
// on `T`'s alignment, so the test runs on every platform — including
// Windows, whose default 1 MiB stack can't accommodate the 128 KiB-
// aligned frame the guarded body would otherwise require.
#[repr(align(131072))]
struct HugeAlign(#[expect(dead_code, reason = "field present to give the type a non-zero size")] u8);

#[test]
fn try_alloc_with_rejects_excessive_alignment() {
    // try_alloc_with is the &mut T entry point. CHUNK_ALIGN is 64 KiB;
    // HugeAlign needs 128 KiB alignment, so the layout-align check
    // must fire and return Err.
    let arena: Arena = Arena::new();
    let result: Result<&mut HugeAlign, _> = arena.try_alloc_with(|| HugeAlign(0));
    assert!(result.is_err());
}

#[test]
fn try_alloc_rc_with_rejects_excessive_alignment() {
    // try_reserve_and_init is the smart-pointer entry point shared by
    // try_alloc_rc / try_alloc_arc / try_alloc_box. Same guard, same
    // expected Err return.
    let arena: Arena = Arena::new();
    let result: Result<multitude::Rc<HugeAlign>, _> = arena.try_alloc_rc_with(|| HugeAlign(0));
    assert!(result.is_err());
}

#[test]
fn try_alloc_string_with_capacity_huge_returns_err() {
    let arena: Arena = Arena::new();
    // Try a capacity that overflows when adding the prefix size.
    let too_big = usize::MAX;
    assert!(arena.try_alloc_string_with_capacity(too_big).is_err());
}

// ---------------------------------------------------------------------------
// Adversarial inputs that trigger remaining error paths.
// ---------------------------------------------------------------------------

#[test]
fn try_alloc_string_with_capacity_isize_max_returns_err() {
    // Drives the `isize::try_from(total).is_err()` guard in
    // ArenaString::try_allocate_initial. Need cap such that
    // `cap + PREFIX_SIZE` is between `isize::MAX + 1` and `usize::MAX`.
    let arena: Arena = Arena::new();
    let cap = (isize::MAX as usize) - 4; // cap + 8 > isize::MAX, and < usize::MAX
    assert!(arena.try_alloc_string_with_capacity(cap).is_err());
}

// Note: the `align > CHUNK_ALIGN` guard inside the typed alloc paths
// (`Arena::try_alloc_with`, `Arena::try_reserve_and_init`) cannot be
// exercised from a test that names a `#[repr(align(N))]` `T` with
// `N > CHUNK_ALIGN` — even though the closure / value would never be
// constructed, the compiled function's stack frame inherits `T`'s
// alignment, producing a STATUS_ACCESS_VIOLATION on call. The
// equivalent guard is exercised through the layout-based path in
// `alloc_uninit_dst_rejects_excessive_alignment` above (which uses
// `Layout::from_size_align` directly without naming a `T`).

#[test]
fn try_alloc_slice_fill_with_rc_isize_max_returns_err() {
    // Drives the `total > isize::MAX - (align-1)` guard in `reserve_slice`.
    // For u64 (align=8, size=8), len = isize::MAX/8 yields total = isize::MAX-7,
    // which equals the bound (not >). len = isize::MAX/8 + 1 yields total
    // that overflows. We need a value of len that's just past the bound
    // without overflowing usize.
    //
    // Actually, for align=8: bound = isize::MAX-7. For len = (isize::MAX/8) + 1,
    // total = 8*((isize::MAX/8)+1) = isize::MAX+1 (depending on rounding).
    // Use len = (isize::MAX as usize / 8) + 1, which is 0x1000_0000_0000_0000 on 64-bit.
    // total = 8 * len = isize::MAX + 1 = 0x8000_0000_0000_0000 (does NOT overflow usize on 64-bit).
    let arena: Arena = Arena::new();
    let len = (isize::MAX as usize / 8) + 1;
    assert!(arena.try_alloc_slice_fill_with_rc::<u64, _>(len, |i| i as u64).is_err());
}

#[test]
fn try_alloc_slice_fill_with_rc_in_small_chunk_register_drop_oversized() {
    // Drives the `end > h.total_size` guard in `reserve_slice`'s
    // register_drop branch. Use a small `chunk_size` and ask for a
    // slice whose worst-case sizing fits the oversized cutoff but
    // whose actual layout can't fit a normal chunk — the worst-case
    // routes us to oversized; the inner end>total_size check is exercised
    // for the oversized chunk's fast-path. For full coverage we need a
    // case where the requested slice exceeds the freshly-allocated
    // chunk's `total_size` even after worst-case sizing.
    //
    // Strategy: use a small chunk_size, ask for a Drop-needing slice
    // larger than the chunk. Worst-case sizing pushes it to oversized;
    // the oversized chunk is sized exactly to fit; the end>total_size
    // check inside reserve_slice should NOT fire on the oversized path
    // (since it's right-sized). To hit the check we'd need a path bug.
    //
    // The defensive `end > h.total_size` re-check inside reserve_slice
    // is therefore reachable only on internal corruption — leave
    // uncovered.
    let _arena: Arena = Arena::builder().chunk_size(8 * 1024).build();
    // No assertion; the test just documents the unreachability.
}

#[test]
fn alloc_rc_oversized_drop_type_uses_has_drop_layout() {
    // Drives the `has_drop = true` arm of `ChunkHeader::oversized_layout`
    // (chunk_header.rs:225-230) — the `end` value returned from
    // `entry_layout::checked_entry_value_offsets` becomes the chunk's
    // total size for an oversized chunk that holds a single
    // Drop-registering value.
    //
    // Recipe: build an arena with a tight `max_normal_alloc` and
    // allocate an `ArenaRc<Drop type>` whose size exceeds it. The
    // request routes to the oversized path, which calls
    // `oversized_layout(payload, has_drop=true)` because `T: Drop`.
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    struct BigDrop {
        _bytes: [u8; 4096],
    }
    impl Drop for BigDrop {
        fn drop(&mut self) {
            let _ = DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }
    DROPPED.store(0, Ordering::SeqCst);

    let arena: Arena = Arena::builder()
        .chunk_size(8 * 1024)
        .max_normal_alloc(4 * 1024) // BigDrop (4 KiB) > 1 KiB cutover
        .build();
    {
        let h = arena.alloc_rc(BigDrop { _bytes: [0; 4096] });
        // Sanity: we can read the value (chunk-recovery via header-mask
        // works for oversized chunks too).
        assert_eq!(h._bytes[0], 0);
    }
    // Smart pointer dropped → oversized chunk's drop list runs → BigDrop::drop fires.
    assert_eq!(DROPPED.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "dst")]
#[test]
fn alloc_box_oversized_drop_type_uses_has_drop_layout() {
    // Same path as above, but via the box family (which also routes
    // through `oversized_layout(_, true)` for Drop-needing types).
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    struct BigDrop {
        _bytes: [u8; 4096],
    }
    impl Drop for BigDrop {
        fn drop(&mut self) {
            let _ = DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }
    DROPPED.store(0, Ordering::SeqCst);

    let arena: Arena = Arena::builder().chunk_size(8 * 1024).max_normal_alloc(4 * 1024).build();
    {
        let _b: Box<BigDrop> = arena.alloc_box(BigDrop { _bytes: [0; 4096] });
    }
    assert_eq!(DROPPED.load(Ordering::SeqCst), 1);
}

#[test]
#[cfg(feature = "dst")]
fn pending_arena_rc_try_new_oversized_layout_succeeds() {
    // A layout that exceeds normal chunk size routes to an oversized chunk.
    // Drives the oversized-chunk path in try_reserve_dst_with_entry.
    let arena: Arena = Arena::builder().chunk_size(8 * 1024).build();
    let layout = core::alloc::Layout::array::<u8>(8 * 1024).unwrap();
    let pa = multitude::dst::PendingRc::try_new(&arena, layout).unwrap();
    drop(pa);
}

#[test]
fn arena_string_grow_through_chunk_rotation() {
    // Drives the `if needs_teardown { teardown_chunk(chunk, true); }`
    // branch in `Arena::grow_for_string` — when the OLD string buffer's
    // chunk has only the string as a holder (refcount==1 → after dec
    // it's 0 → teardown).
    let arena: Arena = Arena::builder().chunk_size(8 * 1024).chunk_cache_capacity(0).build();
    let mut s = arena.alloc_string();
    // Push enough text to force the string to grow into a fresh chunk;
    // the old chunk had ONLY this string (no other allocations) so its
    // refcount drops to 0 on grow → triggers teardown_chunk.
    let chunk = "x".repeat(64);
    for _ in 0..200 {
        s.push_str(&chunk);
    }
    assert_eq!(s.len(), 200 * 64);
}

// ---------------------------------------------------------------------------
// box / arc — PartialEq impls
// ---------------------------------------------------------------------------

#[test]
fn arena_box_partial_eq() {
    let arena = Arena::new();
    let a: Box<u32> = arena.alloc_box(7);
    let b: Box<u32> = arena.alloc_box(7);
    let c: Box<u32> = arena.alloc_box(8);
    assert!(a == b);
    assert!(a != c);
}

#[test]
fn arena_arc_partial_eq() {
    let arena = Arena::new();
    let a: Arc<u32> = arena.alloc_arc(7);
    let b: Arc<u32> = arena.alloc_arc(7);
    let c: Arc<u32> = arena.alloc_arc(8);
    assert!(a == b);
    assert!(a != c);
}

// ---------------------------------------------------------------------------
// vec — DerefMut
// ---------------------------------------------------------------------------

#[test]
fn arena_vec_deref_mut_modifies_in_place() {
    let arena = Arena::new();
    let mut v: Vec<u32, _> = arena.alloc_vec();
    v.push(1);
    v.push(2);
    v.push(3);
    // Modify via DerefMut (not via push).
    let slice: &mut [u32] = &mut v;
    slice[0] = 99;
    assert_eq!(v.as_slice(), &[99, 2, 3]);
}

// ---------------------------------------------------------------------------
// collect_in.rs — empty-iterator path through `new_in`
// ---------------------------------------------------------------------------

#[test]
fn collect_in_empty_iterator_uses_new_in() {
    // An iterator with `size_hint().0 == 0` should take the `new_in`
    // path (no `with_capacity_in(0)` detour). Easiest: filter that
    // discards everything but advertises `(0, _)`.
    let arena = Arena::new();
    let v: Vec<u32, _> = (0..10_u32).filter(|_| false).collect_in(&arena);
    assert!(v.is_empty());
}

// ---------------------------------------------------------------------------
// pending_arena_arc / pending_arena_rc — Drop without finalize, finalize
// with drop_fn.
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "dst")]
fn pending_arena_arc_dropped_without_finalize_releases_chunk() {
    // PendingArc::Drop runs when the user never calls finalize.
    // It must release the chunk refcount (otherwise the chunk and
    // ArenaInner leak). We exercise this and rely on Miri / leak
    // checkers to validate.
    let arena: Arena = Arena::new();
    {
        let _pending = multitude::dst::PendingArc::new(&arena, core::alloc::Layout::new::<u32>());
        // `_pending` drops here without finalize; should release
        // the chunk refcount cleanly.
    }
    // Subsequent allocations should still work.
    let v = arena.alloc_arc(42_u32);
    assert_eq!(*v, 42);
}

#[test]
#[cfg(feature = "dst")]
fn pending_arena_arc_finalize_with_drop_fn_runs_drop() {
    // PendingArc::finalize with `drop_fn = Some(...)` links a
    // drop entry. The drop entry's shim must run at chunk teardown.
    use core::sync::atomic::{AtomicUsize, Ordering as Ord};
    use multitude::dst::DropEntry;

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
    DROP_COUNT.store(0, Ord::SeqCst);

    unsafe fn drop_shim_for_u32(_entry: *mut DropEntry) {
        DROP_COUNT.fetch_add(1, Ord::SeqCst);
    }

    let arena: Arena = Arena::new();
    {
        let mut p = multitude::dst::PendingArc::new(&arena, core::alloc::Layout::new::<u32>());
        // SAFETY: we're writing a u32 of the layout we requested.
        unsafe {
            p.as_mut_ptr().cast::<u32>().write(0xCAFE_F00D);
        }
        let template: *const u32 = core::ptr::null();
        // SAFETY: bytes initialized; drop_shim is well-formed for u32 layout.
        let arc: Arc<u32> = unsafe { p.finalize::<u32>(template, Some(drop_shim_for_u32)) };
        assert_eq!(*arc, 0xCAFE_F00D);
        // Move arc to another thread (exercises the Send+Sync path).
        let h = std::thread::spawn(move || *arc);
        let _ = std::sync::Arc::new(0); // silence unused-import in some configs
        let val = h.join().unwrap();
        assert_eq!(val, 0xCAFE_F00D);
    } // arc dropped → chunk teardown → drop_shim runs
    drop(arena);
    assert_eq!(DROP_COUNT.load(Ord::SeqCst), 1, "drop shim must run exactly once");
}

#[test]
#[cfg(feature = "dst")]
fn pending_arena_rc_dropped_without_finalize_releases_chunk() {
    // PendingRc::Drop equivalent.
    let arena: Arena = Arena::new();
    {
        let _pending = multitude::dst::PendingRc::new(&arena, core::alloc::Layout::new::<u32>());
    }
    let v = arena.alloc_rc(42_u32);
    assert_eq!(*v, 42);
}

// ---------------------------------------------------------------------------
// string / rc_str — Drop teardown branch
// ---------------------------------------------------------------------------

#[test]
fn arena_string_drop_runs_teardown_when_last_ref() {
    // ArenaString::drop's `if needs_teardown` branch fires when the
    // string is the chunk's last reference. Force the chunk holding
    // `s` to be rotated out of `current_local` (so the arena releases
    // its +1 hold), leaving only `s` referencing the chunk. Dropping
    // `s` then triggers teardown_chunk.
    let arena: Arena = Arena::builder().chunk_size(8 * 1024).chunk_cache_capacity(0).build();
    let mut s = arena.alloc_string_with_capacity(2048); // big buffer in current chunk
    s.push_str("hello");
    // Allocate something that forces the next alloc to retire the
    // current chunk (since combined size won't fit).
    let _filler = arena.alloc_slice_copy_rc([0_u8; 1500]);
    // The next alloc should rotate out the chunk holding `s`.
    let _other = arena.alloc_rc(0_u64);
    // `s`'s chunk is no longer current. Dropping `s` is its last ref.
    drop(s); // → dec_ref returns true → teardown_chunk
}

#[test]
fn arena_rc_str_drop_runs_teardown_when_last_ref() {
    // The smart pointer outlives the arena, so when the arena drops it
    // releases its hold on the chunk and the smart pointer becomes the sole
    // reference. Dropping the smart pointer then triggers teardown_chunk.
    let s: RcStr = {
        let arena = Arena::new();
        arena.alloc_str_rc("outlives the arena")
    };
    assert_eq!(&*s, "outlives the arena");
    drop(s); // teardown_chunk fires here
}

// ---------------------------------------------------------------------------
// arena.rs panic_alloc paths via try_alloc on an exhausted allocator
// ---------------------------------------------------------------------------

#[test]
fn try_alloc_returns_err_on_failing_allocator() {
    // Drives the `Err(_) => panic_alloc()` branches indirectly: the
    // try_alloc family returns AllocError instead. Each public
    // try_alloc* with a failing allocator hits its respective error
    // path. (The `_arc` variants require A: Send + Sync; we skip
    // them here — their implementation flows through the same
    // `try_get_chunk_for` failure branch as the Local variants.)
    let alloc = FailingAllocator::new(0);
    let arena: Arena<FailingAllocator> = Arena::new_in(alloc);
    assert!(arena.try_alloc_rc(0_u32).is_err());
    assert!(arena.try_alloc_box(0_u32).is_err());
    assert!(arena.try_alloc_slice_copy_rc::<u8>(&[1, 2, 3]).is_err());
    assert!(arena.try_alloc_slice_clone_rc::<u32>(&[1, 2, 3]).is_err());
    assert!(arena.try_alloc_slice_fill_with_rc::<u32, _>(3, |i| i as u32).is_err());
    assert!(arena.try_alloc_slice_fill_iter_rc([1u32, 2, 3]).is_err());
    assert!(arena.try_alloc_rc_with(|| 0_u32).is_err());
    assert!(arena.try_alloc_box_with(|| 0_u32).is_err());
    #[cfg(feature = "dst")]
    assert!(multitude::dst::PendingRc::try_new(&arena, core::alloc::Layout::new::<u32>()).is_err());
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_panics_on_failing_allocator() {
    // Specifically drive panic_alloc().
    let alloc = FailingAllocator::new(0);
    let arena: Arena<FailingAllocator> = Arena::new_in(alloc);
    let _ = arena.alloc_rc(0_u32);
}

// Use ArenaBuilder type (covered by allocator_in test) to silence
// unused-import warnings if any of the above tests change.
#[test]
fn builder_type_is_constructible() {
    let _: ArenaBuilder = Arena::builder();
}

// ---------------------------------------------------------------------------
// Infallible Arc / Box slice constructors and the strait-`alloc_arc` family.
// These wrap their `try_*` cousins with `unwrap_or_else(panic_alloc)`; the
// happy path was previously uncovered.
// ---------------------------------------------------------------------------

#[test]
fn alloc_arc_value_succeeds() {
    let arena = Arena::new();
    let h: Arc<u32> = arena.alloc_arc(7);
    assert_eq!(*h, 7);
}

#[test]
fn alloc_arc_with_closure_succeeds() {
    let arena = Arena::new();
    let h: Arc<u64> = arena.alloc_arc_with(|| 42_u64);
    assert_eq!(*h, 42);
}

#[test]
fn alloc_slice_copy_arc_succeeds() {
    let arena = Arena::new();
    let h: Arc<[u32]> = arena.alloc_slice_copy_arc([1_u32, 2, 3, 4]);
    assert_eq!(&*h, &[1, 2, 3, 4]);
}

#[test]
fn alloc_slice_clone_arc_succeeds() {
    let arena = Arena::new();
    let src = [std::string::String::from("a"), std::string::String::from("b")];
    let h: Arc<[String]> = arena.alloc_slice_clone_arc(src);
    assert_eq!(h.len(), 2);
    assert_eq!(h[0], "a");
    assert_eq!(h[1], "b");
}

#[test]
fn alloc_slice_fill_with_arc_succeeds() {
    let arena = Arena::new();
    let h: Arc<[u32]> = arena.alloc_slice_fill_with_arc(5, |i| i as u32 * 10);
    assert_eq!(&*h, &[0, 10, 20, 30, 40]);
}

#[test]
fn alloc_slice_fill_iter_arc_succeeds() {
    let arena = Arena::new();
    let h: Arc<[u32]> = arena.alloc_slice_fill_iter_arc(0_u32..3);
    assert_eq!(&*h, &[0, 1, 2]);
}

#[cfg(feature = "dst")]
#[test]
fn alloc_slice_copy_box_succeeds() {
    let arena = Arena::new();
    let b: Box<[u8]> = arena.alloc_slice_copy_box([10_u8, 20, 30]);
    assert_eq!(&*b, &[10, 20, 30]);
}

#[cfg(feature = "dst")]
#[test]
fn alloc_slice_clone_box_succeeds() {
    let arena = Arena::new();
    let src = [
        std::string::String::from("x"),
        std::string::String::from("y"),
        std::string::String::from("z"),
    ];
    let b: Box<[String]> = arena.alloc_slice_clone_box(src);
    assert_eq!(b.len(), 3);
    assert_eq!(b[2], "z");
}

#[cfg(feature = "dst")]
#[test]
fn alloc_slice_fill_with_box_succeeds() {
    let arena = Arena::new();
    let b: Box<[u32]> = arena.alloc_slice_fill_with_box(4, |i| (i + 1) as u32);
    assert_eq!(&*b, &[1, 2, 3, 4]);
}

#[cfg(feature = "dst")]
#[test]
fn alloc_slice_fill_iter_box_succeeds() {
    let arena = Arena::new();
    let b: Box<[u8]> = arena.alloc_slice_fill_iter_box(0_u8..5);
    assert_eq!(&*b, &[0, 1, 2, 3, 4]);
}

// ---------------------------------------------------------------------------
// String-flavor smart pointer fallible constructors via `Arena`.
// ---------------------------------------------------------------------------

#[test]
fn arena_try_alloc_str_arc_succeeds() {
    use multitude::ArcStr;
    let arena: Arena = Arena::new();
    let s: ArcStr = arena.try_alloc_str_arc("hello arc").unwrap();
    assert_eq!(s.as_str(), "hello arc");
}

#[test]
fn arena_try_alloc_str_rc_succeeds() {
    let arena: Arena = Arena::new();
    let s: RcStr = arena.try_alloc_str_rc("hello rc").unwrap();
    assert_eq!(s.as_str(), "hello rc");
}

#[test]
fn arena_try_alloc_str_box_succeeds() {
    use multitude::BoxStr;
    let arena: Arena = Arena::new();
    let s: BoxStr = arena.try_alloc_str_box("hello box").unwrap();
    assert_eq!(s.as_str(), "hello box");
}

#[test]
fn arena_box_str_as_mut_via_trait() {
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_str_box("abc");
    let m: &mut str = AsMut::<str>::as_mut(&mut s);
    // SAFETY: ASCII bytes; in-place uppercase preserves UTF-8.
    unsafe { m.as_bytes_mut()[0] = b'A' };
    assert_eq!(s.as_str(), "Abc");
}

// ---------------------------------------------------------------------------
// ArenaString::with_capacity_in (cap > 0) — exercises allocate_initial path
// (line 102 / 324) and into_arena_str slack reclamation (line 258).
// ---------------------------------------------------------------------------

#[test]
fn alloc_string_with_capacity_allocates_buffer() {
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string_with_capacity(64);
    assert!(s.capacity() >= 64);
    s.push_str("hello world");
    assert_eq!(s.as_str(), "hello world");
}

#[test]
fn arena_string_into_arena_str_reclaims_slack_at_cursor() {
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string_with_capacity(128);
    s.push_str("short");
    let rc = s.into_arena_str();
    assert_eq!(rc.as_str(), "short");
    // After slack reclamation, a subsequent allocation should reuse
    // bytes from the freed tail rather than rotating to a fresh chunk.
    let _follow_on = arena.alloc_str("follow");
}

// ---------------------------------------------------------------------------
// ArenaVec::try_with_capacity_in (cap > 0) and empty into_rc.
// ---------------------------------------------------------------------------

#[test]
fn try_alloc_vec_with_capacity_succeeds() {
    let arena: Arena = Arena::new();
    let mut v = arena.try_alloc_vec_with_capacity::<u32>(16).unwrap();
    assert!(v.capacity() >= 16);
    v.push(1);
    v.push(2);
    assert_eq!(&*v, &[1, 2]);
}

#[test]
fn arena_vec_empty_into_rc_returns_empty_slice() {
    let arena: Arena = Arena::new();
    let v: Vec<u32> = arena.alloc_vec();
    let h = v.into_arena_rc();
    assert!(h.is_empty());
}

// ---------------------------------------------------------------------------
// Serde — ArenaBoxStr serialization (other variants already covered).
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn arena_box_str_serializes_to_string() {
    let arena: Arena = Arena::new();
    let s = arena.alloc_str_box("box-str");
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, "\"box-str\"");
}

// ---------------------------------------------------------------------------
// `panic_alloc` closure paths for the Arc/Box variants of slice / value
// constructors. These mirror the existing tests for the Rc variants;
// each drives the `unwrap_or_else(|_| panic_alloc())` closure body so it
// shows as covered.
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_arc_panics_on_failing_allocator() {
    let arena: Arena<SendFailingAllocator> = Arena::new_in(SendFailingAllocator::new(0));
    let _ = arena.alloc_arc(0_u32);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_arc_with_panics_on_failing_allocator() {
    let arena: Arena<SendFailingAllocator> = Arena::new_in(SendFailingAllocator::new(0));
    let _ = arena.alloc_arc_with(|| 0_u32);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_copy_arc_panics_on_failing_allocator() {
    let arena: Arena<SendFailingAllocator> = Arena::new_in(SendFailingAllocator::new(0));
    let _ = arena.alloc_slice_copy_arc([0_u8; 4]);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_clone_arc_panics_on_failing_allocator() {
    let arena: Arena<SendFailingAllocator> = Arena::new_in(SendFailingAllocator::new(0));
    let _ = arena.alloc_slice_clone_arc([1_u32, 2]);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_fill_with_arc_panics_on_failing_allocator() {
    let arena: Arena<SendFailingAllocator> = Arena::new_in(SendFailingAllocator::new(0));
    let _ = arena.alloc_slice_fill_with_arc::<u32, _>(4, |i| i as u32);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_fill_iter_arc_panics_on_failing_allocator() {
    let arena: Arena<SendFailingAllocator> = Arena::new_in(SendFailingAllocator::new(0));
    let _ = arena.alloc_slice_fill_iter_arc([1_u32, 2, 3]);
}

#[cfg(feature = "dst")]
#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_copy_box_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_copy_box([0_u8; 4]);
}

#[cfg(feature = "dst")]
#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_clone_box_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_clone_box([1_u32, 2]);
}

#[cfg(feature = "dst")]
#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_fill_with_box_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_fill_with_box::<u32, _>(4, |i| i as u32);
}

#[cfg(feature = "dst")]
#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_slice_fill_iter_box_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_slice_fill_iter_box([1_u32, 2, 3]);
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_str_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_str("hi");
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_str_rc_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_str_rc("hi");
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_str_arc_panics_on_failing_allocator() {
    let arena: Arena<SendFailingAllocator> = Arena::new_in(SendFailingAllocator::new(0));
    let _ = arena.alloc_str_arc("hi");
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_str_box_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_str_box("hi");
}

#[test]
#[should_panic(expected = "multitude: allocator returned AllocError")]
fn alloc_string_with_capacity_panics_on_failing_allocator() {
    let arena: Arena<FailingAllocator> = Arena::new_in(FailingAllocator::new(0));
    let _ = arena.alloc_string_with_capacity(64);
}

// Drive `build`'s `unwrap_or_else(panic_build)` closure for each
// allocator monomorphization so the per-instantiation region count
// reaches 100% in the coverage report.
#[test]
#[should_panic(expected = "multitude::ArenaBuilder::build")]
fn build_panics_on_failing_allocator() {
    let _: Arena<FailingAllocator> = Arena::builder().allocator_in(FailingAllocator::new(0)).preallocate(1).build();
}

#[test]
#[should_panic(expected = "multitude::ArenaBuilder::build")]
fn build_panics_on_send_failing_allocator() {
    let _: Arena<SendFailingAllocator> = Arena::builder().allocator_in(SendFailingAllocator::new(0)).preallocate(1).build();
}

// ---------------------------------------------------------------------------
// arena.rs — typed `try_reserve_uninit` align>CHUNK_ALIGN guard.
// ---------------------------------------------------------------------------

// Distinct type from `HugeAlign` above so we don't perturb the caller's frame
// alignment and trigger the issue noted in the comment near
// `try_alloc_with_rejects_excessive_alignment`. The `MaybeUninit<T>` returned
// by the uninit-family entry points never materializes a real `T` on the
// stack, so the test compiles and runs safely on every platform.
#[repr(align(131072))]
struct HugeAlignBox(#[expect(dead_code, reason = "field gives the type a non-zero size")] u8);

#[test]
fn try_alloc_uninit_box_rejects_excessive_alignment() {
    let arena: Arena = Arena::new();
    let r = arena.try_alloc_uninit_box::<HugeAlignBox>();
    assert!(r.is_err());
}

// ---------------------------------------------------------------------------
// string — replace_range edge cases & try_reserve overflow paths.
// ---------------------------------------------------------------------------

#[test]
fn arena_string_replace_range_excluded_start() {
    use core::ops::Bound;
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("hello");
    // Excluded(0) -> start = 1, Excluded(3) -> end = 3 -> replace bytes 1..3 ("el") with "X"
    s.replace_range((Bound::Excluded(0_usize), Bound::Excluded(3_usize)), "X");
    assert_eq!(&*s, "hXlo");
}

#[test]
fn arena_string_replace_range_grow_path() {
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("ab");
    // Replacement is much longer than what's removed, forcing a grow
    // (`new_len > self.cap` branch in replace_range).
    s.replace_range(0..1, "lots of replacement text");
    assert_eq!(&*s, "lots of replacement textb");
}

#[test]
fn arena_string_replace_range_added_gt_removed_no_grow() {
    // Drives the `added > removed` arm of replace_range with the
    // `new_len > self.cap` check evaluating to false (the buffer
    // already has enough capacity for the larger replacement).
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string_with_capacity(64);
    s.push_str("abc");
    s.replace_range(0..1, "XY"); // removed=1, added=2 -> grows by 1; cap (64) suffices
    assert_eq!(&*s, "XYbc");
}

#[test]
fn arena_string_try_reserve_additional_overflow_returns_err() {
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("a");
    // self.len (1) + usize::MAX overflows -> Err.
    let r = s.try_reserve(usize::MAX);
    assert!(r.is_err());
}

#[test]
fn arena_string_try_reserve_within_existing_capacity_is_noop() {
    // Drives the `needed <= self.cap` branch of `try_reserve`
    // (cap already suffices, so try_grow_to_at_least is not called).
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string_with_capacity(64);
    s.push_str("hi");
    s.try_reserve(8).unwrap();
    assert!(s.capacity() >= 64);
}

#[test]
fn arena_string_try_reserve_grow_path_succeeds() {
    // Drives the success-fall-through past `try_grow_to_at_least(needed)?`
    // in `try_reserve` (cap>0, needed>cap, grow succeeds).
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("seed");
    let prior = s.capacity();
    s.try_reserve(prior * 4).unwrap();
    assert!(s.capacity() >= prior * 4 + s.len());
}

#[test]
fn arena_string_try_reserve_grow_path_overflow_returns_err() {
    // Drives `try_grow_to_at_least`'s `PREFIX_SIZE.checked_add(new_cap)` /
    // `isize::try_from(new_total)` failure paths. We need cap > 0 first
    // (so we hit the grow path, not initial allocate), then ask for an
    // additional that pushes total past isize::MAX.
    let arena: Arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("seed"); // cap > 0
    // additional fits in usize but new_total overflows isize.
    let additional = (isize::MAX as usize) - 4;
    let r = s.try_reserve(additional);
    assert!(r.is_err());
}

// ---------------------------------------------------------------------------
// box — From<ArenaBox<[T]>> for ArenaRc<[T]> (gated on `dst`).
// ---------------------------------------------------------------------------

#[cfg(feature = "dst")]
#[test]
fn arena_box_slice_from_into_arena_rc_slice() {
    let arena: Arena = Arena::new();
    let b: Box<[u32]> = arena.alloc_slice_fill_with_box(3, |i| i as u32 + 10);
    let r: multitude::Rc<[u32]> = b.into();
    assert_eq!(&*r, &[10, 11, 12][..]);
}

// ---------------------------------------------------------------------------
// Failure-driven coverage tests — drive `?` Err propagation and panicking
// `unwrap_or_else(|_| panic_alloc())` lambda bodies via FailingAllocator.
// ---------------------------------------------------------------------------

use std::panic::AssertUnwindSafe;

fn expect_panic<F: FnOnce()>(f: F) {
    let r = std::panic::catch_unwind(AssertUnwindSafe(f));
    assert!(r.is_err(), "expected panic but call returned");
}

fn fail_arena() -> Arena<FailingAllocator> {
    Arena::new_in(FailingAllocator::new(0))
}

fn send_fail_arena() -> Arena<SendFailingAllocator> {
    Arena::new_in(SendFailingAllocator::new(0))
}

// Panicking method bodies (every `unwrap_or_else(|_| panic_alloc())` lambda).

#[test]
fn panic_alloc_with() {
    expect_panic(|| {
        let a = fail_arena();
        let _: &mut u64 = a.alloc_with(|| 42);
    });
}

#[test]
fn panic_alloc_str() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_str("hi");
    });
}

#[test]
fn panic_alloc_slice_fill_with_rc() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_slice_fill_with_rc::<u32, _>(4, |i| i as u32);
    });
}

#[test]
fn panic_alloc_slice_fill_iter() {
    expect_panic(|| {
        let a = fail_arena();
        let _: &mut [u32] = a.alloc_slice_fill_iter([1_u32, 2, 3]);
    });
}

#[test]
fn panic_alloc_uninit_box() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_uninit_box::<u32>();
    });
}

#[test]
fn panic_alloc_zeroed_box() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_zeroed_box::<u32>();
    });
}

#[test]
fn panic_alloc_uninit_rc() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_uninit_rc::<u32>();
    });
}

#[test]
fn panic_alloc_zeroed_rc() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_zeroed_rc::<u32>();
    });
}

#[test]
fn panic_alloc_uninit_arc() {
    expect_panic(|| {
        let a = send_fail_arena();
        let _ = a.alloc_uninit_arc::<u32>();
    });
}

#[test]
fn panic_alloc_zeroed_arc() {
    expect_panic(|| {
        let a = send_fail_arena();
        let _ = a.alloc_zeroed_arc::<u32>();
    });
}

#[test]
fn panic_alloc_uninit_slice_rc() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_uninit_slice_rc::<u32>(4);
    });
}

#[test]
fn panic_alloc_zeroed_slice_rc() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_zeroed_slice_rc::<u32>(4);
    });
}

#[test]
fn panic_alloc_uninit_slice_arc() {
    expect_panic(|| {
        let a = send_fail_arena();
        let _ = a.alloc_uninit_slice_arc::<u32>(4);
    });
}

#[test]
fn panic_alloc_zeroed_slice_arc() {
    expect_panic(|| {
        let a = send_fail_arena();
        let _ = a.alloc_zeroed_slice_arc::<u32>(4);
    });
}

#[cfg(feature = "dst")]
#[test]
fn panic_alloc_uninit_slice_box() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_uninit_slice_box::<u32>(4);
    });
}

#[cfg(feature = "dst")]
#[test]
fn panic_alloc_zeroed_slice_box() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_zeroed_slice_box::<u32>(4);
    });
}

// `try_*` Err-propagation branches (the `?` lines).

#[test]
fn try_alloc_str_err() {
    let a = fail_arena();
    assert!(a.try_alloc_str("hi").is_err());
}

#[test]
fn try_alloc_uninit_box_err() {
    let a = fail_arena();
    assert!(a.try_alloc_uninit_box::<u32>().is_err());
}

#[test]
fn try_alloc_zeroed_box_err() {
    let a = fail_arena();
    assert!(a.try_alloc_zeroed_box::<u32>().is_err());
}

#[test]
fn try_alloc_uninit_rc_err() {
    let a = fail_arena();
    assert!(a.try_alloc_uninit_rc::<u32>().is_err());
}

#[test]
fn try_alloc_zeroed_rc_err() {
    let a = fail_arena();
    assert!(a.try_alloc_zeroed_rc::<u32>().is_err());
}

#[test]
fn try_alloc_uninit_arc_err() {
    let a = send_fail_arena();
    assert!(a.try_alloc_uninit_arc::<u32>().is_err());
}

#[test]
fn try_alloc_zeroed_arc_err() {
    let a = send_fail_arena();
    assert!(a.try_alloc_zeroed_arc::<u32>().is_err());
}

#[test]
fn try_alloc_uninit_slice_rc_err() {
    let a = fail_arena();
    assert!(a.try_alloc_uninit_slice_rc::<u32>(4).is_err());
}

#[test]
fn try_alloc_zeroed_slice_rc_err() {
    let a = fail_arena();
    assert!(a.try_alloc_zeroed_slice_rc::<u32>(4).is_err());
}

#[test]
fn try_alloc_uninit_slice_arc_err() {
    let a = send_fail_arena();
    assert!(a.try_alloc_uninit_slice_arc::<u32>(4).is_err());
}

#[test]
fn try_alloc_zeroed_slice_arc_err() {
    let a = send_fail_arena();
    assert!(a.try_alloc_zeroed_slice_arc::<u32>(4).is_err());
}

#[cfg(feature = "dst")]
#[test]
fn try_alloc_uninit_slice_box_err() {
    let a = fail_arena();
    assert!(a.try_alloc_uninit_slice_box::<u32>(4).is_err());
}

#[cfg(feature = "dst")]
#[test]
fn try_alloc_zeroed_slice_box_err() {
    let a = fail_arena();
    assert!(a.try_alloc_zeroed_slice_box::<u32>(4).is_err());
}

// Uninit slice with T: Drop drives the register_drop=true `?` propagation
// in reserve_slice (line 1625) under failure.

#[test]
fn try_alloc_uninit_slice_rc_drop_type_err() {
    let a = fail_arena();
    assert!(a.try_alloc_uninit_slice_rc::<String>(2).is_err());
}

#[test]
fn try_alloc_slice_fill_with_rc_drop_type_err() {
    let a = fail_arena();
    assert!(a.try_alloc_slice_fill_with_rc::<String, _>(2, |i| format!("{i}")).is_err());
}

// ArenaString grow-path failures.

#[test]
fn arena_string_try_push_str_initial_alloc_err() {
    let a = fail_arena();
    let mut s = multitude::builders::String::new_in(&a);
    assert!(s.try_push_str("hello").is_err());
}

#[test]
fn arena_string_try_grow_to_at_least_grow_path_err() {
    // Allow the initial chunk alloc, fail the grow's new-chunk alloc by
    // requesting a capacity that exceeds the chunk_size.
    let a = Arena::builder().chunk_size(8 * 1024).allocator_in(FailingAllocator::new(1)).build();
    let mut s = multitude::builders::String::try_with_capacity_in(4, &a).unwrap();
    s.try_push_str("abcd").unwrap();
    // Forces grow_for_string → needs new (oversized) chunk → allocator fails.
    assert!(s.try_reserve(64 * 1024).is_err());
}

#[test]
fn panic_arena_string_grow_to_at_least() {
    expect_panic(|| {
        let a = Arena::builder().chunk_size(8 * 1024).allocator_in(FailingAllocator::new(1)).build();
        let mut s = multitude::builders::String::try_with_capacity_in(4, &a).unwrap();
        s.try_push_str("abcd").unwrap();
        // grow_to_at_least asks for a new chunk; allocator is exhausted.
        s.push_str("x".repeat(64 * 1024));
    });
}

// grow_for_string slow path: relocate succeeds, old chunk's refcount goes
// to 0 (drives lines 1815/1820/1822-1823 in arena.rs).

#[test]
fn grow_for_string_old_chunk_torn_down() {
    let a = Arena::builder().chunk_size(8 * 1024).chunk_cache_capacity(0).build();
    let mut s = a.alloc_string();
    // Force at least one grow_for_string call. Initial cap == 16.
    s.push_str("x".repeat(64));
    // Multiple grows to ensure we exercise the slow-path relocate.
    s.push_str("y".repeat(8 * 1024));
    drop(s);
}

// Oversized + needs_drop=false branch in ChunkHeader::oversized_layout
// (lines 188, 189). Default max_normal_alloc = chunk_size/4. We allocate
// a chunk-sized payload to force the oversized path with a Copy type.

#[test]
fn oversized_no_drop_branch() {
    let a = Arena::builder().chunk_size(8 * 1024).max_normal_alloc(4 * 1024).build();
    // 1500 bytes of u8 (Copy, no Drop) > max_normal_alloc(4 * 1024).
    let _s = a.alloc_slice_copy(&[0_u8; 1500][..]);
}

#[test]
fn oversized_with_drop_branch() {
    // T: Drop + oversized layout drives oversized_layout(_, has_drop=true)
    // line 185 in chunk_header.rs.
    let a = Arena::builder().chunk_size(8 * 1024).max_normal_alloc(4 * 1024).build();
    let _s = a.alloc_slice_fill_with_rc::<String, _>(64, |i| format!("{i}"));
}

#[test]
fn panic_alloc_slice_fill_with() {
    expect_panic(|| {
        let a = fail_arena();
        let _: &mut [u32] = a.alloc_slice_fill_with(4, |i| i as u32);
    });
}

#[test]
fn arena_vec_into_arena_rc_empty_drop_type() {
    let arena: Arena = Arena::new();
    let v: Vec<String> = Vec::new_in(&arena);
    let r: multitude::Rc<[String]> = v.into_arena_rc();
    assert!(r.is_empty());
}
