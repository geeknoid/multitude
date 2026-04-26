//! Tests for [`PendingBox`] / [`Arena::alloc_uninit_dst_box`] —
//! the owned-smart pointer DST construction path. Gated on the `dst` Cargo
//! feature (which pulls in `ptr_meta` for fat-pointer metadata
//! support).

#![cfg(feature = "dst")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::cast_ptr_alignment, reason = "tests mirror internal allocator pointer math")]
#![allow(clippy::multiple_unsafe_ops_per_block, reason = "tests group related unsafe ops")]
#![allow(clippy::ref_as_ptr, reason = "test exercises raw-pointer based finalize API")]
#![allow(clippy::missing_panics_doc, reason = "test code")]
#![allow(clippy::cast_possible_truncation, reason = "test indices are small and bounded")]

use core::sync::atomic::{AtomicUsize, Ordering};
use multitude::Arena;
use multitude::dst::DropEntry;

#[test]
fn pending_arena_box_finalize_sized_value() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let mut pa = multitude::dst::PendingBox::new(&arena, layout);
    assert_eq!(pa.layout(), layout);
    let p = pa.as_mut_ptr();
    // SAFETY: writing a u32 into the reservation, then finalizing.
    unsafe { p.cast::<u32>().write(0xCAFE_BABE_u32) };
    let template: u32 = 0;
    // SAFETY: u32: !needs_drop, so drop_fn is None.
    let b = unsafe { pa.finalize::<u32>(&raw const template, None) };
    assert_eq!(*b, 0xCAFE_BABE);
}

#[test]
fn pending_arena_box_drop_without_finalize_releases_chunk() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = multitude::dst::PendingBox::new(&arena, layout);
    drop(pa);
    // Arena should still be usable.
    let v = arena.alloc_rc(7_u32);
    assert_eq!(*v, 7);
}

#[test]
fn pending_arena_box_debug_has_layout() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = multitude::dst::PendingBox::new(&arena, layout);
    let s = format!("{pa:?}");
    assert!(s.contains("PendingBox"));
    assert!(s.contains("layout"));
}

#[test]
fn try_alloc_uninit_dst_box_succeeds() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = multitude::dst::PendingBox::try_new(&arena, layout).unwrap();
    drop(pa);
}

// ---------------------------------------------------------------------------
// Sized value with Drop — the box's Drop must run the destructor when the smart pointer is dropped.
// ---------------------------------------------------------------------------

#[test]
fn arena_box_drops_sized_value_on_handle_drop() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Counter(u32);
    impl Drop for Counter {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }
    unsafe fn drop_counter(entry: *mut DropEntry) {
        // Standard sized-T drop shim: value lives immediately after the
        // entry slot, aligned to align_of::<Counter>().
        // SAFETY: caller guarantees `entry` was constructed by the
        // matching reservation path with a Counter immediately after.
        unsafe {
            let after = entry.byte_add(size_of::<DropEntry>());
            let align = align_of::<Counter>();
            let misalign = (after.cast::<u8>() as usize) & (align - 1);
            let padding = if misalign == 0 { 0 } else { align - misalign };
            let value_ptr = after.byte_add(padding).cast::<Counter>();
            core::ptr::drop_in_place(value_ptr);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<Counter>();
    let mut pa = multitude::dst::PendingBox::new(&arena, layout);
    let p = pa.as_mut_ptr();
    // SAFETY: writing the Counter into the reservation.
    unsafe { p.cast::<Counter>().write(Counter(99)) };
    let template = Counter(0);
    // SAFETY: drop_fn matches the layout written.
    let b = unsafe { pa.finalize::<Counter>(&raw const template, Some(drop_counter)) };
    core::mem::forget(template);
    assert_eq!(b.0, 99);
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);

    // Drop the smart pointer BEFORE the arena — ArenaBox must run drop now.
    drop(b);
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
    drop(arena);
    // No additional drops at arena teardown.
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Real DST: a slice value (arena-owned `[u8]`).
// ---------------------------------------------------------------------------

#[test]
fn arena_box_holds_byte_slice_dst() {
    let arena = Arena::new();
    let len = 10_usize;
    let layout = core::alloc::Layout::array::<u8>(len).unwrap();
    let mut pa = multitude::dst::PendingBox::new(&arena, layout);
    let p = pa.as_mut_ptr();
    // SAFETY: writing 10 bytes into the reservation.
    unsafe {
        for i in 0..len {
            p.add(i).write(i as u8);
        }
    }
    // Build a fat-pointer template `*const [u8]` of length `len` whose
    // data half doesn't matter (will be replaced by reconstruct_fat).
    let template_arr = [0_u8; 10];
    let template: *const [u8] = &raw const template_arr[..];
    // SAFETY: bytes initialized; metadata (len=10) matches; [u8]: !needs_drop.
    let b = unsafe { pa.finalize::<[u8]>(template, None) };
    assert_eq!(b.len(), 10);
    for (i, byte) in b.iter().enumerate() {
        assert_eq!(*byte, i as u8);
    }
}

// ---------------------------------------------------------------------------
// Real DST with Drop: a slice of String.
// ---------------------------------------------------------------------------

#[test]
fn arena_box_holds_string_slice_dst_with_drop() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    // Sentinel string we control the drop-count on. Use a String that we
    // can detect via a global counter.
    struct Tracked(String);
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    unsafe fn drop_slice_of_tracked(entry: *mut DropEntry) {
        // Layout of the slice DST inside the chunk: the entry header
        // sits immediately before the slice elements (aligned to
        // align_of::<Tracked>()). The element count is stored in the
        // entry's `slice_len` field by `link_drop_entry` — but
        // PendingBox::finalize calls it with a 0 sentinel for
        // single-value semantics. For this test we pass a custom drop_fn
        // that re-derives `len` from the SAME slice_len field by
        // reading directly from the entry.
        //
        // The chunk's drop registration here always uses slice_len=0
        // because PendingBox uses the single-value link path. So
        // for slice-of-Drop DSTs, the user must pass a drop_fn that
        // *knows* the slice length out of band. dst-factory typically
        // bakes this in via the type's metadata; here we test a manual
        // case where the drop_fn captures len via a closure-equivalent.
        //
        // Concretely: we hard-code len=3 in this test's drop_fn.
        // SAFETY: caller built a [Tracked; 3] starting just past the entry.
        unsafe {
            let after = entry.byte_add(size_of::<DropEntry>());
            let align = align_of::<Tracked>();
            let misalign = (after.cast::<u8>() as usize) & (align - 1);
            let padding = if misalign == 0 { 0 } else { align - misalign };
            let base = after.byte_add(padding).cast::<Tracked>();
            for i in 0..3 {
                core::ptr::drop_in_place(base.add(i));
            }
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    let layout = core::alloc::Layout::array::<Tracked>(3).unwrap();
    let mut pa = multitude::dst::PendingBox::new(&arena, layout);
    let p = pa.as_mut_ptr().cast::<Tracked>();
    // SAFETY: write 3 Tracked values.
    unsafe {
        p.add(0).write(Tracked("a".to_string()));
        p.add(1).write(Tracked("b".to_string()));
        p.add(2).write(Tracked("c".to_string()));
    }
    let template_arr = [
        Tracked(std::string::String::new()),
        Tracked(std::string::String::new()),
        Tracked(std::string::String::new()),
    ];
    let template: *const [Tracked] = &raw const template_arr[..];
    // SAFETY: data initialized; metadata matches.
    let b = unsafe { pa.finalize::<[Tracked]>(template, Some(drop_slice_of_tracked)) };
    // Don't double-drop the templates.
    for t in template_arr {
        core::mem::forget(t);
    }
    assert_eq!(b.len(), 3);
    assert_eq!(b[0].0, "a");
    assert_eq!(b[1].0, "b");
    assert_eq!(b[2].0, "c");

    let before = COUNT.load(Ordering::SeqCst);
    drop(b);
    let after = COUNT.load(Ordering::SeqCst);
    assert_eq!(after - before, 3, "all three Tracked must drop on box drop");
    drop(arena);
    assert_eq!(COUNT.load(Ordering::SeqCst), after, "no extra drops at arena teardown");
}

// ---------------------------------------------------------------------------
// PendingRc::finalize_dst — true DST in an ArenaRc smart pointer.
// ---------------------------------------------------------------------------

#[test]
fn pending_arena_rc_finalize_dst_byte_slice() {
    let arena = Arena::new();
    let len = 5_usize;
    let layout = core::alloc::Layout::array::<u8>(len).unwrap();
    let mut pa = multitude::dst::PendingRc::new(&arena, layout);
    let p = pa.as_mut_ptr();
    // SAFETY: writing 5 bytes.
    unsafe {
        for i in 0..len {
            p.add(i).write((i + 100) as u8);
        }
    }
    let template_arr = [0_u8; 5];
    let template: *const [u8] = &raw const template_arr[..];
    // SAFETY: bytes initialized; metadata matches; [u8]: !needs_drop.
    let r = unsafe { pa.finalize_dst::<[u8]>(template, None) };
    assert_eq!(r.len(), 5);
    for (i, byte) in r.iter().enumerate() {
        assert_eq!(*byte, (i + 100) as u8);
    }
    // ArenaRc smart pointers outlive the arena.
    drop(arena);
    assert_eq!(r[0], 100);
}

#[test]
fn pending_arena_rc_finalize_dst_outlives_arena() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::array::<u32>(4).unwrap();
    let mut pa = multitude::dst::PendingRc::new(&arena, layout);
    let p = pa.as_mut_ptr();
    // SAFETY: write 4 u32s.
    unsafe {
        let q = p.cast::<u32>();
        q.add(0).write(11);
        q.add(1).write(22);
        q.add(2).write(33);
        q.add(3).write(44);
    }
    let template_arr = [0_u32; 4];
    let template: *const [u32] = &raw const template_arr[..];
    // SAFETY: bytes initialized; metadata matches.
    let r = unsafe { pa.finalize_dst::<[u32]>(template, None) };
    drop(arena);
    assert_eq!(&*r, &[11, 22, 33, 44]);
}

// ---------------------------------------------------------------------------
// PendingArc::finalize_dst — true DST in an ArenaArc smart pointer.
// ---------------------------------------------------------------------------

#[test]
fn pending_arena_arc_finalize_dst_byte_slice() {
    let arena = Arena::new();
    let len = 5_usize;
    let layout = core::alloc::Layout::array::<u8>(len).unwrap();
    let mut pa = multitude::dst::PendingArc::new(&arena, layout);
    let p = pa.as_mut_ptr();
    // SAFETY: writing 5 bytes.
    unsafe {
        for i in 0..len {
            p.add(i).write((i + 200) as u8);
        }
    }
    let template_arr = [0_u8; 5];
    let template: *const [u8] = &raw const template_arr[..];
    // SAFETY: bytes initialized; metadata matches; [u8]: !needs_drop.
    let r = unsafe { pa.finalize_dst::<[u8]>(template, None) };
    assert_eq!(r.len(), 5);
    for (i, byte) in r.iter().enumerate() {
        assert_eq!(*byte, (i + 200) as u8);
    }
}

#[test]
fn pending_arena_arc_finalize_dst_send_to_thread() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::array::<u32>(3).unwrap();
    let mut pa = multitude::dst::PendingArc::new(&arena, layout);
    let p = pa.as_mut_ptr().cast::<u32>();
    // SAFETY: write 3 u32s.
    unsafe {
        p.add(0).write(7);
        p.add(1).write(8);
        p.add(2).write(9);
    }
    let template_arr = [0_u32; 3];
    let template: *const [u32] = &raw const template_arr[..];
    // SAFETY: bytes initialized; metadata matches.
    let r = unsafe { pa.finalize_dst::<[u32]>(template, None) };
    let r2 = r.clone();
    let h = std::thread::spawn(move || {
        let sum: u32 = r2.iter().sum();
        sum
    });
    assert_eq!(h.join().unwrap(), 24);
    assert_eq!(&*r, &[7, 8, 9]);
}

// ---------------------------------------------------------------------------
// finalize_dst with drop_fn — exercises the entry-linking branch on Rc/Arc.
// ---------------------------------------------------------------------------

#[test]
fn pending_arena_rc_finalize_dst_with_drop_fn_runs_drop() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked(u32);
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }
    unsafe fn drop_tracked(entry: *mut DropEntry) {
        // SAFETY: entry was constructed by alloc_uninit_dst_rc with a Tracked immediately after.
        unsafe {
            let after = entry.byte_add(size_of::<DropEntry>());
            let align = align_of::<Tracked>();
            let misalign = (after.cast::<u8>() as usize) & (align - 1);
            let padding = if misalign == 0 { 0 } else { align - misalign };
            let value_ptr = after.byte_add(padding).cast::<Tracked>();
            core::ptr::drop_in_place(value_ptr);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    {
        let arena = Arena::new();
        let layout = core::alloc::Layout::new::<Tracked>();
        let mut pa = multitude::dst::PendingRc::new(&arena, layout);
        // SAFETY: writing one Tracked into the reservation.
        unsafe { pa.as_mut_ptr().cast::<Tracked>().write(Tracked(7)) };
        let template = Tracked(0);
        // SAFETY: drop_fn matches the Tracked layout.
        let r = unsafe { pa.finalize_dst::<Tracked>(&raw const template, Some(drop_tracked)) };
        core::mem::forget(template);
        assert_eq!(r.0, 7);
        assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    } // arena drops; chunk teardown runs the drop entry.
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
}

#[test]
fn pending_arena_arc_finalize_dst_with_drop_fn_runs_drop() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked(u32);
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }
    unsafe fn drop_tracked(entry: *mut DropEntry) {
        // SAFETY: see pending_arena_rc_finalize_dst_with_drop_fn_runs_drop.
        unsafe {
            let after = entry.byte_add(size_of::<DropEntry>());
            let align = align_of::<Tracked>();
            let misalign = (after.cast::<u8>() as usize) & (align - 1);
            let padding = if misalign == 0 { 0 } else { align - misalign };
            let value_ptr = after.byte_add(padding).cast::<Tracked>();
            core::ptr::drop_in_place(value_ptr);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    {
        let arena = Arena::new();
        let layout = core::alloc::Layout::new::<Tracked>();
        let mut pa = multitude::dst::PendingArc::new(&arena, layout);
        // SAFETY: writing one Tracked.
        unsafe { pa.as_mut_ptr().cast::<Tracked>().write(Tracked(8)) };
        let template = Tracked(0);
        // SAFETY: drop_fn matches the Tracked layout.
        let r = unsafe { pa.finalize_dst::<Tracked>(&raw const template, Some(drop_tracked)) };
        core::mem::forget(template);
        assert_eq!(r.0, 8);
        assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    }
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Arena::alloc_slice_*_box — owned ArenaBox<[T]> smart pointers via slice helpers.
// These are the dst-feature `_box` symmetry counterparts to the existing
// `_rc` and `_arc` slice constructors on Arena.
// ---------------------------------------------------------------------------

#[test]
fn alloc_slice_copy_box_basic() {
    let arena = Arena::new();
    let b = arena.alloc_slice_copy_box([1_u32, 2, 3]);
    assert_eq!(&*b, &[1, 2, 3]);
}

#[test]
fn alloc_slice_copy_box_mutable() {
    let arena = Arena::new();
    let mut b = arena.alloc_slice_copy_box([10_u32, 20, 30]);
    b[1] = 200;
    assert_eq!(&*b, &[10, 200, 30]);
}

#[test]
fn try_alloc_slice_copy_box_works() {
    let arena = Arena::new();
    let b = arena.try_alloc_slice_copy_box([1_u8, 2, 3]).unwrap();
    assert_eq!(&*b, &[1, 2, 3]);
}

#[test]
fn alloc_slice_clone_box_basic() {
    let arena = Arena::new();
    let originals = [
        std::string::String::from("a"),
        std::string::String::from("b"),
        std::string::String::from("c"),
    ];
    let b = arena.alloc_slice_clone_box(&originals);
    assert_eq!(b.len(), 3);
    assert_eq!(b[0], "a");
    assert_eq!(b[2], "c");
}

#[test]
fn try_alloc_slice_clone_box_works() {
    let arena = Arena::new();
    let b = arena.try_alloc_slice_clone_box([100_u32, 200]).unwrap();
    assert_eq!(&*b, &[100, 200]);
}

#[test]
fn alloc_slice_fill_with_box_basic() {
    let arena = Arena::new();
    let b: multitude::Box<[u64]> = arena.alloc_slice_fill_with_box(5, |i| (i as u64) * 10);
    assert_eq!(&*b, &[0, 10, 20, 30, 40]);
}

#[test]
fn try_alloc_slice_fill_with_box_works() {
    let arena = Arena::new();
    let b: multitude::Box<[u32]> = arena.try_alloc_slice_fill_with_box(3, |i| u32::try_from(i + 100).unwrap()).unwrap();
    assert_eq!(&*b, &[100, 101, 102]);
}

#[test]
fn alloc_slice_fill_iter_box_basic() {
    let arena = Arena::new();
    let b: multitude::Box<[i32]> = arena.alloc_slice_fill_iter_box([7_i32, 8, 9]);
    assert_eq!(&*b, &[7, 8, 9]);
}

#[test]
fn try_alloc_slice_fill_iter_box_works() {
    let arena = Arena::new();
    let b: multitude::Box<[u32]> = arena.try_alloc_slice_fill_iter_box([42_u32, 43, 44]).unwrap();
    assert_eq!(&*b, &[42, 43, 44]);
}

#[test]
fn alloc_slice_fill_iter_box_empty() {
    let arena = Arena::new();
    let b: multitude::Box<[u32]> = arena.alloc_slice_fill_iter_box(core::iter::empty::<u32>());
    assert!(b.is_empty());
}

// ---------------------------------------------------------------------------
// Drop semantics: ArenaBox<[T]>::Drop must run T::drop on each element
// IMMEDIATELY before the chunk reclaims.
// ---------------------------------------------------------------------------

#[test]
fn alloc_slice_clone_box_drops_elements_immediately() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    #[derive(Clone)]
    struct Tracked;
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    let originals = [Tracked, Tracked, Tracked];
    let b = arena.alloc_slice_clone_box(&originals);
    assert_eq!(b.len(), 3);
    let count_before = COUNT.load(Ordering::SeqCst);
    drop(b);
    let count_after = COUNT.load(Ordering::SeqCst);
    assert_eq!(count_after - count_before, 3, "drop_in_place([T;3]) must drop each element");

    // The arena drop must NOT run drops again (entry was unlinked).
    drop(originals);
    drop(arena);
    // After arena drop: count includes the originals drop (3 more), but no
    // double-drop of the box's elements.
    assert_eq!(COUNT.load(Ordering::SeqCst), count_after + 3);
}

#[test]
fn alloc_slice_fill_with_box_drops_elements_immediately() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked;
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    let b: multitude::Box<[Tracked]> = arena.alloc_slice_fill_with_box(5, |_| Tracked);
    assert_eq!(b.len(), 5);
    let before = COUNT.load(Ordering::SeqCst);
    drop(b);
    assert_eq!(COUNT.load(Ordering::SeqCst), before + 5);
    drop(arena); // No double-drop.
    assert_eq!(COUNT.load(Ordering::SeqCst), before + 5);
}

#[test]
fn alloc_slice_copy_box_no_drop_for_copy_types() {
    // T: Copy means no DropEntry slot is reserved. Verify that ArenaBox::Drop
    // for [T: Copy] correctly does NOT try to unlink an entry. (needs_drop::<[T]>
    // is false when T: Copy + !Drop.)
    let arena = Arena::new();
    let b = arena.alloc_slice_copy_box([1_u8, 2, 3, 4, 5]);
    assert_eq!(b.len(), 5);
    drop(b);
    // Arena still works:
    let b2 = arena.alloc_slice_copy_box([9_u8, 8, 7]);
    assert_eq!(&*b2, &[9, 8, 7]);
}

#[test]
fn alloc_slice_fill_with_box_zero_len_works() {
    let arena = Arena::new();
    let b: multitude::Box<[u32]> = arena.alloc_slice_fill_with_box(0, |_| panic!("never called"));
    assert!(b.is_empty());
}

#[test]
fn alloc_slice_fill_with_box_zst_with_drop() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct ZstDrop;
    impl Drop for ZstDrop {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    let b: multitude::Box<[ZstDrop]> = arena.alloc_slice_fill_with_box(7, |_| ZstDrop);
    drop(b);
    assert_eq!(COUNT.load(Ordering::SeqCst), 7);
}

// ---------------------------------------------------------------------------
// Panic safety: SliceInitGuard drops the initialized prefix on init panic.
// ---------------------------------------------------------------------------

#[test]
fn alloc_slice_fill_with_box_panic_drops_initialized_prefix() {
    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
    struct DropCounter;
    impl Drop for DropCounter {
        fn drop(&mut self) {
            let _ = DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    DROP_COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Panic at index 3, after producing 3 DropCounters (indexes 0, 1, 2).
        let _b: multitude::Box<[DropCounter]> = arena.alloc_slice_fill_with_box(10, |i| {
            assert!(i != 3, "intentional");
            DropCounter
        });
    }));
    assert!(result.is_err());
    // Three already-init elements must have been dropped by SliceInitGuard.
    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    // Arena is still usable after the panic.
    let b: multitude::Box<[u32]> = arena.alloc_slice_fill_with_box(2, |i| u32::try_from(i).unwrap());
    assert_eq!(&*b, &[0, 1]);
}

// ---------------------------------------------------------------------------
// Iterator-length-mismatch panic (matches the _rc/_arc family).
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "iterator shorter than ExactSizeIterator len")]
fn alloc_slice_fill_iter_box_panics_on_short_iter() {
    struct Liar(usize);
    impl Iterator for Liar {
        type Item = u32;
        fn next(&mut self) -> Option<u32> {
            None
        }
    }
    impl ExactSizeIterator for Liar {
        fn len(&self) -> usize {
            self.0
        }
    }
    let arena = Arena::new();
    let _b: multitude::Box<[u32]> = arena.alloc_slice_fill_iter_box(Liar(2));
}

// ---------------------------------------------------------------------------
// High-alignment regression: the reserve_slice fix (over-aligning entry to
// max(align_of::<DropEntry>(), align_of::<T>())) must keep the
// ArenaBox<[T]>::Drop reverse formula correct for high-align T with Drop.
// ---------------------------------------------------------------------------

#[test]
fn alloc_slice_box_high_alignment_drop_locates_entry_correctly() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    #[repr(align(32))]
    struct A32(#[expect(dead_code, reason = "field present only to give the type a non-zero size")] u8);
    impl Drop for A32 {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    // Force the bump cursor away from a 32-aligned position so the
    // alignment math is non-trivial.
    let _decoy: &mut u8 = arena.alloc(0_u8);
    let b: multitude::Box<[A32]> = arena.alloc_slice_fill_with_box(4, |_| A32(0));
    assert_eq!(b.len(), 4);
    let before = COUNT.load(Ordering::SeqCst);
    drop(b);
    assert_eq!(COUNT.load(Ordering::SeqCst), before + 4);
}

// ---------------------------------------------------------------------------
// ArenaBox<[T]>::into_rc — slice version of the box → rc conversion.
// ---------------------------------------------------------------------------

#[test]
fn arena_box_slice_into_rc_basic() {
    let arena = Arena::new();
    let b = arena.alloc_slice_copy_box([1_u32, 2, 3]);
    let r = b.into_rc();
    assert_eq!(&*r, &[1, 2, 3]);
}

#[test]
fn arena_box_slice_into_rc_after_mutation() {
    let arena = Arena::new();
    let mut b = arena.alloc_slice_copy_box([10_u32, 20, 30]);
    b[1] = 99;
    let r = b.into_rc();
    assert_eq!(&*r, &[10, 99, 30]);
}

#[test]
fn arena_box_slice_into_rc_outlives_arena() {
    let r = {
        let arena = Arena::new();
        let b = arena.alloc_slice_copy_box([7_u8, 8, 9]);
        b.into_rc()
    };
    assert_eq!(&*r, &[7, 8, 9]);
}

#[test]
fn arena_box_slice_into_rc_preserves_drop_semantics() {
    // Each element drops exactly once — when the rc smart pointer (and its
    // chunk) is finally torn down. Crucially, NOT during the conversion.
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked;
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    let b: multitude::Box<[Tracked]> = arena.alloc_slice_fill_with_box(3, |_| Tracked);
    // Conversion: no drops yet.
    let r = b.into_rc();
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(r.len(), 3);
    drop(r);
    drop(arena);
    // Each element dropped exactly once (no double-drop, no missed drop).
    assert_eq!(COUNT.load(Ordering::SeqCst), 3);
}

#[test]
fn arena_box_slice_into_rc_clones_share_chunk() {
    let arena = Arena::new();
    let b = arena.alloc_slice_copy_box([42_u32; 5]);
    let r = b.into_rc();
    let r2 = r.clone();
    drop(r);
    assert_eq!(&*r2, &[42; 5]);
}
