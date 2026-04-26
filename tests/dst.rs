//! Tests for the in-progress reservation types
//! [`PendingArenaRc`] and [`PendingArenaArc`] (DST construction support).

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::cast_ptr_alignment, reason = "tests mirror internal allocator pointer math")]
#![allow(clippy::multiple_unsafe_ops_per_block, reason = "tests group related unsafe ops")]
#![allow(clippy::ref_as_ptr, reason = "test exercises raw-pointer based finalize API")]

use core::sync::atomic::{AtomicUsize, Ordering};
use harena::Arena;

#[test]
fn pending_arena_rc_finalize_writes_value() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let mut pa = arena.alloc_uninit_dst(layout);
    assert_eq!(pa.layout(), layout);
    let p = pa.as_mut_ptr();
    // SAFETY: we initialize the reservation before finalizing.
    unsafe { p.cast::<u32>().write(1234) };
    let template: u32 = 0;
    // SAFETY: no drop required for u32; bytes initialized.
    let r = unsafe { pa.finalize::<u32>(&raw const template, None) };
    assert_eq!(*r, 1234);
}

#[test]
fn pending_arena_rc_drop_without_finalize_leaks_reservation() {
    // Should NOT crash; the chunk refcount is released and the reserved
    // bytes are leaked (no drop registered).
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = arena.alloc_uninit_dst(layout);
    drop(pa);
    let v = arena.alloc(7_u32);
    assert_eq!(*v, 7);
}

#[test]
fn pending_arena_rc_debug_has_layout() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = arena.alloc_uninit_dst(layout);
    let s = format!("{pa:?}");
    assert!(s.contains("PendingArenaRc"));
    assert!(s.contains("layout"));
}

#[test]
fn try_alloc_uninit_dst_succeeds() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = arena.try_alloc_uninit_dst(layout).unwrap();
    drop(pa);
}

#[test]
fn pending_arena_arc_finalize_writes_value() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u64>();
    let mut pa = arena.alloc_uninit_dst_shared(layout);
    assert_eq!(pa.layout(), layout);
    let p = pa.as_mut_ptr();
    // SAFETY: we initialize before finalizing.
    unsafe { p.cast::<u64>().write(0xDEAD_BEEF) };
    let template: u64 = 0;
    // SAFETY: no drop needed; bytes initialized.
    let r = unsafe { pa.finalize::<u64>(&raw const template, None) };
    assert_eq!(*r, 0xDEAD_BEEF);
}

#[test]
fn pending_arena_arc_drop_without_finalize_leaks_reservation() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = arena.alloc_uninit_dst_shared(layout);
    drop(pa);
    let v = arena.alloc(7_u32);
    assert_eq!(*v, 7);
}

#[test]
fn pending_arena_arc_debug_has_layout() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = arena.alloc_uninit_dst_shared(layout);
    let s = format!("{pa:?}");
    assert!(s.contains("PendingArenaArc"));
    assert!(s.contains("layout"));
}

#[test]
fn try_alloc_uninit_dst_shared_succeeds() {
    let arena = Arena::new();
    let layout = core::alloc::Layout::new::<u32>();
    let pa = arena.try_alloc_uninit_dst_shared(layout).unwrap();
    drop(pa);
}

#[test]
fn finalize_with_drop_fn_runs_drop_at_chunk_teardown() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Counter(u32);
    impl Drop for Counter {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }
    unsafe fn drop_counter(entry: *mut harena::DropEntry) {
        // Replicate the standard sized-T drop shim layout: value lives
        // immediately after the entry slot, aligned to align_of::<T>().
        // SAFETY: caller guarantees `entry` was constructed by the
        // matching reservation path with a Counter immediately after.
        unsafe {
            let after = entry.byte_add(size_of::<harena::DropEntry>());
            let align = align_of::<Counter>();
            let misalign = (after.cast::<u8>() as usize) & (align - 1);
            let padding = if misalign == 0 { 0 } else { align - misalign };
            let value_ptr = after.byte_add(padding).cast::<Counter>();
            core::ptr::drop_in_place(value_ptr);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    {
        let arena = Arena::new();
        let layout = core::alloc::Layout::new::<Counter>();
        let mut pa = arena.alloc_uninit_dst(layout);
        let p = pa.as_mut_ptr();
        // SAFETY: we initialize before finalize.
        unsafe { p.cast::<Counter>().write(Counter(42)) };
        let template = Counter(0);
        // SAFETY: drop_fn matches the layout we wrote.
        let r = unsafe { pa.finalize::<Counter>(&raw const template, Some(drop_counter)) };
        core::mem::forget(template);
        assert_eq!(r.0, 42);
        assert_eq!(COUNT.load(Ordering::SeqCst), 0);
        drop(r);
    }
    // Drop runs at chunk teardown when the arena is released.
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
}
