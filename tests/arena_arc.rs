//! Tests for [`ArenaArc`]: cross-thread shareable refcounted handle.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std for thread/sync primitives")]
#![allow(clippy::unwrap_used, reason = "test code")]

mod common;

use core::cmp::Ordering;
use harena::{Arena, ArenaArc};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

#[test]
fn cross_thread_arena_arc() {
    let arena = Arena::new();
    let shared: ArenaArc<u64> = arena.alloc_shared(99);
    let s2 = shared.clone();
    let h = std::thread::spawn(move || *s2);
    assert_eq!(*shared, 99);
    assert_eq!(99, h.join().unwrap());
}

#[test]
fn shared_alloc_with_drop_type() {
    let arena = Arena::new();
    let counter = Arc::new(AtomicUsize::new(0));
    let value = Arc::clone(&counter);
    let shared = arena.alloc_shared(value);
    let s2 = shared.clone();
    drop(shared);
    drop(s2);
    drop(arena);
    assert_eq!(Arc::strong_count(&counter), 1);
}

#[test]
fn try_alloc_shared_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_shared(99_u32).unwrap();
    assert_eq!(*r, 99);
}

#[test]
fn try_alloc_with_shared_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_with_shared(|| 314_u32).unwrap();
    assert_eq!(*r, 314);
}

#[test]
fn alloc_with_shared_constructs_in_place() {
    let arena = Arena::new();
    let r = arena.alloc_with_shared(|| 271_u32);
    let r2 = r.clone();
    let h = std::thread::spawn(move || *r2);
    assert_eq!(*r, 271);
    assert_eq!(h.join().unwrap(), 271);
}

#[test]
fn alloc_slice_copy_shared_works() {
    let arena = Arena::new();
    let r = arena.alloc_slice_copy_shared(&[7_u32, 8, 9]);
    let r2 = r.clone();
    let h = std::thread::spawn(move || r2[1]);
    assert_eq!(&*r, &[7, 8, 9]);
    assert_eq!(h.join().unwrap(), 8);
}

#[test]
fn try_alloc_slice_copy_shared_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_slice_copy_shared(&[7_u32, 8, 9]).unwrap();
    assert_eq!(&*r, &[7, 8, 9]);
}

#[test]
fn as_ptr_returns_value_ptr() {
    let arena = Arena::new();
    let r = arena.alloc_shared(42_u32);
    let p = ArenaArc::as_ptr(&r);
    // SAFETY: ptr is valid while the handle lives.
    assert_eq!(unsafe { *p }, 42);
}

#[test]
fn chunk_ref_count_tracks_clone_and_drop() {
    let arena = Arena::new();
    let r = arena.alloc_shared(1_u32);
    let initial = ArenaArc::chunk_ref_count(&r);
    let r2 = r.clone();
    assert_eq!(ArenaArc::chunk_ref_count(&r), initial + 1);
    drop(r2);
    assert_eq!(ArenaArc::chunk_ref_count(&r), initial);
}

#[test]
fn ptr_eq_distinguishes_handles() {
    let arena = Arena::new();
    let a = arena.alloc_shared(1_u32);
    let b = a.clone();
    let c = arena.alloc_shared(1_u32);
    assert!(ArenaArc::ptr_eq(&a, &b));
    assert!(!ArenaArc::ptr_eq(&a, &c));
}

#[test]
fn debug_display_compare_hash() {
    let arena = Arena::new();
    let a = arena.alloc_shared(1_u32);
    let b = arena.alloc_shared(2_u32);
    assert_eq!(format!("{a:?}"), "1");
    assert_eq!(format!("{a}"), "1");
    assert!(a < b);
    assert_eq!(a.cmp(&b), Ordering::Less);
    assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    let _ = common::hash_of(&a);
}

#[test]
fn cross_thread_drop_no_use_after_free() {
    // Stress-test: many short-lived shared handles handed across threads.
    // Validates that the dec_ref-on-non-owner-thread path is sound.
    let arena = Arena::new();
    let handles: Vec<_> = (0..100_u32)
        .map(|i| {
            let h = arena.alloc_shared(i);
            std::thread::spawn(move || *h)
        })
        .collect();
    let mut sum = 0_u64;
    for h in handles {
        sum += u64::from(h.join().unwrap());
    }
    assert_eq!(sum, (0..100_u64).sum());

    let _ = AtomicOrdering::SeqCst; // suppress unused-import warning
}
