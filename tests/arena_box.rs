//! Tests for [`ArenaBox`]: owned, mutable single handle whose `Drop`
//! runs `T::drop` immediately on handle drop.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::collection_is_never_read, reason = "tests retain handles to keep chunks alive")]

mod common;

use core::cmp::Ordering;
use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use harena::{Arena, ArenaBox};

#[test]
fn alloc_box_runs_drop_immediately() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Counter;
    impl Drop for Counter {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    COUNT.store(0, AtomicOrdering::SeqCst);
    let arena = Arena::new();
    let b = arena.alloc_box(Counter);
    assert_eq!(COUNT.load(AtomicOrdering::SeqCst), 0);
    drop(b);
    assert_eq!(COUNT.load(AtomicOrdering::SeqCst), 1);
}

#[test]
fn alloc_box_mutable_access() {
    let arena = Arena::new();
    let mut b = arena.alloc_box(vec![1, 2, 3]);
    b.push(4);
    assert_eq!(*b, vec![1, 2, 3, 4]);
}

#[test]
fn alloc_box_with_copy_type_no_panic() {
    // Regression: ArenaBox<T: Copy> originally tried to unlink a non-existent
    // DropEntry. Verify many of them work.
    let arena = Arena::new();
    let mut handles = Vec::new();
    for i in 0..10_000_u64 {
        handles.push(arena.alloc_box(i));
    }
    let sum: u64 = handles.iter().map(|h| **h).sum();
    drop(handles);
    drop(arena);
    assert_eq!(sum, (0..10_000_u64).sum());
}

#[test]
fn try_alloc_box_succeeds() {
    let arena = Arena::new();
    let mut b = arena.try_alloc_box(vec![1_u32, 2, 3]).unwrap();
    b.push(4);
    assert_eq!(&*b, &[1, 2, 3, 4]);
}

#[test]
fn alloc_box_with_constructs_in_place() {
    let arena = Arena::new();
    let b = arena.alloc_box_with(|| String::from("placed-box"));
    assert_eq!(&**b, "placed-box");
}

#[test]
fn try_alloc_box_with_succeeds() {
    let arena = Arena::new();
    let b = arena.try_alloc_box_with(|| 42_u64).unwrap();
    assert_eq!(*b, 42);
}

#[test]
fn as_ptr_and_as_mut_ptr() {
    let arena = Arena::new();
    let mut b = arena.alloc_box(123_u32);
    let p = ArenaBox::as_ptr(&b);
    // SAFETY: ptr valid while the box lives.
    assert_eq!(unsafe { *p }, 123);
    let mp = ArenaBox::as_mut_ptr(&mut b);
    // SAFETY: unique &mut access.
    unsafe {
        *mp = 456;
    }
    assert_eq!(*b, 456);
}

#[test]
fn debug_display_compare_hash() {
    let arena = Arena::new();
    let a = arena.alloc_box(1_u32);
    let b = arena.alloc_box(2_u32);
    assert_eq!(format!("{a:?}"), "1");
    assert_eq!(format!("{a}"), "1");
    assert!(a < b);
    assert_eq!(a.cmp(&b), Ordering::Less);
    assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    let _ = common::hash_of(&a);
}
