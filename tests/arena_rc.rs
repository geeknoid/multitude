//! Tests for [`ArenaRc`]: single-threaded reference-counted handle.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::missing_asserts_for_indexing, reason = "test code is direct")]
#![allow(clippy::cast_possible_truncation, reason = "test code uses small integers")]

mod common;

use core::cmp::Ordering;
use harena::{Arena, ArenaRc};

#[test]
fn alloc_and_clone_basic() {
    let arena = Arena::new();
    let a = arena.alloc(42_u32);
    let b = a.clone();
    assert_eq!(*a, 42);
    assert_eq!(*b, 42);
    assert!(ArenaRc::ptr_eq(&a, &b));
}

#[test]
fn handles_outlive_arena() {
    let s = {
        let arena = Arena::new();
        arena.alloc(String::from("survives"))
    };
    assert_eq!(*s, "survives");
}

#[test]
fn try_alloc_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc(100_u32).unwrap();
    assert_eq!(*r, 100);
}

#[test]
fn alloc_with_constructs_in_place() {
    let arena = Arena::new();
    let r = arena.alloc_with(|| String::from("placed"));
    assert_eq!(&*r, "placed");
}

#[test]
fn try_alloc_with_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_with(|| 200_u32).unwrap();
    assert_eq!(*r, 200);
}

#[test]
fn as_ptr_returns_value_ptr() {
    let arena = Arena::new();
    let r = arena.alloc(42_u32);
    let p = ArenaRc::as_ptr(&r);
    // SAFETY: ptr returned by as_ptr is valid for the lifetime of the handle.
    assert_eq!(unsafe { *p }, 42);
}

#[test]
fn chunk_ref_count_tracks_clone_and_drop() {
    let arena = Arena::new();
    let r = arena.alloc(1_u32);
    let initial = ArenaRc::chunk_ref_count(&r);
    let r2 = r.clone();
    assert_eq!(ArenaRc::chunk_ref_count(&r), initial + 1);
    drop(r2);
    assert_eq!(ArenaRc::chunk_ref_count(&r), initial);
}

#[test]
fn ptr_eq_distinguishes_handles() {
    let arena = Arena::new();
    let a = arena.alloc(1_u32);
    let b = a.clone();
    let c = arena.alloc(1_u32);
    assert!(ArenaRc::ptr_eq(&a, &b));
    assert!(!ArenaRc::ptr_eq(&a, &c));
}

#[test]
fn debug_and_display() {
    let arena = Arena::new();
    let r = arena.alloc(42_u32);
    assert_eq!(format!("{r:?}"), "42");
    assert_eq!(format!("{r}"), "42");
}

#[test]
fn compare_and_hash() {
    let arena = Arena::new();
    let a = arena.alloc(1_u32);
    let b = arena.alloc(2_u32);
    let a2 = arena.alloc(1_u32);
    assert!(a != b);
    assert!(a == a2);
    assert_eq!(a.cmp(&b), Ordering::Less);
    assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    assert_eq!(common::hash_of(&a), common::hash_of(&a2));
}

// ---------------------------------------------------------------------------
// Slice constructors -> ArenaRc<[T]>
// ---------------------------------------------------------------------------

#[test]
fn slice_constructors() {
    let arena = Arena::new();
    let from_copy = arena.alloc_slice_copy(&[1u8, 2, 3, 4, 5]);
    assert_eq!(&*from_copy, &[1, 2, 3, 4, 5]);

    let from_clone = arena.alloc_slice_clone(&[String::from("a"), String::from("b")]);
    assert_eq!(from_clone.len(), 2);
    assert_eq!(&*from_clone[0], "a");

    let filled = arena.alloc_slice_fill_with(5, |i| i * 10);
    assert_eq!(&*filled, &[0, 10, 20, 30, 40]);
}

#[test]
fn try_alloc_slice_copy_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_slice_copy(&[1_u8, 2, 3]).unwrap();
    assert_eq!(&*r, &[1, 2, 3]);
}

#[test]
fn try_alloc_slice_clone_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_slice_clone(&[String::from("x"), String::from("y")]).unwrap();
    assert_eq!(&*r[0], "x");
    assert_eq!(&*r[1], "y");
}

#[test]
fn try_alloc_slice_fill_with_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_slice_fill_with(4, |i| i as u32 + 100).unwrap();
    assert_eq!(&*r, &[100, 101, 102, 103]);
}

#[test]
fn alloc_slice_fill_iter_succeeds() {
    let arena = Arena::new();
    let r = arena.alloc_slice_fill_iter(0_u32..5);
    assert_eq!(&*r, &[0, 1, 2, 3, 4]);
}

#[test]
fn try_alloc_slice_fill_iter_succeeds() {
    let arena = Arena::new();
    let r = arena.try_alloc_slice_fill_iter(10_u32..13).unwrap();
    assert_eq!(&*r, &[10, 11, 12]);
}

#[test]
fn alloc_slice_fill_with_drop_type_registers_drop() {
    let arena = Arena::new();
    let r = arena.alloc_slice_fill_with(3, |i| String::from(["a", "b", "c"][i]));
    assert_eq!(&*r[0], "a");
    assert_eq!(&*r[2], "c");
}

#[test]
fn alloc_slice_copy_empty() {
    let arena = Arena::new();
    let r = arena.alloc_slice_copy::<u32>(&[]);
    assert_eq!(r.len(), 0);
}

#[test]
fn alloc_slice_clone_empty() {
    let arena = Arena::new();
    let r = arena.alloc_slice_clone::<String>(&[]);
    assert_eq!(r.len(), 0);
}

#[test]
fn alloc_slice_fill_with_zero_len() {
    let arena = Arena::new();
    let r = arena.alloc_slice_fill_with::<u32, _>(0, |_| panic!("never called"));
    assert_eq!(r.len(), 0);
}

#[test]
#[should_panic(expected = "iterator shorter than ExactSizeIterator len")]
fn alloc_slice_fill_iter_panics_on_short_iter() {
    struct Liar(u32);
    impl Iterator for Liar {
        type Item = u32;
        fn next(&mut self) -> Option<u32> {
            if self.0 == 0 {
                None
            } else {
                self.0 -= 1;
                Some(0)
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (10, Some(10))
        }
    }
    impl ExactSizeIterator for Liar {
        fn len(&self) -> usize {
            10
        }
    }
    let arena = Arena::new();
    let _ = arena.alloc_slice_fill_iter(Liar(2));
}

#[test]
fn slice_clone_and_compare() {
    let arena = Arena::new();
    let a = arena.alloc_slice_copy(&[1_u32, 2, 3]);
    let b = a.clone();
    assert_eq!(&*a, &*b);
    assert!(ArenaRc::ptr_eq(&a, &b));
    let c = arena.alloc_slice_copy(&[1_u32, 2, 3]);
    assert!(!ArenaRc::ptr_eq(&a, &c));
}

#[test]
fn slice_debug() {
    let arena = Arena::new();
    let r = arena.alloc_slice_copy(&[1_u32, 2]);
    assert_eq!(format!("{r:?}"), "[1, 2]");
}
