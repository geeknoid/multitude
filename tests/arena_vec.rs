//! Tests for [`ArenaVec`]: the growable arena-backed vector.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::missing_asserts_for_indexing, reason = "test code is direct")]

mod common;

use core::cmp::Ordering;
use harena::{Arena, ArenaVec, CollectIn};

#[test]
fn basic_push_index_freeze() {
    let arena = Arena::new();
    let mut v = arena.new_vec();
    for i in 0..100 {
        v.push(i);
    }
    assert_eq!(v.len(), 100);
    assert_eq!(v[42], 42);

    let frozen = v.into_arena_rc();
    assert_eq!(frozen.len(), 100);
    assert_eq!(&frozen[..3], &[0, 1, 2]);
}

#[test]
fn freeze_in_place_for_copy_types() {
    // ArenaVec::into_arena_rc should not copy when T: !Drop and the
    // buffer is at the chunk's bump cursor.
    let arena = Arena::new();
    let mut v = arena.new_vec();
    for i in 0..1000_u32 {
        v.push(i);
    }
    let chunks_before_freeze = arena.stats().chunks_allocated;
    let frozen = v.into_arena_rc();
    let chunks_after_freeze = arena.stats().chunks_allocated;
    assert_eq!(chunks_after_freeze, chunks_before_freeze);
    assert_eq!(frozen.len(), 1000);
    assert_eq!(frozen[42], 42);
    assert_eq!(frozen[999], 999);
}

#[test]
fn freeze_with_drop_type_uses_slow_path() {
    // T: Drop forces the slow path in into_arena_rc.
    let arena = Arena::new();
    let mut v = arena.new_vec::<String>();
    v.push(String::from("a"));
    v.push(String::from("b"));
    v.push(String::from("c"));
    let frozen = v.into_arena_rc();
    assert_eq!(frozen.len(), 3);
    assert_eq!(&*frozen[0], "a");
    assert_eq!(&*frozen[2], "c");
}

#[test]
fn freeze_empty_uses_slow_path() {
    let arena = Arena::new();
    let v = arena.new_vec::<u32>();
    let frozen = v.into_arena_rc();
    assert_eq!(frozen.len(), 0);
}

#[test]
fn freeze_buffer_not_at_cursor_uses_slow_path() {
    // Allocate something between the vec creation and freeze so the
    // vec's buffer isn't at the chunk's cursor anymore.
    let arena = Arena::new();
    let mut v = arena.new_vec::<u32>();
    v.push(1);
    v.push(2);
    let _decoy = arena.alloc(0_u8);
    v.push(3);
    let frozen = v.into_arena_rc();
    assert_eq!(&*frozen, &[1, 2, 3]);
}

#[test]
fn pop_and_clear() {
    let arena = Arena::new();
    let mut v = arena.new_vec::<u32>();
    v.push(1);
    v.push(2);
    v.push(3);
    assert_eq!(v.pop(), Some(3));
    assert_eq!(v.len(), 2);
    let cap = v.capacity();
    v.clear();
    assert!(v.is_empty());
    assert_eq!(v.capacity(), cap);
    assert_eq!(v.pop(), None);
}

#[test]
fn reserve_grows_capacity() {
    let arena = Arena::new();
    let mut v = arena.new_vec::<u32>();
    v.reserve(100);
    assert!(v.capacity() >= 100);
}

#[test]
fn vec_with_capacity_factory() {
    let arena = Arena::new();
    let v = arena.vec_with_capacity::<u32>(50);
    assert!(v.capacity() >= 50);
    assert!(v.is_empty());
}

#[test]
fn as_mut_slice_modifies_elements() {
    let arena = Arena::new();
    let mut v = arena.new_vec();
    v.push(1_u32);
    v.push(2);
    v.as_mut_slice()[0] = 10;
    assert_eq!(v.as_slice(), &[10, 2]);
}

#[test]
fn extend_from_slice() {
    let arena = Arena::new();
    let mut v = arena.new_vec();
    v.extend_from_slice(&[1_u32, 2, 3]);
    v.extend_from_slice(&[4, 5]);
    assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5]);
}

#[test]
fn extend_iter() {
    let arena = Arena::new();
    let mut v = arena.new_vec();
    v.extend(0_u32..5);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
}

#[test]
fn collect_in_works() {
    let arena = Arena::new();
    let v: ArenaVec<i32, _> = (0..10).collect_in(&arena);
    assert_eq!(v.len(), 10);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn traits_compile() {
    let arena = Arena::new();
    let mut a = arena.new_vec();
    a.extend([1_u32, 2, 3]);
    let mut b = arena.new_vec();
    b.extend([1_u32, 2, 3]);
    let mut c = arena.new_vec();
    c.extend([4_u32, 5]);
    let _: &[u32] = a.as_ref();
    let mb: &mut [u32] = a.as_mut();
    mb[0] = 1;
    let r: &[u32] = core::borrow::Borrow::borrow(&a);
    assert_eq!(r, &[1, 2, 3]);
    assert_eq!(format!("{a:?}"), "[1, 2, 3]");
    assert_eq!(a, b);
    assert!(a != c);
    assert_eq!(a.cmp(&c), Ordering::Less);
    assert_eq!(a.partial_cmp(&c), Some(Ordering::Less));
    assert_eq!(common::hash_of(&a), common::hash_of(&b));
}
