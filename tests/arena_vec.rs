#![cfg(feature = "builders")]
//! Tests for [`Vec`]: the growable arena-backed vector.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::missing_asserts_for_indexing, reason = "test code is direct")]

mod common;

use core::cmp::Ordering;
use multitude::Arena;
use multitude::builders::{CollectIn, Vec};
#[test]
fn basic_push_index_freeze() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    for i in 0..100 {
        v.push(i);
    }
    assert_eq!(v.len(), 100);
    assert_eq!(v[42], 42);

    let frozen = v.into_arena_rc();
    assert_eq!(frozen.len(), 100);
    assert_eq!(&frozen[..3], &[0, 1, 2]);
}

#[cfg(feature = "stats")]
#[test]
fn freeze_in_place_for_copy_types() {
    // ArenaVec::into_arena_rc should not copy when T: !Drop and the
    // buffer is at the chunk's bump cursor.
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
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
    let mut v = arena.alloc_vec::<String>();
    v.push(std::string::String::from("a"));
    v.push(std::string::String::from("b"));
    v.push(std::string::String::from("c"));
    let frozen = v.into_arena_rc();
    assert_eq!(frozen.len(), 3);
    assert_eq!(&*frozen[0], "a");
    assert_eq!(&*frozen[2], "c");
}

#[test]
fn freeze_empty_uses_slow_path() {
    let arena = Arena::new();
    let v = arena.alloc_vec::<u32>();
    let frozen = v.into_arena_rc();
    assert_eq!(frozen.len(), 0);
}

#[test]
fn freeze_buffer_not_at_cursor_uses_slow_path() {
    // Allocate something between the vec creation and freeze so the
    // vec's buffer isn't at the chunk's cursor anymore.
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
    v.push(1);
    v.push(2);
    let _decoy = arena.alloc_rc(0_u8);
    v.push(3);
    let frozen = v.into_arena_rc();
    assert_eq!(&*frozen, &[1, 2, 3]);
}

#[test]
fn pop_and_clear() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
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
    let mut v = arena.alloc_vec::<u32>();
    v.reserve(100);
    assert!(v.capacity() >= 100);
}

#[test]
fn vec_with_capacity_factory() {
    let arena = Arena::new();
    let v = arena.alloc_vec_with_capacity::<u32>(50);
    assert!(v.capacity() >= 50);
    assert!(v.is_empty());
}

#[test]
fn as_mut_slice_modifies_elements() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.push(1_u32);
    v.push(2);
    v.as_mut_slice()[0] = 10;
    assert_eq!(v.as_slice(), &[10, 2]);
}

#[test]
fn extend_from_slice() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend_from_slice([1_u32, 2, 3]);
    v.extend_from_slice([4, 5]);
    assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5]);
}

#[test]
fn extend_iter() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend(0_u32..5);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
}

#[test]
fn collect_in_works() {
    let arena = Arena::new();
    let v: Vec<i32, _> = (0..10).collect_in(&arena);
    assert_eq!(v.len(), 10);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn traits_compile() {
    let arena = Arena::new();
    let mut a = arena.alloc_vec();
    a.extend([1_u32, 2, 3]);
    let mut b = arena.alloc_vec();
    b.extend([1_u32, 2, 3]);
    let mut c = arena.alloc_vec();
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

#[test]
fn into_arena_rc_zst_element() {
    // ApiVec for ZST T uses NonNull::dangling() as its buffer; the
    // in-place fast path of into_arena_rc must skip ZSTs (header_for
    // on a dangling pointer would produce a null chunk header).
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<()>();
    for _ in 0..7 {
        v.push(());
    }
    let rc = v.into_arena_rc();
    assert_eq!(rc.len(), 7);
}

#[test]
fn into_arena_rc_zst_drop_element() {
    // ZST that needs drop forces the slow path.
    use core::sync::atomic::{AtomicUsize, Ordering as Ord};
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct DropZst;
    impl Drop for DropZst {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ord::Relaxed);
        }
    }
    {
        let arena = Arena::new();
        let mut v = arena.alloc_vec::<DropZst>();
        for _ in 0..3 {
            v.push(DropZst);
        }
        let rc = v.into_arena_rc();
        assert_eq!(rc.len(), 3);
    }
    assert_eq!(COUNT.load(Ord::Relaxed), 3);
}

// ---------------------------------------------------------------------------
// Fallible mutators: try_push / try_reserve / try_with_capacity_in
// ---------------------------------------------------------------------------

#[test]
fn try_push_succeeds() {
    let arena = Arena::new();
    let mut v = Vec::new_in(&arena);
    v.try_push(1_u32).unwrap();
    v.try_push(2_u32).unwrap();
    assert_eq!(&*v, &[1, 2]);
}

#[test]
fn try_reserve_succeeds() {
    let arena = Arena::new();
    let mut v: Vec<u32> = Vec::new_in(&arena);
    v.try_reserve(64).unwrap();
    assert!(v.capacity() >= 64);
}

#[test]
fn try_with_capacity_in_succeeds() {
    let arena = Arena::new();
    let v: Vec<u32> = Vec::try_with_capacity_in(32, &arena).unwrap();
    assert!(v.capacity() >= 32);
    assert!(v.is_empty());
}

#[test]
fn try_with_capacity_in_zero_does_not_allocate() {
    let arena = Arena::new();
    let v: Vec<u32> = Vec::try_with_capacity_in(0, &arena).unwrap();
    assert_eq!(v.capacity(), 0);
}

#[test]
fn try_push_returns_err_on_alloc_failure() {
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let mut v: Vec<u32, _> = Vec::new_in(&arena);
    let _ = v.try_push(1).unwrap_err();
}

#[test]
fn try_reserve_returns_err_on_alloc_failure() {
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let mut v: Vec<u32, _> = Vec::new_in(&arena);
    let _ = v.try_reserve(16).unwrap_err();
}

#[test]
fn try_with_capacity_in_returns_err_on_alloc_failure() {
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let result: Result<Vec<u32, _>, _> = Vec::try_with_capacity_in(16, &arena);
    let _ = result.unwrap_err();
}

#[test]
fn with_capacity_in_pub_succeeds() {
    let arena = Arena::new();
    let v: Vec<u32> = Vec::with_capacity_in(8, &arena);
    assert!(v.capacity() >= 8);
}

#[test]
fn new_in_pub_succeeds() {
    let arena = Arena::new();
    let v: Vec<u8> = Vec::new_in(&arena);
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 0);
}

// ---------------------------------------------------------------------------
// bumpalo-parity surface
// ---------------------------------------------------------------------------

#[test]
fn from_iter_in_builds_content() {
    let arena = Arena::new();
    let v = Vec::<i32>::from_iter_in(0..5, &arena);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4]);
}

#[test]
fn as_ptr_and_as_mut_ptr_round_trip() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([10_u32, 20, 30]);
    let p = v.as_ptr();
    // SAFETY: pointer is valid for len reads.
    let first = unsafe { *p };
    assert_eq!(first, 10);
    let mp = v.as_mut_ptr();
    // SAFETY: pointer is valid for writes.
    unsafe { *mp = 99 };
    assert_eq!(v.as_slice(), &[99, 20, 30]);
}

#[test]
fn insert_remove_swap_remove() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 4]);
    v.insert(2, 3);
    assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
    let r = v.remove(0);
    assert_eq!(r, 1);
    assert_eq!(v.as_slice(), &[2, 3, 4]);
    let s = v.swap_remove(0);
    assert_eq!(s, 2);
    assert_eq!(v.as_slice(), &[4, 3]);
}

#[test]
#[should_panic(expected = "insertion index")]
fn insert_out_of_bounds_panics() {
    let arena = Arena::new();
    let mut v: Vec<u32> = Vec::new_in(&arena);
    v.insert(99, 1);
}

#[test]
#[should_panic(expected = "removal index")]
fn remove_out_of_bounds_panics() {
    let arena = Arena::new();
    let mut v: Vec<u32> = Vec::new_in(&arena);
    let _ = v.remove(0);
}

#[test]
#[should_panic(expected = "swap_remove index")]
fn swap_remove_out_of_bounds_panics() {
    let arena = Arena::new();
    let mut v: Vec<u32> = Vec::new_in(&arena);
    let _ = v.swap_remove(0);
}

#[test]
fn truncate_shortens() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend(0_u32..10);
    v.truncate(4);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    v.truncate(100);
    assert_eq!(v.len(), 4);
}

#[test]
fn set_len_unsafe() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
    v.reserve(4);
    let p = v.as_mut_ptr();
    for i in 0..4_u32 {
        // SAFETY: capacity >= 4; offset i is in-bounds.
        let slot = unsafe { p.add(i as usize) };
        // SAFETY: slot points to writable spare capacity.
        unsafe { slot.write(i * 2) };
    }
    // SAFETY: the loop above initialized indices 0..4.
    unsafe { v.set_len(4) };
    assert_eq!(v.as_slice(), &[0, 2, 4, 6]);
}

#[test]
fn shrink_to_fit_runs() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
    v.reserve(128);
    v.push(1);
    v.push(2);
    let cap_before = v.capacity();
    v.shrink_to_fit();
    assert!(v.capacity() <= cap_before);
    assert_eq!(v.as_slice(), &[1, 2]);
}

#[test]
fn retain_filters() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend(0_u32..10);
    v.retain(|x| x % 2 == 0);
    assert_eq!(v.as_slice(), &[0, 2, 4, 6, 8]);
}

#[test]
fn retain_mut_filters_and_mutates() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend(0_u32..6);
    v.retain_mut(|x| {
        if *x % 2 == 0 {
            *x *= 10;
            true
        } else {
            false
        }
    });
    assert_eq!(v.as_slice(), &[0, 20, 40]);
}

#[test]
fn dedup_basic() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 1, 2, 3, 3, 3, 4]);
    v.dedup();
    assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn dedup_by_custom() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_i32, -1, 2, -2, 3]);
    v.dedup_by(|a, b| a.abs() == b.abs());
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn dedup_by_key_works() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([10_i32, 11, 21, 22, 30]);
    v.dedup_by_key(|x| *x / 10);
    assert_eq!(v.as_slice(), &[10, 21, 30]);
}

#[test]
fn append_moves_elements() {
    let arena = Arena::new();
    let mut a = arena.alloc_vec();
    a.extend([1_u32, 2]);
    let mut b = arena.alloc_vec();
    b.extend([3_u32, 4, 5]);
    a.append(&mut b);
    assert_eq!(a.as_slice(), &[1, 2, 3, 4, 5]);
    assert!(b.is_empty());
}

#[test]
fn reserve_exact_grows() {
    let arena = Arena::new();
    let mut v: Vec<u32> = Vec::new_in(&arena);
    v.reserve_exact(50);
    assert!(v.capacity() >= 50);
}

#[test]
fn try_reserve_exact_succeeds() {
    let arena = Arena::new();
    let mut v: Vec<u32> = Vec::new_in(&arena);
    v.try_reserve_exact(40).unwrap();
    assert!(v.capacity() >= 40);
}

#[test]
fn try_reserve_exact_returns_err_on_alloc_failure() {
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let mut v: Vec<u32, _> = Vec::new_in(&arena);
    let _ = v.try_reserve_exact(16).unwrap_err();
}

#[test]
fn resize_grow_and_shrink() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    v.resize(5, 9);
    assert_eq!(v.as_slice(), &[1, 2, 3, 9, 9]);
    v.resize(2, 0);
    assert_eq!(v.as_slice(), &[1, 2]);
}

#[test]
fn resize_with_closure() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    let mut counter = 0_u32;
    v.resize_with(4, || {
        counter += 1;
        counter
    });
    assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn split_off_returns_tail() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend(0_u32..6);
    let tail = v.split_off(4);
    assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
    assert_eq!(tail.as_slice(), &[4, 5]);
}

#[test]
fn pop_if_removes_when_true() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    let r = v.pop_if(|x| *x == 3);
    assert_eq!(r, Some(3));
    assert_eq!(v.as_slice(), &[1, 2]);
}

#[test]
fn pop_if_keeps_when_false() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    let r = v.pop_if(|x| *x == 99);
    assert_eq!(r, None);
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn pop_if_empty_returns_none() {
    let arena = Arena::new();
    let mut v: Vec<u32> = Vec::new_in(&arena);
    let r = v.pop_if(|_| true);
    assert_eq!(r, None);
}

#[test]
fn drain_removes_and_yields() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend(0_u32..6);
    let drained: std::vec::Vec<u32> = v.drain(1..4).collect();
    assert_eq!(drained, [1, 2, 3]);
    assert_eq!(v.as_slice(), &[0, 4, 5]);
}

#[test]
fn clone_produces_equal_independent_vec() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    let mut c = v.clone();
    assert_eq!(c.as_slice(), v.as_slice());
    c.push(4);
    assert_eq!(v.as_slice(), &[1, 2, 3]);
    assert_eq!(c.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn into_iter_consumes() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    let collected: std::vec::Vec<u32> = v.into_iter().collect();
    assert_eq!(collected, [1, 2, 3]);
}

#[test]
fn into_iter_borrowed() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    let mut sum = 0_u32;
    for x in &v {
        sum += *x;
    }
    assert_eq!(sum, 6);
    assert_eq!(v.len(), 3);
}

#[test]
fn into_iter_mut_borrowed() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    for x in &mut v {
        *x *= 10;
    }
    assert_eq!(v.as_slice(), &[10, 20, 30]);
}

#[test]
fn extend_ref_for_copy_types() {
    let arena = Arena::new();
    let mut v: Vec<u8> = Vec::new_in(&arena);
    let src = [1_u8, 2, 3];
    v.extend(src.iter());
    assert_eq!(v.as_slice(), &[1, 2, 3]);
}

#[test]
fn borrow_mut_returns_mut_slice() {
    use core::borrow::BorrowMut;
    let arena = Arena::new();
    let mut v = arena.alloc_vec();
    v.extend([1_u32, 2, 3]);
    let s: &mut [u32] = v.borrow_mut();
    s[0] = 9;
    assert_eq!(v.as_slice(), &[9, 2, 3]);
}

// ---------------------------------------------------------------------------
// `multitude::builders::vec!` macro
// ---------------------------------------------------------------------------

#[test]
fn vec_macro_empty() {
    let arena = Arena::new();
    let v: Vec<i32> = multitude::builders::vec![in &arena];
    assert!(v.is_empty());
}

#[test]
fn vec_macro_from_list() {
    let arena = Arena::new();
    let v = multitude::builders::vec![in &arena; 1, 2, 3];
    assert_eq!(&*v, &[1, 2, 3]);
}

#[test]
fn vec_macro_from_list_trailing_comma() {
    let arena = Arena::new();
    let v = multitude::builders::vec![in &arena; 'a', 'b', 'c',];
    assert_eq!(&*v, &['a', 'b', 'c']);
}

#[test]
fn vec_macro_n_copies() {
    let arena = Arena::new();
    let v = multitude::builders::vec![in &arena; 7_u32; 4];
    assert_eq!(&*v, &[7, 7, 7, 7]);
}

#[test]
fn vec_macro_n_copies_zero() {
    let arena = Arena::new();
    let v: Vec<i32> = multitude::builders::vec![in &arena; 0; 0];
    assert!(v.is_empty());
    assert_eq!(v.capacity(), 0);
}

#[test]
fn vec_macro_evaluates_each_expr_once() {
    use core::cell::Cell;
    let arena = Arena::new();
    let n = Cell::new(0_u32);
    let bump = || {
        let v = n.get();
        n.set(v + 1);
        v
    };
    let v = multitude::builders::vec![in &arena; bump(), bump(), bump()];
    assert_eq!(&*v, &[0, 1, 2]);
    assert_eq!(n.get(), 3);
}

#[test]
fn vec_macro_n_copies_evaluates_value_once() {
    use core::cell::Cell;
    let arena = Arena::new();
    let n = Cell::new(0_u32);
    let producer = || {
        n.set(n.get() + 1);
        42_u32
    };
    let v = multitude::builders::vec![in &arena; producer(); 5];
    assert_eq!(&*v, &[42, 42, 42, 42, 42]);
    // `resize` clones the value, so the producer is invoked once.
    assert_eq!(n.get(), 1);
}

#[test]
fn vec_macro_with_typed_expression() {
    let arena = Arena::new();
    let v: Vec<u8> = multitude::builders::vec![in &arena; 1, 2, 3];
    assert_eq!(v.len(), 3);
}

#[test]
fn vec_macro_can_hold_strings() {
    let arena = Arena::new();
    let s1 = std::string::String::from("hello");
    let s2 = std::string::String::from("world");
    let v = multitude::builders::vec![in &arena; s1, s2];
    assert_eq!(&v[0], "hello");
    assert_eq!(&v[1], "world");
}
