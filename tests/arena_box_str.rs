//! Tests for [`BoxStr`] and [`Arena::alloc_str_box`] / `try_alloc_str_box` —
//! the owned, mutable, single-pointer (8 bytes) arena-backed string smart pointer.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]

mod common;

use core::cmp::Ordering;
use multitude::{Arena, BoxStr};
use std::collections::{BTreeMap, HashMap};

#[test]
fn arena_box_str_basic() {
    let arena = Arena::new();
    let s = arena.alloc_str_box("hello, world");
    assert_eq!(&*s, "hello, world");
    assert_eq!(s.len(), 12);
    assert!(!s.is_empty());
}

#[test]
fn arena_box_str_empty() {
    let arena = Arena::new();
    let s = arena.alloc_str_box("");
    assert_eq!(&*s, "");
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

#[test]
fn arena_box_str_is_eight_bytes() {
    // The whole reason `ArenaBoxStr` exists rather than `ArenaBox<str>`
    // (16 bytes via fat pointer): single-pointer compactness.
    assert_eq!(size_of::<BoxStr>(), size_of::<usize>());
}

#[test]
fn arena_box_str_mutable_in_place() {
    let arena = Arena::new();
    let mut s = arena.alloc_str_box("hello");
    s.make_ascii_uppercase();
    assert_eq!(&*s, "HELLO");
}

#[test]
fn arena_box_str_as_mut_via_deref_mut() {
    let arena = Arena::new();
    let mut s = arena.alloc_str_box("Mixed Case");
    let m: &mut str = &mut s;
    m.make_ascii_lowercase();
    assert_eq!(&*s, "mixed case");
}

#[test]
fn arena_box_str_accepts_string() {
    // impl AsRef<str> covers both &str and String.
    let arena = Arena::new();
    let owned = std::string::String::from("from String");
    let s = arena.alloc_str_box(owned);
    assert_eq!(&*s, "from String");
}

#[test]
fn try_alloc_str_box_succeeds() {
    let arena = Arena::new();
    let s = arena.try_alloc_str_box("ok").unwrap();
    assert_eq!(&*s, "ok");
}

#[test]
fn arena_box_str_traits_compile() {
    let arena = Arena::new();
    let s = arena.alloc_str_box("hi");
    let _: &str = s.as_ref();
    let r: &str = core::borrow::Borrow::borrow(&s);
    assert_eq!(r, "hi");
    assert_eq!(format!("{s:?}"), "\"hi\"");
    assert_eq!(format!("{s}"), "hi");
    let other = arena.alloc_str_box("hi");
    let big = arena.alloc_str_box("z");
    assert_eq!(s, other);
    assert!(s < big);
    assert_eq!(s.cmp(&big), Ordering::Less);
    assert_eq!(s.partial_cmp(&big), Some(Ordering::Less));
    assert_eq!(common::hash_of(&s), common::hash_of(&other));
}

#[test]
fn arena_box_str_eq_and_hash_via_hashmap() {
    let arena = Arena::new();
    let key = arena.alloc_str_box("key");
    let mut map: HashMap<BoxStr, i32> = HashMap::new();
    let _ = map.insert(key, 1);
    // Borrow<str> lookup also works, so we don't need the original key.
    assert_eq!(map.get("key"), Some(&1));
}

#[test]
fn arena_box_str_works_as_btreemap_key() {
    let arena = Arena::new();
    let mut m: BTreeMap<BoxStr, u32> = BTreeMap::new();
    let _ = m.insert(arena.alloc_str_box("a"), 1);
    let _ = m.insert(arena.alloc_str_box("b"), 2);
    assert_eq!(m.get("a"), Some(&1));
    assert_eq!(m.get("b"), Some(&2));
}

#[test]
fn arena_box_str_drop_releases_chunk_immediately() {
    // ArenaBoxStr drops its chunk hold the moment the smart pointer is dropped.
    // Subsequent allocations in the arena must still work, exercising
    // the dec_ref + (optional) teardown_chunk path in ArenaBoxStr::Drop.
    let arena = Arena::new();
    let s = arena.alloc_str_box("temporary");
    assert_eq!(&*s, "temporary");
    drop(s);
    // Arena still works.
    let s2 = arena.alloc_str_box("after-drop");
    assert_eq!(&*s2, "after-drop");
}

#[test]
fn arena_box_str_lifetime_bound_to_arena() {
    // The borrow checker must reject use of an `ArenaBoxStr` whose
    // arena has been dropped. We can't write a runtime test for the
    // negative case (it's a compile error), but we can verify positive
    // case: dropping the box BEFORE the arena is fine.
    let arena = Arena::new();
    {
        let s = arena.alloc_str_box("inner");
        assert_eq!(&*s, "inner");
    }
    let s2 = arena.alloc_str_box("outer");
    assert_eq!(&*s2, "outer");
}

#[test]
fn many_arena_box_str_allocations_force_chunk_rotation() {
    let arena = Arena::builder().chunk_size(8 * 1024).chunk_cache_capacity(0).build();
    let mut handles = std::vec::Vec::new();
    for i in 0..200 {
        handles.push(arena.alloc_str_box(format!("item{i}")));
    }
    assert_eq!(&*handles[0], "item0");
    assert_eq!(&*handles[199], "item199");
}

#[test]
fn arena_box_str_round_trip_through_drop_does_not_corrupt() {
    let arena = Arena::new();
    for i in 0..1000 {
        let s = arena.alloc_str_box(format!("transient-{i}"));
        // Each iteration: alloc, mutate, drop. The dec_ref + teardown
        // must keep the arena healthy across many iterations.
        let _ = s.len();
    }
    // Final allocation works too.
    let s = arena.alloc_str_box("final");
    assert_eq!(&*s, "final");
}

#[test]
fn arena_box_str_borrow_mut_and_pointer() {
    use core::borrow::BorrowMut;
    let arena = Arena::new();
    let mut s = arena.alloc_str_box("hello");
    let m: &mut str = s.borrow_mut();
    m.make_ascii_uppercase();
    assert_eq!(&*s, "HELLO");
    let p = format!("{s:p}");
    assert!(p.starts_with("0x"), "Pointer format: {p}");
}

#[test]
fn arena_box_str_into_rc_str_via_from() {
    let arena = Arena::new();
    let b = arena.alloc_str_box("hi");
    let r: multitude::RcStr = b.into();
    let r2 = r.clone();
    assert_eq!(&*r, "hi");
    assert_eq!(&*r, &*r2);
}
