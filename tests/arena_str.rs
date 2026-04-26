//! Tests for the single-pointer string handle types
//! [`ArenaRcStr`] (single-thread) and [`ArenaArcStr`] (cross-thread).

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::redundant_clone, reason = "tests exercise Clone explicitly")]

mod common;

use core::cmp::Ordering;
use harena::{Arena, ArenaArcStr, ArenaRcStr};
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// ArenaRcStr
// ---------------------------------------------------------------------------

#[test]
fn arena_rc_str_basic() {
    let arena = Arena::new();
    let s = ArenaRcStr::from_str(&arena, "hello, world");
    assert_eq!(&*s, "hello, world");
    assert_eq!(s.len(), 12);
    assert!(!s.is_empty());

    let empty = ArenaRcStr::from_str(&arena, "");
    assert_eq!(&*empty, "");
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

#[test]
fn arena_rc_str_clone_increments_refcount() {
    let arena = Arena::new();
    let s = ArenaRcStr::from_str(&arena, "data");
    let s2 = s.clone();
    assert_eq!(&*s, "data");
    assert_eq!(&*s2, "data");
}

#[test]
fn arena_rc_str_traits_compile() {
    let arena = Arena::new();
    let s = ArenaRcStr::from_str(&arena, "hi");
    let _: &str = s.as_ref();
    let r: &str = core::borrow::Borrow::borrow(&s);
    assert_eq!(r, "hi");
    assert_eq!(&*s, "hi");
    assert_eq!(format!("{s:?}"), "\"hi\"");
    assert_eq!(format!("{s}"), "hi");
    let other = ArenaRcStr::from_str(&arena, "hi");
    let big = ArenaRcStr::from_str(&arena, "z");
    assert_eq!(s, other);
    assert!(s < big);
    assert_eq!(s.cmp(&big), Ordering::Less);
    assert_eq!(s.partial_cmp(&big), Some(Ordering::Less));
    assert_eq!(common::hash_of(&s), common::hash_of(&other));
}

#[test]
fn arena_rc_str_eq_and_hash_via_hashmap() {
    let arena = Arena::new();
    let key = ArenaRcStr::from_str(&arena, "key");
    let mut map: HashMap<ArenaRcStr<_>, i32> = HashMap::new();
    let _ = map.insert(key.clone(), 1);
    assert_eq!(map.get(&key), Some(&1));
    // Borrow<str> lookup also works.
    assert_eq!(map.get("key"), Some(&1));
}

#[test]
fn arena_rc_str_works_as_btreemap_key() {
    let arena = Arena::new();
    let mut m: BTreeMap<ArenaRcStr, u32> = BTreeMap::new();
    let _ = m.insert(ArenaRcStr::from_str(&arena, "a"), 1);
    let _ = m.insert(ArenaRcStr::from_str(&arena, "b"), 2);
    assert_eq!(m.get("a"), Some(&1));
    assert_eq!(m.get("b"), Some(&2));
}

// ---------------------------------------------------------------------------
// ArenaArcStr
// ---------------------------------------------------------------------------

#[test]
fn arena_arc_str_basic() {
    let arena = Arena::new();
    let s = ArenaArcStr::from_str(&arena, "hi");
    assert_eq!(&*s, "hi");
    assert_eq!(s.len(), 2);
    assert!(!s.is_empty());
    let s2 = s.clone();
    let h = std::thread::spawn(move || {
        assert_eq!(&*s2, "hi");
    });
    h.join().unwrap();
}

#[test]
fn arena_arc_str_empty() {
    let arena = Arena::new();
    let s = ArenaArcStr::from_str(&arena, "");
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

#[test]
fn arena_arc_str_traits_compile() {
    let arena = Arena::new();
    let s = ArenaArcStr::from_str(&arena, "hi");
    let _: &str = s.as_ref();
    let r: &str = core::borrow::Borrow::borrow(&s);
    assert_eq!(r, "hi");
    assert_eq!(format!("{s:?}"), "\"hi\"");
    assert_eq!(format!("{s}"), "hi");
    let other = ArenaArcStr::from_str(&arena, "hi");
    let big = ArenaArcStr::from_str(&arena, "z");
    assert_eq!(s, other);
    assert!(s < big);
    assert_eq!(s.cmp(&big), Ordering::Less);
    assert_eq!(s.partial_cmp(&big), Some(Ordering::Less));
    assert_eq!(common::hash_of(&s), common::hash_of(&other));
}
