#![cfg(feature = "builders")]
//! Tests for the single-pointer string smart pointer types
//! [`RcStr`] (single-thread) and [`ArenaArcStr`] (cross-thread).

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::redundant_clone, reason = "tests exercise Clone explicitly")]

mod common;

use core::cmp::Ordering;
use multitude::{Arena, RcStr};
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// ArenaRcStr
// ---------------------------------------------------------------------------

#[test]
fn arena_rc_str_basic() {
    let arena = Arena::new();
    let s = arena.alloc_str_rc("hello, world");
    assert_eq!(&*s, "hello, world");
    assert_eq!(s.len(), 12);
    assert!(!s.is_empty());

    let empty = arena.alloc_str_rc("");
    assert_eq!(&*empty, "");
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

#[test]
fn arena_rc_str_clone_increments_refcount() {
    let arena = Arena::new();
    let s = arena.alloc_str_rc("data");
    let s2 = s.clone();
    assert_eq!(&*s, "data");
    assert_eq!(&*s2, "data");
}

#[test]
fn arena_rc_str_traits_compile() {
    let arena = Arena::new();
    let s = arena.alloc_str_rc("hi");
    let _: &str = s.as_ref();
    let r: &str = core::borrow::Borrow::borrow(&s);
    assert_eq!(r, "hi");
    assert_eq!(&*s, "hi");
    assert_eq!(format!("{s:?}"), "\"hi\"");
    assert_eq!(format!("{s}"), "hi");
    let other = arena.alloc_str_rc("hi");
    let big = arena.alloc_str_rc("z");
    assert_eq!(s, other);
    assert!(s < big);
    assert_eq!(s.cmp(&big), Ordering::Less);
    assert_eq!(s.partial_cmp(&big), Some(Ordering::Less));
    assert_eq!(common::hash_of(&s), common::hash_of(&other));
}

#[test]
fn arena_rc_str_eq_and_hash_via_hashmap() {
    let arena = Arena::new();
    let key = arena.alloc_str_rc("key");
    let mut map: HashMap<RcStr<_>, i32> = HashMap::new();
    let _ = map.insert(key.clone(), 1);
    assert_eq!(map.get(&key), Some(&1));
    // Borrow<str> lookup also works.
    assert_eq!(map.get("key"), Some(&1));
}

#[test]
fn arena_rc_str_works_as_btreemap_key() {
    let arena = Arena::new();
    let mut m: BTreeMap<RcStr, u32> = BTreeMap::new();
    let _ = m.insert(arena.alloc_str_rc("a"), 1);
    let _ = m.insert(arena.alloc_str_rc("b"), 2);
    assert_eq!(m.get("a"), Some(&1));
    assert_eq!(m.get("b"), Some(&2));
}

// ---------------------------------------------------------------------------
// ArenaArcStr
// ---------------------------------------------------------------------------

#[test]
fn arena_arc_str_basic() {
    let arena = Arena::new();
    let s = arena.alloc_str_arc("hi");
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
    let s = arena.alloc_str_arc("");
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

#[test]
fn arena_arc_str_traits_compile() {
    let arena = Arena::new();
    let s = arena.alloc_str_arc("hi");
    let _: &str = s.as_ref();
    let r: &str = core::borrow::Borrow::borrow(&s);
    assert_eq!(r, "hi");
    assert_eq!(format!("{s:?}"), "\"hi\"");
    assert_eq!(format!("{s}"), "hi");
    let other = arena.alloc_str_arc("hi");
    let big = arena.alloc_str_arc("z");
    assert_eq!(s, other);
    assert!(s < big);
    assert_eq!(s.cmp(&big), Ordering::Less);
    assert_eq!(s.partial_cmp(&big), Some(Ordering::Less));
    assert_eq!(common::hash_of(&s), common::hash_of(&other));
}

#[test]
fn arena_arc_str_outlives_arena() {
    // Drives the `teardown_chunk(chunk, false)` branch in
    // ArenaArcStr::Drop when this is the LAST reference and the arena
    // itself has already been dropped.
    let s: multitude::ArcStr = {
        let arena = Arena::new();
        arena.alloc_str_arc("survives the arena")
    };
    assert_eq!(&*s, "survives the arena");
    drop(s); // teardown_chunk(chunk, false) for the Shared chunk.
}

#[test]
fn unpin_impl_rc_str() {
    fn assert_unpin<T: Unpin>() {}
    assert_unpin::<RcStr>();
    assert_unpin::<multitude::ArcStr>();
}

#[test]
fn from_arena_string_freezes_to_rc_str() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("frozen");
    let r: RcStr = s.into();
    assert_eq!(&*r, "frozen");
    let r2 = r.clone();
    assert_eq!(&*r, &*r2);
}

// ---------------------------------------------------------------------------
// Str → byte-slice conversions (mirroring `From<Rc<str>> for Rc<[u8]>`)
// ---------------------------------------------------------------------------

#[test]
fn from_arena_rc_str_to_arena_rc_byte_slice() {
    use multitude::Rc;
    let arena = Arena::new();
    let s = arena.alloc_str_rc("héllo"); // includes a multi-byte char
    let bytes: Rc<[u8]> = s.into();
    assert_eq!(&*bytes, "héllo".as_bytes());
}

#[test]
fn from_arena_arc_str_to_arena_arc_byte_slice() {
    use multitude::{Arc, ArcStr};
    let arena = Arena::new();
    let s: ArcStr = arena.alloc_str_arc("payload");
    let bytes: Arc<[u8]> = s.into();
    assert_eq!(&*bytes, b"payload");
}

#[test]
fn arena_arc_byte_slice_is_send_sync() {
    use multitude::{Arc, ArcStr};
    let arena = Arena::new();
    let s: ArcStr = arena.alloc_str_arc("threaded");
    let bytes: Arc<[u8]> = s.into();
    let bytes2 = bytes.clone();
    let h = std::thread::spawn(move || bytes2.len());
    assert_eq!(h.join().unwrap(), bytes.len());
}
