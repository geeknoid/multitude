#![cfg(all(feature = "utf16", feature = "builders"))]
//! Smoke tests for the UTF-16 string surface.

use multitude::{ArcUtf16Str, Arena, BoxUtf16Str, RcUtf16Str};
use widestring::utf16str;

#[test]
fn alloc_utf16_str_rc() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_rc(utf16str!("hello"));
    assert_eq!(&*s, utf16str!("hello"));
    assert_eq!(s.len(), 5);
    let s2 = s.clone();
    assert_eq!(&*s, &*s2);
}

#[test]
fn alloc_utf16_str_box() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_box(utf16str!("hello"));
    assert_eq!(&*s, utf16str!("hello"));
    assert_eq!(s.len(), 5);
}

#[test]
fn alloc_utf16_str_arc() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_arc(utf16str!("shared"));
    let _: ArcUtf16Str = s;
    assert_eq!(&*s, utf16str!("shared"));
}

#[test]
fn alloc_utf16_str_rc_from_str() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_rc_from_str("hello, world");
    assert_eq!(&*s, utf16str!("hello, world"));
}

#[test]
fn alloc_utf16_str_box_from_str() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_box_from_str("hello");
    assert_eq!(&*s, utf16str!("hello"));
}

#[test]
fn alloc_utf16_str_arc_from_str() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_arc_from_str("hello");
    assert_eq!(&*s, utf16str!("hello"));
}

#[test]
fn empty_string_round_trip() {
    let arena = Arena::new();
    let r: RcUtf16Str = arena.alloc_utf16_str_rc(utf16str!(""));
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
}

#[test]
fn box_into_rc() {
    let arena = Arena::new();
    let b: BoxUtf16Str = arena.alloc_utf16_str_box(utf16str!("hello"));
    let r: RcUtf16Str = b.into();
    assert_eq!(&*r, utf16str!("hello"));
}

#[test]
fn rc_into_arena_rc_slice() {
    let arena = Arena::new();
    let r = arena.alloc_utf16_str_rc(utf16str!("abc"));
    let bytes: multitude::Rc<[u16]> = r.into();
    assert_eq!(&*bytes, &[u16::from(b'a'), u16::from(b'b'), u16::from(b'c')][..]);
}

#[test]
fn arc_into_arena_arc_slice() {
    let arena = Arena::new();
    let a = arena.alloc_utf16_str_arc(utf16str!("abc"));
    let bytes: multitude::Arc<[u16]> = a.into();
    assert_eq!(&*bytes, &[u16::from(b'a'), u16::from(b'b'), u16::from(b'c')][..]);
}

#[test]
fn surrogate_pair_round_trip() {
    // U+1F496 SPARKLING HEART encodes as a UTF-16 surrogate pair.
    let arena = Arena::new();
    let r = arena.alloc_utf16_str_rc(utf16str!("💖"));
    assert_eq!(r.len(), 2); // 2 u16 units
    assert_eq!(&*r, utf16str!("💖"));
    let s = r.as_utf16_str().to_string();
    assert_eq!(s, "💖");
}

#[test]
fn empty_builder_into_arena_str() {
    let arena = Arena::new();
    let b = arena.alloc_utf16_string();
    let r = b.into_arena_utf16_str();
    assert!(r.is_empty());
}

#[test]
fn comparison_traits() {
    let arena = Arena::new();
    let a = arena.alloc_utf16_str_rc(utf16str!("aaa"));
    let b = arena.alloc_utf16_str_rc(utf16str!("aab"));
    assert!(a < b);
    assert_ne!(a, b);
    assert_eq!(a, a.clone());
    let _ = format!("{a}");
    let _ = format!("{a:?}");
}
