#![cfg(all(feature = "utf16", feature = "serde", feature = "builders"))]
//! Serde round-trip tests for the UTF-16 string types. They are
//! `Serialize`-only — serialized as a UTF-8 string — so this test
//! verifies the wire format directly.

use multitude::Arena;
use widestring::utf16str;

#[test]
fn rc_serializes_as_utf8_string() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_rc(utf16str!("hello, 💖"));
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, r#""hello, 💖""#);
}

#[test]
fn arc_serializes_as_utf8_string() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_arc(utf16str!("shared"));
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, r#""shared""#);
}

#[test]
fn box_serializes_as_utf8_string() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_box(utf16str!("boxed"));
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, r#""boxed""#);
}

#[test]
fn string_serializes_as_utf8_string() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("growable"));
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, r#""growable""#);
}
