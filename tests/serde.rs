//! Tests for the optional serde Serialize impls (gated on the
//! `serde` feature).
//!
//! Multitude provides Serialize but not Deserialize — deserializing into
//! arena-backed types requires an arena context that serde's stock
//! Deserialize trait cannot carry. These tests verify only Serialize
//! round-trips.

#![cfg(all(feature = "serde", feature = "builders"))]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]

use multitude::Arena;

#[test]
fn arena_rc_str_serializes_to_string() {
    let arena = Arena::new();
    let s = arena.alloc_str_rc("hello world");
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, "\"hello world\"");
}

#[test]
fn arena_arc_str_serializes_to_string() {
    let arena = Arena::new();
    let s = arena.alloc_str_arc("shared");
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, "\"shared\"");
}

#[test]
fn arena_string_serializes_to_string() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("growable");
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, "\"growable\"");
}

#[test]
fn arena_string_empty_serializes() {
    let arena = Arena::new();
    let s = arena.alloc_string();
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(json, "\"\"");
}

#[test]
fn arena_vec_serializes_to_array() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<u32>();
    v.push(1);
    v.push(2);
    v.push(3);
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(json, "[1,2,3]");
}

#[test]
fn arena_vec_empty_serializes_to_array() {
    let arena = Arena::new();
    let v: multitude::builders::Vec<u32, _> = arena.alloc_vec();
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(json, "[]");
}

#[test]
fn arena_vec_of_strings_serializes() {
    let arena = Arena::new();
    let mut v = arena.alloc_vec::<String>();
    v.push("a".to_string());
    v.push("b".to_string());
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(json, "[\"a\",\"b\"]");
}

#[test]
fn nested_serialization_in_struct() {
    use serde::Serialize;

    #[derive(Serialize)]
    struct Wrapper<'a> {
        name: multitude::RcStr,
        items: multitude::builders::Vec<'a, i32>,
    }

    let arena = Arena::new();
    let mut items = arena.alloc_vec::<i32>();
    items.push(10);
    items.push(20);
    let w = Wrapper {
        name: arena.alloc_str_rc("widget"),
        items,
    };
    let json = serde_json::to_string(&w).unwrap();
    assert_eq!(json, "{\"name\":\"widget\",\"items\":[10,20]}");
}
