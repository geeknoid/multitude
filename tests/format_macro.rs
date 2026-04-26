//! Tests for the [`format!`](harena::format) macro.

#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]

use harena::Arena;

#[test]
fn format_macro_basic() {
    let arena = Arena::new();
    let name = "world";
    let s = harena::format!(in &arena, "hello, {name}!");
    assert_eq!(&*s, "hello, world!");
}

#[test]
fn format_macro_returns_arena_string() {
    // The macro returns ArenaString, not a frozen ArenaRcStr.
    let arena = Arena::new();
    let mut s = harena::format!(in &arena, "Hello, {}!", "Alice");
    s.push_str(" extended");
    assert_eq!(s.as_str(), "Hello, Alice! extended");
}

#[test]
fn format_macro_freeze_to_arena_str() {
    let arena = Arena::new();
    let name = "Alice";
    let s = harena::format!(in &arena, "Hello, {name}!");
    let frozen = s.into_arena_str();
    assert_eq!(&*frozen, "Hello, Alice!");
}

#[test]
fn format_macro_with_multiple_args() {
    let arena = Arena::new();
    let s = harena::format!(in &arena, "{}+{}={}", 2, 3, 5);
    assert_eq!(&*s, "2+3=5");
}

#[test]
fn format_macro_empty_format_string() {
    let arena = Arena::new();
    let s = harena::format!(in &arena, "");
    assert_eq!(&*s, "");
}

#[test]
fn arena_string_is_a_fmt_write_target() {
    use core::fmt::Write;
    let arena = Arena::new();
    let mut s = arena.new_string();
    write!(&mut s, "x={}", 42).unwrap();
    assert_eq!(s.as_str(), "x=42");

    s.write_char('!').unwrap();
    assert_eq!(s.as_str(), "x=42!");
}
