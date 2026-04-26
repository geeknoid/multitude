#![cfg(all(feature = "utf16", feature = "builders"))]
//! Tests for `format_utf16!`.

use core::fmt;
use multitude::Arena;
use multitude::builders::format_utf16;
use widestring::utf16str;

#[test]
fn format_utf16_basic() {
    let arena = Arena::new();
    let n = 42_i32;
    let s = format_utf16!(in &arena, "n = {n}");
    assert_eq!(s.as_utf16_str(), utf16str!("n = 42"));
}

#[test]
fn format_utf16_with_unicode() {
    let arena = Arena::new();
    let s = format_utf16!(in &arena, "love {}", '💖');
    assert_eq!(s.as_utf16_str(), utf16str!("love 💖"));
}

/// A `Display` impl that fragments output across multiple `write_str`
/// calls — verifies that fragmenting a sequence of code points across
/// `write_str` boundaries produces correct UTF-16 output. Each `&str`
/// passed to `write_str` is itself a complete UTF-8 fragment so no
/// cross-call surrogate state is needed; this test exercises that
/// invariant.
struct Fragmented<'a>(&'a [&'a str]);

impl fmt::Display for Fragmented<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for piece in self.0 {
            f.write_str(piece)?;
        }
        Ok(())
    }
}

#[test]
fn format_utf16_fragmented_writes() {
    let arena = Arena::new();
    let pieces = ["he", "llo, ", "💖", " bye"];
    let f = Fragmented(&pieces);
    let s = format_utf16!(in &arena, "{f}");
    assert_eq!(s.as_utf16_str(), utf16str!("hello, 💖 bye"));
}

#[test]
fn format_utf16_freeze() {
    let arena = Arena::new();
    let s = format_utf16!(in &arena, "x={}", 7);
    let frozen: multitude::RcUtf16Str = s.into_arena_utf16_str();
    assert_eq!(&*frozen, utf16str!("x=7"));
}
