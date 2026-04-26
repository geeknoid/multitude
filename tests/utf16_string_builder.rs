#![cfg(all(feature = "utf16", feature = "builders"))]
//! Builder tests for `ArenaUtf16String`.

use multitude::Arena;
use widestring::utf16str;

#[test]
fn push_and_pop() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push('a');
    s.push('b');
    s.push('💖'); // surrogate pair: +2 u16
    assert_eq!(s.len(), 4);
    assert_eq!(s.pop(), Some('💖'));
    assert_eq!(s.len(), 2);
    s.push_str(utf16str!("xyz"));
    assert_eq!(s.as_utf16_str(), utf16str!("abxyz"));
}

#[test]
fn push_from_str_transcodes() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_from_str("hello, 💖");
    assert_eq!(s.as_utf16_str(), utf16str!("hello, 💖"));
    assert_eq!(s.len(), 9); // 7 ascii + 2 surrogate
}

#[test]
fn truncate_and_clear() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("hello"));
    s.truncate(3);
    assert_eq!(s.as_utf16_str(), utf16str!("hel"));
    s.clear();
    assert!(s.is_empty());
}

#[test]
#[should_panic(expected = "is not on a UTF-16 char boundary")]
fn truncate_mid_surrogate_panics() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push('💖'); // 2 u16 units
    s.truncate(1);
}

#[test]
fn insert_and_remove() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("hello"));
    s.insert(0, 'X');
    assert_eq!(s.as_utf16_str(), utf16str!("Xhello"));
    s.insert_utf16_str(2, utf16str!("YY"));
    assert_eq!(s.as_utf16_str(), utf16str!("XhYYello"));
    let removed = s.remove(0);
    assert_eq!(removed, 'X');
    assert_eq!(s.as_utf16_str(), utf16str!("hYYello"));
}

#[test]
fn replace_range_grows_and_shrinks() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("Hello, world!"));
    s.replace_range(7..12, utf16str!("everyone"));
    assert_eq!(s.as_utf16_str(), utf16str!("Hello, everyone!"));
    s.replace_range(7..15, utf16str!("X"));
    assert_eq!(s.as_utf16_str(), utf16str!("Hello, X!"));
}

#[test]
fn retain_drops_predicate_failures() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_from_str("Hello, World!");
    s.retain(|c| c.is_ascii_alphabetic());
    assert_eq!(s.as_utf16_str(), utf16str!("HelloWorld"));
}

#[test]
fn capacity_growth() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    for _ in 0..1000 {
        s.push('a');
    }
    assert_eq!(s.len(), 1000);
    assert!(s.capacity() >= 1000);
}

#[test]
fn shrink_to_fit_at_bump_cursor() {
    let arena = Arena::builder().chunk_cache_capacity(0).build();
    let mut s = arena.alloc_utf16_string_with_capacity(64);
    s.push_str(utf16str!("hi"));
    let cap_before = s.capacity();
    assert!(cap_before >= 64);
    s.shrink_to_fit();
    assert_eq!(s.capacity(), 2);
}

#[test]
fn extend_with_chars_str_and_utf16str() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.extend(['a', 'b', 'c'].iter().copied());
    s.extend(["12", "34"].iter().copied());
    s.extend([utf16str!("XY"), utf16str!("Z")].iter().copied());
    assert_eq!(s.as_utf16_str(), utf16str!("abc1234XYZ"));
}

#[test]
fn from_str_in_and_from_utf16_str_in() {
    let arena = Arena::new();
    let a = multitude::builders::Utf16String::from_str_in("hello", &arena);
    assert_eq!(a.as_utf16_str(), utf16str!("hello"));
    let b = multitude::builders::Utf16String::from_utf16_str_in(utf16str!("world"), &arena);
    assert_eq!(b.as_utf16_str(), utf16str!("world"));
}

#[test]
fn clone_builder() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("hello"));
    let c = s.clone();
    assert_eq!(c.as_utf16_str(), s.as_utf16_str());
}

#[test]
fn freeze_tail_reclamation() {
    let arena = Arena::builder().chunk_cache_capacity(0).build();
    let stats_before = arena.alloc_utf16_string_with_capacity(64);
    let cap_before = stats_before.capacity();
    drop(stats_before);
    let mut s = arena.alloc_utf16_string_with_capacity(64);
    s.push_str(utf16str!("hi"));
    let r = s.into_arena_utf16_str();
    // Tail reclaim should let a subsequent allocation reuse those bytes.
    let s2 = arena.alloc_utf16_string_with_capacity(50);
    assert!(s2.capacity() >= 50);
    assert_eq!(&*r, utf16str!("hi"));
    assert!(cap_before >= 64);
}
