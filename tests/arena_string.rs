//! Tests for [`ArenaString`]: the growable arena-backed string builder.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]

mod common;

use core::cmp::Ordering;
use harena::{Arena, ArenaString, CollectIn};

#[test]
fn build_then_freeze_zero_copy() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.push_str("hello, world");
    let s_addr_before = s.as_str().as_ptr() as usize;
    let frozen = s.into_arena_str();
    let s_addr_after = frozen.as_ptr() as usize;
    assert_eq!(s_addr_before, s_addr_after, "freeze should be zero-copy");
    assert_eq!(&*frozen, "hello, world");
}

#[test]
fn freeze_reclaims_slack() {
    let arena = Arena::new();
    let mut s = arena.string_with_capacity(1024);
    s.push_str("short");
    assert!(s.capacity() >= 1024);
    let chunks_before = arena.stats().chunks_allocated;
    let _frozen = s.into_arena_str();
    // The slack we just reclaimed should be available for the next alloc
    // without needing a fresh chunk.
    let _other = arena.alloc_slice_copy(&[0_u8; 1000]);
    assert_eq!(arena.stats().chunks_allocated, chunks_before);
}

#[test]
fn grow_through_reallocation() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.push_str("original ");
    let _decoy = arena.alloc(0_u64);
    // This grow can no longer extend in place — must reallocate. The
    // inline len prefix must be preserved through the reallocation.
    s.push_str("grown ");
    s.push_str("a lot ");
    for _ in 0..100 {
        s.push_str("more text ");
    }
    let frozen = s.into_arena_str();
    assert!(frozen.starts_with("original grown a lot "));
    assert!(frozen.len() > 1000);
}

#[test]
fn empty_freeze() {
    let arena = Arena::new();
    let s = arena.new_string();
    let frozen = s.into_arena_str();
    assert_eq!(&*frozen, "");
    assert_eq!(frozen.len(), 0);
}

#[test]
fn clear_and_reuse() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.push_str("hello");
    let cap_before = s.capacity();
    s.clear();
    assert_eq!(s.len(), 0);
    assert_eq!(s.capacity(), cap_before);
    s.push_str("world");
    assert_eq!(s.as_str(), "world");
}

#[test]
fn clear_when_unallocated_is_noop() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.clear();
    assert!(s.is_empty());
}

#[test]
fn size_24_bytes_on_64bit() {
    if size_of::<usize>() == 8 {
        assert_eq!(size_of::<ArenaString<'_>>(), 24);
    }
}

#[test]
fn push_single_char() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.push('a');
    s.push('é');
    s.push('日');
    assert_eq!(s.as_str(), "aé日");
}

#[test]
fn reserve_grows_capacity() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.reserve(100);
    assert!(s.capacity() >= 100);
    s.push_str("hi");
    assert_eq!(s.as_str(), "hi");
}

#[test]
fn reserve_noop_when_already_large() {
    let arena = Arena::new();
    let mut s = arena.string_with_capacity(200);
    let cap = s.capacity();
    s.reserve(50);
    assert_eq!(s.capacity(), cap);
}

#[test]
fn as_str_when_empty_returns_empty_slice() {
    let arena = Arena::new();
    let s = arena.new_string();
    assert_eq!(s.as_str(), "");
    assert_eq!(s.len(), 0);
    assert_eq!(s.capacity(), 0);
}

#[test]
fn extend_chars() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.extend(['a', 'b', 'c'].iter().copied());
    assert_eq!(s.as_str(), "abc");
}

#[test]
fn extend_chars_empty_iter() {
    // Hits the `lower == 0` branch in Extend<char>::extend.
    let arena = Arena::new();
    let mut s = arena.new_string();
    let empty: [char; 0] = [];
    s.extend(empty.iter().copied());
    assert_eq!(s.as_str(), "");
}

#[test]
fn extend_strs() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.extend(["foo", "bar", "baz"].iter().copied());
    assert_eq!(s.as_str(), "foobarbaz");
}

#[test]
fn collect_in_chars() {
    let arena = Arena::new();
    let s: ArenaString<'_> = "héllo".chars().collect_in(&arena);
    assert_eq!(s.as_str(), "héllo");
}

#[test]
fn traits_compile() {
    let arena = Arena::new();
    let mut s = arena.new_string();
    s.push_str("hi");
    let _: &str = s.as_ref();
    let r: &str = core::borrow::Borrow::borrow(&s);
    assert_eq!(r, "hi");
    assert_eq!(format!("{s:?}"), "\"hi\"");
    assert_eq!(format!("{s}"), "hi");
    let mut other = arena.new_string();
    other.push_str("hi");
    let mut big = arena.new_string();
    big.push_str("z");
    assert_eq!(s, other);
    assert!(s < big);
    assert_eq!(s.cmp(&big), Ordering::Less);
    assert_eq!(s.partial_cmp(&big), Some(Ordering::Less));
    assert_eq!(common::hash_of(&s), common::hash_of(&other));
}
