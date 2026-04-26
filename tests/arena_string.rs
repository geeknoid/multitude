#![cfg(feature = "builders")]
//! Tests for [`String`]: the growable arena-backed string builder.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]

mod common;

use core::cmp::Ordering;
use multitude::Arena;
use multitude::builders::{CollectIn, String};
#[test]
fn build_then_freeze_zero_copy() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("hello, world");
    let s_addr_before = s.as_str().as_ptr() as usize;
    let frozen = s.into_arena_str();
    let s_addr_after = frozen.as_ptr() as usize;
    assert_eq!(s_addr_before, s_addr_after, "freeze should be zero-copy");
    assert_eq!(&*frozen, "hello, world");
}

#[cfg(feature = "stats")]
#[test]
fn freeze_reclaims_slack() {
    let arena = Arena::new();
    let mut s = arena.alloc_string_with_capacity(1024);
    s.push_str("short");
    assert!(s.capacity() >= 1024);
    let chunks_before = arena.stats().chunks_allocated;
    let _frozen = s.into_arena_str();
    // The slack we just reclaimed should be available for the next alloc
    // without needing a fresh chunk.
    let _other = arena.alloc_slice_copy_rc([0_u8; 1000]);
    assert_eq!(arena.stats().chunks_allocated, chunks_before);
}

#[test]
fn grow_through_reallocation() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("original ");
    let _decoy = arena.alloc_rc(0_u64);
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
    let s = arena.alloc_string();
    let frozen = s.into_arena_str();
    assert_eq!(&*frozen, "");
    assert_eq!(frozen.len(), 0);
}

#[test]
fn clear_and_reuse() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
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
    let mut s = arena.alloc_string();
    s.clear();
    assert!(s.is_empty());
}

#[test]
fn size_32_bytes_on_64bit() {
    // 32 bytes = data ptr + cached len + cap + arena ref. The cached len
    // (added for perf) avoids a chunk-memory load on every read; it costs
    // 8 extra bytes vs. the previous 24-byte builder.
    if size_of::<usize>() == 8 {
        assert_eq!(size_of::<String<'_>>(), 32);
    }
}

#[test]
fn push_single_char() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push('a');
    s.push('é');
    s.push('日');
    assert_eq!(s.as_str(), "aé日");
}

#[test]
fn reserve_grows_capacity() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.reserve(100);
    assert!(s.capacity() >= 100);
    s.push_str("hi");
    assert_eq!(s.as_str(), "hi");
}

#[test]
fn reserve_noop_when_already_large() {
    let arena = Arena::new();
    let mut s = arena.alloc_string_with_capacity(200);
    let cap = s.capacity();
    s.reserve(50);
    assert_eq!(s.capacity(), cap);
}

#[test]
fn as_str_when_empty_returns_empty_slice() {
    let arena = Arena::new();
    let s = arena.alloc_string();
    assert_eq!(s.as_str(), "");
    assert_eq!(s.len(), 0);
    assert_eq!(s.capacity(), 0);
}

#[test]
fn extend_chars() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.extend(['a', 'b', 'c'].iter().copied());
    assert_eq!(s.as_str(), "abc");
}

#[test]
fn extend_chars_empty_iter() {
    // Hits the `lower == 0` branch in Extend<char>::extend.
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    let empty: [char; 0] = [];
    s.extend(empty.iter().copied());
    assert_eq!(s.as_str(), "");
}

#[test]
fn extend_strs() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.extend(["foo", "bar", "baz"].iter().copied());
    assert_eq!(s.as_str(), "foobarbaz");
}

#[test]
fn collect_in_chars() {
    let arena = Arena::new();
    let s: String<'_> = "héllo".chars().collect_in(&arena);
    assert_eq!(s.as_str(), "héllo");
}

#[test]
fn traits_compile() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.push_str("hi");
    let _: &str = s.as_ref();
    let r: &str = core::borrow::Borrow::borrow(&s);
    assert_eq!(r, "hi");
    assert_eq!(format!("{s:?}"), "\"hi\"");
    assert_eq!(format!("{s}"), "hi");
    let mut other = arena.alloc_string();
    other.push_str("hi");
    let mut big = arena.alloc_string();
    big.push_str("z");
    assert_eq!(s, other);
    assert!(s < big);
    assert_eq!(s.cmp(&big), Ordering::Less);
    assert_eq!(s.partial_cmp(&big), Some(Ordering::Less));
    assert_eq!(common::hash_of(&s), common::hash_of(&other));
}

// ---------------------------------------------------------------------------
// Fallible mutators: try_push / try_push_str / try_reserve / try_with_capacity_in
// ---------------------------------------------------------------------------

#[test]
fn try_push_succeeds() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.try_push('a').unwrap();
    s.try_push('b').unwrap();
    assert_eq!(&*s, "ab");
}

#[test]
fn try_push_str_succeeds() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.try_push_str("hello").unwrap();
    s.try_push_str(" world").unwrap();
    assert_eq!(&*s, "hello world");
}

#[test]
fn try_push_str_empty_is_noop_ok() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.try_push_str("").unwrap();
    assert!(s.is_empty());
}

#[test]
fn try_reserve_succeeds() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.try_reserve(64).unwrap();
    assert!(s.capacity() >= 64);
}

#[test]
fn try_with_capacity_in_succeeds() {
    let arena = Arena::new();
    let s = String::try_with_capacity_in(32, &arena).unwrap();
    assert!(s.capacity() >= 32);
    assert!(s.is_empty());
}

#[test]
fn try_with_capacity_in_zero_does_not_allocate() {
    let arena = Arena::new();
    let s = String::try_with_capacity_in(0, &arena).unwrap();
    assert_eq!(s.capacity(), 0);
}

#[test]
fn try_push_str_returns_err_on_alloc_failure() {
    use allocator_api2::alloc::AllocError;
    // FailingAllocator with 0 budget: every allocate() fails.
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let mut s = arena.alloc_string();
    let err: AllocError = s.try_push_str("x").unwrap_err();
    let _ = err;
}

#[test]
fn try_push_returns_err_on_alloc_failure() {
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let mut s = arena.alloc_string();
    let _ = s.try_push('x').unwrap_err();
}

#[test]
fn try_reserve_returns_err_on_alloc_failure() {
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let mut s = arena.alloc_string();
    let _ = s.try_reserve(16).unwrap_err();
}

#[test]
fn try_with_capacity_in_returns_err_on_alloc_failure() {
    let alloc = common::FailingAllocator::new(0);
    let arena = Arena::new_in(alloc);
    let result = String::try_with_capacity_in(16, &arena);
    let _ = result.unwrap_err();
}

#[test]
fn try_grow_path_via_push_str_after_initial() {
    // Drives try_grow_to_at_least's slow path (cap > 0 branch).
    let arena = Arena::new();
    let mut s = String::try_with_capacity_in(4, &arena).unwrap();
    s.try_push_str("abcd").unwrap(); // fills initial cap exactly
    s.try_push_str("e").unwrap(); // forces grow
    assert_eq!(&*s, "abcde");
    assert!(s.capacity() >= 5);
}

// ---------------------------------------------------------------------------
// from_str_in / as_bytes / as_mut_str / as_ptr / as_mut_ptr
// ---------------------------------------------------------------------------

#[test]
fn from_str_in_copies_content() {
    let arena = Arena::new();
    let s = String::from_str_in("hello, world", &arena);
    assert_eq!(s.as_str(), "hello, world");
    assert!(s.capacity() >= "hello, world".len());
}

#[test]
fn from_str_in_empty() {
    let arena = Arena::new();
    let s = String::from_str_in("", &arena);
    assert!(s.is_empty());
    assert_eq!(s.capacity(), 0);
    assert_eq!(s.as_str(), "");
}

#[test]
fn as_bytes_returns_correct_bytes() {
    let arena = Arena::new();
    let s = String::from_str_in("héllo", &arena);
    assert_eq!(s.as_bytes(), "héllo".as_bytes());
}

#[test]
fn as_bytes_empty() {
    let arena = Arena::new();
    let s = arena.alloc_string();
    assert_eq!(s.as_bytes(), b"");
}

#[test]
fn as_mut_str_allows_mutation() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello", &arena);
    s.as_mut_str().make_ascii_uppercase();
    assert_eq!(s.as_str(), "HELLO");
}

#[test]
fn as_mut_str_empty() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    assert_eq!(s.as_mut_str(), "");
}

#[test]
fn as_ptr_and_as_mut_ptr() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hi", &arena);
    let p = s.as_ptr();
    let q = s.as_mut_ptr();
    assert_eq!(p, q.cast_const());
    // SAFETY: valid pointer to len bytes.
    unsafe {
        assert_eq!(*p, b'h');
    }
    // SAFETY: valid pointer to len bytes; offset 1 is in bounds.
    let p1 = unsafe { p.add(1) };
    // SAFETY: valid pointer to a byte.
    unsafe {
        assert_eq!(*p1, b'i');
    }
}

// ---------------------------------------------------------------------------
// pop / truncate / shrink_to_fit
// ---------------------------------------------------------------------------

#[test]
fn pop_returns_chars_in_reverse() {
    let arena = Arena::new();
    let mut s = String::from_str_in("a💖é", &arena);
    assert_eq!(s.pop(), Some('é'));
    assert_eq!(s.pop(), Some('💖'));
    assert_eq!(s.pop(), Some('a'));
    assert_eq!(s.pop(), None);
    assert!(s.is_empty());
}

#[test]
fn truncate_shortens() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello", &arena);
    let cap = s.capacity();
    s.truncate(3);
    assert_eq!(s.as_str(), "hel");
    assert_eq!(s.capacity(), cap, "capacity unchanged");
}

#[test]
fn truncate_noop_when_longer() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hi", &arena);
    s.truncate(50);
    assert_eq!(s.as_str(), "hi");
}

#[test]
#[should_panic(expected = "char boundary")]
fn truncate_panics_on_non_boundary() {
    let arena = Arena::new();
    let mut s = String::from_str_in("é", &arena); // 2 bytes
    s.truncate(1);
}

#[test]
fn shrink_to_fit_reclaims_when_at_cursor() {
    let arena = Arena::new();
    let mut s = String::with_capacity_in(1024, &arena);
    s.push_str("short");
    let _len = s.len();
    s.shrink_to_fit();
    // Buffer is at the bump cursor, so shrink should succeed.
    assert_eq!(s.capacity(), 5);
    assert_eq!(s.as_str(), "short");
}

#[test]
fn shrink_to_fit_noop_when_not_at_cursor() {
    let arena = Arena::new();
    let mut s = String::with_capacity_in(1024, &arena);
    s.push_str("short");
    let _decoy = arena.alloc_rc(0_u64);
    s.shrink_to_fit();
    // Decoy moved the bump cursor past us; capacity unchanged.
    assert!(s.capacity() >= 1024);
    assert_eq!(s.as_str(), "short");
}

#[test]
fn shrink_to_fit_empty_or_full_noop() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    s.shrink_to_fit();
    assert_eq!(s.capacity(), 0);

    let mut s2 = String::with_capacity_in(4, &arena);
    s2.push_str("abcd");
    let cap = s2.capacity();
    s2.shrink_to_fit();
    assert_eq!(s2.capacity(), cap);
}

// ---------------------------------------------------------------------------
// insert / insert_str / remove
// ---------------------------------------------------------------------------

#[test]
fn insert_at_various_positions() {
    let arena = Arena::new();
    let mut s = String::from_str_in("ac", &arena);
    s.insert(1, 'b');
    assert_eq!(s.as_str(), "abc");
    s.insert(0, 'Z');
    assert_eq!(s.as_str(), "Zabc");
    s.insert(s.len(), '!');
    assert_eq!(s.as_str(), "Zabc!");
}

#[test]
fn insert_multibyte_char() {
    let arena = Arena::new();
    let mut s = String::from_str_in("ab", &arena);
    s.insert(1, '💖');
    assert_eq!(s.as_str(), "a💖b");
}

#[test]
fn insert_str_grows() {
    let arena = Arena::new();
    let mut s = String::from_str_in("ad", &arena);
    s.insert_str(1, "bc");
    assert_eq!(s.as_str(), "abcd");
}

#[test]
fn insert_str_empty_is_noop() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hi", &arena);
    s.insert_str(1, "");
    assert_eq!(s.as_str(), "hi");
}

#[test]
#[should_panic(expected = "char boundary")]
fn insert_panics_on_bad_index() {
    let arena = Arena::new();
    let mut s = String::from_str_in("é", &arena);
    s.insert(1, 'x');
}

#[test]
#[should_panic(expected = "char boundary")]
fn insert_panics_when_idx_past_end() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hi", &arena);
    s.insert(99, 'x');
}

#[test]
fn remove_returns_char() {
    let arena = Arena::new();
    let mut s = String::from_str_in("a💖c", &arena);
    let ch = s.remove(1);
    assert_eq!(ch, '💖');
    assert_eq!(s.as_str(), "ac");
}

#[test]
fn remove_first_and_last() {
    let arena = Arena::new();
    let mut s = String::from_str_in("abcd", &arena);
    assert_eq!(s.remove(0), 'a');
    assert_eq!(s.as_str(), "bcd");
    assert_eq!(s.remove(s.len() - 1), 'd');
    assert_eq!(s.as_str(), "bc");
}

#[test]
#[should_panic(expected = "out of bounds")]
fn remove_panics_when_empty() {
    let arena = Arena::new();
    let mut s = arena.alloc_string();
    let _ = s.remove(0);
}

// ---------------------------------------------------------------------------
// retain / replace_range
// ---------------------------------------------------------------------------

#[test]
fn retain_filters_chars() {
    let arena = Arena::new();
    let mut s = String::from_str_in("a1b2c3", &arena);
    s.retain(|c| c.is_ascii_alphabetic());
    assert_eq!(s.as_str(), "abc");
}

#[test]
fn retain_removes_all() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello", &arena);
    s.retain(|_| false);
    assert!(s.is_empty());
}

#[test]
fn retain_keeps_all() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello", &arena);
    s.retain(|_| true);
    assert_eq!(s.as_str(), "hello");
}

#[test]
fn retain_with_multibyte() {
    let arena = Arena::new();
    let mut s = String::from_str_in("a💖b💖c", &arena);
    s.retain(|c| c != '💖');
    assert_eq!(s.as_str(), "abc");
}

#[test]
fn replace_range_same_length() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello world", &arena);
    s.replace_range(6..11, "earth");
    assert_eq!(s.as_str(), "hello earth");
}

#[test]
fn replace_range_grow() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hi world", &arena);
    s.replace_range(0..2, "hello");
    assert_eq!(s.as_str(), "hello world");
}

#[test]
fn replace_range_shrink() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello world", &arena);
    s.replace_range(0..5, "hi");
    assert_eq!(s.as_str(), "hi world");
}

#[test]
fn replace_range_unbounded() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello", &arena);
    s.replace_range(.., "goodbye");
    assert_eq!(s.as_str(), "goodbye");
}

#[test]
fn replace_range_empty_replacement() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello world", &arena);
    s.replace_range(5..11, "");
    assert_eq!(s.as_str(), "hello");
}

#[test]
fn replace_range_inclusive() {
    let arena = Arena::new();
    let mut s = String::from_str_in("abcdef", &arena);
    s.replace_range(1..=3, "XYZW");
    assert_eq!(s.as_str(), "aXYZWef");
}

#[test]
#[should_panic(expected = "char boundary")]
fn replace_range_panics_on_non_boundary() {
    let arena = Arena::new();
    let mut s = String::from_str_in("é", &arena);
    s.replace_range(0..1, "x");
}

// ---------------------------------------------------------------------------
// Clone / DerefMut / AsMut / BorrowMut
// ---------------------------------------------------------------------------

#[test]
fn clone_produces_equal_independent_string() {
    let arena = Arena::new();
    let original = String::from_str_in("hello", &arena);
    let mut cloned = original.clone();
    assert_eq!(original.as_str(), cloned.as_str());
    // Independent buffers
    assert_ne!(original.as_ptr(), cloned.as_ptr());
    cloned.push_str(" world");
    assert_eq!(original.as_str(), "hello");
    assert_eq!(cloned.as_str(), "hello world");
}

#[test]
fn clone_empty() {
    let arena = Arena::new();
    let original = arena.alloc_string();
    let cloned = original.clone();
    assert_eq!(cloned.as_str(), "");
    assert_eq!(cloned.capacity(), 0);
}

#[test]
fn deref_mut_allows_mutation() {
    let arena = Arena::new();
    let mut s = String::from_str_in("hello", &arena);
    let r: &mut str = &mut s;
    r.make_ascii_uppercase();
    assert_eq!(s.as_str(), "HELLO");
}

#[test]
fn as_mut_trait_allows_mutation() {
    let arena = Arena::new();
    let mut s = String::from_str_in("abc", &arena);
    let r: &mut str = AsMut::as_mut(&mut s);
    r.make_ascii_uppercase();
    assert_eq!(s.as_str(), "ABC");
}

#[test]
fn borrow_mut_trait_allows_mutation() {
    let arena = Arena::new();
    let mut s = String::from_str_in("xyz", &arena);
    let r: &mut str = core::borrow::BorrowMut::borrow_mut(&mut s);
    r.make_ascii_uppercase();
    assert_eq!(s.as_str(), "XYZ");
}
