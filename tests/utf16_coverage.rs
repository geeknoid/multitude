#![cfg(all(feature = "utf16", feature = "builders"))]
//! Coverage-driven tests for the UTF-16 surface. Targets every
//! UTF-16-related line: trait impls, panicking variants, fallible Err
//! propagation, surrogate-pair edge cases, and the new `PartialEq<str>`
//! / `PartialEq<&str>` impls on the UTF-8 surface.

#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(unused_results, reason = "test code")]

use core::sync::atomic::AtomicUsize;
use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::AssertUnwindSafe;

use allocator_api2::alloc::{AllocError, Allocator, Global};
use multitude::builders::{String, Utf16String};
use multitude::{ArcStr, Arena, BoxStr, RcStr, RcUtf16Str};
use widestring::{Utf16Str, utf16str};

// ---------------------------------------------------------------------------
// Allocator helpers (mirrors of tests/common/mod.rs since utf16_coverage
// doesn't include the common shim).
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FailingAllocator {
    remaining: std::rc::Rc<Cell<usize>>,
}

impl FailingAllocator {
    fn new(allow: usize) -> Self {
        Self {
            remaining: std::rc::Rc::new(Cell::new(allow)),
        }
    }
}

// SAFETY: forwards to Global on success.
unsafe impl Allocator for FailingAllocator {
    fn allocate(&self, layout: core::alloc::Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        let r = self.remaining.get();
        if r == 0 {
            return Err(AllocError);
        }
        self.remaining.set(r - 1);
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: core::alloc::Layout) {
        // SAFETY: forwarded.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

#[derive(Clone)]
struct SendFailingAllocator {
    remaining: std::sync::Arc<AtomicUsize>,
}
impl SendFailingAllocator {
    fn new(allow: usize) -> Self {
        Self {
            remaining: std::sync::Arc::new(AtomicUsize::new(allow)),
        }
    }
}
// SAFETY: forwards to Global on success.
unsafe impl Allocator for SendFailingAllocator {
    fn allocate(&self, layout: core::alloc::Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        use core::sync::atomic::Ordering;
        loop {
            let r = self.remaining.load(Ordering::Relaxed);
            if r == 0 {
                return Err(AllocError);
            }
            if self
                .remaining
                .compare_exchange(r, r - 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Global.allocate(layout);
            }
        }
    }
    unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: core::alloc::Layout) {
        // SAFETY: forwarded.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

fn expect_panic<F: FnOnce()>(f: F) {
    let r = std::panic::catch_unwind(AssertUnwindSafe(f));
    assert!(r.is_err(), "expected panic but call returned");
}

fn fail_arena() -> Arena<FailingAllocator> {
    Arena::new_in(FailingAllocator::new(0))
}
fn send_fail_arena() -> Arena<SendFailingAllocator> {
    Arena::new_in(SendFailingAllocator::new(0))
}

// ---------------------------------------------------------------------------
// Panicking arena UTF-16 alloc methods → drive `unwrap_or_else(panic_alloc)`.
// ---------------------------------------------------------------------------

#[test]
fn panic_alloc_utf16_str_rc() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_utf16_str_rc(utf16str!("x"));
    });
}

#[test]
fn panic_alloc_utf16_str_arc() {
    expect_panic(|| {
        let a = send_fail_arena();
        let _ = a.alloc_utf16_str_arc(utf16str!("x"));
    });
}

#[test]
fn panic_alloc_utf16_str_box() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_utf16_str_box(utf16str!("x"));
    });
}

#[test]
fn panic_alloc_utf16_str_rc_from_str() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_utf16_str_rc_from_str("x");
    });
}

#[test]
fn panic_alloc_utf16_str_arc_from_str() {
    expect_panic(|| {
        let a = send_fail_arena();
        let _ = a.alloc_utf16_str_arc_from_str("x");
    });
}

#[test]
fn panic_alloc_utf16_str_box_from_str() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_utf16_str_box_from_str("x");
    });
}

#[test]
fn panic_alloc_utf16_string_with_capacity() {
    expect_panic(|| {
        let a = fail_arena();
        let _ = a.alloc_utf16_string_with_capacity(64);
    });
}

// ---------------------------------------------------------------------------
// Fallible try_* err paths.
// ---------------------------------------------------------------------------

#[test]
fn try_alloc_utf16_str_rc_err() {
    let a = fail_arena();
    a.try_alloc_utf16_str_rc(utf16str!("x")).unwrap_err();
}

#[test]
fn try_alloc_utf16_str_arc_err() {
    let a = send_fail_arena();
    a.try_alloc_utf16_str_arc(utf16str!("x")).unwrap_err();
}

#[test]
fn try_alloc_utf16_str_box_err() {
    let a = fail_arena();
    a.try_alloc_utf16_str_box(utf16str!("x")).unwrap_err();
}

#[test]
fn try_alloc_utf16_str_rc_from_str_err() {
    let a = fail_arena();
    a.try_alloc_utf16_str_rc_from_str("x").unwrap_err();
}

#[test]
fn try_alloc_utf16_str_arc_from_str_err() {
    let a = send_fail_arena();
    a.try_alloc_utf16_str_arc_from_str("x").unwrap_err();
}

#[test]
fn try_alloc_utf16_str_box_from_str_err() {
    let a = fail_arena();
    a.try_alloc_utf16_str_box_from_str("x").unwrap_err();
}

#[test]
fn try_alloc_utf16_string_with_capacity_err() {
    let a = fail_arena();
    a.try_alloc_utf16_string_with_capacity(64).unwrap_err();
}

#[test]
fn try_alloc_utf16_string_with_capacity_zero_no_alloc() {
    // cap == 0 — no allocation, no failure.
    let a = fail_arena();
    let s = a.try_alloc_utf16_string_with_capacity(0).unwrap();
    assert_eq!(s.capacity(), 0);
}

// ---------------------------------------------------------------------------
// ArenaUtf16String empty-string accessors and edge cases.
// ---------------------------------------------------------------------------

#[test]
fn empty_builder_accessors() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_string();
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
    assert_eq!(s.capacity(), 0);
    assert_eq!(s.as_utf16_str(), utf16str!(""));
    assert_eq!(s.as_slice(), &[][..]);
    let p: *const u16 = s.as_ptr();
    assert!(!p.is_null());
}

#[test]
fn empty_builder_as_mut() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    let m: &mut Utf16Str = s.as_mut_utf16_str();
    assert_eq!(m, utf16str!(""));
    let p: *mut u16 = s.as_mut_ptr();
    assert!(!p.is_null());
}

#[test]
fn pop_on_empty_returns_none() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    assert_eq!(s.pop(), None);
}

#[test]
fn truncate_noop_when_new_len_ge_len() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abc"));
    s.truncate(10);
    assert_eq!(s.as_utf16_str(), utf16str!("abc"));
    s.truncate(3);
    assert_eq!(s.as_utf16_str(), utf16str!("abc"));
}

#[test]
fn shrink_to_fit_noop_when_full() {
    let arena = Arena::new();
    let mut s = Utf16String::with_capacity_in(0, &arena);
    // cap==0 path
    s.shrink_to_fit();
    assert_eq!(s.capacity(), 0);
    // cap == len path
    s.push_str(utf16str!("hi"));
    let cap = s.capacity();
    let len = s.len();
    if cap == len {
        s.shrink_to_fit(); // still no-op
        assert_eq!(s.capacity(), cap);
    }
    let mut s2 = Utf16String::with_capacity_in(8, &arena);
    s2.push_str(utf16str!("ab"));
    drop(s); // ensure s2 is no longer at the bump cursor
    let _other = arena.alloc_utf16_str_rc(utf16str!("xx"));
    let cap_before = s2.capacity();
    s2.shrink_to_fit(); // not at bump cursor → no-op
    assert_eq!(s2.capacity(), cap_before);
}

#[test]
fn insert_str_empty_is_noop() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abc"));
    s.insert_utf16_str(1, utf16str!(""));
    assert_eq!(s.as_utf16_str(), utf16str!("abc"));
}

#[test]
fn insert_at_end() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abc"));
    s.insert_utf16_str(3, utf16str!("XY"));
    assert_eq!(s.as_utf16_str(), utf16str!("abcXY"));
}

#[test]
fn replace_range_unbounded() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("hello"));
    s.replace_range(.., utf16str!("world"));
    assert_eq!(s.as_utf16_str(), utf16str!("world"));
}

#[test]
fn replace_range_inclusive_excluded_bounds() {
    use core::ops::Bound;
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abcdef"));
    // (Excluded(0), Included(2)) ≡ 1..=2
    s.replace_range((Bound::Excluded(0), Bound::Included(2)), utf16str!("X"));
    assert_eq!(s.as_utf16_str(), utf16str!("aXdef"));
}

#[test]
fn replace_range_equal_size() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abcdef"));
    s.replace_range(2..4, utf16str!("XY")); // same length
    assert_eq!(s.as_utf16_str(), utf16str!("abXYef"));
}

#[test]
fn try_push_err() {
    let a = fail_arena();
    let mut s = Utf16String::new_in(&a);
    assert!(s.try_push('a').is_err());
}

#[test]
fn try_push_str_err() {
    let a = fail_arena();
    let mut s = Utf16String::new_in(&a);
    assert!(s.try_push_str(utf16str!("abc")).is_err());
}

#[test]
fn try_push_from_str_err() {
    let a = fail_arena();
    let mut s = Utf16String::new_in(&a);
    assert!(s.try_push_from_str("abc").is_err());
}

#[test]
fn try_reserve_zero_is_noop() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.try_reserve(0).unwrap();
    assert_eq!(s.capacity(), 0);
}

#[test]
fn reserve_panics_when_alloc_fails() {
    expect_panic(|| {
        let a = fail_arena();
        let mut s = Utf16String::new_in(&a);
        s.reserve(64);
    });
}

#[test]
fn try_reserve_err() {
    let a = fail_arena();
    let mut s = Utf16String::new_in(&a);
    assert!(s.try_reserve(64).is_err());
}

#[test]
fn push_empty_str_is_noop() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!(""));
    assert!(s.is_empty());
    s.push_from_str("");
    assert!(s.is_empty());
}

#[test]
fn try_push_from_str_with_surrogate_pair() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.try_push_from_str("a💖b").unwrap();
    assert_eq!(s.as_utf16_str(), utf16str!("a💖b"));
}

// ---------------------------------------------------------------------------
// Trait impls on ArenaUtf16String — exercise every Deref/Borrow/etc. branch.
// ---------------------------------------------------------------------------

#[test]
fn arena_utf16_string_traits() {
    use core::borrow::{Borrow, BorrowMut};
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("hello"));

    // Deref / DerefMut
    let _: &Utf16Str = &s;
    let _: &mut Utf16Str = &mut s;
    // AsRef / AsMut
    let _: &Utf16Str = AsRef::as_ref(&s);
    let _: &mut Utf16Str = AsMut::as_mut(&mut s);
    // Borrow / BorrowMut
    let _: &Utf16Str = Borrow::borrow(&s);
    let _: &mut Utf16Str = BorrowMut::borrow_mut(&mut s);

    // Clone
    let c = s.clone();
    assert_eq!(c, s);
    assert_eq!(c.as_utf16_str(), s.as_utf16_str());

    // Ord / PartialOrd
    let mut other = arena.alloc_utf16_string();
    other.push_str(utf16str!("hellp"));
    assert!(s < other);
    assert!(s.partial_cmp(&other).is_some());
    assert_eq!(s.cmp(&s.clone()), core::cmp::Ordering::Equal);

    // PartialEq vs Self / Utf16Str / &Utf16Str
    let lit = utf16str!("hello");
    assert_eq!(s, lit);
    assert!(s == lit);

    // Hash
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    let _ = h.finish();

    // Display / Debug
    let _ = format!("{s}");
    let _ = format!("{s:?}");
}

// ---------------------------------------------------------------------------
// Trait impls / methods on ArenaRcUtf16Str / ArenaArcUtf16Str / ArenaBoxUtf16Str.
// ---------------------------------------------------------------------------

#[test]
fn rc_utf16_traits_and_pointer() {
    use core::borrow::Borrow;
    let arena = Arena::new();
    let r = arena.alloc_utf16_str_rc(utf16str!("hello"));
    let _: &Utf16Str = &r;
    let _: &Utf16Str = AsRef::as_ref(&r);
    let _: &Utf16Str = Borrow::borrow(&r);

    // PartialEq vs Utf16Str / &Utf16Str
    assert!(r == *utf16str!("hello"));
    assert!(r == utf16str!("hello"));

    // Display / Debug / Pointer formatters
    let _ = format!("{r}");
    let _ = format!("{r:?}");
    let _ = format!("{r:p}");

    // Hash
    let mut h = DefaultHasher::new();
    r.hash(&mut h);
    let _ = h.finish();

    // Ord / PartialOrd via clones
    let r2 = r.clone();
    assert_eq!(r.cmp(&r2), core::cmp::Ordering::Equal);
    assert_eq!(r.partial_cmp(&r2), Some(core::cmp::Ordering::Equal));

    // From<ArenaRcUtf16Str> for ArenaRc<[u16]>
    let bytes: multitude::Rc<[u16]> = r.into();
    assert_eq!(
        &*bytes,
        &[u16::from(b'h'), u16::from(b'e'), u16::from(b'l'), u16::from(b'l'), u16::from(b'o')][..]
    );
}

#[test]
fn arc_utf16_traits_and_pointer() {
    use core::borrow::Borrow;
    let arena = Arena::new();
    let a = arena.alloc_utf16_str_arc(utf16str!("hello"));
    let _: &Utf16Str = &a;
    let _: &Utf16Str = AsRef::as_ref(&a);
    let _: &Utf16Str = Borrow::borrow(&a);

    assert!(a == *utf16str!("hello"));
    assert!(a == utf16str!("hello"));

    let _ = format!("{a}");
    let _ = format!("{a:?}");
    let _ = format!("{a:p}");

    let mut h = DefaultHasher::new();
    a.hash(&mut h);
    let _ = h.finish();

    let a2 = a.clone();
    assert_eq!(a.cmp(&a2), core::cmp::Ordering::Equal);
    assert_eq!(a.partial_cmp(&a2), Some(core::cmp::Ordering::Equal));

    let bytes: multitude::Arc<[u16]> = a.into();
    assert_eq!(
        &*bytes,
        &[u16::from(b'h'), u16::from(b'e'), u16::from(b'l'), u16::from(b'l'), u16::from(b'o')][..]
    );
}

#[test]
fn box_utf16_traits_pointer_and_mutators() {
    use core::borrow::{Borrow, BorrowMut};
    let arena = Arena::new();
    let mut b = arena.alloc_utf16_str_box(utf16str!("hello"));

    // Deref / DerefMut / AsRef / AsMut / Borrow / BorrowMut
    let _: &Utf16Str = &b;
    let _: &mut Utf16Str = &mut b;
    let _: &Utf16Str = AsRef::as_ref(&b);
    let _: &mut Utf16Str = AsMut::as_mut(&mut b);
    let _: &Utf16Str = Borrow::borrow(&b);
    let _: &mut Utf16Str = BorrowMut::borrow_mut(&mut b);

    // PartialEq variants
    assert!(b == *utf16str!("hello"));
    assert!(b == utf16str!("hello"));

    // Display / Debug / Pointer
    let _ = format!("{b}");
    let _ = format!("{b:?}");
    let _ = format!("{b:p}");

    // Hash
    let mut h = DefaultHasher::new();
    b.hash(&mut h);
    let _ = h.finish();

    // Ord / PartialOrd against another box
    let b2 = arena.alloc_utf16_str_box(utf16str!("hello"));
    assert_eq!(b.cmp(&b2), core::cmp::Ordering::Equal);
    assert_eq!(b.partial_cmp(&b2), Some(core::cmp::Ordering::Equal));

    // From<ArenaBoxUtf16Str> for ArenaRcUtf16Str
    let r: RcUtf16Str = b.into();
    assert_eq!(&*r, utf16str!("hello"));
}

// ---------------------------------------------------------------------------
// New PartialEq<str>/<&str> on the UTF-8 surface — added in the review fix.
// ---------------------------------------------------------------------------

#[test]
fn utf8_smart_pointer_partial_eq_str() {
    let arena = Arena::new();

    let r: RcStr = arena.alloc_str_rc("hi");
    assert!(r == *"hi");
    assert!(r == "hi");

    let a: ArcStr = arena.alloc_str_arc("hi");
    assert!(a == *"hi");
    assert!(a == "hi");

    let b: BoxStr = arena.alloc_str_box("hi");
    assert!(b == *"hi");
    assert!(b == "hi");

    let mut g: String = arena.alloc_string();
    g.push_str("hi");
    assert!(g == *"hi");
    assert!(g == "hi");
}

// ---------------------------------------------------------------------------
// `from_utf16_str_in` and `from_str_in` constructors.
// ---------------------------------------------------------------------------

#[test]
fn from_utf16_str_in_and_from_str_in() {
    let arena = Arena::new();
    let a = Utf16String::from_str_in("hello, 💖", &arena);
    assert_eq!(a.as_utf16_str(), utf16str!("hello, 💖"));
    let b = Utf16String::from_utf16_str_in(utf16str!("world"), &arena);
    assert_eq!(b.as_utf16_str(), utf16str!("world"));
}

// ---------------------------------------------------------------------------
// Extend impls.
// ---------------------------------------------------------------------------

#[test]
fn extend_chars_with_size_hint() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    let chars = ['a', 'b', 'c', '💖'];
    s.extend(chars.iter().copied()); // size_hint > 0 → reserve path
    assert_eq!(s.as_utf16_str(), utf16str!("abc💖"));
}

#[test]
fn extend_chars_zero_lower_bound() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.extend(core::iter::empty::<char>()); // lower == 0 → skip reserve branch
    assert!(s.is_empty());
}

#[test]
fn extend_str_slices() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.extend(["ab", "cd"]);
    assert_eq!(s.as_utf16_str(), utf16str!("abcd"));
}

#[test]
fn extend_utf16_str_slices() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.extend([utf16str!("ab"), utf16str!("cd")]);
    assert_eq!(s.as_utf16_str(), utf16str!("abcd"));
}

// ---------------------------------------------------------------------------
// Boundary panics.
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "not on a UTF-16 char boundary")]
fn insert_at_mid_surrogate_panics() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push('💖');
    s.insert(1, 'X');
}

#[test]
#[should_panic(expected = "not on a UTF-16 char boundary")]
fn remove_at_mid_surrogate_panics() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push('💖');
    s.remove(1);
}

#[test]
#[should_panic(expected = "not on a UTF-16 char boundary")]
fn replace_range_start_mid_surrogate_panics() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("a💖b"));
    s.replace_range(2..3, utf16str!(""));
}

#[test]
#[should_panic(expected = "Utf16String::replace_range")]
fn replace_range_end_oob_panics() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abc"));
    s.replace_range(0..10, utf16str!(""));
}

#[test]
#[should_panic(expected = "Utf16String::replace_range")]
fn replace_range_start_greater_than_end_panics() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abc"));
    #[expect(clippy::reversed_empty_ranges, reason = "intentionally inverted to trigger panic")]
    s.replace_range(2..1, utf16str!(""));
}

// ---------------------------------------------------------------------------
// Capacity growth hits the doubling-vs-min_cap path.
// ---------------------------------------------------------------------------

#[test]
fn grow_doubling_path() {
    let arena = Arena::new();
    let mut s = Utf16String::with_capacity_in(4, &arena);
    s.push_str(utf16str!("abcd"));
    s.push('e'); // forces grow; doubling 4*2 = 8 covers needed 5
    assert!(s.capacity() >= 8);
    assert_eq!(s.as_utf16_str(), utf16str!("abcde"));
}

#[test]
fn grow_uses_min_cap_when_doubling_too_small() {
    let arena = Arena::new();
    let mut s = Utf16String::with_capacity_in(4, &arena);
    s.push_str(utf16str!("abcd"));
    let big = "x".repeat(100);
    s.push_from_str(&big); // forces grow; min_cap > 2*old_cap
    assert!(s.capacity() >= 104);
}

// ---------------------------------------------------------------------------
// Drop: ensure non-empty ArenaUtf16String releases its chunk refcount.
// ---------------------------------------------------------------------------

#[test]
fn arena_utf16_string_drop_releases_chunk() {
    let arena = Arena::new();
    {
        let mut s = arena.alloc_utf16_string();
        s.push_str(utf16str!("data"));
        // explicit drop covers the cap > 0 branch in Drop
    }
    // arena drop later covers the cap == 0 branch via the empty builder used elsewhere
}

// ---------------------------------------------------------------------------
// retain — exercise both keep and drop branches.
// ---------------------------------------------------------------------------

#[test]
fn retain_keep_all() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abc"));
    s.retain(|_| true);
    assert_eq!(s.as_utf16_str(), utf16str!("abc"));
}

#[test]
fn retain_drop_all() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("abc"));
    s.retain(|_| false);
    assert!(s.is_empty());
}

#[test]
fn retain_with_surrogate() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("a💖b"));
    s.retain(|c| c.is_ascii());
    assert_eq!(s.as_utf16_str(), utf16str!("ab"));
}

// ---------------------------------------------------------------------------
// clear()
// ---------------------------------------------------------------------------

#[test]
fn clear_resets_len_only() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("hello"));
    let cap = s.capacity();
    s.clear();
    assert_eq!(s.len(), 0);
    assert_eq!(s.capacity(), cap);
}

// ---------------------------------------------------------------------------
// into_arena_utf16_str empty-builder shortcut path.
// ---------------------------------------------------------------------------

#[test]
fn into_arena_utf16_str_zero_cap() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_string();
    let r = s.into_arena_utf16_str();
    assert!(r.is_empty());
}

#[test]
fn into_arena_utf16_str_with_data_reclaims_tail() {
    let arena = Arena::builder().chunk_cache_capacity(0).build();
    let mut s = arena.alloc_utf16_string_with_capacity(64);
    s.push_str(utf16str!("hi"));
    let r = s.into_arena_utf16_str();
    assert_eq!(&*r, utf16str!("hi"));
    let _follow = arena.alloc_utf16_string_with_capacity(50);
}

// ---------------------------------------------------------------------------
// From<ArenaUtf16String> for ArenaRcUtf16Str (exercises the
// `into` arrow that the From impl wraps).
// ---------------------------------------------------------------------------

#[test]
fn from_arena_utf16_string_for_rc() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("frozen"));
    let r: RcUtf16Str = s.into();
    assert_eq!(&*r, utf16str!("frozen"));
}

// ---------------------------------------------------------------------------
// Display impl chain: ArenaArcUtf16Str / ArenaBoxUtf16Str / ArenaRcUtf16Str
// pass through to Utf16Str's Display.
// ---------------------------------------------------------------------------

#[test]
fn display_passes_through() {
    let arena = Arena::new();
    let r = arena.alloc_utf16_str_rc(utf16str!("hi💖"));
    assert_eq!(format!("{r}"), "hi💖");
    let a = arena.alloc_utf16_str_arc(utf16str!("hi💖"));
    assert_eq!(format!("{a}"), "hi💖");
    let b = arena.alloc_utf16_str_box(utf16str!("hi💖"));
    assert_eq!(format!("{b}"), "hi💖");
}

// ---------------------------------------------------------------------------
// Insert / replace_range that force the builder's grow path.
// ---------------------------------------------------------------------------

#[test]
fn insert_grows_capacity() {
    let arena = Arena::new();
    let mut s = Utf16String::with_capacity_in(4, &arena);
    s.push_str(utf16str!("abcd"));
    s.insert_utf16_str(2, utf16str!("XYZW"));
    assert_eq!(s.as_utf16_str(), utf16str!("abXYZWcd"));
    assert!(s.capacity() >= 8);
}

#[test]
fn replace_range_grows_capacity() {
    let arena = Arena::new();
    let mut s = Utf16String::with_capacity_in(4, &arena);
    s.push_str(utf16str!("abcd"));
    s.replace_range(0..1, utf16str!("XXXXX")); // adds 4 → needs cap 8
    assert_eq!(s.as_utf16_str(), utf16str!("XXXXXbcd"));
    assert!(s.capacity() >= 8);
}

// ---------------------------------------------------------------------------
// as_slice on a non-empty builder (covers the post-empty branch).
// ---------------------------------------------------------------------------

#[test]
fn as_slice_non_empty() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("ab"));
    let slice: &[u16] = s.as_slice();
    assert_eq!(slice, &[u16::from(b'a'), u16::from(b'b')][..]);
}

// ---------------------------------------------------------------------------
// PartialEq<Utf16Str> (by-value rhs) on the ArenaUtf16String — explicitly
// invoke the impl so its body is hit.
// ---------------------------------------------------------------------------

#[test]
fn arena_utf16_string_eq_utf16str_value() {
    let arena = Arena::new();
    let mut s = arena.alloc_utf16_string();
    s.push_str(utf16str!("hi"));
    let lit: &Utf16Str = utf16str!("hi");
    // Both PartialEq<&Utf16Str> (s == lit) and PartialEq<Utf16Str> (s == *lit)
    // are reachable through method-call form; using the bare op resolves to
    // whichever impl rustc prefers, so call eq directly to nail the by-value one.
    assert!(<Utf16String<'_, _> as PartialEq<Utf16Str>>::eq(&s, lit));
    assert!(<Utf16String<'_, _> as PartialEq<&Utf16Str>>::eq(&s, &lit));
}

// ---------------------------------------------------------------------------
// Panicking from_str helpers' second-stage Err: cap allocation succeeds, the
// secondary alloc (Shared chunk for arc; Local for box) fails. This drives
// the `?` propagation past `try_with_capacity_in` in the from_str helpers.
// ---------------------------------------------------------------------------

#[test]
fn try_alloc_utf16_str_arc_from_str_err_on_second_alloc() {
    // FailingAllocator(1): first alloc (the Local builder chunk) succeeds,
    // second alloc (the Shared chunk for the Arc) fails.
    let arena = Arena::new_in(SendFailingAllocator::new(1));
    arena.try_alloc_utf16_str_arc_from_str("xyz").unwrap_err();
}

#[test]
fn try_alloc_utf16_str_rc_from_str_success() {
    let arena = Arena::new();
    let r = arena.try_alloc_utf16_str_rc_from_str("hello, 💖").unwrap();
    assert_eq!(&*r, utf16str!("hello, 💖"));
}

#[test]
fn try_alloc_utf16_str_arc_from_str_success() {
    let arena = Arena::new();
    let a = arena.try_alloc_utf16_str_arc_from_str("hello").unwrap();
    assert_eq!(&*a, utf16str!("hello"));
}

#[test]
fn try_alloc_utf16_str_box_from_str_success() {
    let arena = Arena::new();
    let b = arena.try_alloc_utf16_str_box_from_str("hello").unwrap();
    assert_eq!(&*b, utf16str!("hello"));
}

#[test]
fn try_alloc_utf16_str_box_from_str_err_on_second_alloc() {
    // For Local box alloc, the buffer chunk and the box chunk are both
    // Local. The box reuses the builder's chunk via bump alloc, so a
    // single FailingAllocator(1) lets it succeed. Force a separate alloc
    // by demanding a string that won't fit in one chunk.
    let arena = Arena::builder().chunk_size(8 * 1024).allocator_in(FailingAllocator::new(1)).build();
    // 5000 ASCII bytes → 5000 u16s = 10 KB > 4 KiB chunk. Builder grows
    // (or relocates) → second alloc fails.
    let big = "x".repeat(5000);
    arena.try_alloc_utf16_str_box_from_str(&big).unwrap_err();
}

// ---------------------------------------------------------------------------
// Panic-driving variants of the same path (drive the `panic_alloc` lambda
// in `alloc_utf16_str_*_from_str`).
// ---------------------------------------------------------------------------

#[test]
fn panic_alloc_utf16_str_arc_from_str_when_only_local_allowed() {
    expect_panic(|| {
        let arena = Arena::new_in(SendFailingAllocator::new(1));
        let _ = arena.alloc_utf16_str_arc_from_str("xyz");
    });
}

#[test]
fn panic_alloc_utf16_str_box_from_str_when_only_first_alloc_allowed() {
    expect_panic(|| {
        let arena = Arena::builder().chunk_size(8 * 1024).allocator_in(FailingAllocator::new(1)).build();
        let big = "x".repeat(5000);
        let _ = arena.alloc_utf16_str_box_from_str(&big);
    });
}

// ---------------------------------------------------------------------------
// grow_for_string err inside try_grow_to_at_least: cap > 0 path that
// re-allocates and the allocator fails on the relocation.
// ---------------------------------------------------------------------------

#[test]
fn grow_for_string_err_on_relocation() {
    // FailingAllocator(1): builder gets initial chunk (cap=4), then a huge
    // try_reserve would need a fresh oversized chunk → second alloc fails.
    let arena = Arena::builder().chunk_size(8 * 1024).allocator_in(FailingAllocator::new(1)).build();
    let mut s = Utf16String::try_with_capacity_in(4, &arena).unwrap();
    s.try_push_str(utf16str!("abcd")).unwrap();
    assert!(s.try_reserve(64 * 1024).is_err());
}

#[test]
fn panic_grow_to_at_least() {
    expect_panic(|| {
        let arena = Arena::builder().chunk_size(8 * 1024).allocator_in(FailingAllocator::new(1)).build();
        let mut s = Utf16String::try_with_capacity_in(4, &arena).unwrap();
        s.try_push_str(utf16str!("abcd")).unwrap();
        // grow_to_at_least's panic_alloc lambda fires here.
        s.push_from_str(&"x".repeat(64 * 1024));
    });
}

// ---------------------------------------------------------------------------
// Panic paths through push / push_str / push_from_str when the allocator
// fails — drives the `unwrap_or_else(panic_alloc)` lambdas inside push_slice
// and push_from_str, plus the `reserve` path with no growth needed.
// ---------------------------------------------------------------------------

#[test]
fn panic_push_when_alloc_fails() {
    expect_panic(|| {
        let a = fail_arena();
        let mut s = Utf16String::new_in(&a);
        s.push('a');
    });
}

#[test]
fn panic_push_str_when_alloc_fails() {
    expect_panic(|| {
        let a = fail_arena();
        let mut s = Utf16String::new_in(&a);
        s.push_str(utf16str!("abc"));
    });
}

#[test]
fn panic_push_from_str_when_alloc_fails() {
    expect_panic(|| {
        let a = fail_arena();
        let mut s = Utf16String::new_in(&a);
        s.push_from_str("abc");
    });
}

#[test]
fn reserve_no_growth_path() {
    let arena = Arena::new();
    let mut s = Utf16String::with_capacity_in(16, &arena);
    s.reserve(4); // already have cap >= len + 4; no-op branch
    assert_eq!(s.capacity(), 16);
    assert_eq!(s.len(), 0);
}

// ---------------------------------------------------------------------------
// `total > isize::MAX` guard: cap whose payload byte count overflows isize
// (but not usize) — we never allocate; the function returns AllocError
// before reaching the allocator.
// ---------------------------------------------------------------------------

#[test]
fn try_with_capacity_isize_overflow_guard() {
    let arena = Arena::new();
    // cap*2 > isize::MAX but ≤ usize::MAX on 64-bit. checked_mul succeeds,
    // checked_add succeeds, isize::try_from fails → AllocError.
    let cap = (isize::MAX.unsigned_abs() / 2) + 1000;
    let r = Utf16String::try_with_capacity_in(cap, &arena);
    r.unwrap_err();
}

#[test]
fn try_grow_isize_overflow_guard() {
    let arena = Arena::new();
    let mut s = Utf16String::try_with_capacity_in(4, &arena).unwrap();
    // Force try_grow_to_at_least → new_cap such that new_cap*2 > isize::MAX
    // but ≤ usize::MAX. The isize::try_from check on new_total returns Err.
    let huge = (isize::MAX.unsigned_abs() / 2) + 1000;
    let r = s.try_reserve(huge);
    assert!(r.is_err());
}
