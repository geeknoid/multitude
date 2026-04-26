#![cfg(feature = "utf16")]
//! Cross-thread tests for `ArenaArcUtf16Str`.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use multitude::Arena;
use widestring::utf16str;

#[test]
fn arc_send_across_threads() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_arc(utf16str!("shared utf16"));
    let s2 = s.clone();
    let h = thread::spawn(move || s2.len());
    assert_eq!(h.join().unwrap(), 12);
    assert_eq!(&*s, utf16str!("shared utf16"));
}

#[test]
fn arc_concurrent_clone_drop() {
    let arena = Arena::new();
    let s = arena.alloc_utf16_str_arc(utf16str!("concurrent"));
    let counter = std::sync::Arc::new(AtomicUsize::new(0));
    let mut handles = std::vec::Vec::new();
    for _ in 0..8 {
        let s = s.clone();
        let c = std::sync::Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let copy = s.clone();
                let _ = c.fetch_add(copy.len(), Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(counter.load(Ordering::Relaxed), 8 * 100 * 10);
    assert_eq!(&*s, utf16str!("concurrent"));
}
