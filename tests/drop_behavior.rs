//! Tests for drop-related guarantees: chunk-teardown drop ordering,
//! per-handle drop semantics, and lifetime guarantees when handles
//! outlive the arena.

#![allow(clippy::clone_on_ref_ptr, reason = "tests prefer concise method-call form")]
#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::items_after_statements, reason = "test-local types are clearer near use sites")]

use core::sync::atomic::{AtomicUsize, Ordering};
use harena::Arena;
use std::sync::Arc;

#[test]
fn alloc_drop_runs_at_chunk_teardown() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Counter;
    impl Drop for Counter {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let arena = Arena::new();
    {
        let _a = arena.alloc(Counter);
        let _b = arena.alloc(Counter);
        let _c = arena.alloc(Counter);
        assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    }
    drop(arena);
    assert_eq!(COUNT.load(Ordering::SeqCst), 3);
}

#[test]
fn drops_in_lifo_order() {
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    struct Logger {
        id: u32,
        log: Arc<std::sync::Mutex<Vec<u32>>>,
    }
    impl Drop for Logger {
        fn drop(&mut self) {
            self.log.lock().unwrap().push(self.id);
        }
    }

    let arena = Arena::new();
    let a = arena.alloc(Logger { id: 1, log: log.clone() });
    let b = arena.alloc(Logger { id: 2, log: log.clone() });
    let c = arena.alloc(Logger { id: 3, log: log.clone() });
    drop(a);
    drop(b);
    drop(c);
    drop(arena);
    assert_eq!(*log.lock().unwrap(), vec![3, 2, 1]);
}

#[test]
fn handles_keep_arena_storage_alive() {
    // The arena drops, but live handles keep their chunk's backing
    // storage alive.
    let s = {
        let arena = Arena::new();
        arena.alloc(String::from("survives the arena"))
    };
    assert_eq!(*s, "survives the arena");
}
