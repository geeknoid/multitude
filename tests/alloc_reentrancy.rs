//! Regression test: a user `Drop` impl invoked during alloc-time chunk
//! eviction may re-enter `arena.alloc_*`. The re-entrant call must
//! observe a populated `current_*` slot so it cannot race the outer
//! call's `current_slot.set(...)`. Previously the evicted, non-pinned
//! `OwnedChunk` was dropped *before* the new chunk was installed,
//! letting the re-entrant call populate the slot first; the outer
//! call's `set` then triggered a debug assertion (release builds
//! silently leaked the previous chunk).

#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]

use core::cell::Cell;
use multitude::Arena;

#[test]
fn reentrant_alloc_from_drop_during_eviction() {
    struct DropAlloc<'a> {
        arena: &'a Arena,
        fired: &'a Cell<usize>,
    }
    impl Drop for DropAlloc<'_> {
        fn drop(&mut self) {
            self.fired.set(self.fired.get() + 1);
            // Re-entrant allocation while the outer alloc is mid-eviction.
            let _r = self.arena.alloc_rc(42_u32);
        }
    }

    let arena: Arena = Arena::builder().chunk_size(8 * 1024).max_normal_alloc(4 * 1024).build();
    let fired = Cell::new(0);

    {
        let r = arena.alloc_rc(DropAlloc {
            arena: &arena,
            fired: &fired,
        });
        // Release the outer Rc; the entry remains linked into the chunk's
        // drop list (Rc only released the slot's +1).
        drop(r);

        // Allocate enough small smart-pointer values to force the chunk
        // holding our `DropAlloc` entry to be evicted from `current_local`.
        for _ in 0..200 {
            let r = arena.alloc_rc([0_u8; 100]);
            drop(r);
        }
    }
    // Drop runs at arena teardown; the inner re-entrant alloc must succeed
    // without panicking and without leaking chunks.
    assert!(fired.get() >= 1);
}
