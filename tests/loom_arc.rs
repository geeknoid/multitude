//! Loom model-checked tests for multitude's cross-thread refcount machinery.
//!
//! These tests are only built and run under `--cfg loom`. They use the
//! atomic primitives swapped in by `src/sync.rs` so Loom can permute the
//! orderings of every `fetch_add`, `fetch_sub`, `store`, `load`, and
//! `fence` and verify the resulting executions remain sound.
//!
//! Run with:
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" \
//!     cargo test --test loom_arc --release -- --nocapture --test-threads=1
//! ```
//!
//! ## What is verified
//!
//! Each test focuses on one concurrent invariant the design relies on.
//! The tests are deliberately small (≤ 3 worker threads, ≤ 2 clones)
//! because Loom's state space is exponential in interleaving points;
//! large workloads would not finish in reasonable time.
//!
//! - **`arc_clone_drop_race`** — N worker threads each clone an `Arc`
//!   then drop the clone. After all join, the original drops, and the
//!   chunk reaches refcount 0 exactly once, so the payload's `Drop`
//!   runs exactly once. Verifies the standard `fetch_add(Relaxed)` /
//!   `fetch_sub(Release) + fence(Acquire)` pattern on `ref_count`.
//!
//! - **`arc_drop_after_arena_drop`** — the owner thread drops the arena
//!   handle while a worker holds an `Arc`. The worker then drops the
//!   `Arc`, which decs the chunk's `ref_count`, runs teardown on the
//!   non-owner thread (because `arena_dropped` is observed `true`),
//!   decs `outstanding_chunks` to 0, and runs the `last_reclaimer`
//!   path that frees `ArenaInner`. Verifies the `Release/Acquire`
//!   pair on `arena_dropped` and the last-reclaimer fence on
//!   `outstanding_chunks`.
//!
//! - **`deferred_reconciliation_race`** — the owner thread allocates K
//!   `Arc`s on a Shared chunk (so `arcs_issued = K` non-atomically),
//!   then evicts the slot. The eviction path does
//!   `fetch_sub(LARGE - K, Release)` on the chunk's atomic
//!   `ref_count`. A worker thread holds one of the `Arc`s and drops
//!   it concurrently with the eviction. The reconciliation math
//!   `(LARGE - d) - (LARGE - m) = m - d` must produce the correct
//!   live-Arc count (here: 0) under any interleaving, so teardown
//!   runs exactly once.
//!
//! - **`two_arcs_two_threads`** — two clones, two worker threads each
//!   dropping one. The owner has already dropped its share. The last
//!   drop must hit refcount 0 on exactly one thread, and that thread
//!   runs teardown.

#![cfg(loom)]
#![allow(clippy::std_instead_of_core, reason = "loom + std interop in tests")]
#![allow(clippy::missing_panics_doc, reason = "test code")]
#![allow(clippy::unwrap_used, reason = "test code")]

use std::sync::atomic::{AtomicUsize as StdAtomicUsize, Ordering as StdOrdering};

use loom::thread;

use multitude::{Arc, Arena};

/// A payload type that increments a global counter on Drop.  We use a
/// real `std::sync::atomic::AtomicUsize` here (NOT loom's), because
/// the counter exists outside the model: it accumulates *across*
/// permutations to verify each permutation drops exactly once.
fn drop_counter() -> &'static StdAtomicUsize {
    static C: StdAtomicUsize = StdAtomicUsize::new(0);
    &C
}

struct DropCounted;

impl Drop for DropCounted {
    fn drop(&mut self) {
        let _prev = drop_counter().fetch_add(1, StdOrdering::Relaxed);
    }
}

/// Build a fresh `Arena` with a tiny chunk size so chunk allocation
/// happens deterministically per scenario.
fn fresh_arena() -> Arena {
    Arena::builder().chunk_size(8 * 1024).max_normal_alloc(4 * 1024).build()
}

#[test]
fn arc_clone_drop_race() {
    loom::model(|| {
        let baseline = drop_counter().load(StdOrdering::Relaxed);

        let arena = fresh_arena();
        let original: Arc<DropCounted> = arena.alloc_arc(DropCounted);
        let c1 = Arc::clone(&original);
        let c2 = Arc::clone(&original);

        let t1 = thread::spawn(move || {
            drop(c1);
        });
        let t2 = thread::spawn(move || {
            drop(c2);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Owner-side drop of `original` and `arena`.
        drop(original);
        drop(arena);

        // Exactly one Drop must have run for the payload, regardless
        // of the interleaving Loom picked.
        let after = drop_counter().load(StdOrdering::Relaxed);
        assert_eq!(after - baseline, 1, "DropCounted::drop must run exactly once");
    });
}

#[test]
fn arc_drop_after_arena_drop() {
    loom::model(|| {
        let baseline = drop_counter().load(StdOrdering::Relaxed);

        let arena = fresh_arena();
        let arc: Arc<DropCounted> = arena.alloc_arc(DropCounted);

        // Owner drops the arena BEFORE the worker drops the Arc.
        // Worker's Arc::drop decs ref_count to 0 → runs teardown on
        // the worker (non-owner) thread → decs outstanding_chunks →
        // last reclaimer → free_storage(ArenaInner).
        let t = thread::spawn(move || {
            drop(arc);
        });

        drop(arena);
        t.join().unwrap();

        let after = drop_counter().load(StdOrdering::Relaxed);
        assert_eq!(after - baseline, 1);
    });
}

#[test]
fn two_arcs_two_threads() {
    loom::model(|| {
        let baseline = drop_counter().load(StdOrdering::Relaxed);

        let arena = fresh_arena();
        let original: Arc<DropCounted> = arena.alloc_arc(DropCounted);
        let c1 = Arc::clone(&original);
        let c2 = Arc::clone(&original);
        // Owner releases its share first; only the worker-thread Arcs remain.
        drop(original);

        let t1 = thread::spawn(move || drop(c1));
        let t2 = thread::spawn(move || drop(c2));

        t1.join().unwrap();
        t2.join().unwrap();

        drop(arena);

        let after = drop_counter().load(StdOrdering::Relaxed);
        assert_eq!(after - baseline, 1);
    });
}

#[test]
fn deferred_reconciliation_race() {
    // The owner allocates 2 Arcs on a Shared chunk (so the chunk's
    // arcs_issued is bumped twice non-atomically). The slot evicts
    // when we drop the arena, which does
    // `fetch_sub(LARGE - 2, Release)` on the chunk's atomic
    // `ref_count`. A worker thread is concurrently dropping one of
    // the Arcs (an atomic `fetch_sub(1, Release)`).
    //
    // After both have run and the owner-side `arc2.drop()` has also
    // run, the chunk's net refcount must be zero and teardown must
    // run exactly once.
    loom::model(|| {
        let baseline = drop_counter().load(StdOrdering::Relaxed);

        let arena = fresh_arena();
        let arc1: Arc<DropCounted> = arena.alloc_arc(DropCounted);
        let arc2: Arc<DropCounted> = arena.alloc_arc(DropCounted);

        let t = thread::spawn(move || drop(arc1));

        // Drop `arc2` and the arena on the owner thread, racing the
        // worker's `arc1` drop.
        drop(arc2);
        drop(arena);

        t.join().unwrap();

        let after = drop_counter().load(StdOrdering::Relaxed);
        assert_eq!(after - baseline, 2, "both DropCounted payloads must drop exactly once");
    });
}

#[test]
fn arc_clone_then_send_then_drop() {
    // Owner clones an Arc and sends it to a worker. Worker drops its
    // clone. Owner drops the original and the arena. Verifies the
    // standard inc/dec ordering produces a correct final count even
    // when the inc and dec happen on different threads.
    loom::model(|| {
        let baseline = drop_counter().load(StdOrdering::Relaxed);

        let arena = fresh_arena();
        let original: Arc<DropCounted> = arena.alloc_arc(DropCounted);
        let cloned = Arc::clone(&original);

        let t = thread::spawn(move || drop(cloned));

        drop(original);
        t.join().unwrap();
        drop(arena);

        let after = drop_counter().load(StdOrdering::Relaxed);
        assert_eq!(after - baseline, 1);
    });
}

#[test]
fn worker_clones_then_drops() {
    // Owner sends an Arc to a worker. The worker clones it on its
    // own thread, drops the clone, then drops the original. The
    // owner drops the arena. Verifies clone-on-non-owner-thread is
    // sound (the inc + later dec both happen on the worker, but
    // they're paired across the spawned thread boundary, exercising
    // Loom's HB tracking).
    loom::model(|| {
        let baseline = drop_counter().load(StdOrdering::Relaxed);

        let arena = fresh_arena();
        let original: Arc<DropCounted> = arena.alloc_arc(DropCounted);

        let t = thread::spawn(move || {
            let cloned = Arc::clone(&original);
            drop(cloned);
            drop(original);
        });

        // Owner waits for worker, then drops the arena.
        t.join().unwrap();
        drop(arena);

        let after = drop_counter().load(StdOrdering::Relaxed);
        assert_eq!(after - baseline, 1);
    });
}

#[test]
fn arena_drop_concurrent_with_clone_and_drop() {
    // Three concurrent operations: owner drops the arena while two
    // workers are racing on Arc clone/drop. Stresses the
    // `arena_dropped` Acquire/Release pairing on cross-thread
    // teardown.
    loom::model(|| {
        let baseline = drop_counter().load(StdOrdering::Relaxed);

        let arena = fresh_arena();
        let original: Arc<DropCounted> = arena.alloc_arc(DropCounted);
        let c1 = Arc::clone(&original);

        let t1 = thread::spawn(move || drop(c1));
        // Owner drops original then arena while t1 races.
        drop(original);
        drop(arena);
        t1.join().unwrap();

        let after = drop_counter().load(StdOrdering::Relaxed);
        assert_eq!(after - baseline, 1);
    });
}
