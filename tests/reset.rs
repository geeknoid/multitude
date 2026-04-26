//! Tests for `Arena::reset`.
//!
//! `reset` releases the arena's hold on every chunk it currently owns.
//! Per chunk:
//!   - If no refcounted smart pointer still references it, the chunk's drop
//!     list runs and the chunk returns to the cache (or is freed).
//!   - If at least one smart pointer is still alive, the chunk *detaches* from
//!     the arena and lives on under smart pointer ownership, exactly as it
//!     would after `Arena::drop`. When the last smart pointer eventually
//!     drops, the chunk's drop list runs and the chunk is freed (or
//!     returned to the arena's cache).
//!
//! These tests cover:
//! - Empty / idempotent reset.
//! - Destructors run for non-smart pointer allocations on reset.
//! - Lifetime stats survive reset (`chunks_allocated` accumulates).
//! - Cache reuse: post-reset allocation doesn't allocate a fresh chunk.
//! - Byte budget gets reset to "still committed in the cache".
//! - Reset works while `ArenaRc` / `ArenaArc` smart pointers are outstanding;
//!   the smart pointers keep working; their chunks rejoin the cache when the
//!   smart pointers drop.
//! - Cross-thread `ArenaArc` smart pointers don't perturb reset.
//! - Pinned chunks (chunks held by `Arena::alloc`-style refs) are
//!   handled correctly once those refs are out of scope.

#![allow(clippy::std_instead_of_core, reason = "tests use std")]
#![allow(clippy::unwrap_used, reason = "test code")]

use core::sync::atomic::{AtomicUsize, Ordering};
use multitude::{Arc, Arena, Rc};
#[test]
fn reset_on_empty_arena_is_a_noop() {
    let mut arena = Arena::new();
    arena.reset();
    // Still usable.
    let r = arena.alloc_rc(1_u32);
    assert_eq!(*r, 1);
}

#[test]
fn reset_idempotent() {
    let mut arena: Arena = Arena::new();
    arena.reset();
    arena.reset();
    arena.reset();
    let _ = arena.alloc(0_u8);
    arena.reset();
    arena.reset();
}

#[test]
fn reset_runs_destructors_for_alloc_style_values() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked;
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let mut arena = Arena::new();
    {
        let _v: &mut Tracked = arena.alloc(Tracked);
    }
    assert_eq!(COUNT.load(Ordering::SeqCst), 0, "drop hasn't fired yet");
    arena.reset();
    assert_eq!(COUNT.load(Ordering::SeqCst), 1, "destructor must run during reset");
}

#[test]
fn reset_runs_destructors_for_all_chunk_residents() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked;
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let mut arena = Arena::new();
    for _ in 0..5 {
        let _: &mut Tracked = arena.alloc(Tracked);
    }
    arena.reset();
    assert_eq!(COUNT.load(Ordering::SeqCst), 5);
}

#[cfg(feature = "stats")]
#[test]
fn reset_returns_chunks_to_cache_and_avoids_fresh_alloc() {
    let mut arena = Arena::builder().chunk_cache_capacity(4).build();
    let _ = arena.alloc(0_u64);

    let stats_before = arena.stats();
    assert!(stats_before.chunks_allocated >= 1);

    arena.reset();

    let stats_after_reset = arena.stats();
    assert_eq!(stats_after_reset.chunks_allocated, stats_before.chunks_allocated);

    // Should reuse the cached chunk.
    let _ = arena.alloc(1_u64);
    let stats_after_realloc = arena.stats();
    assert_eq!(
        stats_after_realloc.chunks_allocated, stats_before.chunks_allocated,
        "reset should not allocate a fresh chunk; cache reuse expected"
    );
}

#[cfg(feature = "stats")]
#[test]
fn reset_preserves_lifetime_chunk_count_across_phases() {
    let mut arena = Arena::new();
    let mut last = 0;
    for _ in 0..3 {
        for _ in 0..10 {
            let _ = arena.alloc(0_u64);
        }
        let now = arena.stats().chunks_allocated;
        assert!(now >= last, "lifetime chunks_allocated must be monotonic across resets");
        last = now;
        arena.reset();
    }
}

#[test]
fn reset_clears_byte_budget_for_cached_chunks() {
    // Tight budget: only one chunk worth.
    let mut arena: Arena = Arena::builder()
        .chunk_size(8 * 1024)
        .chunk_cache_capacity(4)
        .byte_budget(8 * 1024)
        .build();

    let _ = arena.alloc(0_u8); // forces fresh chunk allocation
    arena.reset();
    // Should be able to allocate again from the cached chunk without
    // tripping the budget.
    let _ = arena.alloc(1_u8);
}

#[cfg(feature = "stats")]
#[test]
fn reset_works_with_pinned_chunks() {
    // Force chunk rotation by allocating multiple buffers that fill the
    // chunk.
    let mut arena: Arena = Arena::builder()
        .chunk_size(16 * 1024)
        .max_normal_alloc(4 * 1024)
        .chunk_cache_capacity(8)
        .build();
    let _ = arena.alloc([0_u8; 4000]);
    let _ = arena.alloc([0_u8; 4000]);
    let _ = arena.alloc([0_u8; 4000]);
    let _ = arena.alloc([0_u8; 4000]);
    let _ = arena.alloc([0_u8; 4000]);
    let chunks_before = arena.stats().chunks_allocated;
    assert!(chunks_before >= 2, "expected chunk rotation, got chunks_allocated={chunks_before}");

    arena.reset();
    let _ = arena.alloc(0_u64);
    assert_eq!(arena.stats().chunks_allocated, chunks_before, "no fresh chunk allocation expected");
}

#[test]
fn reset_works_after_alloc_style_refs_drop() {
    let mut arena = Arena::new();
    {
        let r: &mut u64 = arena.alloc(123);
        *r += 1;
    }
    arena.reset();
    let r = arena.alloc(1_u64);
    assert_eq!(*r, 1);
}

// ---------------------------------------------------------------------------
// Reset with outstanding refcounted smart pointers: the chunk *detaches*, the
// smart pointer keeps working, no destructor is skipped.
// ---------------------------------------------------------------------------

#[test]
fn reset_with_outstanding_arena_rc_keeps_handle_valid() {
    let mut arena = Arena::new();
    let r: Rc<u32> = arena.alloc_rc(42);
    arena.reset();
    // Smart pointer still works.
    assert_eq!(*r, 42);
    let r2 = r.clone();
    assert_eq!(*r2, 42);
    drop(r);
    drop(r2);
    // Arena still works after.
    let _ = arena.alloc_rc(99_u32);
}

#[test]
fn reset_with_outstanding_arena_arc_keeps_handle_valid() {
    let mut arena = Arena::new();
    let r: Arc<u32> = arena.alloc_arc(7);
    arena.reset();
    assert_eq!(*r, 7);
    drop(r);
    let _ = arena.alloc_arc(11_u32);
}

#[test]
fn reset_runs_destructor_when_last_handle_drops_post_reset() {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked;
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let mut arena = Arena::new();
    let r: Rc<Tracked> = arena.alloc_rc(Tracked);
    arena.reset();
    // Smart pointer outlived reset; destructor not yet run.
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    drop(r);
    // Now the chunk's last smart pointer dropped → destructor runs.
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
}

#[test]
fn reset_runs_chunk_residents_drops_only_once_with_handle_outstanding() {
    // Subtle: a single chunk hosts both an Arena::alloc-style value and
    // an ArenaRc smart pointer. The ArenaRc keeps the chunk alive past reset,
    // so the alloc-style value's destructor doesn't run at reset. It
    // runs when the last smart pointer drops.
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    struct Tracked;
    impl Drop for Tracked {
        fn drop(&mut self) {
            let _ = COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    COUNT.store(0, Ordering::SeqCst);
    let mut arena = Arena::new();
    let r: Rc<u8> = arena.alloc_rc(0);
    let _ = arena.alloc(Tracked); // pinned alloc-style; same chunk
    arena.reset();
    assert_eq!(
        COUNT.load(Ordering::SeqCst),
        0,
        "destructor must NOT run yet — chunk is detached but alive"
    );
    drop(r);
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
}

#[test]
fn reset_with_arena_arc_held_on_another_thread() {
    use std::sync::Arc as StdArc;
    use std::sync::Barrier;

    let mut arena = Arena::new();
    let r: Arc<u32> = arena.alloc_arc(99);

    let barrier = StdArc::new(Barrier::new(2));
    let b = StdArc::clone(&barrier);
    let h = std::thread::spawn(move || {
        let _ = b.wait();
        assert_eq!(*r, 99);
        let _ = b.wait();
        // Drop here.
    });
    let _ = barrier.wait();
    arena.reset();
    let _ = barrier.wait();
    h.join().unwrap();
    // Arena still usable.
    let _ = arena.alloc_arc(11_u32);
}

#[cfg(feature = "stats")]
#[test]
fn reset_returns_chunk_to_cache_when_handles_drop_after_reset() {
    let mut arena = Arena::builder().chunk_cache_capacity(4).build();
    let r: Rc<u64> = arena.alloc_rc(1);
    let chunks_before = arena.stats().chunks_allocated;
    arena.reset();
    drop(r); // last handle: chunk is reclaimed → cached.
    let _ = arena.alloc(0_u64);
    assert_eq!(
        arena.stats().chunks_allocated,
        chunks_before,
        "chunk should have rejoined the cache when handle dropped"
    );
}

#[test]
fn reset_handles_destructor_that_drops_other_smart_pointer() {
    // Regression: a destructor running during `Arena::reset` may drop
    // another smart pointer that triggers re-entrant chunk teardown,
    // potentially installing new chunks in `current_*` mid-drain.
    // The fixed-point loop must drain those re-entrantly-installed
    // chunks too, so all destructors run exactly once.
    static OUTER: AtomicUsize = AtomicUsize::new(0);
    static INNER: AtomicUsize = AtomicUsize::new(0);

    struct Outer<A: allocator_api2::alloc::Allocator + Clone + Send + Sync + 'static> {
        inner: Option<Arc<Inner, A>>,
    }
    struct Inner;
    impl Drop for Inner {
        fn drop(&mut self) {
            let _ = INNER.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl<A: allocator_api2::alloc::Allocator + Clone + Send + Sync + 'static> Drop for Outer<A> {
        fn drop(&mut self) {
            let _ = OUTER.fetch_add(1, Ordering::SeqCst);
            let _ = self.inner.take();
        }
    }

    OUTER.store(0, Ordering::SeqCst);
    INNER.store(0, Ordering::SeqCst);
    let mut arena: Arena = Arena::builder().chunk_size(8 * 1024).max_normal_alloc(4 * 1024).build();
    let inner = arena.alloc_arc(Inner);
    let _ = arena.alloc(Outer { inner: Some(inner) });
    // Force chunk rotation so Outer's chunk goes onto the pinned list.
    let _ = arena.alloc([0_u8; 1500]);
    let _ = arena.alloc([0_u8; 1500]);
    let _ = arena.alloc([0_u8; 1500]);
    arena.reset();
    assert_eq!(OUTER.load(Ordering::SeqCst), 1, "Outer::drop must run");
    assert_eq!(INNER.load(Ordering::SeqCst), 1, "Inner::drop must run");
    // Arena is still usable after reset.
    let r = arena.alloc_rc(42_u32);
    assert_eq!(*r, 42);
}
