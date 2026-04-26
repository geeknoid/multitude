//! Shared test helpers. Each integration test file includes this via
//! `mod common;` and uses items as `common::*`.

#![allow(dead_code, reason = "shared between multiple test binaries; some helpers may be unused per-file")]

use core::alloc::Layout;
use core::cell::Cell;
use core::hash::{Hash, Hasher};
use core::ptr::NonNull;
use std::collections::hash_map::DefaultHasher;

use allocator_api2::alloc::{AllocError, Allocator, Global};

pub fn hash_of<T: Hash>(v: &T) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// Allocator that fails its `allocate` calls after the first `n`
/// successes. Used to drive the `try_alloc*` / `panic_alloc` Err
/// branches that aren't reachable with the global allocator under
/// normal test conditions.
///
/// Cloned references share the same underlying counter (interior
/// mutability via `Rc<Cell<usize>>` on the test side; here we wrap
/// the count in a heap pointer for `Clone` to copy).
#[derive(Clone)]
pub struct FailingAllocator {
    remaining: std::rc::Rc<Cell<usize>>,
}

impl FailingAllocator {
    pub fn new(allow_n_allocs: usize) -> Self {
        Self {
            remaining: std::rc::Rc::new(Cell::new(allow_n_allocs)),
        }
    }
}

// SAFETY: forwards to Global on success; fails atomically on exhaustion.
unsafe impl Allocator for FailingAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let r = self.remaining.get();
        if r == 0 {
            return Err(AllocError);
        }
        self.remaining.set(r - 1);
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded — caller's contract.
        unsafe { Global.deallocate(ptr, layout) };
    }
}

/// Allocator that tracks live allocations (count and bytes) so tests
/// can detect leaks across an `Arena`'s lifetime. Tracks `allocate`
/// vs. `deallocate` and `grow`/`shrink` deltas.
#[derive(Clone)]
pub struct TrackingAllocator {
    live_chunks: std::rc::Rc<Cell<isize>>,
    live_bytes: std::rc::Rc<Cell<isize>>,
}

impl TrackingAllocator {
    pub fn new() -> Self {
        Self {
            live_chunks: std::rc::Rc::new(Cell::new(0)),
            live_bytes: std::rc::Rc::new(Cell::new(0)),
        }
    }

    pub fn live_chunks(&self) -> isize {
        self.live_chunks.get()
    }

    pub fn live_bytes(&self) -> isize {
        self.live_bytes.get()
    }
}

// SAFETY: forwards to Global; counters are interior-mutable bookkeeping only.
unsafe impl Allocator for TrackingAllocator {
    #[expect(clippy::cast_possible_wrap, reason = "test allocator: chunk sizes fit in isize")]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let p = Global.allocate(layout)?;
        self.live_chunks.set(self.live_chunks.get() + 1);
        self.live_bytes.set(self.live_bytes.get() + layout.size() as isize);
        Ok(p)
    }

    #[expect(clippy::cast_possible_wrap, reason = "test allocator: chunk sizes fit in isize")]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded — caller's contract.
        unsafe { Global.deallocate(ptr, layout) };
        self.live_chunks.set(self.live_chunks.get() - 1);
        self.live_bytes.set(self.live_bytes.get() - layout.size() as isize);
    }
}
/// constructor families (which require `A: Send + Sync`).
#[derive(Clone)]
pub struct SendFailingAllocator {
    remaining: std::sync::Arc<core::sync::atomic::AtomicUsize>,
}

impl SendFailingAllocator {
    pub fn new(allow_n_allocs: usize) -> Self {
        Self {
            remaining: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(allow_n_allocs)),
        }
    }
}

// SAFETY: forwards to Global on success; fails atomically on exhaustion.
unsafe impl Allocator for SendFailingAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
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

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded — caller's contract.
        unsafe { Global.deallocate(ptr, layout) };
    }
}
