use core::alloc::Layout;
use core::cell::Cell;
use core::fmt;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr::NonNull;

use allocator_api2::alloc::Allocator;

use crate::arena_inner::ArenaInner;
use crate::chunk_ref::ChunkRef;
use crate::chunk_sharing::ChunkSharing;
use crate::chunk_size_class::ChunkSizeClass;
use crate::constants::{CHUNK_ALIGN, checked_align_up, padded_header_size};
use crate::drop_entry::DropEntry;
use crate::entry_layout::checked_entry_value_offsets;
use crate::sync::{AtomicUsize, Ordering, fence};

/// Refcount overflow ceiling, mirroring `std::sync::Arc`'s `MAX_REFCOUNT`.
/// Crossing this is a sign of a malicious `mem::forget(clone())` loop and
/// must terminate the process — wrapping the count to zero would
/// produce a use-after-free when the next legitimate drop "frees"
/// still-referenced memory.
const MAX_REFCOUNT: usize = isize::MAX as usize;

/// Abort the process via a double-panic. Used when a refcount overflow
/// is observed; matches the soundness contract of `std::sync::Arc::clone`
/// (panic must not be catchable by the caller, since the count is now
/// in an unsafe state).
#[cold]
#[inline(never)]
#[expect(
    clippy::panic,
    reason = "double-panic abort path is the established pattern for refcount overflow on no_std"
)]
fn refcount_overflow() -> ! {
    struct Bomb;
    impl Drop for Bomb {
        fn drop(&mut self) {
            panic!("multitude: refcount overflow");
        }
    }
    let _bomb = Bomb;
    panic!("multitude: refcount overflow");
}

/// Assert that `count` has not crossed [`MAX_REFCOUNT`]. Aborts the
/// process (via [`refcount_overflow`]) if it has. Centralizes the check
/// so refcount-bump sites need only one line each.
#[inline]
fn assert_valid_refcount(count: usize) {
    if count > MAX_REFCOUNT {
        refcount_overflow();
    }
}

/// Refcount storage that is either a non-atomic [`Cell<usize>`] (for
/// `Local` chunks — saves the LLVM atomic intrinsic on every clone/drop)
/// or an [`AtomicUsize`] (for `Shared` chunks — required for cross-thread
/// safety). The active variant is determined by the chunk's `sharing` field
/// and never changes after the chunk is published.
///
/// On x86 a `Relaxed` atomic load/store compiles to the same instruction as
/// a plain load/store, but using `AtomicUsize` defeats some LLVM peephole
/// optimizations (CSE across atomic ops, hoisting through atomic
/// boundaries). The union avoids that for `Local` chunks, which are by
/// construction the only kind reachable from `Rc` / `RcStr`.
/// Aligned to a cache line (`CACHE_LINE_SIZE`) to prevent the
/// cross-thread atomic refcount on `Shared` chunks from false-sharing
/// the cache line that holds the owner-thread-only hot fields (`bump`,
/// `total_size`, `pinned`). The added padding (~120 B) is negligible
/// against the 64 KiB default chunk size (~0.2%).
#[repr(C, align(128))]
pub union RefCount {
    pub local: ManuallyDrop<Cell<usize>>,
    pub shared: ManuallyDrop<AtomicUsize>,
}

impl RefCount {
    #[inline]
    const fn new_local(initial: usize) -> Self {
        Self {
            local: ManuallyDrop::new(Cell::new(initial)),
        }
    }

    #[inline]
    #[cfg_attr(not(loom), expect(clippy::missing_const_for_fn, reason = "loom's AtomicUsize::new is not const"))]
    fn new_shared(initial: usize) -> Self {
        Self {
            shared: ManuallyDrop::new(AtomicUsize::new(initial)),
        }
    }

    #[inline]
    pub(crate) fn new_for(sharing: ChunkSharing, initial: usize) -> Self {
        match sharing {
            ChunkSharing::Local => Self::new_local(initial),
            ChunkSharing::Shared => Self::new_shared(initial),
        }
    }
}

/// Project a raw pointer to the `Cell<usize>` half of the refcount union.
///
/// # Safety
///
/// `chunk` must point to a live header whose `sharing` is `Local`.
#[inline]
unsafe fn local_rc<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>) -> *const Cell<usize> {
    // SAFETY: caller guarantees the union variant is `Local`.
    let rc_ptr: *const RefCount = unsafe { &raw const (*chunk.as_ptr()).ref_count };
    rc_ptr.cast::<Cell<usize>>()
}

/// Project a raw pointer to the `AtomicUsize` half of the refcount union.
///
/// # Safety
///
/// `chunk` must point to a live header whose `sharing` is `Shared`.
#[inline]
unsafe fn shared_rc<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>) -> *const AtomicUsize {
    // SAFETY: caller guarantees the union variant is `Shared`.
    let rc_ptr: *const RefCount = unsafe { &raw const (*chunk.as_ptr()).ref_count };
    rc_ptr.cast::<AtomicUsize>()
}

/// Per-chunk metadata, stored at the very start of every chunk allocation.
///
/// Hot fields touched on every owner-thread allocation (`bump`,
/// `total_size`, `pinned`, `size_class`, `sharing`) are clustered at
/// the front so they share the same cache line. The cross-thread
/// atomic `ref_count` is at the end and aligned to its own cache line
/// (via `RefCount`'s `#[repr(align(128))]`) to avoid false sharing with
/// the hot fields when other threads bump `Arc` refcounts on
/// `Shared`-flavored chunks.
#[repr(C)]
pub struct ChunkHeader<A: Allocator + Clone> {
    /// Current bump offset, measured from the chunk's base address.
    pub bump: Cell<usize>,
    /// Owner-thread-only counter of `Arc`s issued from this chunk while
    /// it sits in `Arena::current_shared`. Only meaningful for
    /// `Shared`-flavored chunks; left at 0 for `Local`. Used by the
    /// deferred-reconciliation refcount scheme:
    ///
    /// - On chunk install in `current_shared`, the atomic `ref_count`
    ///   is initialized to `LARGE_INITIAL_SHARED_REFCOUNT` (a very
    ///   large constant).
    /// - Each `alloc_arc` non-atomically increments `arcs_issued`. No
    ///   atomic touch on `ref_count`.
    /// - On chunk eviction from `current_shared`, the owner does one
    ///   `fetch_sub(LARGE - arcs_issued)` to reconcile: the unused
    ///   pre-payment is returned, leaving `ref_count` equal to the
    ///   number of outstanding live `Arc`s (clones minus drops).
    ///
    /// This eliminates the per-allocation LOCK RMW entirely on the
    /// `alloc_arc` hot path. Co-located with `bump` on the first cache
    /// line to keep the alloc-`Arc` hot path single-cache-line.
    pub arcs_issued: Cell<usize>,
    /// Total chunk size, in bytes.
    pub total_size: usize,
    /// Next-pointer when this chunk sits in the chunk cache or the
    /// pinned list; `None` otherwise. Storing the `NonNull` directly
    /// (rather than smuggling it through `bump` as an integer)
    /// preserves provenance under strict-provenance Miri.
    pub next_in_list: Cell<Option<NonNull<Self>>>,
    /// Head of the intrusive doubly-linked drop list.
    pub drop_head: Cell<Option<NonNull<DropEntry>>>,
    /// Back-pointer to the owning arena. Only safe to deref from owner thread.
    pub arena: NonNull<ArenaInner<A>>,
    /// `Local` vs `Shared` — determines which refcount variant is active.
    pub sharing: Cell<ChunkSharing>,
    /// `Normal` vs `Oversized` — determines cache eligibility.
    pub size_class: ChunkSizeClass,
    /// Set to `true` once any bump-style allocation targets this chunk.
    /// Pinned chunks stay alive until arena drop even after rotating out.
    pub pinned: Cell<bool>,
    /// Clone of the backing allocator. Each chunk can free itself independently.
    pub allocator: A,
    /// Live smart pointer references into this chunk (plus the arena's own
    /// reference while it's the current chunk). Dispatched via `sharing`.
    /// Aligned to its own cache line to avoid false sharing with the hot
    /// fields above on `Shared` chunks.
    pub ref_count: RefCount,
}

impl fmt::Debug for RefCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RefCount(<opaque union; active variant per chunk's sharing flag>)")
    }
}

impl<A: Allocator + Clone> fmt::Debug for ChunkHeader<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkHeader")
            .field("bump", &self.bump.get())
            .field("total_size", &self.total_size)
            .field("sharing", &self.sharing.get())
            .field("size_class", &self.size_class)
            .field("pinned", &self.pinned.get())
            .finish_non_exhaustive()
    }
}

impl<A: Allocator + Clone> ChunkHeader<A> {
    #[inline]
    #[must_use]
    pub const fn header_padded_size() -> usize {
        padded_header_size(size_of::<Self>())
    }

    /// Layout for a normal chunk. Always aligned to [`CHUNK_ALIGN`]
    /// for the address-mask trick.
    #[inline]
    #[must_use]
    pub const fn normal_layout(chunk_size: usize) -> Layout {
        // SAFETY: CHUNK_ALIGN is a power of 2; builder validates chunk_size ≤ CHUNK_ALIGN.
        unsafe { Layout::from_size_align_unchecked(chunk_size, CHUNK_ALIGN) }
    }

    /// Layout for an oversized chunk holding one allocation of `payload`,
    /// optionally preceded by a `DropEntry`. Returns `None` on overflow
    /// or if `payload.align() >= CHUNK_ALIGN`.
    #[must_use]
    #[inline]
    pub fn oversized_layout(payload: Layout, has_drop: bool) -> Option<Layout> {
        if payload.align() >= CHUNK_ALIGN {
            return None;
        }
        let header = Self::header_padded_size();
        let total = if has_drop {
            let (_entry_addr, _value_addr, end) = checked_entry_value_offsets(header, payload)?;
            end
        } else {
            let value_offset = checked_align_up(header, payload.align())?;
            value_offset.checked_add(payload.size())?
        };
        Layout::from_size_align(total, CHUNK_ALIGN).ok()
    }

    /// Bump-allocate `layout` bytes from this chunk. Caller must have
    /// verified the request fits (e.g. via worst-case sizing).
    ///
    /// # Safety
    ///
    /// - `layout.align() <= CHUNK_ALIGN`
    /// - `align_up(h.bump.get(), layout.align()) + layout.size()` fits and is `<= h.total_size`
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub unsafe fn alloc_unchecked(chunk: NonNull<Self>, layout: Layout) -> NonNull<u8> {
        // SAFETY: chunk is live (caller's contract).
        let h = unsafe { chunk.as_ref() };
        let cur = h.bump.get();
        let align = layout.align();
        let aligned = (cur + (align - 1)) & !(align - 1);
        let end = aligned + layout.size();
        h.bump.set(end);
        let base = chunk.as_ptr().cast::<u8>();
        // SAFETY: end <= total_size (caller's contract).
        unsafe { NonNull::new_unchecked(base.add(aligned)) }
    }

    /// Bump-allocate a `DropEntry` plus a `T` value. Caller must have
    /// verified the request fits.
    ///
    /// # Safety
    ///
    /// Same as [`Self::alloc_unchecked`], plus worst-case sizing must
    /// include `DropEntry`'s size and alignment.
    #[inline]
    pub unsafe fn alloc_with_drop_entry_unchecked<T>(chunk: NonNull<Self>) -> (NonNull<DropEntry>, NonNull<T>) {
        // SAFETY: chunk is live; worst-case sizing covers the layout.
        let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
        // SAFETY: caller's worst-case sizing covers the entry + value layout.
        let (entry, value) = unsafe { chunk_ref.alloc_entry_value_slot_unchecked(Layout::new::<T>()) };
        (entry, value.cast::<T>())
    }

    /// Increment the chunk refcount. Takes `NonNull<Self>` to avoid
    /// strong protector issues under Stacked/Tree Borrows.
    ///
    /// # Safety
    ///
    /// `chunk` must point to a live chunk header.
    #[cfg(test)]
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "sharing read + flavored inc share one live-chunk invariant"
    )]
    pub unsafe fn inc_ref(chunk: NonNull<Self>) {
        // SAFETY: chunk is live; sharing read picks the active variant.
        unsafe {
            match (*chunk.as_ptr()).sharing.get() {
                ChunkSharing::Local => Self::inc_ref_local(chunk),
                ChunkSharing::Shared => Self::inc_ref_shared(chunk),
            }
        }
    }

    /// `Local`-flavor specialization of [`Self::inc_ref`]. Skips the
    /// runtime sharing dispatch.
    ///
    /// # Safety
    ///
    /// `chunk` must point to a live `Local`-flavored chunk header.
    #[inline]
    #[expect(clippy::multiple_unsafe_ops_per_block, reason = "refcount-cell access shares one safety invariant")]
    pub unsafe fn inc_ref_local(chunk: NonNull<Self>) {
        // SAFETY: caller's contract.
        unsafe {
            let cell = &*local_rc::<A>(chunk);
            let v = cell.get();
            // No `assert_valid_refcount` on the Local path: an
            // overflow would require ~2^63 simultaneous `Rc<T>`
            // clones of a single value, which is unreachable in any
            // realistic program. The Shared path keeps the check
            // (cross-thread `mem::forget(clone())` loops are
            // theoretically reachable).
            cell.set(v + 1);
        }
    }

    /// `Shared`-flavor specialization of [`Self::inc_ref`]. Skips the
    /// runtime sharing dispatch.
    ///
    /// # Safety
    ///
    /// `chunk` must point to a live `Shared`-flavored chunk header.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "atomic deref + fetch_add share one safety invariant"
    )]
    pub unsafe fn inc_ref_shared(chunk: NonNull<Self>) {
        // SAFETY: caller's contract.
        unsafe {
            let prev = (*shared_rc::<A>(chunk)).fetch_add(1, Ordering::Relaxed);
            assert_valid_refcount(prev);
        }
    }

    /// Increment the atomic refcount of a `Shared`-flavored chunk by
    /// `n`. Used by the deferred-reconciliation path when
    /// `arcs_issued` approaches the LARGE pre-payment, to fold the
    /// non-atomic count into the atomic refcount and reset
    /// `arcs_issued`.
    ///
    /// # Safety
    ///
    /// `chunk` must point to a live `Shared`-flavored chunk header.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "atomic deref + fetch_add share one safety invariant"
    )]
    pub unsafe fn add_ref_shared(chunk: NonNull<Self>, n: usize) {
        // SAFETY: caller's contract.
        unsafe {
            let prev = (*shared_rc::<A>(chunk)).fetch_add(n, Ordering::Relaxed);
            assert_valid_refcount(prev);
        }
    }

    /// Decrement the chunk refcount. Returns `true` iff the count
    /// reached zero (caller should run teardown). Takes `NonNull<Self>`
    /// to avoid strong protector issues.
    ///
    /// # Safety
    ///
    /// `chunk` must point to a live chunk header.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "sharing read + cell access share the live-chunk safety invariant"
    )]
    pub unsafe fn dec_ref(chunk: NonNull<Self>) -> bool {
        // SAFETY: chunk is live; sharing read picks the active variant.
        let (was, sharing) = unsafe {
            let sharing = (*chunk.as_ptr()).sharing.get();
            let was = match sharing {
                ChunkSharing::Local => {
                    let cell = &*local_rc::<A>(chunk);
                    let v = cell.get();
                    debug_assert!(v >= 1, "chunk refcount underflow in dec_ref");
                    cell.set(v - 1);
                    v
                }
                ChunkSharing::Shared => (*shared_rc::<A>(chunk)).fetch_sub(1, Ordering::Release),
            };
            (was, sharing)
        };
        if was == 1 {
            if matches!(sharing, ChunkSharing::Shared) {
                fence(Ordering::Acquire);
            }
            true
        } else {
            false
        }
    }

    /// Push `entry` onto the front of the doubly-linked drop list.
    ///
    /// # Safety
    ///
    /// `entry` must be a freshly-allocated, uninitialized `DropEntry`
    /// inside this chunk.
    #[inline]
    pub unsafe fn link_drop_entry(&self, entry: NonNull<DropEntry>, drop_fn: unsafe fn(*mut DropEntry), slice_len: usize) {
        let old_head = self.drop_head.get();
        // SAFETY: entry slot is writable (caller's contract).
        unsafe {
            entry.as_ptr().write(DropEntry {
                drop_fn,
                prev: None,
                next: old_head,
                slice_len,
            });
        }
        if let Some(old) = old_head {
            // SAFETY: old_head is a valid entry in this chunk.
            unsafe {
                (*old.as_ptr()).prev = Some(entry);
            }
        }
        self.drop_head.set(Some(entry));
    }

    /// Unlink `entry` from the drop list.
    ///
    /// # Safety
    ///
    /// `entry` must currently be in this chunk's drop list.
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    #[inline]
    pub unsafe fn unlink_drop_entry(&self, entry: NonNull<DropEntry>) {
        // SAFETY: entry is in our list (caller's contract).
        unsafe {
            let prev = (*entry.as_ptr()).prev;
            let next = (*entry.as_ptr()).next;
            if let Some(p) = prev {
                (*p.as_ptr()).next = next;
            } else {
                self.drop_head.set(next);
            }
            if let Some(n) = next {
                (*n.as_ptr()).prev = prev;
            }
        }
    }

    #[inline]
    #[must_use]
    pub const fn data_start_offset() -> usize {
        Self::header_padded_size()
    }

    /// Reset bump cursor, drop list, pinned flag, and `arcs_issued`.
    /// Used when pulling a chunk from the cache.
    #[inline]
    pub fn reset(&self) {
        self.bump.set(Self::data_start_offset());
        self.drop_head.set(None);
        self.pinned.set(false);
        self.arcs_issued.set(0);
    }

    /// Set the refcount to `n`. Used during chunk repurposing.
    ///
    /// # Safety
    ///
    /// `chunk` must point to a live header; caller has exclusive access.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "sharing read + cell write share the exclusive-access safety invariant"
    )]
    pub unsafe fn set_ref_count(chunk: NonNull<Self>, n: usize) {
        // SAFETY: chunk is live; sharing read picks the active variant.
        unsafe {
            match (*chunk.as_ptr()).sharing.get() {
                ChunkSharing::Local => (*local_rc::<A>(chunk)).set(n),
                ChunkSharing::Shared => (*shared_rc::<A>(chunk)).store(n, Ordering::Relaxed),
            }
        }
    }
    /// Reinitialize the refcount to a fresh `RefCount` of the given
    /// sharing flavor and starting value. Used during chunk repurposing
    /// (cache pop) when the previous chunk's flavor differs from the
    /// caller's needs.
    ///
    /// Writes a brand-new `RefCount` over the existing one via a raw
    /// pointer to avoid forming any reference into the union (which is
    /// itself not wrapped in `UnsafeCell`).
    ///
    /// # Safety
    ///
    /// - `chunk` must point to a live header.
    /// - Caller must hold exclusive access to `chunk` — no other live
    ///   reference (shared or unique, on this thread or any other) into
    ///   the chunk's `ref_count` field.
    /// - The previous `RefCount` variant must already be logically dead
    ///   (no outstanding `Cell`/`AtomicUsize` reads or writes against
    ///   it). Cache-popped chunks satisfy this because their refcount
    ///   reached zero before being cached.
    #[inline]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
    )]
    pub unsafe fn reinit_refcount(chunk: NonNull<Self>, sharing: ChunkSharing, value: usize) {
        // SAFETY: caller has exclusive access.
        unsafe {
            let rc_ptr: *mut RefCount = &raw mut (*chunk.as_ptr()).ref_count;
            rc_ptr.write(RefCount::new_for(sharing, value));
        }
    }
}

/// Recover the [`ChunkHeader<A>`] address for any pointer allocated
/// from a chunk by masking off the low bits.
///
/// # Safety
///
/// `ptr` must be a pointer from an `Arena<A>` allocation.
#[inline]
#[must_use]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
pub unsafe fn header_for<T: ?Sized, A: Allocator + Clone>(ptr: NonNull<T>) -> NonNull<ChunkHeader<A>> {
    let raw = ptr.as_ptr().cast::<u8>();
    let offset_within_chunk = (raw as usize) & (CHUNK_ALIGN - 1);
    // SAFETY: offset_within_chunk is raw - base; subtracting returns to chunk base.
    unsafe {
        let header = raw.byte_sub(offset_within_chunk).cast::<ChunkHeader<A>>();
        NonNull::new_unchecked(header)
    }
}

/// Initialize a chunk's header in place.
///
/// The chunk's `ref_count` starts at `initial_refcount`:
/// - `1` for normal chunks (arena's transient hold)
/// - `0` for oversized chunks (no arena hold)
///
/// # Safety
///
/// `chunk_ptr` must point to an allocation of at least `total_size` bytes,
/// aligned to `CHUNK_ALIGN`. `arena` must outlive the chunk's owner-side usage.
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
pub unsafe fn init_chunk<A: Allocator + Clone>(
    chunk_ptr: NonNull<u8>,
    total_size: usize,
    sharing: ChunkSharing,
    size_class: ChunkSizeClass,
    arena: NonNull<ArenaInner<A>>,
    allocator: A,
    initial_refcount: usize,
) -> NonNull<ChunkHeader<A>> {
    let header_ptr = chunk_ptr.cast::<MaybeUninit<ChunkHeader<A>>>();
    // SAFETY: chunk allocation is large enough.
    unsafe {
        header_ptr.as_ptr().write(MaybeUninit::new(ChunkHeader {
            ref_count: RefCount::new_for(sharing, initial_refcount),
            bump: Cell::new(ChunkHeader::<A>::data_start_offset()),
            arcs_issued: Cell::new(0),
            next_in_list: Cell::new(None),
            drop_head: Cell::new(None),
            total_size,
            sharing: Cell::new(sharing),
            size_class,
            pinned: Cell::new(false),
            arena,
            allocator,
        }));
        NonNull::new_unchecked(header_ptr.as_ptr().cast::<ChunkHeader<A>>())
    }
}

/// Initialize a chunk header in place as an inert "sentinel" header.
///
/// Sentinels live embedded in [`ArenaInner`](crate::arena_inner::ArenaInner)
/// rather than in a heap-allocated chunk. They carry `total_size = 0`,
/// `bump = 1`, and `pinned = false`, so the alloc fast path's natural
/// fit-check (`end <= total_size`) returns "doesn't fit" for *every*
/// layout — including ZSTs, where `end = aligned + 0` would otherwise
/// be `0` and tie the comparison. (`bump = 1` raises `aligned`/`end` to
/// at least `1`, breaking the tie without a special-case branch.) This
/// lets a [`ChunkSlot`](crate::owned_chunk::ChunkSlot) treat the sentinel
/// as a stand-in for "empty slot" without an `Option` discriminant on
/// the hot path. Sentinels are never reachable from any smart pointer
/// (the fit-check fails before any allocation can land in one), so
/// their refcount and teardown machinery is never exercised — the
/// `ref_count` initial value below is therefore irrelevant.
///
/// # Safety
///
/// `slot` must point at writable, sentinel-header-sized memory whose
/// alignment satisfies `ChunkHeader<A>`'s requirements. The memory must
/// outlive every `ChunkSlot<A>` that points at this sentinel.
pub unsafe fn init_sentinel_header<A: Allocator + Clone>(
    slot: NonNull<ChunkHeader<A>>,
    arena: NonNull<ArenaInner<A>>,
    sharing: ChunkSharing,
    allocator: A,
) {
    // SAFETY: caller's contract.
    unsafe {
        slot.as_ptr().write(ChunkHeader {
            ref_count: RefCount::new_for(sharing, 0),
            // `bump = 1` ensures any fit-check (`end <= total_size`)
            // fails against the sentinel, including ZSTs whose
            // `end = aligned + 0` would otherwise tie.
            bump: Cell::new(1),
            arcs_issued: Cell::new(0),
            next_in_list: Cell::new(None),
            drop_head: Cell::new(None),
            total_size: 0,
            sharing: Cell::new(sharing),
            size_class: ChunkSizeClass::Normal,
            pinned: Cell::new(false),
            arena,
            allocator,
        });
    }
}

/// Decrement a chunk's refcount and tear it down if it reached zero.
///
/// # Safety
///
/// `chunk` must point to a live chunk; caller holds at least one refcount.
/// `on_owner_thread` must be true for `Local` chunks.
#[inline]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "header recovery + dec_ref + teardown share one safety invariant"
)]
pub unsafe fn release_chunk_ref<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>, on_owner_thread: bool) {
    // SAFETY: caller holds a refcount.
    unsafe {
        if ChunkHeader::dec_ref(chunk) {
            teardown_chunk(chunk, on_owner_thread);
        }
    }
}

/// Bulk variant of [`release_chunk_ref`]: release `n` refcount units in
/// a single atomic op (or a single `Cell` write for `Local`). Used by
/// [`OwnedChunk`](crate::owned_chunk::OwnedChunk)'s `Drop` impl on
/// `Shared` chunks, where the hold to release is `LARGE - arcs_issued`
/// (the unused portion of the deferred-reconciliation pre-payment).
///
/// # Safety
///
/// `chunk` must point to a live chunk; caller holds at least `n`
/// refcount units. `on_owner_thread` must be true for `Local` chunks.
#[inline]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "sharing read + atomic/cell op + cold teardown share one safety invariant"
)]
pub unsafe fn release_chunk_ref_n<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>, n: usize, on_owner_thread: bool) {
    debug_assert!(n >= 1, "release_chunk_ref_n called with n=0");
    // SAFETY: chunk is live; sharing read picks the active variant.
    let (was, sharing) = unsafe {
        let sharing = (*chunk.as_ptr()).sharing.get();
        let was = match sharing {
            ChunkSharing::Local => {
                let cell = &*local_rc::<A>(chunk);
                let v = cell.get();
                debug_assert!(v >= n, "chunk refcount underflow in release_chunk_ref_n");
                cell.set(v - n);
                v
            }
            ChunkSharing::Shared => (*shared_rc::<A>(chunk)).fetch_sub(n, Ordering::Release),
        };
        (was, sharing)
    };
    if was == n {
        if matches!(sharing, ChunkSharing::Shared) {
            fence(Ordering::Acquire);
        }
        // SAFETY: refcount reached zero; caller's contract on n.
        unsafe { teardown_chunk::<A>(chunk, on_owner_thread) };
    }
}

/// `Local`-flavor specialization for releasing a chunk refcount.
/// Skips the `sharing` dispatch and inlines a `Cell<usize>` decrement.
///
/// # Safety
///
/// `value` must be a pointer from an `Arena<A>` allocation; caller holds
/// a refcount; the chunk MUST be `Local`-flavored. Owner-thread only.
#[inline]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "header recovery + cell decrement + cold teardown share one safety invariant"
)]
pub unsafe fn release_chunk_ref_local<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
    // SAFETY: caller's contract.
    unsafe {
        let chunk: NonNull<ChunkHeader<A>> = header_for(value);
        release_chunk_header_local(chunk);
    }
}

/// Release a refcount on a `Local`-flavored chunk via the chunk pointer
/// directly (no `header_for` masking). Used by panic-release guards in
/// `arena.rs` that operate on a chunk pointer they already have.
///
/// # Safety
///
/// `chunk` must point to a live `Local`-flavored chunk on the owner
/// thread; caller holds a refcount.
#[inline]
#[expect(
    clippy::redundant_pub_crate,
    reason = "shared with arena module's RefcountReleaseGuard / SliceInitFailGuard panic releases"
)]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "cell decrement + cold teardown share one safety invariant"
)]
pub(crate) unsafe fn release_chunk_header_local<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>) {
    // SAFETY: caller's contract.
    unsafe {
        let cell = &*local_rc::<A>(chunk);
        let v = cell.get();
        debug_assert!(v >= 1, "chunk refcount underflow in release_chunk_header_local");
        cell.set(v - 1);
        if v == 1 {
            teardown_chunk::<A>(chunk, true);
        }
    }
}

/// `Shared`-flavor specialization for releasing a chunk refcount.
/// Skips the `sharing` dispatch and inlines a Release `fetch_sub` + Acquire fence.
///
/// # Safety
///
/// `value` must be a pointer from an `Arena<A>` allocation; caller holds
/// a refcount; the chunk MUST be `Shared`-flavored.
#[inline]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "header recovery + atomic fetch_sub + cold teardown share one safety invariant"
)]
pub unsafe fn release_chunk_ref_shared<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
    // SAFETY: caller's contract.
    unsafe {
        let chunk: NonNull<ChunkHeader<A>> = header_for(value);
        let was = (*shared_rc::<A>(chunk)).fetch_sub(1, Ordering::Release);
        if was == 1 {
            fence(Ordering::Acquire);
            teardown_chunk::<A>(chunk, false);
        }
    }
}

/// `Local`-flavor specialization for incrementing a chunk refcount.
/// Skips the `sharing` dispatch and inlines a `Cell<usize>` increment.
///
/// # Safety
///
/// `value` must be a pointer from an `Arena<A>` allocation; caller holds
/// a refcount; the chunk MUST be `Local`-flavored.
#[inline]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "header recovery + cell increment share one safety invariant"
)]
pub unsafe fn inc_chunk_ref_local<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
    // SAFETY: caller's contract.
    unsafe {
        let chunk: NonNull<ChunkHeader<A>> = header_for(value);
        let cell = &*local_rc::<A>(chunk);
        let v = cell.get();
        // No overflow check on the Local path; see `ChunkHeader::inc_ref_local`.
        cell.set(v + 1);
    }
}

/// `Shared`-flavor specialization for incrementing a chunk refcount.
/// Skips the `sharing` dispatch and inlines a Relaxed `fetch_add`.
///
/// # Safety
///
/// `value` must be a pointer from an `Arena<A>` allocation; caller holds
/// a refcount; the chunk MUST be `Shared`-flavored.
#[inline]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "header recovery + atomic increment share one safety invariant"
)]
pub unsafe fn inc_chunk_ref_shared<T: ?Sized, A: Allocator + Clone>(value: NonNull<T>) {
    // SAFETY: caller's contract.
    unsafe {
        let chunk: NonNull<ChunkHeader<A>> = header_for(value);
        let prev = (*shared_rc::<A>(chunk)).fetch_add(1, Ordering::Relaxed);
        assert_valid_refcount(prev);
    }
}

/// Tear down a chunk: walk its drop list, then cache or free the memory.
///
/// # Safety
///
/// Caller must have observed `ref_count == 0` (after appropriate fence
/// on Shared chunks). The chunk must not be touched after this returns.
#[cold]
pub unsafe fn teardown_chunk<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>, on_owner_thread: bool) {
    // SAFETY: we own the chunk exclusively (refcount==0).
    let chunk_ref = unsafe { ChunkRef::<A>::new(chunk) };
    let mut node = chunk_ref.drop_head();
    while let Some(entry) = node {
        // SAFETY: entry was linked via link_drop_entry.
        let next = unsafe { (*entry.as_ptr()).next };
        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            reason = "deref + indirect call share the entry-validity invariant"
        )]
        // SAFETY: drop_fn applies to the co-allocated value.
        unsafe {
            ((*entry.as_ptr()).drop_fn)(entry.as_ptr());
        }
        node = next;
    }

    let size_class = chunk_ref.size_class();
    let chunk_sharing = chunk_ref.sharing();
    let inner_ptr = chunk_ref.arena();
    let can_cache_class = matches!(size_class, ChunkSizeClass::Normal) && (matches!(chunk_sharing, ChunkSharing::Local) || on_owner_thread);

    if can_cache_class {
        // SAFETY: chunk's back-pointer was set at init_chunk.
        let arena = unsafe { inner_ptr.as_ref() };
        if !arena.arena_dropped() {
            // SAFETY: chunk is exclusively owned at refcount=0.
            let retired = unsafe { crate::owned_chunk::RetiredChunk::from_raw(chunk) };
            match arena.try_push_to_cache(retired) {
                Ok(()) => return,
                Err(retired) => {
                    drop(retired);
                    return;
                }
            }
        }
    }

    // SAFETY: chunk is exclusively owned; drop list walked.
    unsafe { free_chunk(chunk) };
}

/// Free a chunk's backing memory. Decrements the parent `ArenaInner`'s
/// outstanding-chunks counter; if this was the last chunk after arena drop,
/// also frees the `ArenaInner` storage.
///
/// # Safety
///
/// Caller owns the chunk exclusively (refcount==0); drop list already walked.
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "unsafe operations within form a single tightly-coupled sequence with a unified safety invariant documented at the block level"
)]
#[cold]
pub unsafe fn free_chunk<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>) {
    // SAFETY: caller owns chunk exclusively.
    unsafe {
        let header_ptr = chunk.as_ptr();
        let total_size = (*header_ptr).total_size;
        let inner_ptr: NonNull<ArenaInner<A>> = (*header_ptr).arena;
        let allocator: A = core::ptr::read(&raw const (*header_ptr).allocator);
        let layout = Layout::from_size_align_unchecked(total_size, CHUNK_ALIGN);
        allocator.deallocate(chunk.cast::<u8>(), layout);

        let last_reclaimer = ArenaInner::register_chunk_freed(inner_ptr);
        if last_reclaimer {
            ArenaInner::free_storage(inner_ptr);
        }
    }
}

/// Revive a cached chunk: reset state, reinitialize refcount, and set
/// sharing flavor.
///
/// # Safety
///
/// Caller has just popped this chunk from cache and holds it exclusively.
pub unsafe fn revive_cached_chunk<A: Allocator + Clone>(chunk: NonNull<ChunkHeader<A>>, sharing: ChunkSharing, initial_refcount: usize) {
    // SAFETY: caller has exclusive access.
    let h = unsafe { chunk.as_ref() };
    h.reset();
    // SAFETY: exclusively owned; reinit before set to avoid mismatch.
    unsafe { ChunkHeader::reinit_refcount(chunk, sharing, initial_refcount) };
    h.sharing.set(sharing);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arc, Arena};
    use allocator_api2::alloc::Global;

    #[test]
    fn refcount_debug_is_opaque() {
        let rc = RefCount::new_local(5);
        let s = alloc::format!("{rc:?}");
        assert!(s.contains("RefCount"));
        assert!(s.contains("opaque"));
    }

    #[test]
    #[expect(clippy::multiple_unsafe_ops_per_block, reason = "tightly-coupled pointer-recovery sequence")]
    #[expect(clippy::undocumented_unsafe_blocks, reason = "test code")]
    fn chunk_header_debug_includes_fields() {
        use crate::Box;
        let arena = Arena::<Global>::new();
        let v: Box<u32> = arena.alloc_box(0_u32);
        let chunk: NonNull<ChunkHeader<Global>> = unsafe { header_for(NonNull::new_unchecked(Box::as_ptr(&v).cast_mut())) };
        let s = alloc::format!("{:?}", unsafe { chunk.as_ref() });
        assert!(s.contains("ChunkHeader"));
        assert!(s.contains("bump"));
        assert!(s.contains("total_size"));
    }

    #[test]
    #[cfg_attr(miri, ignore = "test deliberately leaks the arena to avoid Arena::drop underflow")]
    #[expect(clippy::multiple_unsafe_ops_per_block, reason = "tightly-coupled refcount-manipulation sequence")]
    fn set_ref_count_shared_branch() {
        let arena = Arena::<Global>::new();
        let h: Arc<u32> = arena.alloc_arc(42);
        // SAFETY: h is an in-arena allocation.
        let chunk: NonNull<ChunkHeader<Global>> = unsafe { header_for(NonNull::new_unchecked(Arc::as_ptr(&h).cast_mut())) };
        // SAFETY: chunk alive; inc/set pair preserves refcount.
        unsafe {
            ChunkHeader::inc_ref(chunk);
            ChunkHeader::set_ref_count(chunk, 2);
        }
        assert_eq!(*h, 42);
        // The arena's `Drop` would otherwise try to release its
        // outstanding `current_shared` credit batch and underflow this
        // chunk's freshly clamped refcount. The test deliberately
        // corrupts refcount accounting to exercise the Shared branch
        // of `set_ref_count`, so leak the arena rather than running
        // its drop.
        core::mem::forget(h);
        core::mem::forget(arena);
    }

    #[test]
    fn oversized_layout_rejects_align_at_or_above_chunk_align() {
        // align > CHUNK_ALIGN is rejected.
        let big_align = CHUNK_ALIGN * 2;
        let payload = Layout::from_size_align(8, big_align).expect("valid layout");
        assert!(ChunkHeader::<Global>::oversized_layout(payload, false).is_none());
        assert!(ChunkHeader::<Global>::oversized_layout(payload, true).is_none());

        // align == CHUNK_ALIGN is also rejected: with the chunk base
        // aligned to CHUNK_ALIGN and the value placed at the next
        // CHUNK_ALIGN-aligned offset (= CHUNK_ALIGN), `header_for`'s
        // address-mask trick would round back to the value pointer
        // itself rather than the chunk base.
        let exactly_chunk_align = Layout::from_size_align(8, CHUNK_ALIGN).expect("valid layout");
        assert!(ChunkHeader::<Global>::oversized_layout(exactly_chunk_align, false).is_none());
        assert!(ChunkHeader::<Global>::oversized_layout(exactly_chunk_align, true).is_none());

        // align < CHUNK_ALIGN is accepted.
        let half_chunk_align = Layout::from_size_align(8, CHUNK_ALIGN / 2).expect("valid layout");
        assert!(ChunkHeader::<Global>::oversized_layout(half_chunk_align, false).is_some());
    }

    #[test]
    fn oversized_layout_overflow_returns_none() {
        // `checked_entry_value_offsets` (has_drop=true) and `checked_add`
        // (no-drop) return None when adding header to size overflows. Use a
        // size near isize::MAX so Layout::from_size_align_unchecked's debug
        // checks pass while still triggering the overflow inside.
        let isize_max_usize = isize::MAX.unsigned_abs();
        // SAFETY: align=1 is a power of two; size == isize::MAX satisfies
        // `size + (align-1) <= isize::MAX`, the precondition for
        // from_size_align_unchecked.
        let payload = unsafe { Layout::from_size_align_unchecked(isize_max_usize, 1) };
        assert!(ChunkHeader::<Global>::oversized_layout(payload, true).is_none());
        assert!(ChunkHeader::<Global>::oversized_layout(payload, false).is_none());
    }
}
