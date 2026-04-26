//! **A chunked, reference-counted bump allocation arena for Rust.**
//!
//! ## Bump allocation
//!
//! Bump allocation is a fast, restricted approach to memory management.
//! The allocator owns one or more contiguous chunks of memory and a cursor
//! ("bump pointer") inside each. To allocate a value, it checks that the
//! chunk has enough room, then advances the cursor by the value's size.
//! That's it — no metadata to maintain, no free-list walks, no
//! fragmentation. The cost is a handful of cycles compared to dozens or
//! hundreds for a general-purpose allocator like jemalloc or the system
//! allocator.
//!
//! The trade-off: a pure bump allocator can't free individual values.
//! Memory is reclaimed in bulk — typically by dropping the entire arena
//! and reusing or releasing all of its chunks at once.
//!
//! Bump allocation shines for **phase-oriented workloads**: groups of
//! related allocations that live and die together. Compilers, parsers,
//! request handlers, and graph builders are classic fits.
//!
//! ## What `harena` adds
//!
//! `harena` is a bump allocator that addresses two of the usual
//! restrictions:
//!
//! 1. **Per-chunk reclamation.** Each chunk is reference-counted. As soon
//!    as nothing references a chunk, its memory is returned to the
//!    backing allocator (or held in a small free-list cache for reuse).
//!    You don't have to wait for the whole arena to die.
//! 2. **Handles outlive the arena.** The smart-pointer handles
//!    [`ArenaRc`] and [`ArenaArc`] keep their owning chunk alive on their
//!    own. You can drop the arena and keep using the values you've
//!    allocated.
//!
//! Plus the things you'd expect from a modern arena:
//!
//! - **Automatic per-object `Drop`** with zero overhead for trivial
//!   types. `Drop` runs exactly once at chunk teardown.
//! - **Cross-thread sharing** of individual values via [`ArenaArc`]
//!   (atomic refcount, opt-in).
//! - **Owned, mutable single handle** ([`ArenaBox`]) when you want
//!   `Drop` on handle drop and `&mut T` access.
//! - **Single-pointer string handles** ([`ArenaRcStr`]) — 8 bytes on
//!   64-bit, with the length stored inline in the chunk.
//! - **`Allocator` trait integration**: `&Arena<A>` implements
//!   `allocator-api2`'s `Allocator`, so collections like
//!   `hashbrown::HashMap` can store their internal buffers in the arena.
//! - **Pluggable backing allocator** via [`Arena<A>`].
//! - **`#![no_std]`** with `alloc`.
//!
//! ## Quick start
//!
//! ```
//! use harena::Arena;
//!
//! let arena = Arena::new();
//!
//! // Cheap reference-counted allocation.
//! let s = arena.alloc(String::from("hello"));
//! let s2 = s.clone();
//! assert_eq!(*s, *s2);
//!
//! // Single-pointer immutable strings.
//! let name = harena::ArenaRcStr::from_str(&arena, "Alice");
//! assert_eq!(&*name, "Alice");
//!
//! // format! macro returning an ArenaString (call .into_arena_str() to
//! // freeze into a compact 8-byte ArenaRcStr).
//! let greeting = harena::format!(in &arena, "Hello, {}!", "world");
//! assert_eq!(&*greeting, "Hello, world!");
//! ```
//!
//! ### Handles outlive the arena
//!
//! Each handle keeps its memory chunk alive via the chunk's refcount.
//! The arena can be dropped and the handles remain valid:
//!
//! ```
//! use harena::ArenaRc;
//!
//! let s: ArenaRc<String> = {
//!     let arena = harena::Arena::new();
//!     arena.alloc(String::from("I outlive my arena!"))
//!     // arena dropped here
//! };
//! assert_eq!(*s, "I outlive my arena!");
//! ```
//!
//! ### Cross-thread sharing
//!
//! [`Arena`] is single-threaded (`!Send` and `!Sync`). The handles it
//! produces come in two flavors:
//!
//! - [`ArenaRc<T, A>`] / [`ArenaRcStr<A>`] — `!Send`/`!Sync`, cheap
//!   non-atomic refcount. The default.
//! - [`ArenaArc<T, A>`] / [`ArenaArcStr<A>`] — `Send + Sync` when `T` and
//!   `A` are, atomic reference counting safe for cross-thread sharing.
//!
//! ```
//! let arena = harena::Arena::new();
//! let shared = arena.alloc_shared(42_u64);
//! let h = std::thread::spawn(move || *shared);
//! assert_eq!(42, h.join().unwrap());
//! ```
//!
//! ### Owned single handle ([`ArenaBox`])
//!
//! [`ArenaBox`] is a unique owner whose `Drop` runs `T::drop` immediately
//! on handle drop and provides `&mut T` access, similar to
//! [`alloc::boxed::Box`] but backed by the arena.
//!
//! ```
//! let arena = harena::Arena::new();
//! let mut v = arena.alloc_box(vec![1, 2, 3]);
//! v.push(4);
//! assert_eq!(*v, vec![1, 2, 3, 4]);
//! drop(v); // Vec's drop runs here, freeing its heap buffer.
//! ```
//!
//! ### Collections in the arena
//!
//! [`ArenaVec`] and [`ArenaString`] are growable collections that live in
//! the arena. You can use the arena directly as
//! the allocator for any type from the [`allocator_api2`] ecosystem
//! (including `hashbrown::HashMap`).
//!
//! ```
//! use harena::{Arena, ArenaVec, CollectIn};
//!
//! let arena = Arena::new();
//!
//! let mut v = arena.new_vec::<i32>();
//! for i in 0..5 { v.push(i); }
//!
//! // CollectIn trait for iterator collection.
//! let squares: ArenaVec<i32, _> = (1..=5).map(|i| i * i).collect_in(&arena);
//! assert_eq!(squares.as_slice(), &[1, 4, 9, 16, 25]);
//! ```
//!
//! ### Build-then-freeze: shrink long-lived collections to a single pointer
//!
//! [`ArenaString`] and [`ArenaVec`] are designed as **transient
//! builders**. They carry a data pointer + capacity + arena reference
//! because every `push` may need to grow, and growth needs the
//! allocator.
//!
//! Once you're done building, **freeze them** into immutable handles:
//!
//! - [`ArenaString::into_arena_str`] → [`ArenaRcStr`] (**8 bytes**, length
//!   stored inline in the chunk). The freeze itself is **O(1)** — no
//!   copy, no new allocation; the data pointer is just rewrapped.
//! - [`ArenaVec::into_arena_rc`] → [`ArenaRc<[T]>`] (16-byte slice fat
//!   pointer; immutable, cloneable, refcount-based). For `T: !Drop`,
//!   the freeze is **O(1)** too (the buffer is rewrapped in place).
//!
//! Both freezes also reclaim any unused capacity at the buffer's tail
//! when the buffer happens to sit at the chunk's bump cursor — those
//! bytes become available for the next allocation.
//!
//! ```
//! use harena::{Arena, ArenaRcStr};
//!
//! let arena = Arena::new();
//!
//! // Build phase: 24-byte builder, alive briefly.
//! let mut builder = arena.new_string();
//! builder.push_str("hello, ");
//! builder.push_str("world");
//!
//! // Freeze for storage: 8-byte single-pointer handle. O(1) — no copy.
//! let stored: ArenaRcStr = builder.into_arena_str();
//! assert_eq!(&*stored, "hello, world");
//! # let _ = stored.clone();
//! ```
//!
//! Use this pattern whenever you'd be storing many strings or slices
//! long-term — the per-handle savings (16 bytes for strings, 8 for
//! slices) add up quickly across millions of items, and the frozen
//! handles are also cheaper to clone.
//!
//! ## Comparison with `bumpalo`
//!
//! [`bumpalo`](https://crates.io/crates/bumpalo) is the closest crate in
//! spirit; here's how harena2 differs.
//!
//! | Capability | `bumpalo` | `harena` |
//! |---|---|---|
//! | Bump allocation | ✅ | ✅ |
//! | `Allocator` trait integration | ✅ via `allocator-api2` | ✅ via `allocator-api2` |
//! | Reclamation granularity | Whole arena at reset | **Per chunk**, as refcounts hit 0 |
//! | Smart-pointer handles | ❌ (raw `&'bump T`) | ✅ [`ArenaRc`], [`ArenaArc`], [`ArenaRcStr`] |
//! | Handles outlive the arena | ❌ | ✅ |
//! | Cross-thread sharing of individual values | ❌ | ✅ via [`ArenaArc`] |
//! | Automatic per-object `Drop` | Only via `bumpalo::boxed::Box` | ✅ Automatic |
//! | Owned single handle (`Drop` on handle drop) | `bumpalo::boxed::Box` | [`ArenaBox`] |
//! | Single-pointer string handles | ❌ (`&str` is 16 bytes) | ✅ [`ArenaRcStr`] is 8 bytes |
//! | Bumpalo-style growable collections | ✅ `bumpalo::collections` | ✅ [`ArenaVec`], [`ArenaString`] |
//! | `format!`-style macro | ✅ | ✅ |
//! | `#![no_std]` | ✅ | ✅ |
//!
//! `bumpalo` wins on raw allocation speed in tight loops where you don't
//! need `Drop`, refcounting, or chunk reuse — its `&'bump T` is just an
//! advanced bump pointer with no per-handle bookkeeping.
//!
//! `harena` wins when you need any of: drop-on-chunk-teardown, handles
//! that outlive the arena, cross-thread sharing of individual values,
//! incremental memory reclamation, or compact ref-counted string handles.
//!
//! ## Crate features
//!
//! - **`std`** (default) — enables `std::io::Write`-style integration
//!   where applicable. Disable for `#![no_std]` environments (the crate
//!   still requires `alloc`).
//! - **`serde`** — adds `Serialize` impls for [`ArenaRcStr`],
//!   [`ArenaArcStr`], [`ArenaString`], and [`ArenaVec`].
//!
//! ## Thread support
//!
//! [`Arena`] is `!Send` and `!Sync` regardless of `A`. Cross-thread
//! sharing happens at the *handle* level via [`ArenaArc`] /
//! [`ArenaArcStr`], not at the arena level. This pay-for-what-you-use
//! split keeps the common single-threaded path on cheap non-atomic
//! refcount ops, and only the explicitly-shared values pay for atomics.
//!
//! ## FAQ
//!
//! ### Why no small-string optimization for [`ArenaString`]?
//!
//! SSO in `std::String` exists to skip a **heap allocation** (~30-100
//! cycles) for short strings. In `ArenaString`, the "allocation" being
//! skipped is a **bump-pointer increment** (~3 cycles, hot in L1). SSO
//! would add a discriminator branch on every read and write — likely
//! costing more than it saves.
//!
//! It would also inflate the transient builder (32 → ~48 bytes) to
//! optimize the path that's already cheapest, and it doesn't compose
//! with [`ArenaRcStr`]'s 8-byte handle invariant.
//!
//! For interning many short strings, [`ArenaRcStr::from_str`] is already
//! the right answer: one bump op, no SSO needed.
//!
//! ### Why no `SmallVec`-style inline storage for [`ArenaVec`]?
//!
//! Same reasoning as above — the bump allocation we'd be skipping is
//! near-free, while inline storage would bloat the builder by N elements
//! per instance.
//!
//! For embedded "usually empty/tiny" collection fields (the classic
//! `SmallVec` use case), the better answer in this ecosystem is the
//! sibling `dst-factory` crate: build the parent struct as a DST with
//! the collection as an inline trailing field. One allocation, true
//! inline storage, no per-field handle overhead.
//!
//! ### Why no `ArenaSlice<T>` (single-pointer slice handle)?
//!
//! [`ArenaRcStr`] is 8 bytes via inline length-prefix; an `ArenaSlice<T>`
//! could be too. We don't ship one because:
//!
//! 1. The slice-interning workload that would justify it is rare.
//! 2. [`ArenaRc<[T]>`](crate::ArenaRc) already exists at 16 bytes —
//!    matching `Box<[T]>` / `Arc<[T]>` user expectations.
//! 3. Aligning the inline length prefix correctly for arbitrary `T` is
//!    fiddlier than for `str` (whose alignment is 1).
//!
//! Can be added without breaking changes if a use case emerges.
//!

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Crate-wide lint adjustments. We document the rationale for each so a
// future reader understands the deliberate exception.
#![allow(
    clippy::redundant_pub_crate,
    reason = "one-type-per-file org puts every internal type behind a private module; \
              `pub(crate)` is the correct visibility regardless of module scope"
)]
#![allow(
    clippy::multiple_unsafe_ops_per_block,
    reason = "tightly-coupled unsafe sequences (e.g., write+link+inc-ref) are easier to \
              reason about as one block with a single SAFETY comment than as fragmented blocks"
)]
#![allow(
    clippy::missing_const_for_fn,
    reason = "many small fns could be const but adding it eagerly hurts forward compatibility \
              when the body grows to use non-const operations"
)]
#![allow(
    clippy::option_if_let_else,
    reason = "explicit `if let { Some } else { None }` is often clearer than \
              `map_or_else` when the closure isn't trivial"
)]
#![allow(
    clippy::cast_ptr_alignment,
    reason = "we manually align bump-allocated addresses; the cast is the result of \
              that alignment, not a violation of it"
)]
#![allow(
    clippy::module_name_repetitions,
    reason = "type names like ArenaRc/ArenaArc/ArenaBox in modules of the same name read \
              naturally"
)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

// ---------------------------------------------------------------------------
// Public modules — one type per file.
// ---------------------------------------------------------------------------

mod allocator_impl;
mod arena;
mod arena_arc;
mod arena_arc_str;
mod arena_box;
mod arena_builder;
mod arena_inner;
mod arena_rc;
mod arena_rc_str;
mod arena_string;
mod arena_vec;
mod chunk_header;
mod chunk_sharing;
mod chunk_size_class;
mod collect_in;
mod constants;
mod drop_entry;
mod dst_helpers;
mod format_macro;
mod pending_arena_arc;
mod pending_arena_rc;
mod stats;

#[cfg(feature = "serde")]
mod serde_impls;

// ---------------------------------------------------------------------------
// Public re-exports.
// ---------------------------------------------------------------------------

pub use crate::arena::Arena;
pub use crate::arena_arc::ArenaArc;
pub use crate::arena_arc_str::ArenaArcStr;
pub use crate::arena_box::ArenaBox;
pub use crate::arena_builder::{ArenaBuilder, BuildError};
pub use crate::arena_rc::ArenaRc;
pub use crate::arena_rc_str::ArenaRcStr;
pub use crate::arena_string::ArenaString;
pub use crate::arena_vec::ArenaVec;
pub use crate::collect_in::{CollectIn, FromIteratorIn};
pub use crate::drop_entry::DropEntry;
pub use crate::pending_arena_arc::PendingArenaArc;
pub use crate::pending_arena_rc::PendingArenaRc;
pub use crate::stats::ArenaStats;
