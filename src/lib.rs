//! Fast and flexible arena-based bump allocator.
//!
//! `multitude` is an arena-based bump allocator designed to improve the performance of applications that have **phase-oriented logic**, which
//! is when groups of related allocations live and die together. Service request handling and parsers are two examples of this pattern which usually
//! benefit from a bump allocator.
//!
//! `multitude` works by accumulating large chunks of memory allocated from the system and then carving out smaller pieces of it for application use
//! using a fast bump allocation strategy, which is considerably faster than allocating from the system. The downside however is that the individual allocations
//! can't be freed separately. Instead, memory is reclaimed and returned to the system in bulk when the entire arena is dropped.
//!
//! # Why Another Bump Allocator?
//!
//! The Rust ecosystem has a few bump allocators, the most popular being [`bumpalo`](https://crates.io/crates/bumpalo).
//! `multitude` uses a different implementation strategy and has a richer API surface making it suitable for more
//! use cases. The main features that set `multitude` apart are:
//!
//! 1. **Flexibility.** `multitude` provides multiple allocation styles, all of
//!    which can coexist in the same arena:
//!
//!    - Mutable references with lifetimes tied to the arena (`&mut T`,
//!      `&mut str`, `&mut [T]`).
//!    - Reference-counted smart pointers ([`Rc`], [`RcStr`]) for
//!      single-threaded sharing.
//!    - Atomic reference-counted smart pointers ([`Arc`], [`ArcStr`])
//!      for cross-thread sharing.
//!    - Owned, mutable smart pointers ([`Box`], [`BoxStr`]).
//!
//! 2. **Early Reclamation.** In many situations, `multitude` can reclaim memory from individual chunks as soon as their reference counts drop to zero,
//!    without waiting for the entire arena to be dropped. This allows for more efficient memory usage in long-running arenas with many short-lived allocations.
//!
//! 3. **Smart Pointers Can Outlive the Arena.** The reference-counted smart pointers produced by `multitude` can keep their owning chunk alive even after the arena itself has been dropped,
//!    allowing for more flexible memory management and longer-lived data structures.
//!
//! 4. **Drop Support.** `multitude` automatically runs `Drop` for allocated values at the appropriate time.
//!
//! 5. **Efficient Immutable String References.** `multitude` provides [`RcStr`], [`ArcStr`], and
//!    [`BoxStr`] — single-pointer (8 bytes) smart pointers to UTF-8 strings stored in the arena. Refcounted, atomic-refcounted,
//!    and owned-mutable variants respectively, all sharing the same compact layout.
//!
//! 6. **Efficient Mutable Strings and Vectors.** With the `builders` Cargo feature (default-on), `multitude` provides [`String`](crate::builders::String) and [`Vec`](crate::builders::Vec) which are growable collections that live in the arena and can be frozen into compact
//!    reference-counted smart pointers when you're done building.
//!
//! 7. **Dynamically-Sized Types.** `multitude` supports dynamically-sized types (DSTs) like slices and strings, allowing you to allocate and manage them in the
//!    arena with the same flexibility as sized types. The [`dst-factory`](https://crates.io/crates/dst-factory) crate is a great companion for building DSTs in the arena.
//!
//! 8. **`format!`-style Macro.** `multitude` includes a `format!`-style macro that allows you to create formatted strings directly in the arena, avoiding intermediate allocations and copies.
//!
//! 9. **UTF-16 Support.** With the `utf16` Cargo feature, `multitude` provides a parallel set of arena-resident UTF-16 string types ([`RcUtf16Str`],
//!    [`ArcUtf16Str`], [`BoxUtf16Str`], [`Utf16String`](crate::builders::Utf16String)) and a `format_utf16!` macro for FFI / Windows / JS-engine
//!    interop without per-call transcoding at every boundary.
//!
//! 10. **`#![no_std]` Support.** `multitude` can be used in `#![no_std]` environments, making it suitable for embedded systems and other resource-constrained contexts.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "builders")] {
//! use multitude::Arena;
//!
//! let arena = Arena::new();
//!
//! // Cheap reference-counted allocation of any user type.
//! struct Point { x: f64, y: f64 }
//! let p = arena.alloc_rc(Point { x: 3.0, y: 4.0 });
//! let p2 = p.clone();
//! assert_eq!(p.x, p2.x);
//!
//! // Single-pointer immutable strings.
//! let name = arena.alloc_str_rc("Alice");
//! assert_eq!(&*name, "Alice");
//!
//! // format! macro returning an String (call .into_arena_str() to
//! // freeze into a compact 8-byte RcStr).
//! let greeting = multitude::builders::format!(in &arena, "Hello, {}!", "world");
//! assert_eq!(&*greeting, "Hello, world!");
//! # }
//! ```
//! # Flexibility
//!
//! `multitude` supports a variety of ways to allocate data and track it over time.
//!
//! ## Simple References
//!
//! The simplest use of the arena is to get plain mutable references. The lifetime of those references is then tied
//! to the arena's own lifetime.
//!
//! ```
//! let arena = multitude::Arena::new();
//! let x: &mut u32 = arena.alloc(42);
//! let y: &mut u32 = arena.alloc(100);
//! *x += 1;
//! *y += 1;
//! assert_eq!(*x, 43);
//! assert_eq!(*y, 101);
//!
//! // Strings and slices too:
//! let s: &mut str = arena.alloc_str("hello");
//! let v: &mut [i32] = arena.alloc_slice_copy(&[1, 2, 3]);
//! ```
//!
//! These references can't outlive the arena, which limits their use. But they are the fastest and
//! most efficient way to allocate from the arena, so if the lifetime constraints are tolerable, simple
//! references are the way to go.
//!
//! ## Smart Pointers
//!
//! Smart pointers ([`Rc`], [`Arc`], [`Box`] and their `str` variations) work in a way similar to the like-named types
//! in the standard library, except that they reference addresses within an arena.
//!
//! ```
//! use multitude::Rc;
//!
//! struct Point { x: f64, y: f64 }
//!
//! let p: Rc<Point> = {
//!     let arena = multitude::Arena::new();
//!     arena.alloc_rc(Point { x: 3.0, y: 4.0 })
//!     // arena dropped here
//! };
//! assert_eq!(p.x, 3.0);
//! ```
//!
//! Although [`Arena`] itself is single-threaded (`!Send` and `!Sync`), the arc-family of types (e.g., [`Arc`]) enable cross-thread sharing.
//!
//! ```
//! let arena = multitude::Arena::new();
//! let shared = arena.alloc_arc(42_u64);
//! let h = std::thread::spawn(move || *shared);
//! assert_eq!(42, h.join().unwrap());
//! ```
//!
//! [`Box`] is a unique owner whose `Drop` runs `T::drop` immediately
//! when the smart pointer is dropped and provides `&mut T` access, similar to
//! [`alloc::boxed::Box`] but backed by the arena.
//!
//! ```
//! let arena = multitude::Arena::new();
//! let mut v = arena.alloc_box(vec![1, 2, 3]);
//! v.push(4);
//! assert_eq!(*v, vec![1, 2, 3, 4]);
//! drop(v); // Vec's drop runs here, freeing its heap buffer.
//! ```
//!
//! ## Collections
//!
//! [`Vec`](crate::builders::Vec) and [`String`](crate::builders::String) are growable collections that live in
//! the arena.
//!
//! Additionally, you can use an arena as
//! the allocator for any type from the [`allocator_api2`] ecosystem
//! (including `hashbrown::HashMap`).
//!
//! ```
//! # #[cfg(feature = "builders")] {
//! use multitude::Arena;
//! use multitude::builders::{Vec, CollectIn};
//!
//! let arena = Arena::new();
//!
//! let mut v = arena.alloc_vec::<i32>();
//! for i in 0..5 { v.push(i); }
//!
//! // CollectIn trait for iterator collection.
//! let squares: Vec<i32, _> = (1..=5).map(|i| i * i).collect_in(&arena);
//! assert_eq!(squares.as_slice(), &[1, 4, 9, 16, 25]);
//! # }
//! ```
//!
//! ## Freezing
//!
//! [`String`](crate::builders::String) and [`Vec`](crate::builders::Vec) are designed as **transient
//! builders**. They carry a data pointer + length + capacity + arena reference.
//!
//! Once you're done building, you can **freeze them** into immutable smart pointers:
//!
//! - [`String::into_arena_str`](crate::builders::String::into_arena_str) → [`RcStr`] (**8 bytes**). The
//!   freeze itself is **O(1)** — no copy, no new allocation.
//! - [`Vec::into_arena_rc`](crate::builders::Vec::into_arena_rc) → [`Rc<[T]>`](crate::Rc) (16-byte slice fat
//!   pointer; immutable, cloneable, refcount-based). For `T: !Drop`,
//!   the freeze is **O(1)** too.
//!
//! Both freezes also reclaim any unused capacity left in the buffer
//! when the conditions allow it, so those bytes become available for
//! the next allocation.
//!
//! ```
//! # #[cfg(feature = "builders")] {
//! use multitude::{Arena, RcStr};
//!
//! let arena = Arena::new();
//!
//! // Build phase: 32-byte builder, alive briefly.
//! let mut builder = arena.alloc_string();
//! builder.push_str("hello, ");
//! builder.push_str("world");
//!
//! // Freeze for storage: 8-byte single-pointer smart pointer. O(1) — no copy.
//! let stored: RcStr = builder.into_arena_str();
//! assert_eq!(&*stored, "hello, world");
//! # let _ = stored.clone();
//! # }
//! ```
//!
//! Use this pattern whenever you'd be storing many strings or slices
//! long-term — the per-pointer savings (16 bytes for strings, 8 for
//! slices) add up quickly across millions of items, and the frozen
//! smart pointers are also cheaper to clone.
//!
//! # Comparison with `bumpalo`
//!
//! [`bumpalo`](https://crates.io/crates/bumpalo) is the closest crate in
//! spirit; here's how multitude differs.
//!
//! | Capability | `bumpalo` | `multitude` |
//! |---|---|---|
//! | Bump allocation | ✅ | ✅ |
//! | Simple references (`&'arena mut T`) | ✅ `Bump::alloc` | ✅ [`Arena::alloc`] |
//! | `Allocator` trait integration | ✅ via `allocator-api2` | ✅ via `allocator-api2` |
//! | Reclamation granularity | Whole arena at reset | **Per chunk**, as refcounts hit 0 (refcount smart pointers); whole-arena (simple references) |
//! | Smart pointers | ❌ (raw `&'bump T`) | ✅ [`Rc`], [`Arc`], [`RcStr`] |
//! | Smart pointers outlive the arena | ❌ | ✅ ([`Rc`] / [`Arc`] / [`Box`] and their `str` variants — simple references are lifetime-bound) |
//! | Cross-thread sharing of individual values | ❌ | ✅ via [`Arc`] |
//! | Automatic per-object `Drop` | Only via `bumpalo::boxed::Box` | ✅ Automatic (refcount smart pointers drop at chunk teardown; [`Box`] / [`BoxStr`] drop at smart pointer drop; simple references drop at arena drop) |
//! | Owned single smart pointer (`Drop` on drop) | `bumpalo::boxed::Box` | [`Box`] |
//! | Single-pointer string smart pointers | ❌ (`&str` is 16 bytes) | ✅ [`RcStr`] / [`ArcStr`] / [`BoxStr`] are 8 bytes |
//! | Growable collections | ✅ `bumpalo::collections` | ✅ [`Vec`](crate::builders::Vec), [`String`](crate::builders::String) |
//! | `format!`-style macro | ✅ | ✅ |
//! | UTF-16 strings | ❌ | ✅ via `RcUtf16Str` / `ArcUtf16Str` / `BoxUtf16Str` / `Utf16String` (gated on the `utf16` feature) |
//! | Dynamically-sized types (DSTs, e.g. `dyn Trait`, `[T]`) | ❌ | ✅ via the `dst` module (gated on the `dst` feature) |
//! | `#![no_std]` | ✅ | ✅ |
//!
//! # Crate Features
//!
//! - **`std`** — enables `std::io::Write`-style integration
//!   where applicable. Disable for `#![no_std]` environments (the crate
//!   still requires `alloc`).
//! - **`builders`** — enables the [`builders`] module,
//!   which contains the growable collection types
//!   ([`String`](crate::builders::String),
//!   [`Vec`](crate::builders::Vec),
//!   `Utf16String`), the `CollectIn` / `FromIteratorIn` traits, and
//!   the `format!` / `vec!` / `format_utf16!` macros. Also adds the
//!   matching `Arena::alloc_string` / `alloc_vec` / `alloc_utf16_string`
//!   methods. Disable for the leanest builds — the smart pointers
//!   ([`RcStr`], [`Rc`], etc.) and `Arena::alloc_str_*` /
//!   `alloc_*_slice_*` still work without it.
//! - **`stats`** — enables runtime instrumentation counters
//!   returned by `Arena::stats`. Disable for the tightest allocation
//!   throughput when you don't need observability.
//! - **`serde`** — adds `Serialize` impls for [`RcStr`],
//!   [`ArcStr`], and (with `builders`)
//!   [`String`](crate::builders::String) and
//!   [`Vec`](crate::builders::Vec). With
//!   `serde + utf16`, also adds `Serialize` impls for the UTF-16
//!   types (transcoded to UTF-8 on the wire).
//! - **`dst`** — enables the `dst` module (`PendingRc`,
//!   `PendingArc`, `PendingBox`) for constructing
//!   true dynamically-sized types and trait objects in the arena, plus
//!   eight `Arena::alloc_slice_*_box` methods.
//! - **`utf16`** — adds a parallel UTF-16 string surface
//!   (`RcUtf16Str`, `ArcUtf16Str`, `BoxUtf16Str`, plus
//!   `Utf16String` and `format_utf16!` when combined with
//!   `builders`) backed by the
//!   [`widestring`](https://crates.io/crates/widestring) crate. All
//!   length and capacity values are counted in `u16` elements (matching
//!   `widestring::Utf16Str::len()`).
//!
//! The default-enabled features are `std` and `builders`.
//!
//! # UTF-16 Strings
//!
//! With the `utf16` Cargo feature enabled, `multitude` exposes a
//! parallel set of arena-resident UTF-16 string types that mirror the
//! UTF-8 surface element-for-element:
//!
//! | UTF-8 type        | UTF-16 analogue          |
//! |-------------------|--------------------------|
//! | [`RcStr`]    | `RcUtf16Str`        |
//! | [`ArcStr`]   | `ArcUtf16Str`       |
//! | [`BoxStr`]   | `BoxUtf16Str`       |
//! | [`String`](crate::builders::String)   | `Utf16String`       |
//!
//! Strict (validated) UTF-16 only — lone surrogates are rejected. The
//! types are interoperable with `widestring::Utf16Str` /
//! `widestring::Utf16String` for I/O and FFI bridging.
//!
//! ```
//! # #[cfg(all(feature = "utf16", feature = "builders"))] {
//! use multitude::Arena;
//! use widestring::utf16str;
//!
//! let arena = Arena::new();
//!
//! // From a validated &Utf16Str literal:
//! let s = arena.alloc_utf16_str_rc(utf16str!("hello, world"));
//! assert_eq!(&*s, utf16str!("hello, world"));
//!
//! // Or transcode from a &str:
//! let s2 = arena.alloc_utf16_str_rc_from_str("hello");
//! assert_eq!(&*s2, utf16str!("hello"));
//!
//! // Build incrementally and freeze:
//! let mut b = arena.alloc_utf16_string();
//! b.push_str(utf16str!("abc"));
//! b.push_from_str("123");
//! let frozen = b.into_arena_utf16_str();
//! assert_eq!(&*frozen, utf16str!("abc123"));
//!
//! // format!-style:
//! let name = "Alice";
//! let greeting = multitude::builders::format_utf16!(in &arena, "Hello, {name}!");
//! assert_eq!(greeting.as_utf16_str(), utf16str!("Hello, Alice!"));
//! # }
//! ```
//!
//! # Building DSTs
//!
//! With the `dst` Cargo feature enabled, `multitude` exposes the
//! `dst` module containing three pending-reservation
//! builders — `PendingRc`, `PendingArc`, and
//! `PendingBox` — for constructing
//! values whose layout is only known at runtime (custom DSTs, fat
//! pointers, trait objects).
//!
//! The two-phase pattern (reserve raw bytes → initialize → finalize)
//! lets you build any value the standard `alloc_*` methods can't
//! express. For most users, the `dst-factory` companion crate is the
//! recommended high-level driver; the low-level interface looks like:
//!
//! ```
//! # #[cfg(feature = "dst")] {
//! use core::alloc::Layout;
//! use multitude::{Arena, dst::PendingBox};
//!
//! let arena = Arena::new();
//!
//! // Reserve room for a u32 (any layout works — typically a custom DST).
//! let layout = Layout::new::<u32>();
//! let mut pending = PendingBox::new(&arena, layout);
//!
//! // Initialize the reservation.
//! // Writing a fully-initialized u32 into the reservation is safe
//! // because the layout above matches `u32` exactly.
//! unsafe { pending.as_mut_ptr().cast::<u32>().write(0xCAFE_BABE); }
//!
//! // Finalize. For non-Drop types, drop_fn is None.
//! let template: u32 = 0;
//! // Bytes are initialized above; u32 doesn't need Drop, so drop_fn is None.
//! let b = unsafe { pending.finalize::<u32>(&raw const template, None) };
//! assert_eq!(*b, 0xCAFE_BABE);
//! # }
//! ```
//!
//! The same feature also enables eight `Arena::alloc_slice_*_box`
//! methods that produce `Box<[T]>` directly (mirroring the
//! existing `_rc`/`_arc` slice methods).
//!
//! # How it Works
//!
//! A high-level sketch of what happens when you use the crate. For
//! the gory implementation details, read the source.
//!
//! ## Chunks
//!
//! The arena owns a list of large memory **chunks** allocated from the
//! system. Each user allocation carves out a piece of the current
//! chunk via a fast bump-pointer increment. When the current chunk
//! can't fit the next request, the arena rotates to a fresh chunk.
//! Allocations larger than a chunk's usable capacity get their own
//! dedicated **oversized chunks**.
//!
//! ## Smart Pointers
//!
//! The smart-pointer types ([`Rc`], [`Arc`], [`Box`])
//! are deliberately compact — typically one pointer plus zero-sized
//! markers. Reference counts live with the chunk, not with each smart
//! pointer, so cloning and dropping cost a single counter update.
//!
//! The single-pointer string types ([`RcStr`], [`ArcStr`],
//! [`BoxStr`]) go further: 8 bytes per smart pointer (vs 16
//! for `&str`).
//!
//! ## Reclamation
//!
//! When the last smart pointer into a chunk is dropped, the arena
//! runs every destructor for values it hosted, then either parks the
//! chunk in a free-list cache for fast reuse or returns its memory to
//! the system allocator. This **per-chunk reclamation** lets
//! long-lived arenas with churn keep their memory footprint bounded.
//!
//! Smart pointers ([`Rc`] / [`Arc`] / [`Box`] and their
//! `str` variants) may **outlive the arena**: their chunks stay alive
//! as long as any smart pointer references them, even after the arena
//! itself has been dropped.
//!
//! References returned by [`Arena::alloc`] don't carry refcounts;
//! their chunks live until arena drop. The borrow checker keeps them
//! lifetime-bound to the arena.
//!
//! ## Cross-Thread Sharing
//!
//! The arena itself is `!Send` and `!Sync`. Cross-thread sharing
//! happens at the smart-pointer level via [`Arc`] /
//! [`ArcStr`], which use atomic refcounts. Single-threaded code
//! that uses [`Rc`] only stays on cheap non-atomic operations.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod allocator_impl;
mod arc;
mod arc_str;
#[cfg(feature = "utf16")]
mod arc_utf16_str;
mod arena;
mod arena_builder;
mod arena_handle_macros;
mod arena_inner;
#[cfg(feature = "stats")]
mod arena_stats;
mod arena_str_helpers;
mod arena_str_macros;
mod r#box;
mod box_str;
#[cfg(feature = "utf16")]
mod box_utf16_str;
#[cfg(feature = "builders")]
#[cfg_attr(docsrs, doc(cfg(feature = "builders")))]
pub mod builders;
mod chunk_header;
mod chunk_ref;
mod chunk_sharing;
mod chunk_size_class;
mod constants;
mod drop_entry;
#[cfg(feature = "dst")]
#[cfg_attr(docsrs, doc(cfg(feature = "dst")))]
pub mod dst;
mod entry_layout;
mod flavor;
mod owned_chunk;
mod raw_handle;
mod rc;
mod rc_str;
#[cfg(feature = "utf16")]
mod rc_utf16_str;
mod sync;

pub use crate::arc::Arc;
pub use crate::arc_str::ArcStr;
#[cfg(feature = "utf16")]
#[cfg_attr(docsrs, doc(cfg(feature = "utf16")))]
pub use crate::arc_utf16_str::ArcUtf16Str;
pub use crate::arena::Arena;
pub use crate::arena_builder::{ArenaBuilder, BuildError};
#[cfg(feature = "stats")]
#[cfg_attr(docsrs, doc(cfg(feature = "stats")))]
pub use crate::arena_stats::ArenaStats;
pub use crate::r#box::Box;
pub use crate::box_str::BoxStr;
#[cfg(feature = "utf16")]
#[cfg_attr(docsrs, doc(cfg(feature = "utf16")))]
pub use crate::box_utf16_str::BoxUtf16Str;
pub use crate::rc::Rc;
pub use crate::rc_str::RcStr;
#[cfg(feature = "utf16")]
#[cfg_attr(docsrs, doc(cfg(feature = "utf16")))]
pub use crate::rc_utf16_str::RcUtf16Str;
