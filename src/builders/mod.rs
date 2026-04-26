//! Growable, mutable arena-resident builders that can be frozen into the
//! crate's smart-pointer types.
//!
//! These types are **transient builders** — small structs (32 bytes on
//! 64-bit) carrying a data pointer + length + capacity + arena reference.
//! Once built, freeze them into compact, immutable smart pointers via
//! `into_arena_str` / `into_arena_utf16_str` / `into_arena_rc`.
//!
//! Gated on the `builders` Cargo feature (default-on).

mod collect_in;
mod format_macro;
#[cfg(feature = "utf16")]
mod format_utf16_macro;
mod from_iterator_in;
mod string;
#[cfg(feature = "utf16")]
mod utf16_string;
mod vec;
mod vec_macro;

pub use collect_in::CollectIn;
pub use from_iterator_in::FromIteratorIn;
pub use string::String;
#[cfg(feature = "utf16")]
#[cfg_attr(docsrs, doc(cfg(feature = "utf16")))]
pub use utf16_string::Utf16String;
pub use vec::{Drain, IntoIter, Vec};

// Re-export the macros under `multitude::builders::*`. The `#[macro_export]`
// at the macro definition site puts the underscore-prefixed names at the
// crate root (hidden from docs); these `pub use` lines surface the
// user-facing names on `multitude::builders`.
#[doc(inline)]
pub use crate::__multitude_format as format;
#[cfg(feature = "utf16")]
#[cfg_attr(docsrs, doc(cfg(feature = "utf16")))]
#[doc(inline)]
pub use crate::__multitude_format_utf16 as format_utf16;
#[doc(inline)]
pub use crate::__multitude_vec as vec;
