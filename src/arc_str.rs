use allocator_api2::alloc::{Allocator, Global};

use crate::arena_str_macros::{impl_str_accessors, impl_str_handle_core, impl_str_read_traits};
use crate::flavor::Shared;
use crate::raw_handle::RawHandle;

/// An immutable, single-pointer reference-counted UTF-8 string stored in
/// an [`Arena`](crate::Arena), safe to share across threads.
///
/// 8 bytes on 64-bit (one pointer); contrast with `&'arena str`'s 16
/// bytes. Cloning is **O(1)** (one atomic refcount bump). For
/// single-threaded code, prefer [`RcStr`](crate::RcStr) — it has the
/// same layout and same cost model with a non-atomic refcount.
///
/// Build via either:
///
/// - [`Arena::alloc_str_arc`](crate::Arena::alloc_str_arc) — copy a
///   `&str` directly into the arena.
/// - [`String`](crate::builders::String) +
///   [`into_arena_str`](crate::builders::String::into_arena_str), then
///   convert with `.into()` — build incrementally, then freeze.
///
/// # Example
///
/// ```
/// use multitude::{Arena, ArcStr};
/// use std::thread;
///
/// let arena = Arena::new();
/// let s = arena.alloc_str_arc("shared");
/// let s2 = s.clone();
/// let h = thread::spawn(move || s2.len());
/// assert_eq!(s.len(), h.join().unwrap());
/// ```
pub struct ArcStr<A: Allocator + Clone = Global> {
    inner: RawHandle<u8, Shared, A>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "RawHandle holds a NonNull which is structurally !Send; the Shared flavor's atomic refcount makes cross-thread sharing sound"
)]
// SAFETY: backed by a Shared chunk with atomic refcount.
unsafe impl<A: Allocator + Clone + Send + Sync> Send for ArcStr<A> {}
// SAFETY: see above.
unsafe impl<A: Allocator + Clone + Send + Sync> Sync for ArcStr<A> {}

impl_str_handle_core!(ArcStr, Shared);
impl_str_accessors!([<A: Allocator + Clone>], ArcStr<A>);
impl_str_read_traits!([<A: Allocator + Clone>], ArcStr<A>);

impl<A: Allocator + Clone> ::core::convert::From<ArcStr<A>> for crate::Arc<[u8], A> {
    /// Convert an [`ArcStr<A>`] into an [`Arc<[u8], A>`](crate::Arc).
    /// O(1) — the chunk's +1 refcount transfers; the new handle's slice
    /// covers the underlying UTF-8 bytes (excluding the inline length
    /// prefix). Mirrors `std::sync::Arc::<[u8]>::from(Arc<str>)`.
    #[inline]
    fn from(s: ArcStr<A>) -> Self {
        let len = s.len();
        let data = s.data_ptr();
        core::mem::forget(s);
        // SAFETY: `data` is `len` bytes of UTF-8 in a Shared chunk; refcount transferred.
        unsafe { Self::from_raw_slice(data, len) }
    }
}
