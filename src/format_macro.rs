//! The [`format!`](crate::format) macro.

/// `format!`-style macro that formats into a fresh
/// [`ArenaString`](crate::ArenaString) in the given arena.
///
/// The result is a mutable, growable string. To freeze it into an
/// immutable [`ArenaRcStr`](crate::ArenaRcStr) (8 bytes on 64-bit), call
/// [`ArenaString::into_arena_str`](crate::ArenaString::into_arena_str)
/// — that's an O(1) operation thanks to the inline length prefix.
///
/// # Panics
///
/// Panics on allocator failure (consistent with [`std::format!`]) or if
/// a `{...}` formatter's `Display`/`Debug` impl returns `Err`.
///
/// # Example
///
/// ```
/// let arena = harena::Arena::new();
/// let name = "Alice";
/// let s = harena::format!(in &arena, "Hello, {name}!");
/// assert_eq!(&*s, "Hello, Alice!");
///
/// // Freeze for compact long-term storage if desired:
/// let frozen: harena::ArenaRcStr = s.into_arena_str();
/// assert_eq!(&*frozen, "Hello, Alice!");
/// ```
#[macro_export]
macro_rules! format {
    (in $arena:expr, $($arg:tt)*) => {{
        let mut __harena2_buf = $crate::Arena::new_string($arena);
        ::core::fmt::Write::write_fmt(
            &mut __harena2_buf,
            ::core::format_args!($($arg)*),
        )
        .expect("a formatting trait implementation returned an error");
        __harena2_buf
    }};
}

// `core::fmt::Write` impl for `ArenaString`. Lives here to keep all the
// `format!`-related infrastructure in one place.
use allocator_api2::alloc::Allocator;
use core::fmt;

impl<A: Allocator + Clone> fmt::Write for crate::arena_string::ArenaString<'_, A> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.push(c);
        Ok(())
    }
}
