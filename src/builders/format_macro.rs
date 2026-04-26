/// `format!`-style macro that formats into a fresh
/// [`String`](crate::builders::String) in the given arena.
///
/// The result is a mutable, growable string. To freeze it into an
/// immutable [`RcStr`](crate::RcStr) (8 bytes on 64-bit), call
/// [`String::into_arena_str`](crate::builders::String::into_arena_str)
/// — that's an O(1) operation thanks to the inline length prefix.
///
/// # Panics
///
/// Panics if the backing allocator fails (consistent with `std::format!`),
/// or if a `{...}` formatter's `Display`/`Debug` impl returns `Err`.
///
/// # Example
///
/// ```
/// let arena = multitude::Arena::new();
/// let name = "Alice";
/// let s = multitude::builders::format!(in &arena, "Hello, {name}!");
/// assert_eq!(&*s, "Hello, Alice!");
///
/// // Freeze for compact long-term storage if desired:
/// let frozen: multitude::RcStr = s.into_arena_str();
/// assert_eq!(&*frozen, "Hello, Alice!");
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __multitude_format {
    (in $arena:expr, $($arg:tt)*) => {{
        let mut __multitude_buf = $crate::Arena::alloc_string($arena);
        ::core::fmt::Write::write_fmt(
            &mut __multitude_buf,
            ::core::format_args!($($arg)*),
        )
        .expect("a formatting trait implementation returned an error");
        __multitude_buf
    }};
}

// `core::fmt::Write` impl for `String`.
use allocator_api2::alloc::Allocator;
use core::fmt;

use crate::builders::String;

impl<A: Allocator + Clone> fmt::Write for String<'_, A> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.push(c);
        Ok(())
    }
}
