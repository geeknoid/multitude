/// `format!`-style macro that formats into a fresh
/// [`Utf16String`](crate::builders::Utf16String) in the given arena.
///
/// The result is a mutable, growable UTF-16 string. To freeze it into
/// an immutable [`RcUtf16Str`](crate::RcUtf16Str) (8 bytes
/// on 64-bit), call
/// [`Utf16String::into_arena_utf16_str`](crate::builders::Utf16String::into_arena_utf16_str)
/// — that's an O(1) operation thanks to the inline length prefix.
///
/// # Panics
///
/// Panics if the backing allocator fails, or if a `{...}` formatter's
/// `Display`/`Debug` impl returns `Err`.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "utf16")] {
/// use widestring::utf16str;
/// let arena = multitude::Arena::new();
/// let name = "Alice";
/// let s = multitude::builders::format_utf16!(in &arena, "Hello, {name}!");
/// assert_eq!(s.as_utf16_str(), utf16str!("Hello, Alice!"));
/// # }
/// ```
#[doc(hidden)]
#[macro_export]
#[cfg(feature = "utf16")]
macro_rules! __multitude_format_utf16 {
    (in $arena:expr, $($arg:tt)*) => {{
        let mut __multitude_buf = $crate::Arena::alloc_utf16_string($arena);
        ::core::fmt::Write::write_fmt(
            &mut __multitude_buf,
            ::core::format_args!($($arg)*),
        )
        .expect("a formatting trait implementation returned an error");
        __multitude_buf
    }};
}

use allocator_api2::alloc::Allocator;
use core::fmt;

use crate::builders::Utf16String;

/// `core::fmt::Write` impl for [`Utf16String`]. Each `&str` chunk
/// passed to `write_str` is independently a complete, valid UTF-8
/// fragment, which means transcoding it via `encode_utf16()` produces a
/// complete UTF-16 sequence with no cross-call surrogate state. Hence
/// pushing via the bulk transcode path is sound.
impl<A: Allocator + Clone> fmt::Write for Utf16String<'_, A> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_from_str(s);
        Ok(())
    }
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.push(c);
        Ok(())
    }
}
