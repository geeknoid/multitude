/// Construct an [`Vec`](crate::builders::Vec) inside a given arena.
///
/// Three forms are supported:
///
/// - `multitude::builders::vec![in &arena]` — an empty `Vec`.
/// - `multitude::builders::vec![in &arena; a, b, c]` — a vec from a literal list.
/// - `multitude::builders::vec![in &arena; value; n]` — `n` copies of `value`
///   (requires `T: Clone`; `value` is evaluated once and then cloned).
///
/// The trailing comma in the list form is optional. The arena expression
/// can be any `&Arena<A>` reference (or anything that derefs to one).
///
/// # Panics
///
/// Panics if the backing allocator fails. Use the explicit `try_*`
/// constructors on [`Vec`](crate::builders::Vec) for fallible variants.
///
/// # Examples
///
/// ```
/// use multitude::Arena;
///
/// let arena = Arena::new();
///
/// // Empty:
/// let _v: multitude::builders::Vec<i32> = multitude::builders::vec![in &arena];
///
/// // From a list:
/// let v = multitude::builders::vec![in &arena; 1, 2, 3];
/// assert_eq!(&*v, &[1, 2, 3]);
///
/// // N copies:
/// let zeros = multitude::builders::vec![in &arena; 0_u32; 5];
/// assert_eq!(&*zeros, &[0, 0, 0, 0, 0]);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __multitude_vec {
    (in $arena:expr) => {
        $crate::builders::Vec::new_in($arena)
    };
    (in $arena:expr; $elem:expr; $n:expr) => {{
        let __multitude_n: ::core::primitive::usize = $n;
        let mut __multitude_buf = $crate::builders::Vec::with_capacity_in(__multitude_n, $arena);
        __multitude_buf.resize(__multitude_n, $elem);
        __multitude_buf
    }};
    (in $arena:expr; $($x:expr),+ $(,)?) => {{
        $crate::builders::Vec::from_iter_in([$($x),+], $arena)
    }};
}
