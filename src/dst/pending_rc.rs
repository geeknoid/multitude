use crate::dst::pending_macro::define_pending_arena;

define_pending_arena!(
    /// In-progress reservation in a `Local`-flavor chunk that finalizes
    /// into an [`Rc<T, A>`](crate::Rc).
    ///
    /// Construct via [`Self::new`] / [`Self::try_new`], passing the arena
    /// and the desired byte layout. The caller initializes the reserved
    /// bytes through [`Self::as_mut_ptr`], then calls [`Self::finalize`]
    /// (or [`Self::finalize_dst`] for true DSTs) to obtain an
    /// [`Rc<T, A>`](crate::Rc). Forgetting to call `finalize`
    /// decrements the chunk refcount; the reserved bytes are leaked
    /// until chunk teardown but never dropped.
    PendingRc,
    handle = [crate::Rc],
    sharing = Local,
    on_owner_thread = true,
    debug_name = "PendingRc",
    kind = sized_and_dst,
);
