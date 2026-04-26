use crate::dst::pending_macro::define_pending_arena;

define_pending_arena!(
    /// In-progress reservation in a `Local`-flavor chunk that finalizes
    /// into an [`Box<T, A>`](crate::Box).
    ///
    /// Mirrors [`PendingRc`](crate::dst::PendingRc), but the
    /// resulting smart pointer is *owned*: its `Drop` impl runs the value's
    /// destructor immediately when the smart pointer is dropped (the refcount
    /// smart pointers defer drop to chunk teardown).
    ///
    /// Construct via [`Self::new`] / [`Self::try_new`], passing the arena
    /// and the desired byte layout. The caller initializes the reserved
    /// bytes through [`Self::as_mut_ptr`], then calls [`Self::finalize`]
    /// (with a fat-pointer template for DSTs) to obtain an
    /// [`Box<T, A>`](crate::Box). Forgetting to call
    /// `finalize` decrements the chunk refcount; the reserved bytes are
    /// leaked until chunk teardown but never dropped.
    PendingBox,
    handle = [crate::Box],
    sharing = Local,
    on_owner_thread = true,
    debug_name = "PendingBox",
    kind = dst_only,
);
