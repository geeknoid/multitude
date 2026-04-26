use crate::dst::pending_macro::define_pending_arena;

define_pending_arena!(
    /// In-progress reservation in a `Shared`-flavor chunk that finalizes
    /// into an [`Arc<T, A>`](crate::Arc).
    ///
    /// Mirrors [`PendingRc`](crate::dst::PendingRc), but
    /// finalizes into a `Send + Sync`-capable
    /// [`Arc`](crate::Arc) backed by an atomic refcount.
    ///
    /// Construct via [`Self::new`] / [`Self::try_new`], initialize via
    /// [`Self::as_mut_ptr`], then call [`Self::finalize`] (or
    /// [`Self::finalize_dst`] for true DSTs) to obtain an
    /// [`Arc<T, A>`](crate::Arc). Forgetting to call
    /// `finalize` decrements the chunk refcount; the reserved bytes are
    /// leaked until chunk teardown but never dropped.
    PendingArc,
    handle = [crate::Arc],
    sharing = Shared,
    on_owner_thread = true,
    debug_name = "PendingArc",
    kind = sized_and_dst,
);
