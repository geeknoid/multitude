//! Support for creating dynamically-sized types in an [`Arena`](crate::Arena).

mod helpers;
mod pending_arc;
mod pending_box;
mod pending_rc;
#[macro_use]
pub(crate) mod pending_macro;

pub use pending_arc::PendingArc;
pub use pending_box::PendingBox;
pub use pending_rc::PendingRc;

pub use crate::drop_entry::DropEntry;
