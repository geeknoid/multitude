//! Optional Serde support, gated on the `serde` feature.
//!
//! We provide `Serialize` only — not `Deserialize`. Deserializing into
//! arena-backed types requires an arena context that serde's stock
//! `Deserialize` trait cannot carry. Users wanting deserialization should
//! write a `DeserializeSeed` impl that captures `&Arena<A>`.

use allocator_api2::alloc::Allocator;
use serde::ser::{Serialize, SerializeSeq, Serializer};

use crate::arena_arc_str::ArenaArcStr;
use crate::arena_rc_str::ArenaRcStr;
use crate::arena_string::ArenaString;
use crate::arena_vec::ArenaVec;

impl<A: Allocator + Clone> Serialize for ArenaRcStr<A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<A: Allocator + Clone> Serialize for ArenaArcStr<A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<A: Allocator + Clone> Serialize for ArenaString<'_, A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<T: Serialize, A: Allocator + Clone> Serialize for ArenaVec<'_, T, A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let slice = self.as_slice();
        let mut seq = serializer.serialize_seq(Some(slice.len()))?;
        for elem in slice {
            seq.serialize_element(elem)?;
        }
        seq.end()
    }
}
