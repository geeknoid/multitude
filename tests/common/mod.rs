//! Shared test helpers. Each integration test file includes this via
//! `mod common;` and uses items as `common::*`.

#![allow(dead_code, reason = "shared between multiple test binaries; some helpers may be unused per-file")]

use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub fn hash_of<T: Hash>(v: &T) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}
