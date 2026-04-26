//! Arena-backed strings: the various flavors and their use cases.

#![allow(clippy::unwrap_used, reason = "example code")]
#![allow(clippy::missing_panics_doc, reason = "example code")]
#![allow(clippy::std_instead_of_core, reason = "example uses std::thread/HashMap")]
#![allow(clippy::items_after_statements, reason = "example uses inline imports")]

use harena::{Arena, ArenaArcStr, ArenaRcStr};

fn main() {
    let arena = Arena::new();

    let s1: ArenaRcStr = ArenaRcStr::from_str(&arena, "Hello, world!");
    println!("ArenaRcStr: {} (len={})", &*s1, s1.len());

    let mut builder = arena.new_string();
    builder.push_str("built ");
    builder.push_str("up ");
    builder.push_str("incrementally");
    println!("ArenaString (mutable): {}", builder.as_str());
    let s2: ArenaRcStr = builder.into_arena_str();
    println!("frozen: {}", &*s2);

    let name = "Alice";
    let s3 = harena::format!(in &arena, "format! says hi, {name}");
    println!("format!: {}", &*s3);

    let shared: ArenaArcStr = ArenaArcStr::from_str(&arena, "shared across threads");
    let cloned = shared.clone();
    let h = std::thread::spawn(move || cloned.len());
    println!("ArenaArcStr: {} (len from thread: {})", &*shared, h.join().unwrap());

    use std::collections::HashMap;
    let mut counts: HashMap<ArenaRcStr, i32> = HashMap::new();
    let _ = counts.insert(ArenaRcStr::from_str(&arena, "alpha"), 1);
    let _ = counts.insert(ArenaRcStr::from_str(&arena, "beta"), 2);
    println!("map alpha: {:?}", counts.get("alpha"));
}
