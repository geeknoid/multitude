//! Basic usage of `harena::Arena` and the various handle types.

#![allow(clippy::unwrap_used, reason = "example code")]
#![allow(clippy::missing_panics_doc, reason = "example code")]
#![allow(clippy::std_instead_of_core, reason = "example uses std::thread")]

use harena::{Arena, ArenaArc, ArenaBox, ArenaRc, ArenaRcStr, ArenaVec, CollectIn};

fn main() {
    let arena = Arena::new();

    // -- Reference-counted local handle --------------------------------
    let a: ArenaRc<u32> = arena.alloc(42);
    let b = a.clone();
    println!("a = {}, b = {}, ptr_eq = {}", *a, *b, ArenaRc::ptr_eq(&a, &b));

    // -- Owned single handle (mutable, immediate Drop) ----------------
    let mut owned: ArenaBox<Vec<i32>> = arena.alloc_box(vec![1, 2, 3]);
    owned.push(4);
    println!("owned = {:?}", *owned);
    drop(owned);

    // -- Single-pointer immutable string handle -----------------------
    let name: ArenaRcStr = ArenaRcStr::from_str(&arena, "Alice");
    println!("name = {} (len = {})", &*name, name.len());

    // -- format! macro returning ArenaRcStr -----------------------------
    let greeting = harena::format!(in &arena, "Hello, {name}!");
    println!("greeting = {}", &*greeting);

    // -- Arena-backed vector ------------------------------------------
    let mut v: ArenaVec<u64, _> = arena.new_vec();
    for i in 0..5 {
        v.push(i * 100);
    }
    println!("v = {v:?}");
    let frozen: ArenaRc<[u64], _> = v.into_arena_rc();
    println!("frozen = {frozen:?}");

    // -- Collect from an iterator -------------------------------------
    let squares: ArenaVec<i64, _> = (1..=5).map(|i| i * i).collect_in(&arena);
    println!("squares = {squares:?}");

    // -- Cross-thread sharing via ArenaArc ----------------------------
    let shared: ArenaArc<u64> = arena.alloc_shared(99);
    let s2 = shared.clone();
    let h = std::thread::spawn(move || *s2);
    println!("from main: {}, from thread: {}", *shared, h.join().unwrap());

    println!("arena stats: {:?}", arena.stats());
}
