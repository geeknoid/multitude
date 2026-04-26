//! Benchmarks comparing harena2 vs. bumpalo vs. the global allocator.
//!
//! Run with: `cargo bench --bench alloc_compare`

#![allow(clippy::unwrap_used, reason = "benchmark code")]
#![allow(clippy::missing_panics_doc, reason = "benchmark code")]
#![allow(clippy::clone_on_ref_ptr, reason = "we want explicit method calls")]
#![allow(deprecated, reason = "criterion::black_box is deprecated in favor of std::hint::black_box")]
#![allow(unused_results, reason = "benchmark code")]
#![allow(clippy::similar_names, reason = "intentional test-local names")]
#![allow(clippy::std_instead_of_core, reason = "benchmark code")]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const N: usize = 10_000;

fn bench_alloc_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_u64");

    group.bench_function("global_box", |b| {
        b.iter(|| {
            let mut handles = Vec::with_capacity(N);
            for i in 0..N {
                handles.push(Box::new(black_box(i as u64)));
            }
            black_box(handles);
        });
    });

    group.bench_function("bumpalo", |b| {
        b.iter(|| {
            let bump = bumpalo::Bump::new();
            let mut refs = Vec::with_capacity(N);
            for i in 0..N {
                refs.push(bump.alloc(black_box(i as u64)));
            }
            black_box(refs);
            black_box(bump);
        });
    });

    group.bench_function("harena2_arena_rc", |b| {
        b.iter(|| {
            let arena = harena::Arena::new();
            let mut refs = Vec::with_capacity(N);
            for i in 0..N {
                refs.push(arena.alloc(black_box(i as u64)));
            }
            black_box(refs);
            black_box(arena);
        });
    });

    group.bench_function("harena2_arena_box", |b| {
        b.iter(|| {
            let arena = harena::Arena::new();
            let mut refs = Vec::with_capacity(N);
            for i in 0..N {
                refs.push(arena.alloc_box(black_box(i as u64)));
            }
            black_box(refs);
        });
    });

    group.finish();
}

fn bench_alloc_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_string");

    group.bench_function("global_box", |b| {
        b.iter(|| {
            let mut handles = Vec::with_capacity(N);
            for i in 0..N {
                handles.push(Box::new(format!("item {i}")));
            }
            black_box(handles);
        });
    });

    group.bench_function("bumpalo", |b| {
        b.iter(|| {
            let bump = bumpalo::Bump::new();
            let mut refs = Vec::with_capacity(N);
            for i in 0..N {
                refs.push(bump.alloc(format!("item {i}")));
            }
            black_box(refs);
            black_box(bump);
        });
    });

    group.bench_function("harena2", |b| {
        b.iter(|| {
            let arena = harena::Arena::new();
            let mut refs = Vec::with_capacity(N);
            for i in 0..N {
                refs.push(arena.alloc(format!("item {i}")));
            }
            black_box(refs);
            black_box(arena);
        });
    });

    group.finish();
}

fn bench_strings(c: &mut Criterion) {
    let mut group = c.benchmark_group("strings");
    let words: Vec<String> = (0..N).map(|i| format!("word{i}")).collect();

    group.bench_function("global_box_str", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(N);
            for w in &words {
                out.push(w.clone().into_boxed_str());
            }
            black_box(out);
        });
    });

    group.bench_function("bumpalo_str", |b| {
        b.iter(|| {
            let bump = bumpalo::Bump::new();
            let mut out = Vec::with_capacity(N);
            for w in &words {
                out.push(bump.alloc_str(w));
            }
            black_box(out);
            black_box(bump);
        });
    });

    group.bench_function("harena2_arena_str", |b| {
        b.iter(|| {
            let arena = harena::Arena::new();
            let mut out = Vec::with_capacity(N);
            for w in &words {
                out.push(harena::ArenaRcStr::from_str(&arena, w));
            }
            black_box(out);
            black_box(arena);
        });
    });

    group.finish();
}

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone");

    let std_arc = std::sync::Arc::new(42_u64);
    group.bench_function("std_arc_clone", |b| {
        b.iter(|| {
            for _ in 0..N {
                black_box(std_arc.clone());
            }
        });
    });

    let std_rc = std::rc::Rc::new(42_u64);
    group.bench_function("std_rc_clone", |b| {
        b.iter(|| {
            for _ in 0..N {
                black_box(std_rc.clone());
            }
        });
    });

    let arena = harena::Arena::new();
    let h_rc = arena.alloc(42_u64);
    group.bench_function("harena_rc_clone", |b| {
        b.iter(|| {
            for _ in 0..N {
                black_box(h_rc.clone());
            }
        });
    });

    let h_arc = arena.alloc_shared(42_u64);
    group.bench_function("harena_arc_clone", |b| {
        b.iter(|| {
            for _ in 0..N {
                black_box(h_arc.clone());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_alloc_u64, bench_alloc_string, bench_strings, bench_clone);
criterion_main!(benches);
