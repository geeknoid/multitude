#!/usr/bin/env bash
# Run multitude's Loom model-checked tests.
#
# Loom is a permutation tester: it explores every legal interleaving of the
# atomic operations the C++/Rust memory model allows. This catches memory-
# ordering bugs that hardware-stressing won't reproduce reliably.
#
# Cost:
#   - Each test is bounded by Loom's `max_branches` budget (default 1000).
#     Small tests (≤ 3 threads, ≤ 5 atomic ops/thread) finish in milliseconds.
#   - Larger scenarios can take seconds to minutes because the state space
#     is exponential in the number of yield points.
#
# Usage:
#
#     scripts/loom.sh                    # Run all loom_* test crates.
#     scripts/loom.sh arc_clone_drop_race  # Run a single test.
#
# Tuning:
#
#     LOOM_MAX_PREEMPTIONS=3   # Bound how many context switches Loom tries.
#                              # Smaller = faster, less coverage. Default 2.
#     LOOM_LOG=info            # Print per-iteration thread schedules.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# `--release` makes Loom's exploration substantially faster because the
# instrumented atomic primitives are themselves expensive in debug mode.
RUSTFLAGS="${RUSTFLAGS:-}--cfg loom" \
    cargo test --release --tests \
    -- "$@" --test-threads=1
