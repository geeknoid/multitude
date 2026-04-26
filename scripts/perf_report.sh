#!/usr/bin/env bash
# Run criterion + gungraun benchmark suites and rebuild PERF.md.
#
# Requirements:
#   - Linux host with valgrind installed (for gungraun / Callgrind).
#   - python3 on PATH (for the report builder).
#
# Usage:
#   scripts/perf_report.sh                    # full run (30 samples, 2s measurement)
#   FAST=1 scripts/perf_report.sh             # quick run (10 samples, 1s)
#   SAMPLES=50 MEAS=3 scripts/perf_report.sh  # custom criterion settings

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v valgrind >/dev/null 2>&1; then
    echo "error: valgrind is required for the gungraun benchmarks" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to build the report" >&2
    exit 1
fi

if [[ "${FAST:-0}" == "1" ]]; then
    SAMPLES="${SAMPLES:-10}"
    MEAS="${MEAS:-1}"
    WARMUP="${WARMUP:-1}"
else
    SAMPLES="${SAMPLES:-30}"
    MEAS="${MEAS:-2}"
    WARMUP="${WARMUP:-1}"
fi

artifacts_dir="$(mktemp -d -t multitude-perf-XXXXXX)"
trap 'rm -rf "$artifacts_dir"' EXIT

crit_alloc_log="$artifacts_dir/criterion_alloc.log"
crit_drop_log="$artifacts_dir/criterion_drop.log"
gung_alloc_log="$artifacts_dir/gungraun_alloc.log"
gung_drop_log="$artifacts_dir/gungraun_drop.log"

echo "==> Running criterion_alloc: ${SAMPLES} samples, ${MEAS}s measurement"
cargo bench --bench criterion_alloc -- \
    --warm-up-time "$WARMUP" \
    --measurement-time "$MEAS" \
    --sample-size "$SAMPLES" \
    > "$crit_alloc_log" 2>&1

echo "==> Running criterion_drop: ${SAMPLES} samples, ${MEAS}s measurement"
cargo bench --bench criterion_drop -- \
    --warm-up-time "$WARMUP" \
    --measurement-time "$MEAS" \
    --sample-size "$SAMPLES" \
    > "$crit_drop_log" 2>&1

echo "==> Running gungraun_alloc"
cargo bench --bench gungraun_alloc > "$gung_alloc_log" 2>&1

echo "==> Running gungraun_drop"
cargo bench --bench gungraun_drop > "$gung_drop_log" 2>&1

echo "==> Building PERF.md"
python3 "$repo_root/scripts/build_perf_report.py" \
    --criterion-alloc "$crit_alloc_log" \
    --criterion-drop "$crit_drop_log" \
    --gungraun-alloc "$gung_alloc_log" \
    --gungraun-drop "$gung_drop_log" \
    --output "$repo_root/PERF.md"

echo "==> Done. Report written to PERF.md"
