#!/usr/bin/env python3
"""Build PERF.md from criterion + gungraun benchmark logs.

Used by `scripts/perf_report.sh`. Both bench suites must be aligned 1:1: each
criterion `<group>/<variant>` corresponds to a gungraun `<group>_<variant>`.
The variant order below mirrors the order benches are defined in
`benches/alloc_compare.rs` and `benches/gungraun_*.rs`; if a bench is added
or removed, update this file to match.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ordered (group, variants) — must match the criterion bench order.
# Each entry is (criterion_group, [(criterion_variant, gungraun_func_name_or_None), ...]).
# The criterion variant name is the string passed to `g.bench_function(...)` in
# `benches/criterion_*.rs`; the gungraun function name is the `fn` name in
# `benches/gungraun_*.rs` (its `library_benchmark` symbol). `None` means the
# variant has no gungraun counterpart and the columns will show "—".
GROUPS: list[tuple[str, list[tuple[str, str | None]]]] = [
    ("arena_creation", [
        ("multitude", "arena_creation_multitude"),
        ("bumpalo", "arena_creation_bumpalo"),
    ]),
    ("alloc_u64", [
        ("alloc", "alloc"),
        ("alloc_with", "alloc_with"),
        ("alloc_box", "alloc_box"),
        ("alloc_box_with", "alloc_box_with"),
        ("alloc_uninit_box", "alloc_uninit_box"),
        ("alloc_zeroed_box", "alloc_zeroed_box"),
        ("alloc_rc", "alloc_rc"),
        ("alloc_rc_with", "alloc_rc_with"),
        ("alloc_uninit_rc", "alloc_uninit_rc"),
        ("alloc_zeroed_rc", "alloc_zeroed_rc"),
        ("alloc_arc", "alloc_arc"),
        ("alloc_arc_with", "alloc_arc_with"),
        ("alloc_uninit_arc", "alloc_uninit_arc"),
        ("alloc_zeroed_arc", "alloc_zeroed_arc"),
        ("bumpalo", "alloc_u64_bumpalo"),
        ("bumpalo_with", "alloc_u64_bumpalo_with"),
    ]),
    ("alloc_str", [
        ("alloc_str", "alloc_str"),
        ("alloc_str_box", "alloc_str_box"),
        ("alloc_str_rc", "alloc_str_rc"),
        ("alloc_str_arc", "alloc_str_arc"),
        ("bumpalo", "alloc_str_bumpalo"),
    ]),
    ("alloc_slice", [
        ("alloc_slice_copy", "alloc_slice_copy"),
        ("alloc_slice_clone", "alloc_slice_clone"),
        ("alloc_slice_fill_with", "alloc_slice_fill_with"),
        ("alloc_slice_fill_iter", "alloc_slice_fill_iter"),
        ("alloc_slice_copy_box", "alloc_slice_copy_box"),
        ("alloc_slice_clone_box", "alloc_slice_clone_box"),
        ("alloc_slice_fill_with_box", "alloc_slice_fill_with_box"),
        ("alloc_slice_fill_iter_box", "alloc_slice_fill_iter_box"),
        ("alloc_uninit_slice_box", "alloc_uninit_slice_box"),
        ("alloc_zeroed_slice_box", "alloc_zeroed_slice_box"),
        ("alloc_slice_copy_rc", "alloc_slice_copy_rc"),
        ("alloc_slice_clone_rc", "alloc_slice_clone_rc"),
        ("alloc_slice_fill_with_rc", "alloc_slice_fill_with_rc"),
        ("alloc_slice_fill_iter_rc", "alloc_slice_fill_iter_rc"),
        ("alloc_uninit_slice_rc", "alloc_uninit_slice_rc"),
        ("alloc_zeroed_slice_rc", "alloc_zeroed_slice_rc"),
        ("alloc_slice_copy_arc", "alloc_slice_copy_arc"),
        ("alloc_slice_clone_arc", "alloc_slice_clone_arc"),
        ("alloc_slice_fill_with_arc", "alloc_slice_fill_with_arc"),
        ("alloc_slice_fill_iter_arc", "alloc_slice_fill_iter_arc"),
        ("alloc_uninit_slice_arc", "alloc_uninit_slice_arc"),
        ("alloc_zeroed_slice_arc", "alloc_zeroed_slice_arc"),
        ("bumpalo_copy", "alloc_slice_bumpalo_copy"),
        ("bumpalo_clone", "alloc_slice_bumpalo_clone"),
        ("bumpalo_fill_with", "alloc_slice_bumpalo_fill_with"),
        ("bumpalo_fill_iter", "alloc_slice_bumpalo_fill_iter"),
    ]),
    ("string_builder", [
        ("alloc_string", "alloc_string"),
        ("alloc_string_with_capacity", "alloc_string_with_capacity"),
        ("bumpalo_grow", "string_builder_bumpalo_grow"),
        ("bumpalo_with_cap", "string_builder_bumpalo_with_cap"),
    ]),
    ("vec_builder", [
        ("alloc_vec", "alloc_vec"),
        ("alloc_vec_with_capacity", "alloc_vec_with_capacity"),
        ("bumpalo_grow", "vec_builder_bumpalo_grow"),
        ("bumpalo_with_cap", "vec_builder_bumpalo_with_cap"),
    ]),
    ("drop", [
        ("box_u64", None),
        ("rc_u64", None),
        ("arc_u64", None),
        ("box_droppy", "drop_box_droppy"),
        ("rc_droppy", "drop_rc_droppy"),
        ("arc_droppy", "drop_arc_droppy"),
        ("str_box", "drop_str_box"),
        ("str_rc", "drop_str_rc"),
        ("str_arc", "drop_str_arc"),
        ("slice_box_u64", None),
        ("slice_rc_u64", None),
        ("slice_arc_u64", None),
        ("slice_box_droppy", "drop_slice_box_droppy"),
        ("slice_rc_droppy", "drop_slice_rc_droppy"),
        ("slice_arc_droppy", "drop_slice_arc_droppy"),
        ("alloc", "drop_alloc"),
    ]),
]

UNIT_TO_NS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}

TIME_RE = re.compile(
    r"time:\s+\[([\d.]+)\s+(\w+)\s+([\d.]+)\s+(\w+)\s+([\d.]+)\s+(\w+)\]"
)


def parse_criterion(path: Path, expected: list[tuple[str, str]]) -> dict[str, float]:
    """Parse criterion log and return {group/variant: median_ns}.

    In non-TTY mode criterion only writes the per-bench `time:` summary
    line, in execution order. We zip those lines against the expected
    bench list in order.
    """
    text = path.read_text()
    medians: list[float] = []
    for line in text.splitlines():
        m = TIME_RE.search(line)
        if m:
            medians.append(float(m.group(3)) * UNIT_TO_NS[m.group(4)])

    if len(medians) != len(expected):
        print(
            f"warning: {path.name} has {len(medians)} time entries, "
            f"expected {len(expected)}",
            file=sys.stderr,
        )
    return {f"{g}/{v}": t for (g, v), t in zip(expected, medians)}


GUNG_NAME_RE_TMPL = r"^{prefix}::\w+::([\w]+) run"
GUNG_METRIC_RE = re.compile(
    r"^\s+(Instructions|L1 Hits|LL Hits|RAM Hits|Bcm):\s+(\d+)\|"
)


def parse_gungraun(path: Path, prefix: str) -> dict[str, dict[str, int]]:
    text = path.read_text()
    name_re = re.compile(GUNG_NAME_RE_TMPL.format(prefix=prefix))
    out: dict[str, dict[str, int]] = {}
    cur: str | None = None
    metrics: dict[str, int] = {}
    for line in text.splitlines():
        m = name_re.match(line)
        if m:
            if cur is not None:
                out[cur] = metrics
            cur = m.group(1)
            metrics = {}
            continue
        m = GUNG_METRIC_RE.match(line)
        if m and cur is not None:
            metrics[m.group(1)] = int(m.group(2))
    if cur is not None:
        out[cur] = metrics
    return out


def fmt_ns(ns: float | None) -> str:
    if ns is None:
        return "—"
    if ns < 1000:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    return f"{ns / 1e6:.2f} ms"


def fmt_int(n: int | None) -> str:
    return "—" if n is None else f"{n:,}"


COMPARISONS = [
    ("alloc_u64", "alloc", "bumpalo"),
    ("alloc_str", "alloc_str", "bumpalo"),
    ("alloc_slice", "alloc_slice_copy", "bumpalo_copy"),
    ("alloc_slice", "alloc_slice_clone", "bumpalo_clone"),
    ("alloc_slice", "alloc_slice_fill_with", "bumpalo_fill_with"),
    ("alloc_slice", "alloc_slice_fill_iter", "bumpalo_fill_iter"),
    ("string_builder", "alloc_string", "bumpalo_grow"),
    ("string_builder", "alloc_string_with_capacity", "bumpalo_with_cap"),
    ("vec_builder", "alloc_vec", "bumpalo_grow"),
    ("vec_builder", "alloc_vec_with_capacity", "bumpalo_with_cap"),
]


def build_report(
    crit: dict[str, float],
    g_alloc: dict[str, dict[str, int]],
    g_drop: dict[str, dict[str, int]],
) -> str:
    out: list[str] = []
    out.append("# Multitude Performance Report\n\n")
    out.append("Generated by `scripts/perf_report.sh`:\n")
    out.append(
        "- `cargo bench --bench alloc_compare` — criterion wall-clock timings.\n"
    )
    out.append(
        "- `cargo bench --bench gungraun_alloc` and `gungraun_drop` — "
        "Callgrind instruction-precise counts.\n\n"
    )
    out.append(
        "**Workload:** N = 1000 operations per measurement; slice element count = 8.  \n"
    )
    out.append(
        "Criterion median is reported (default 30 samples, 1 s warm-up, "
        "2 s measurement; override with `SAMPLES=` / `MEAS=`).  \n"
    )
    out.append(
        "Memory accesses = L1 Hits + LL Hits + RAM Hits "
        "(Callgrind D-cache references).  \n"
    )
    out.append(
        "Bench names are aligned between criterion and gungraun via the "
        "`GROUPS` table in `scripts/build_perf_report.py`.\n\n"
    )

    for group, variants in GROUPS:
        out.append(f"## `{group}`\n\n")
        out.append(
            "| Variant | Time (criterion) | Instructions | "
            "Branch misses | Mem accesses |\n"
        )
        out.append("|---|---:|---:|---:|---:|\n")
        src = g_drop if group == "drop" else g_alloc
        for variant, gung_name in variants:
            t = crit.get(f"{group}/{variant}")
            gm = src.get(gung_name, {}) if gung_name else {}
            instr = gm.get("Instructions")
            bcm = gm.get("Bcm")
            mem = None
            if all(k in gm for k in ("L1 Hits", "LL Hits", "RAM Hits")):
                mem = gm["L1 Hits"] + gm["LL Hits"] + gm["RAM Hits"]
            out.append(
                f"| `{variant}` | {fmt_ns(t)} | {fmt_int(instr)} | "
                f"{fmt_int(bcm)} | {fmt_int(mem)} |\n"
            )
        out.append("\n")

    out.append("## Multitude vs Bumpalo Head-to-Head\n\n")
    out.append(
        "Direct comparisons of multitude versus bumpalo on identical "
        "workloads (the multitude variant chosen is the closest "
        "semantic equivalent to bumpalo's plain bump-allocation).\n\n"
    )
    out.append(
        "| Workload | Multitude time | Bumpalo time | Δ time | "
        "Multitude instr | Bumpalo instr | Δ instr |\n"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for group, mvar, bvar in COMPARISONS:
        mt = crit.get(f"{group}/{mvar}")
        bt = crit.get(f"{group}/{bvar}")
        mi = _gung_for(group, mvar, g_alloc)
        bi = _gung_for(group, bvar, g_alloc)
        dt = f"{(mt / bt - 1) * 100:+.1f}%" if mt and bt else "—"
        di = f"{(mi / bi - 1) * 100:+.1f}%" if mi and bi else "—"
        out.append(
            f"| `{group}/{mvar}` vs `{bvar}` | {fmt_ns(mt)} | {fmt_ns(bt)} | {dt} | "
            f"{fmt_int(mi)} | {fmt_int(bi)} | {di} |\n"
        )
    out.append("\n")
    return "".join(out)


def _gung_for(group: str, variant: str, g_alloc: dict[str, dict[str, int]]) -> int | None:
    """Look up Instructions for (group, variant) via the GROUPS mapping."""
    for g, vs in GROUPS:
        if g != group:
            continue
        for v, gname in vs:
            if v == variant and gname is not None:
                return g_alloc.get(gname, {}).get("Instructions")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion-alloc", required=True, type=Path)
    parser.add_argument("--criterion-drop", required=True, type=Path)
    parser.add_argument("--gungraun-alloc", required=True, type=Path)
    parser.add_argument("--gungraun-drop", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    alloc_groups = [g for g in GROUPS if g[0] != "drop"]
    drop_groups = [g for g in GROUPS if g[0] == "drop"]
    alloc_keys = [(g, v) for g, vs in alloc_groups for (v, _) in vs]
    drop_keys = [(g, v) for g, vs in drop_groups for (v, _) in vs]

    crit = parse_criterion(args.criterion_alloc, alloc_keys)
    crit.update(parse_criterion(args.criterion_drop, drop_keys))
    g_alloc = parse_gungraun(args.gungraun_alloc, "gungraun_alloc")
    g_drop = parse_gungraun(args.gungraun_drop, "gungraun_drop")
    args.output.write_text(build_report(crit, g_alloc, g_drop))
    print(
        f"Wrote {args.output} "
        f"({len(crit)} criterion, {len(g_alloc)} gungraun_alloc, "
        f"{len(g_drop)} gungraun_drop benches)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
