#!/usr/bin/env python3
"""Aggregate the sampling-mode sweep into a comparison table.

Auto-discovers every scenario from <results_dir>/<label>.meta.txt, averages the
per-tree buckets, pulls row counts from the DEDUP line, and prints:
  - a per-scenario table (any scenarios present)
  - named comparison pairs (ratio num/den) for every pair where both sides exist
"""
import csv
import glob
import os
import re
import sys

BUCKETS = [
    "projection_ms", "split_finding_ms", "example_mgmt_ms", "partition_ms",
    "compute_offsets_ms", "overhead_ms", "tree_total_ms",
]

# num, den, note. Printed only when both scenarios are present in results/.
COMPARISONS = [
    ("A_repl", "A_noboot",
     "nid: replacement(~0.632N) vs no-replacement(N) — pure row-count effect"),
    ("A_wor063", "A_repl",
     "nid: without-repl(0.632N, no weights) vs with-repl(0.632N, weighted) — weighting/dedup cost at EQUAL rows"),
    ("A_wor1", "A_noboot",
     "nid: without-repl ratio=1 vs no-bootstrap — sanity, should be ~identical"),
    ("B_repl", "B_noboot",
     "SRDW dedup: replacement(0.632N) vs no-replacement(N)"),
    ("B_wor063", "B_repl",
     "SRDW dedup: without-repl(0.632N) vs with-repl(0.632N) — weighting cost at EQUAL rows"),
    ("C_repl", "B_repl",
     "SRDW: dedup OFF(N w/ repeats) vs dedup ON(0.632N unique) at replacement"),
    ("A_repl", "B_repl",
     "replacement: node_id_drsw path vs SRDW baseline"),
]


def read_pertree(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return None
    out = {}
    for b in BUCKETS:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[b]))
            except (KeyError, ValueError):
                pass
        out[b] = sum(vals) / len(vals) if vals else float("nan")
    out["_n_trees"] = len(rows)
    return out


def read_dedup(path):
    if not os.path.exists(path):
        return None
    pat = re.compile(r"before=(\d+)\s+after=(\d+)\s+dedup=([\d.]+)ms")
    b, a, m = [], [], []
    for line in open(path):
        g = pat.search(line)
        if g:
            b.append(int(g.group(1))); a.append(int(g.group(2))); m.append(float(g.group(3)))
    if not b:
        return None
    n = len(b)
    return {"before": sum(b) / n, "after": sum(a) / n, "dedup_ms": sum(m) / n}


def read_meta(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
    return d


def discover(results_dir):
    labels = sorted(
        os.path.basename(p)[: -len(".meta.txt")]
        for p in glob.glob(os.path.join(results_dir, "*.meta.txt"))
    )
    data = {}
    for lab in labels:
        data[lab] = {
            "pt": read_pertree(os.path.join(results_dir, f"{lab}.pertree.csv")),
            "dd": read_dedup(os.path.join(results_dir, f"{lab}.dedup.txt")),
            "meta": read_meta(os.path.join(results_dir, f"{lab}.meta.txt")),
        }
    return labels, data


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "sweeps/results"
    labels, data = discover(results_dir)
    if not labels:
        print(f"no scenarios found in {results_dir}"); return

    print("\n================ per-scenario (mean over trees) ================\n")
    hdr = (f"{'scenario':14} {'boot/swr/ratio':16} {'rows(after)':>12} {'dedup_ms':>9} "
           + " ".join(f"{b.replace('_ms',''):>12}" for b in BUCKETS))
    print(hdr); print("-" * len(hdr))
    for lab in labels:
        pt, dd, meta = data[lab]["pt"], data[lab]["dd"], data[lab]["meta"]
        cfg = f"{meta.get('bootstrap','?')[:1]}/{meta.get('swr','?')[:1]}/{meta.get('ratio','?')}"
        if pt is None:
            rc = meta.get("rc", "?")
            print(f"{lab:14} {cfg:16}  <no data — run rc={rc} / header-only CSV>")
            continue
        rows_after = str(int(dd["after"])) if dd else "n/a(dedup off)"
        dedup_ms = f"{dd['dedup_ms']:.2f}" if dd else "-"
        print(f"{lab:14} {cfg:16} {rows_after:>12} {dedup_ms:>9} "
              + " ".join(f"{pt[b]:12.2f}" for b in BUCKETS))

    print("\n================ comparisons (ratio = num / den) ================")
    any_pair = False
    for num, den, note in COMPARISONS:
        if num not in data or den not in data:
            continue
        pt_n, pt_d = data[num]["pt"], data[den]["pt"]
        if pt_n is None or pt_d is None:
            continue
        any_pair = True
        dd_n, dd_d = data[num]["dd"], data[den]["dd"]
        ra_n = int(dd_n["after"]) if dd_n else "n/a"
        ra_d = int(dd_d["after"]) if dd_d else "n/a"
        print(f"\n  [{num}] / [{den}]")
        print(f"    {note}")
        print(f"    rows: {num}={ra_n}  {den}={ra_d}")
        for b in BUCKETS:
            r = pt_n[b] / pt_d[b] if pt_d[b] else float("nan")
            print(f"    {b:18} {pt_n[b]:10.2f} {pt_d[b]:10.2f}   ratio {r:6.3f}")
    if not any_pair:
        print("  (no comparison pair has both sides present in this run)")
    print()


if __name__ == "__main__":
    main()
