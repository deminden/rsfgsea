#!/usr/bin/env python3
"""Compare native Rust blitz speed/parity against Python blitzgsea.

This is an opt-in local benchmark harness. It prefers the copied DESeq2 positive
rank files under data/deseq2_positive_ranks and falls back only when the caller
provides explicit paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKS = REPO_ROOT / "data/deseq2_positive_ranks/lung_vs_muscle.rnk"
DEFAULT_GMT = REPO_ROOT / "data/GO_Biological_Process_2025.gmt"
DEFAULT_RUST_BIN = REPO_ROOT / "target/release/rsfgsea"
DEFAULT_PYTHON = Path("/home/den/miniforge3/bin/python")


PYTHON_BLITZ_CODE = r"""
import json
import math
import sys
import time
from pathlib import Path

import blitzgsea
import pandas as pd

ranks_path = Path(sys.argv[1])
gmt_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
warm = sys.argv[4] == "1"

signature = pd.read_csv(ranks_path, sep=r"\s+", header=None, names=["i", "v"])
library = {}
with gmt_path.open(encoding="utf-8") as handle:
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            library[parts[0]] = parts[2:]

def run_once(signature_cache):
    start = time.perf_counter()
    res = blitzgsea.gsea(
        signature,
        library,
        permutations=1000,
        anchors=40,
        min_size=5,
        max_size=4000,
        processes=4,
        plotting=False,
        verbose=False,
        progress=False,
        symmetric=False,
        signature_cache=signature_cache,
        seed=0,
        accuracy=40,
        deep_accuracy=50,
        center=True,
    )
    return time.perf_counter() - start, res

if warm:
    run_once(True)
    elapsed, result = run_once(True)
else:
    elapsed, result = run_once(False)

out = result.reset_index().rename(
    columns={
        "Term": "pathway",
        "fdr": "padj",
        "geneset_size": "size",
    }
)
for name in ["pathway", "size", "es", "nes", "pval", "padj", "leading_edge"]:
    if name not in out.columns:
        out[name] = ""
out[["pathway", "size", "es", "nes", "pval", "padj", "leading_edge"]].to_csv(
    out_path, sep="\t", index=False
)
print(json.dumps({"elapsed_s": elapsed, "rows": int(len(out))}))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="lung_vs_muscle_go_bp")
    parser.add_argument("--ranks", type=Path, default=DEFAULT_RANKS)
    parser.add_argument("--gmt", type=Path, default=DEFAULT_GMT)
    parser.add_argument("--rust-bin", type=Path, default=DEFAULT_RUST_BIN)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--build", action="store_true", help="cargo build --release -p rsfgsea first")
    parser.add_argument("--json", action="store_true", help="print JSON instead of TSV")
    parser.add_argument("--skip-python", action="store_true")
    return parser.parse_args()


def ensure_inputs(args: argparse.Namespace) -> None:
    for label, path in [("ranks", args.ranks), ("gmt", args.gmt)]:
        if not path.exists():
            raise SystemExit(f"{label} file not found: {path}")
    if args.build:
        subprocess.run(
            ["cargo", "build", "--release", "-p", "rsfgsea"],
            cwd=REPO_ROOT,
            check=True,
        )
    if not args.rust_bin.exists():
        raise SystemExit(f"Rust binary not found: {args.rust_bin}; pass --build or run cargo build --release -p rsfgsea")


def count_rank_genes(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_pathways(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def run_rust(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    env = os.environ.copy()
    start = time.perf_counter()
    proc = subprocess.run(
        [
            str(args.rust_bin),
            "--ranks",
            str(args.ranks),
            "--gmt",
            str(args.gmt),
            "--output",
            str(output),
            "--mode",
            "blitz",
            "--nPermSimple",
            "1000",
            "--minSize",
            "5",
            "--maxSize",
            "4000",
            "--seed",
            "0",
            "--nproc",
            "4",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    wall = time.perf_counter() - start
    match = re.search(r"GSEA_COMP_TIME_MS:\s*(\d+)", proc.stdout)
    compute_s = float(match.group(1)) / 1000.0 if match else math.nan
    return {"wall_s": wall, "compute_s": compute_s}


def run_python(args: argparse.Namespace, output: Path, *, warm: bool) -> dict[str, Any]:
    if not args.python.exists():
        return {"elapsed_s": math.nan, "rows": 0, "error": f"missing python: {args.python}"}
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(
        [
            str(args.python),
            "-c",
            PYTHON_BLITZ_CODE,
            str(args.ranks),
            str(args.gmt),
            str(output),
            "1" if warm else "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return {
            "elapsed_s": math.nan,
            "rows": 0,
            "error": (proc.stderr or proc.stdout).strip().splitlines()[-1],
        }
    return json.loads(proc.stdout)


def read_result(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pathway = row.get("pathway") or row.get("Term")
            if not pathway:
                continue
            rows[pathway] = row
    return rows


def parse_float(value: Any) -> float:
    if value is None or value == "" or value == "NA":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def parity_summary(rust_path: Path, python_path: Path) -> dict[str, Any]:
    rust = read_result(rust_path)
    python = read_result(python_path)
    shared = sorted(set(rust) & set(python))
    summary: dict[str, Any] = {
        "rust_rows": len(rust),
        "python_rows": len(python),
        "shared_rows": len(shared),
        "missing_in_rust": len(set(python) - set(rust)),
        "missing_in_python": len(set(rust) - set(python)),
        "size_mismatches": 0,
        "leading_edge_order_mismatches": 0,
        "leading_edge_set_mismatches": 0,
    }
    for field in ["es", "nes", "pval", "padj"]:
        summary[f"max_abs_diff_{field}"] = 0.0
        summary[f"max_rel_diff_{field}"] = 0.0
    for pathway in shared:
        r = rust[pathway]
        p = python[pathway]
        if str(r.get("size", "")) != str(p.get("size", "")):
            summary["size_mismatches"] += 1
        r_le = [x for x in str(r.get("leading_edge", "")).split(",") if x]
        p_le = [x for x in str(p.get("leading_edge", "")).replace(";", ",").split(",") if x]
        if r_le != p_le:
            summary["leading_edge_order_mismatches"] += 1
        if set(r_le) != set(p_le):
            summary["leading_edge_set_mismatches"] += 1
        for field in ["es", "nes", "pval", "padj"]:
            rv = parse_float(r.get(field))
            pv = parse_float(p.get(field))
            if math.isfinite(rv) and math.isfinite(pv):
                diff = abs(rv - pv)
                rel = diff / max(abs(pv), sys.float_info.min)
                summary[f"max_abs_diff_{field}"] = max(summary[f"max_abs_diff_{field}"], diff)
                summary[f"max_rel_diff_{field}"] = max(summary[f"max_rel_diff_{field}"], rel)
    return summary


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def main() -> None:
    args = parse_args()
    ensure_inputs(args)
    result: dict[str, Any] = {
        "input": args.name,
        "genes": count_rank_genes(args.ranks),
        "pathways": count_pathways(args.gmt),
    }
    rust_wall: list[float] = []
    rust_compute: list[float] = []
    python_cold: list[float] = []
    python_warm: list[float] = []
    parity: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="rsfgsea_blitz_bench_") as tmp:
        tmp_path = Path(tmp)
        rust_output = tmp_path / "rust.tsv"
        python_output = tmp_path / "python.tsv"
        for _ in range(args.reps):
            rust = run_rust(args, rust_output)
            rust_wall.append(rust["wall_s"])
            rust_compute.append(rust["compute_s"])
            if not args.skip_python:
                py_cold = run_python(args, python_output, warm=False)
                python_cold.append(float(py_cold.get("elapsed_s", math.nan)))
                py_warm = run_python(args, python_output, warm=True)
                python_warm.append(float(py_warm.get("elapsed_s", math.nan)))
                if "error" in py_cold:
                    parity["python_error"] = py_cold["error"]
                elif python_output.exists():
                    parity = parity_summary(rust_output, python_output)
    result.update(
        {
            "rust_wall_s": mean(rust_wall),
            "rust_compute_s": mean(rust_compute),
            "python_cold_s": mean(python_cold),
            "python_warm_cache_s": mean(python_warm),
        }
    )
    result.update(parity)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        fields = list(result)
        print("\t".join(fields))
        print("\t".join(str(result[field]) for field in fields))


if __name__ == "__main__":
    main()
