#!/usr/bin/env python3
"""Paired, alternating A/B benchmark for native Blitz CLI binaries.

The benchmark keeps model calibration cold (the CLI does not enable the
in-process signature cache), alternates execution order to reduce thermal and
time-drift bias, records every raw timing and output hash, and reports a paired
bootstrap confidence interval for the candidate/baseline ratio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKS = ROOT / "data/deseq2_positive_ranks/lung_vs_muscle.rnk"
DEFAULT_GMT = ROOT / "data/GO_Biological_Process_2025.gmt"
COMPUTE_RE = re.compile(r"GSEA_COMP_TIME_MS:\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin", type=Path, required=True)
    parser.add_argument("--candidate-bin", type=Path, required=True)
    parser.add_argument("--ranks", type=Path, default=DEFAULT_RANKS)
    parser.add_argument("--gmt", type=Path, default=DEFAULT_GMT)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu-list", default="8,10,12,14")
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--equivalence-margin-pct",
        type=float,
        default=1.0,
        help="Allowed upper confidence-bound slowdown for the no-regression gate",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise SystemExit("--cpu-list must name at least one CPU")
    return cpus


def run_cli(binary: Path, ranks: Path, gmt: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        [
            str(binary.resolve()),
            "--ranks",
            str(ranks.resolve()),
            "--gmt",
            str(gmt.resolve()),
            "--output",
            str(output),
            "--mode",
            "blitz",
            "--nPermSimple",
            "1000",
            "--blitz-anchors",
            "40",
            "--minSize",
            "5",
            "--maxSize",
            "4000",
            "--seed",
            "0",
            "--nproc",
            "4",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    wall = time.perf_counter() - started
    match = COMPUTE_RE.search(proc.stdout)
    if match is None:
        raise RuntimeError(f"{binary} did not report GSEA_COMP_TIME_MS")
    return {
        "compute_s": int(match.group(1)) / 1000.0,
        "wall_s": wall,
        # Hashing happens after the wall timer, so determinism checking is not
        # charged to either binary's end-to-end measurement.
        "output_sha256": sha256(output),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "mean": statistics.fmean(values),
        "median": median,
        "mad": statistics.median(deviations),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def geometric_mean_ratio(candidate: list[float], baseline: list[float]) -> float:
    return math.exp(
        statistics.fmean(math.log(cand / base) for cand, base in zip(candidate, baseline))
    )


def bootstrap_ratio_interval(
    candidate: list[float],
    baseline: list[float],
    resamples: int,
    confidence: float,
) -> tuple[float, float]:
    rng = random.Random(0xB117_25EA)
    count = len(candidate)
    log_ratios = [math.log(cand / base) for cand, base in zip(candidate, baseline)]
    samples = []
    for _ in range(resamples):
        samples.append(
            math.exp(statistics.fmean(log_ratios[rng.randrange(count)] for _ in range(count)))
        )
    alpha = (1.0 - confidence) / 2.0
    return percentile(samples, alpha), percentile(samples, 1.0 - alpha)


def metric_summary(
    rows: list[dict[str, Any]], metric: str, args: argparse.Namespace
) -> dict[str, Any]:
    baseline = [float(row["baseline"][metric]) for row in rows]
    candidate = [float(row["candidate"][metric]) for row in rows]
    paired = [cand / base for cand, base in zip(candidate, baseline)]
    ratio = geometric_mean_ratio(candidate, baseline)
    ci_low, ci_high = bootstrap_ratio_interval(
        candidate,
        baseline,
        args.bootstrap_resamples,
        args.confidence,
    )
    margin = 1.0 + args.equivalence_margin_pct / 100.0
    return {
        "baseline": summarize(baseline),
        "candidate": summarize(candidate),
        "paired_ratios": paired,
        "geometric_mean_ratio": ratio,
        "change_pct": 100.0 * (ratio - 1.0),
        "confidence_interval": [ci_low, ci_high],
        "candidate_point_estimate_not_slower": ratio <= 1.0,
        "upper_bound_within_equivalence_margin": ci_high <= margin,
        "passes_no_regression_gate": ratio <= 1.0 and ci_high <= margin,
    }


def output_determinism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ["baseline", "candidate"]:
        hashes = sorted({str(row[label]["output_sha256"]) for row in rows})
        result[label] = {
            "deterministic": len(hashes) == 1,
            "unique_hashes": hashes,
        }
    result["all_deterministic"] = all(
        bool(result[label]["deterministic"]) for label in ["baseline", "candidate"]
    )
    return result


def command_output(command: list[str]) -> str:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return (proc.stdout or proc.stderr).strip()


def main() -> None:
    args = parse_args()
    if args.reps < 3:
        raise SystemExit("--reps must be at least 3")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.bootstrap_resamples < 1000:
        raise SystemExit("--bootstrap-resamples must be at least 1000")
    if not 0.5 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between 0.5 and 1")
    for path in [args.baseline_bin, args.candidate_bin, args.ranks, args.gmt]:
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")

    cpus = parse_cpu_list(args.cpu_list)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)

    metadata: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "loadavg_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "rustc": command_output(["rustc", "-Vv"]),
        "build_environment": {
            variable: os.environ.get(variable)
            for variable in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"]
        },
        "baseline_binary": str(args.baseline_bin.resolve()),
        "baseline_sha256": sha256(args.baseline_bin),
        "candidate_binary": str(args.candidate_bin.resolve()),
        "candidate_sha256": sha256(args.candidate_bin),
        "ranks": str(args.ranks.resolve()),
        "ranks_sha256": sha256(args.ranks),
        "gmt": str(args.gmt.resolve()),
        "gmt_sha256": sha256(args.gmt),
        "reps": args.reps,
        "warmups": args.warmups,
        "bootstrap_resamples": args.bootstrap_resamples,
        "confidence": args.confidence,
        "equivalence_margin_pct": args.equivalence_margin_pct,
    }

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rsfgsea_blitz_ab_") as tmp:
        tmpdir = Path(tmp)
        for warmup in range(args.warmups):
            for label, binary in [
                ("baseline", args.baseline_bin),
                ("candidate", args.candidate_bin),
            ]:
                run_cli(binary, args.ranks, args.gmt, tmpdir / f"warmup-{warmup}-{label}.tsv")
            print(f"completed warmup pair {warmup + 1}/{args.warmups}", file=sys.stderr, flush=True)

        for rep in range(args.reps):
            order = ["baseline", "candidate"] if rep % 2 == 0 else ["candidate", "baseline"]
            row: dict[str, Any] = {"rep": rep, "order": order}
            for label in order:
                binary = args.baseline_bin if label == "baseline" else args.candidate_bin
                row[label] = run_cli(
                    binary,
                    args.ranks,
                    args.gmt,
                    tmpdir / f"rep-{rep:03d}-{label}.tsv",
                )
            rows.append(row)
            print(f"completed measured pair {rep + 1}/{args.reps}", file=sys.stderr, flush=True)

    result = {
        "metadata": metadata,
        "raw_pairs": rows,
        "compute": metric_summary(rows, "compute_s", args),
        "wall": metric_summary(rows, "wall_s", args),
        "output_determinism": output_determinism(rows),
        "loadavg_end": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }
    result["passes_no_regression_gate"] = bool(
        result["compute"]["passes_no_regression_gate"]
        and result["wall"]["passes_no_regression_gate"]
    )
    result["passes_acceptance_gate"] = bool(
        result["passes_no_regression_gate"]
        and result["output_determinism"]["all_deterministic"]
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passes_acceptance_gate"]:
        raise SystemExit("benchmark acceptance gate failed")


if __name__ == "__main__":
    main()
