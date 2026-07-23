#!/usr/bin/env python3
"""Generate reproducible, full-precision large Blitz reference artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Iterable


for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import blitzgsea  # noqa: E402
import pandas as pd  # noqa: E402

import generate_blitz_reference as fixture  # noqa: E402
from blitz_reference_env import validate_reference_environment  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKS = ROOT / "data/deseq2_positive_ranks/lung_vs_muscle.rnk"
DEFAULT_GMT = ROOT / "data/GO_Biological_Process_2025.gmt"
DEFAULT_OUTPUT = ROOT / "data/derived/blitz_precision/lung_vs_muscle_go_bp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=Path, default=DEFAULT_RANKS)
    parser.add_argument("--gmt", type=Path, default=DEFAULT_GMT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", default="python_blitzgsea_1_3_54")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--fixed-schedule-trace",
        action="store_true",
        help="also generate deterministic anchor/model traces",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_environment() -> dict[str, object]:
    return validate_reference_environment(ROOT)


def read_inputs(ranks_path: Path, gmt_path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    signature = pd.read_csv(
        ranks_path,
        sep=r"\s+",
        header=None,
        names=["i", "v"],
        dtype={"i": str, "v": float},
        float_precision="round_trip",
    )
    library: dict[str, list[str]] = {}
    with gmt_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                library[parts[0]] = parts[2:]
    return signature, library


def run_reference(
    signature: pd.DataFrame, library: dict[str, Iterable[str]]
) -> tuple[float, pd.DataFrame]:
    started = time.perf_counter()
    result = blitzgsea.gsea(
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
        signature_cache=False,
        seed=0,
        accuracy=40,
        deep_accuracy=50,
        center=True,
    )
    return time.perf_counter() - started, result


def write_result(path: Path, result: pd.DataFrame) -> None:
    rows = result.reset_index()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["Term", "es", "nes", "pval", "sidak", "fdr", "geneset_size", "leading_edge"]
        )
        for row in rows.itertuples(index=False):
            writer.writerow(
                [
                    str(row.Term),
                    format(float(row.es), ".17g"),
                    format(float(row.nes), ".17g"),
                    format(float(row.pval), ".17g"),
                    format(float(row.sidak), ".17g"),
                    format(float(row.fdr), ".17g"),
                    str(int(row.geneset_size)),
                    str(row.leading_edge),
                ]
            )


def main() -> None:
    args = parse_args()
    if args.reps < 1:
        raise SystemExit("--reps must be positive")
    for path in [args.ranks, args.gmt]:
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    reference_environment = check_environment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature, library = read_inputs(args.ranks, args.gmt)

    runs = []
    result_hashes = []
    for rep in range(args.reps):
        elapsed, result = run_reference(signature, library)
        path = args.output_dir / f"{args.prefix}.rep{rep + 1}.tsv"
        write_result(path, result)
        digest = sha256(path)
        result_hashes.append(digest)
        runs.append({"rep": rep + 1, "elapsed_s": elapsed, "path": str(path), "sha256": digest})
    if len(set(result_hashes)) != 1:
        raise SystemExit(f"large Blitz reference was not deterministic: {result_hashes}")

    fixed_artifacts: dict[str, str] = {}
    if args.fixed_schedule_trace:
        fixture.OUT = args.output_dir
        prepared_signature, prepared_library, absolute, signature_map = fixture.prepare_blitz_inputs(
            signature, library
        )
        fixed_prefix = f"{args.prefix}.fixed_schedule"
        fixture.write_reference_files(
            fixed_prefix,
            prepared_signature,
            prepared_library,
            absolute,
            signature_map,
            emit_trace=True,
        )
        for path in sorted(args.output_dir.glob(f"{fixed_prefix}.*")):
            fixed_artifacts[path.name] = sha256(path)

    manifest = {
        "created_unix": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "thread_environment": {
            variable: os.environ.get(variable)
            for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "reference_environment": reference_environment,
        "versions": reference_environment["packages"],
        "inputs": {
            "ranks": str(args.ranks.resolve()),
            "ranks_sha256": sha256(args.ranks),
            "rank_rows": len(signature),
            "gmt": str(args.gmt.resolve()),
            "gmt_sha256": sha256(args.gmt),
            "gmt_pathways": len(library),
        },
        "options": {
            "permutations": 1000,
            "anchors": 40,
            "min_size": 5,
            "max_size": 4000,
            "processes": 4,
            "symmetric": False,
            "signature_cache": False,
            "seed": 0,
            "accuracy": 40,
            "deep_accuracy": 50,
            "center": True,
        },
        "runs": runs,
        "deterministic_sha256": result_hashes[0],
        "fixed_schedule_artifacts": fixed_artifacts,
    }
    manifest_path = args.output_dir / f"{args.prefix}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
