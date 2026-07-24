#!/usr/bin/env python3
"""Compare a full-precision rsfgsea Blitz TSV with a Python reference TSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any


FIELDS = {
    "es": ("es", "es"),
    "nes": ("nes", "nes"),
    "pval": ("pval", "pval"),
    "fdr": ("padj", "fdr"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def pathway_column(fieldnames: list[str] | None) -> str:
    names = set(fieldnames or [])
    for candidate in ["Term", "pathway"]:
        if candidate in names:
            return candidate
    raise ValueError("TSV is missing a Term/pathway column")


def read_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        key = pathway_column(reader.fieldnames)
        order: list[str] = []
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            pathway = row[key]
            order.append(pathway)
            rows[pathway] = row
    return order, rows


def parse_float(value: str | None) -> float:
    if value is None or value in {"", "NA", "None"}:
        return math.nan
    return float(value)


def ordered_float_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits >> 63:
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def ulp_distance(lhs: float, rhs: float) -> int | None:
    if not (math.isfinite(lhs) and math.isfinite(rhs)):
        return 0 if lhs == rhs else None
    return abs(ordered_float_bits(lhs) - ordered_float_bits(rhs))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def leading_edge(row: dict[str, str]) -> list[str]:
    value = row.get("leading_edge", "")
    return [item for item in value.replace(";", ",").split(",") if item]


def size_value(row: dict[str, str]) -> str:
    return row.get("size", row.get("geneset_size", ""))


def field_summary(
    name: str,
    observed_field: str,
    reference_field: str,
    shared: list[str],
    observed: dict[str, dict[str, str]],
    reference: dict[str, dict[str, str]],
    top: int,
) -> dict[str, Any]:
    finite: list[tuple[float, int, str, float, float]] = []
    finite_class_mismatches = 0
    zero_mismatches = 0
    nonfinite_mismatches = 0
    log10_differences: list[float] = []
    for pathway in shared:
        observed_value = observed[pathway].get(observed_field)
        if observed_value is None and observed_field == "padj":
            observed_value = observed[pathway].get("fdr")
        lhs = parse_float(observed_value)
        rhs = parse_float(reference[pathway].get(reference_field))
        lhs_finite = math.isfinite(lhs)
        rhs_finite = math.isfinite(rhs)
        if lhs_finite != rhs_finite:
            finite_class_mismatches += 1
        if (lhs == 0.0) != (rhs == 0.0):
            zero_mismatches += 1
        if not lhs_finite or not rhs_finite:
            if lhs != rhs:
                nonfinite_mismatches += 1
            continue
        distance = ulp_distance(lhs, rhs)
        assert distance is not None
        finite.append((abs(lhs - rhs), distance, pathway, lhs, rhs))
        if name in {"pval", "fdr"} and lhs > 0.0 and rhs > 0.0:
            log10_differences.append(abs(math.log10(lhs) - math.log10(rhs)))

    finite.sort(reverse=True)
    absolute = [row[0] for row in finite]
    ulps = [row[1] for row in finite]
    result: dict[str, Any] = {
        "finite_pairs": len(finite),
        "finite_class_mismatches": finite_class_mismatches,
        "nonfinite_mismatches": nonfinite_mismatches,
        "zero_mismatches": zero_mismatches,
        "exact_finite": sum(diff == 0.0 for diff in absolute),
        "nonexact_finite": sum(diff != 0.0 for diff in absolute),
        "max_abs": max(absolute, default=math.nan),
        "mean_abs": statistics.fmean(absolute) if absolute else math.nan,
        "p50_abs": percentile(absolute, 0.50),
        "p95_abs": percentile(absolute, 0.95),
        "p99_abs": percentile(absolute, 0.99),
        "max_ulp": max(ulps, default=None),
        "p95_ulp": percentile([float(value) for value in ulps], 0.95),
        "p99_ulp": percentile([float(value) for value in ulps], 0.99),
        "worst": [
            {
                "pathway": pathway,
                "observed": lhs,
                "reference": rhs,
                "abs_diff": difference,
                "ulp_diff": distance,
            }
            for difference, distance, pathway, lhs, rhs in finite[:top]
        ],
    }
    if log10_differences:
        result["max_abs_log10_diff"] = max(log10_differences)
        result["p99_abs_log10_diff"] = percentile(log10_differences, 0.99)
    return result


def main() -> None:
    args = parse_args()
    reference_order, reference = read_rows(args.reference)
    observed_order, observed = read_rows(args.observed)
    shared = sorted(set(reference) & set(observed))
    result: dict[str, Any] = {
        "reference": str(args.reference.resolve()),
        "observed": str(args.observed.resolve()),
        "reference_rows": len(reference),
        "observed_rows": len(observed),
        "shared_rows": len(shared),
        "missing_in_observed": sorted(set(reference) - set(observed)),
        "missing_in_reference": sorted(set(observed) - set(reference)),
        "order_mismatches": sum(
            lhs != rhs for lhs, rhs in zip(observed_order, reference_order)
        )
        + abs(len(observed_order) - len(reference_order)),
        "size_mismatches": sum(
            size_value(observed[pathway]) != size_value(reference[pathway])
            for pathway in shared
        ),
        "leading_edge_order_mismatches": sum(
            leading_edge(observed[pathway]) != leading_edge(reference[pathway])
            for pathway in shared
        ),
        "leading_edge_set_mismatches": sum(
            set(leading_edge(observed[pathway]))
            != set(leading_edge(reference[pathway]))
            for pathway in shared
        ),
    }
    for name, (observed_field, reference_field) in FIELDS.items():
        result[name] = field_summary(
            name,
            observed_field,
            reference_field,
            shared,
            observed,
            reference,
            args.top,
        )

    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
