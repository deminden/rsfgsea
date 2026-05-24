#!/usr/bin/env python3
"""Generate small blitzgsea 1.3.54 reference fixtures.

Run with the pinned/local reference stack and PYTHONHASHSEED=0, for example:

    PYTHONHASHSEED=0 /home/den/miniforge3/bin/python scripts/generate_blitz_reference.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import random
from pathlib import Path
from typing import Iterable

import blitzgsea
from mpmath import mp
import numpy as np
import pandas as pd
from scipy.stats import gamma
from statsmodels.stats.multitest import multipletests


def _estimate_anchor_trace_star(args):
    set_size = args[3]
    return set_size, blitzgsea.estimate_anchor(*args)


EXPECTED = {
    "blitzgsea": "1.3.54",
    "numpy": "2.4.0",
    "scipy": "1.16.3",
    "statsmodels": "0.14.6",
    "pandas": "2.3.3",
    "mpmath": "1.4.1",
}

OUT = Path("crates/rsfgsea/tests/data/blitz_reference")
TAIL_TRACE = OUT / "tail_fallback.trace_gamma.tsv"


def check_environment() -> dict[str, str]:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 before generating blitz fixtures.")

    observed = {pkg: metadata.version(pkg) for pkg in EXPECTED}
    mismatches = {
        pkg: (EXPECTED[pkg], observed[pkg])
        for pkg in EXPECTED
        if observed[pkg] != EXPECTED[pkg]
    }
    if mismatches:
        details = ", ".join(
            f"{pkg}: expected {expected}, observed {actual}"
            for pkg, (expected, actual) in mismatches.items()
        )
        raise SystemExit(f"Reference environment mismatch: {details}")
    return observed


def build_inputs() -> tuple[pd.DataFrame, dict[str, Iterable[str]], list[str], np.ndarray]:
    genes = [f"g{i:03d}" for i in range(1, 121)]
    scores = np.linspace(4, -4, len(genes)) + np.sin(np.arange(len(genes)) * 0.7) * 0.15
    signature = pd.DataFrame({"i": genes, "v": scores})
    library = {
        "TOP_12": set(genes[:12]),
        "MID_MIX": set(genes[10:20] + genes[70:80]),
        "BOTTOM_15": set(genes[-15:]),
        "SPARSE_14": set(genes[::9]),
    }
    return signature, library, genes, scores


def build_edge_inputs() -> tuple[pd.DataFrame, dict[str, Iterable[str]]]:
    genes = [f"e{i:02d}" for i in range(1, 61)]
    scores = np.cos(np.arange(len(genes)) * 0.37) * 2.5 + np.linspace(-1.2, 1.4, len(genes))
    rows = list(zip(genes, scores))
    rows = rows[18:33] + [("dup", -2.0)] + rows[:18] + [("dup", 4.25)] + rows[33:]
    signature = pd.DataFrame(rows, columns=["i", "v"])
    library = {
        "DUP_AND_ALIEN": ["dup", "e01", "e01", "missing", "e02", "e03", "e04", "e05"],
        "TOP_HEAVY": ["dup", "e60", "e59", "e58", "e57", "e56"],
        "BOTTOM_HEAVY": ["e22", "e23", "e24", "e25", "e26", "e27", "e28"],
        "MIXED_DUPLICATE_ORDER": [
            "e03",
            "e08",
            "e13",
            "e18",
            "e23",
            "e28",
            "e33",
            "e38",
            "e43",
            "e48",
            "e53",
            "e58",
            "e58",
            "not_in_signature",
        ],
        "FILTER_TOO_SMALL": ["e01", "e02", "e03", "missing"],
        "WIDE_REALISTIC": genes[5:35:2] + ["dup", "ghost"],
    }
    return signature, library


def load_publication_inputs() -> tuple[pd.DataFrame, dict[str, Iterable[str]], dict[str, str]]:
    source_dir = Path("r_libs/publication/fgsea/extdata")
    ranks_path = source_dir / "naive.vs.th1.rnk"
    gmt_path = source_dir / "mouse.reactome.gmt"
    ranks = pd.read_csv(ranks_path, sep="\t")
    ranks["ID"] = ranks["ID"].astype(str)
    signature = (
        ranks.assign(abs_t=ranks["t"].abs())
        .nlargest(600, "abs_t")[["ID", "t"]]
        .rename(columns={"ID": "i", "t": "v"})
        .reset_index(drop=True)
    )
    signature_genes = set(signature["i"])
    candidates: list[tuple[str, int, list[str]]] = []
    with gmt_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genes = parts[2:]
            overlap = len(set(genes) & signature_genes)
            if 5 <= overlap <= 80:
                candidates.append((parts[0], overlap, genes))

    wanted_sizes = [5, 7, 10, 15, 20, 30, 50, 70]
    library: dict[str, Iterable[str]] = {}
    used: set[str] = set()
    for wanted_size in wanted_sizes:
        name, _, genes = min(
            (candidate for candidate in candidates if candidate[0] not in used),
            key=lambda candidate: (abs(candidate[1] - wanted_size), candidate[0]),
        )
        library[name] = genes
        used.add(name)

    metadata = {
        "rank_source": str(ranks_path),
        "gmt_source": str(gmt_path),
        "rank_selection": "top 600 genes by absolute t-statistic from naive.vs.th1.rnk",
        "pathway_selection": "closest cleaned sizes to 5,7,10,15,20,30,50,70 from mouse.reactome.gmt",
    }
    return signature, library, metadata


def anchor_set_sizes(library: dict[str, Iterable[str]], signature_len: int) -> list[int]:
    sizes = [len(value) for value in library.values()]
    max_size = max(sizes)
    anchors = [int(x) for x in list(np.linspace(1, max_size, 40))]
    anchors.extend(
        [1, 2, 3, 4, 5, 6, 7, 12, 16, 20, 30, 40, 50, 60, 70, 80, 100, max_size + 10, max_size + 30]
    )
    return [size for size in sorted(set(anchors)) if size <= signature_len]


def prepare_blitz_inputs(
    signature: pd.DataFrame, library: dict[str, Iterable[str]]
) -> tuple[pd.DataFrame, dict[str, set[str]], np.ndarray, dict[str, int]]:
    random.seed(0)
    np.random.seed(0)
    signature = signature.copy()
    signature.columns = ["i", "v"]
    signature = signature.sort_values("v", ascending=False).set_index("i")
    signature = signature[~signature.index.duplicated(keep="first")]
    library = {key: set(value) for key, value in library.items()}
    library = blitzgsea.clean_library(library, signature)
    signature.loc[:, "v"] -= np.mean(signature.loc[:, "v"])
    abs_signature = np.array(np.abs(signature.loc[:, "v"]))
    signature_map = {gene: idx for idx, gene in enumerate(signature.index)}
    return signature, library, abs_signature, signature_map


def estimate_anchor_traces(
    signature: pd.DataFrame,
    abs_signature: np.ndarray,
    signature_map: dict[str, int],
    anchors: list[int],
) -> list[tuple[int, tuple[float, float, float, float, float, float, float]]]:
    # blitzgsea's multiprocessing inherits the parent NumPy RandomState in each
    # worker, then advances it according to the Pool task schedule. Repeated
    # local runs can choose different schedules, so the fixture records the
    # deterministic schedule used by the Rust parity target.
    parent_state = np.random.get_state()
    worker_states = [parent_state for _ in range(4)]
    per_worker: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for idx, size in enumerate(anchors):
        per_worker[reference_worker_index(idx)].append((idx, size))

    results: list[tuple[int, tuple[float, float, float, float, float, float, float]] | None] = [
        None
    ] * len(anchors)
    for worker_idx, jobs in enumerate(per_worker):
        np.random.set_state(worker_states[worker_idx])
        for idx, size in jobs:
            args = (signature, abs_signature, signature_map, size, 1000, False, int(size), True)
            results[idx] = (size, blitzgsea.estimate_anchor(*args))
        worker_states[worker_idx] = np.random.get_state()
    np.random.set_state(parent_state)
    return [result for result in results if result is not None]


def reference_worker_index(task_idx: int) -> int:
    if task_idx < 4:
        return task_idx
    return [1, 2, 0, 3][(task_idx - 4) % 4]


def write_reference_files(
    prefix: str,
    signature: pd.DataFrame,
    library: dict[str, set[str]],
    abs_signature: np.ndarray,
    signature_map: dict[str, int],
    emit_trace: bool,
) -> None:
    anchors = anchor_set_sizes(library, len(abs_signature))
    anchor_results = estimate_anchor_traces(signature, abs_signature, signature_map, anchors)
    anchor_by_size = {size: result for size, result in anchor_results}

    alpha_pos = []
    beta_pos = []
    alpha_neg = []
    beta_neg = []
    pos_ratio = []
    for size in anchors:
        fit = anchor_by_size[size]
        alpha_pos.append(fit[0])
        beta_pos.append(fit[1])
        alpha_neg.append(fit[3])
        beta_neg.append(fit[4])
        pos_ratio.append(fit[6])

    x = np.array(anchors, dtype=float)
    alpha_pos_smooth = blitzgsea.lowess(alpha_pos, x, frac=0.6)[:, 1]
    beta_pos_smooth = blitzgsea.lowess(beta_pos, x, frac=0.15)[:, 1]
    alpha_neg_smooth = blitzgsea.lowess(alpha_neg, x, frac=0.6)[:, 1]
    beta_neg_smooth = blitzgsea.lowess(beta_neg, x, frac=0.15)[:, 1]
    jittered_pos_ratio = np.array(pos_ratio) - np.abs(0.0001 * np.random.randn(len(pos_ratio)))
    pos_ratio_smooth = blitzgsea.lowess(jittered_pos_ratio, x, frac=0.5)[:, 1]

    f_alpha_pos = blitzgsea.interpolate.interp1d(
        x, alpha_pos_smooth, bounds_error=False, fill_value="extrapolate"
    )
    f_beta_pos = blitzgsea.interpolate.interp1d(
        x, beta_pos_smooth, bounds_error=False, fill_value="extrapolate"
    )
    f_alpha_neg = blitzgsea.interpolate.interp1d(
        x, alpha_neg_smooth, bounds_error=False, fill_value="extrapolate"
    )
    f_beta_neg = blitzgsea.interpolate.interp1d(
        x, beta_neg_smooth, bounds_error=False, fill_value="extrapolate"
    )
    f_pos_ratio = blitzgsea.interpolate.interp1d(
        x, pos_ratio_smooth, bounds_error=False, fill_value="extrapolate"
    )

    if emit_trace:
        (OUT / f"{prefix}.trace_signature.tsv").write_text(
            "gene\tcentered_score\tabs_score\n"
            + "".join(
                f"{gene}\t{float(score):.17g}\t{float(abs(score)):.17g}\n"
                for gene, score in zip(signature.index, signature["v"])
            ),
            encoding="utf-8",
        )
        (OUT / f"{prefix}.trace_pathways.tsv").write_text(
            "pathway\tgenes\n"
            + "".join(
                f"{name}\t{','.join(blitzgsea.strip_gene_set(set(signature.index), genes))}\n"
                for name, genes in library.items()
            ),
            encoding="utf-8",
        )

    anchor_lines = [
        "set_size\talpha_pos\tbeta_pos\talpha_neg\tbeta_neg\tpos_ratio\t"
        "alpha_pos_smooth\tbeta_pos_smooth\talpha_neg_smooth\tbeta_neg_smooth\t"
        "pos_ratio_jittered\tpos_ratio_smooth\n"
    ]
    for idx, size in enumerate(anchors):
        fit = anchor_by_size[size]
        anchor_lines.append(
            f"{size}\t{float(fit[0]):.17g}\t{float(fit[1]):.17g}\t"
            f"{float(fit[3]):.17g}\t{float(fit[4]):.17g}\t{float(fit[6]):.17g}\t"
            f"{float(alpha_pos_smooth[idx]):.17g}\t{float(beta_pos_smooth[idx]):.17g}\t"
            f"{float(alpha_neg_smooth[idx]):.17g}\t{float(beta_neg_smooth[idx]):.17g}\t"
            f"{float(jittered_pos_ratio[idx]):.17g}\t{float(pos_ratio_smooth[idx]):.17g}\n"
        )
    if emit_trace:
        (OUT / f"{prefix}.trace_anchors.tsv").write_text("".join(anchor_lines), encoding="utf-8")

    rows = []
    gamma_rows = []
    signature_genes = set(signature.index)
    for name, genes in library.items():
        stripped_set = blitzgsea.strip_gene_set(signature_genes, genes)
        if len(stripped_set) < 5 or len(stripped_set) > 4000:
            continue
        running_sum, es = blitzgsea.enrichment_score(abs_signature, signature_map, stripped_set)
        leading_edge = blitzgsea.get_leading_edge(running_sum, signature, stripped_set, signature_map)
        size = len(stripped_set)
        pos_alpha = float(f_alpha_pos(size))
        pos_beta = float(f_beta_pos(size))
        interpolated_pos_ratio = float(f_pos_ratio(size))
        clipped_pos_ratio = max(0.0, min(1.0, interpolated_pos_ratio))
        neg_alpha = float(f_alpha_neg(size))
        neg_beta = float(f_beta_neg(size))
        if es > 0:
            raw_gamma_prob = gamma.cdf(es, pos_alpha, scale=pos_beta)
            gamma_prob = raw_gamma_prob
            fallback_used = gamma_prob > 0.999999999 or gamma_prob < 0.00000000001
            if fallback_used:
                gamma_prob = blitzgsea.gammacdf(es, pos_alpha, pos_beta, dps=50)
            prob_two_tailed = np.min(
                [0.5, (1 - np.min([gamma_prob * clipped_pos_ratio + 1 - clipped_pos_ratio, 1]))]
            )
            nes = blitzgsea.invcdf(1 - np.min([1, prob_two_tailed]))
            gamma_rows.append(
                {
                    "pathway": name,
                    "branch": "pos",
                    "es_or_minus_es": float(es),
                    "alpha": pos_alpha,
                    "beta": pos_beta,
                    "z": float(es / pos_beta),
                    "raw_gamma_prob": float(raw_gamma_prob),
                    "fallback_used": fallback_used,
                    "gamma_prob": float(gamma_prob),
                    "pval": float(2 * prob_two_tailed),
                    "nes": -float(nes),
                }
            )
        else:
            raw_gamma_prob = gamma.cdf(-es, neg_alpha, scale=neg_beta)
            gamma_prob = raw_gamma_prob
            fallback_used = gamma_prob > 0.999999999 or gamma_prob < 0.00000000001
            if fallback_used:
                gamma_prob = blitzgsea.gammacdf(-es, neg_alpha, neg_beta, dps=50)
            prob_two_tailed = np.min(
                [0.5, (1 - np.min([((gamma_prob - (gamma_prob * clipped_pos_ratio)) + clipped_pos_ratio), 1]))]
            )
            if prob_two_tailed == 0.5:
                prob_two_tailed = prob_two_tailed - gamma_prob
            nes = blitzgsea.invcdf(np.min([1, prob_two_tailed]))
            gamma_rows.append(
                {
                    "pathway": name,
                    "branch": "neg",
                    "es_or_minus_es": float(-es),
                    "alpha": neg_alpha,
                    "beta": neg_beta,
                    "z": float((-es) / neg_beta),
                    "raw_gamma_prob": float(raw_gamma_prob),
                    "fallback_used": fallback_used,
                    "gamma_prob": float(gamma_prob),
                    "pval": float(2 * prob_two_tailed),
                    "nes": -float(nes),
                }
            )
        pval = 2 * prob_two_tailed
        rows.append(
            {
                "pathway": name,
                "set_size": size,
                "es": float(es),
                "pos_alpha": pos_alpha,
                "pos_beta": pos_beta,
                "pos_ratio": interpolated_pos_ratio,
                "pos_ratio_clipped": clipped_pos_ratio,
                "neg_alpha": neg_alpha,
                "neg_beta": neg_beta,
                "gamma_prob": float(gamma_prob),
                "prob_two_tailed": float(prob_two_tailed),
                "nes": -float(nes),
                "pval": float(pval),
                "leading_edge": leading_edge,
            }
        )

    fdr_values = multipletests([row["pval"] for row in rows], method="fdr_bh")[1]
    for row, gamma_row, fdr in zip(rows, gamma_rows, fdr_values):
        row["fdr"] = float(fdr)
        gamma_row["fdr"] = float(fdr)
    rows.sort(key=lambda row: abs(row["pval"]))
    gamma_rows.sort(key=lambda row: abs(row["pval"]))

    gamma_header = [
        "pathway",
        "branch",
        "es_or_minus_es",
        "alpha",
        "beta",
        "z",
        "raw_gamma_prob",
        "fallback_used",
        "gamma_prob",
        "pval",
        "nes",
        "fdr",
    ]
    (OUT / f"{prefix}.trace_gamma.tsv").write_text(
        "\t".join(gamma_header)
        + "\n"
        + "".join(
            "\t".join(
                str(row[key])
                if isinstance(row[key], str)
                else str(row[key]).lower()
                if isinstance(row[key], (bool, np.bool_))
                else repr(float(row[key]))
                for key in gamma_header
            )
            + "\n"
            for row in gamma_rows
        ),
        encoding="utf-8",
    )

    result_header = [
        "pathway",
        "set_size",
        "es",
        "pos_alpha",
        "pos_beta",
        "pos_ratio",
        "pos_ratio_clipped",
        "neg_alpha",
        "neg_beta",
        "gamma_prob",
        "prob_two_tailed",
        "nes",
        "pval",
        "fdr",
        "leading_edge",
    ]
    if emit_trace:
        (OUT / f"{prefix}.trace_results.tsv").write_text(
            "\t".join(result_header)
            + "\n"
            + "".join(
                "\t".join(
                    str(row[key]) if isinstance(row[key], str) else f"{float(row[key]):.17g}"
                    for key in result_header
                )
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
    sidak_values = multipletests([row["pval"] for row in rows], method="sidak")[1]
    expected_header = ["Term", "es", "nes", "pval", "sidak", "fdr", "geneset_size", "leading_edge"]
    expected_lines = ["\t".join(expected_header) + "\n"]
    for row, sidak in zip(rows, sidak_values):
        expected_lines.append(
            "\t".join(
                [
                    str(row["pathway"]),
                    repr(float(row["es"])),
                    repr(float(row["nes"])),
                    repr(float(row["pval"])),
                    repr(float(sidak)),
                    repr(float(row["fdr"])),
                    str(int(row["set_size"])),
                    str(row["leading_edge"]),
                ]
            )
            + "\n"
        )
    (OUT / f"{prefix}.expected.tsv").write_text(
        "".join(expected_lines),
        encoding="utf-8",
    )


def write_input_files(prefix: str, signature: pd.DataFrame, library: dict[str, Iterable[str]]) -> None:
    (OUT / f"{prefix}.rnk").write_text(
        "".join(f"{gene}\t{repr(float(score))}\n" for gene, score in zip(signature["i"], signature["v"])),
        encoding="utf-8",
    )
    (OUT / f"{prefix}.gmt").write_text(
        "".join(
            f"{name}\tdesc\t" + "\t".join(str(gene) for gene in gene_set) + "\n"
            for name, gene_set in library.items()
        ),
        encoding="utf-8",
    )


def read_input_files(prefix: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    signature = pd.read_csv(
        OUT / f"{prefix}.rnk",
        sep="\t",
        header=None,
        names=["i", "v"],
        dtype={"i": str, "v": float},
        float_precision="round_trip",
    )
    library: dict[str, list[str]] = {}
    with (OUT / f"{prefix}.gmt").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            library[parts[0]] = parts[2:]
    return signature, library


def write_fixture(
    prefix: str,
    signature: pd.DataFrame,
    library: dict[str, Iterable[str]],
    emit_trace: bool = False,
) -> None:
    write_input_files(prefix, signature, library)
    fixture_signature, fixture_library = read_input_files(prefix)
    trace_signature, trace_library, abs_signature, signature_map = prepare_blitz_inputs(
        fixture_signature, fixture_library
    )
    write_reference_files(prefix, trace_signature, trace_library, abs_signature, signature_map, emit_trace)


def write_tail_fallback_trace() -> None:
    """Record focused rows that force blitzgsea's mpmath gamma fallback.

    This is not a random fuzz fixture: the rows are deterministic sentinels
    for practical blitz fallback regimes. They cover lower/upper tails,
    threshold-adjacent inputs, default and non-default deep_accuracy values,
    integer/half-integer/non-integer shape parameters, both sign branches, and
    nextafter perturbations that catch f64 rounding edge drift.
    """
    cases: list[dict[str, object]] = []
    seen: set[tuple[str, float, float, float, int]] = set()

    def add_case(
        name: str,
        branch: str,
        tail: str,
        alpha: float,
        z: float,
        beta: float,
        pos_ratio: float,
        deep_accuracy: int,
    ) -> None:
        if not np.isfinite(z) or z <= 0:
            return
        key = (branch, float(alpha), float(z * beta), float(beta), int(deep_accuracy))
        if key in seen:
            return
        seen.add(key)
        cases.append(
            {
                "case": name,
                "branch": branch,
                "tail": tail,
                "x": float(z * beta),
                "alpha": float(alpha),
                "beta": float(beta),
                "pos_ratio": float(pos_ratio),
                "deep_accuracy": int(deep_accuracy),
            }
        )

    manual_cases = [
        ("pos_lower_half", "pos", "lower", 0.5, 1.0e-30, 2.0, 0.20, 50),
        ("pos_upper_integer", "pos", "upper", 5.0, 50.0, 1.0, 0.73, 50),
        ("pos_lower_noninteger", "pos", "lower", 1.2, 1.0e-12, 1.0, 0.43, 50),
        ("neg_lower_integer", "neg", "lower", 20.0, 1.0, 1.0, 0.31, 50),
        ("neg_upper_integer", "neg", "upper", 100.0, 200.0, 1.0, 0.64, 50),
        ("neg_upper_noninteger", "neg", "upper", 1.2, 50.0, 1.0, 0.58, 50),
        # Positive-control DESeq2 contrasts can drive blitz into a more extreme
        # upper tail than the synthetic sweep. Python blitz rounds gammacdf()
        # to exactly 1.0 at deep_accuracy=50 here, so p-value becomes 0 and
        # NES is infinite.
        (
            "pos_upper_python_underflow_real_lung_vs_muscle",
            "pos",
            "upper",
            52.98247998666223,
            0.6236670695830199 / 0.0017743600379359313,
            0.0017743600379359313,
            0.5370602886045087,
            50,
        ),
        (
            "neg_upper_python_underflow_real_lung_vs_muscle",
            "neg",
            "upper",
            52.98247998666223,
            0.6236670695830199 / 0.0017743600379359313,
            0.0017743600379359313,
            0.5370602886045087,
            50,
        ),
        (
            "pos_upper_python_underflow_response_external_stimulus",
            "pos",
            "upper",
            50.71670836797934,
            0.686089595988157 / 0.002632954190910833,
            0.002632954190910833,
            0.5370416540690217,
            50,
        ),
        (
            "pos_upper_python_underflow_neuron_projection",
            "pos",
            "upper",
            53.88755278180769,
            0.4900408755626833 / 0.0019681114609304733,
            0.0019681114609304733,
            0.5370837511002126,
            50,
        ),
    ]
    for case in manual_cases:
        add_case(*case)

    alpha_cases = [
        ("int_small", 1.0),
        ("int_large", 100.0),
        ("half_small", 0.5),
        ("half_mid", 10.5),
        ("nonint_lt1", 0.75),
        ("nonint_mid", 1.2),
        ("nonint_large", 80.25),
    ]
    lower_targets = [9.5e-12, 1.0e-20]
    upper_survival_targets = [9.5e-10, 1.0e-20]
    dps_by_alpha = {1.2: [30, 50, 80], 0.75: [30, 50], 80.25: [50, 80]}
    ratio_by_idx = [0.17, 0.31, 0.43, 0.58, 0.73, 0.91]
    beta_by_idx = [1.0, 0.25, 2.0]

    for alpha_idx, (alpha_name, alpha) in enumerate(alpha_cases):
        dps_values = dps_by_alpha.get(alpha, [50])
        for dps in dps_values:
            for target_idx, target in enumerate(lower_targets):
                z = float(gamma.ppf(target, alpha))
                if not np.isfinite(z) or z <= 0:
                    continue
                variants = [("base", z)]
                if dps == 50 and target_idx == 0:
                    variants.extend(
                        [
                            ("down", float(np.nextafter(z, 0.0))),
                            ("up", float(np.nextafter(z, np.inf))),
                        ]
                    )
                for variant_name, variant_z in variants:
                    branch = "pos" if (target_idx + alpha_idx) % 2 == 0 else "neg"
                    add_case(
                        f"{branch}_lower_{alpha_name}_{target_idx}_{variant_name}_dps{dps}",
                        branch,
                        "lower",
                        alpha,
                        variant_z,
                        beta_by_idx[(alpha_idx + target_idx) % len(beta_by_idx)],
                        ratio_by_idx[(alpha_idx + target_idx) % len(ratio_by_idx)],
                        dps,
                    )
            for target_idx, survival_target in enumerate(upper_survival_targets):
                z = float(gamma.isf(survival_target, alpha))
                if not np.isfinite(z) or z <= 0:
                    continue
                variants = [("base", z)]
                if dps == 50 and target_idx == 0:
                    variants.extend(
                        [
                            ("down", float(np.nextafter(z, 0.0))),
                            ("up", float(np.nextafter(z, np.inf))),
                        ]
                    )
                for variant_name, variant_z in variants:
                    branch = "neg" if (target_idx + alpha_idx) % 2 == 0 else "pos"
                    add_case(
                        f"{branch}_upper_{alpha_name}_{target_idx}_{variant_name}_dps{dps}",
                        branch,
                        "upper",
                        alpha,
                        variant_z,
                        beta_by_idx[(alpha_idx + target_idx + 1) % len(beta_by_idx)],
                        ratio_by_idx[(alpha_idx + target_idx + 2) % len(ratio_by_idx)],
                        dps,
                    )

    rows = []
    fallback_count = 0
    coverage = {
        "lower": 0,
        "upper": 0,
        "pos": 0,
        "neg": 0,
        "integer": 0,
        "half": 0,
        "noninteger": 0,
        "nondefault_dps": 0,
        "python_underflow_zero_pval": 0,
    }
    for case in cases:
        dps = int(case["deep_accuracy"])
        x_value = float(case["x"])
        alpha = float(case["alpha"])
        beta = float(case["beta"])
        raw_gamma_prob = gamma.cdf(x_value, alpha, scale=beta)
        fallback_used = bool(raw_gamma_prob > 0.999999999 or raw_gamma_prob < 0.00000000001)
        fallback_count += int(fallback_used)
        if not fallback_used:
            raise SystemExit(f"tail case {case['case']} did not trigger blitz fallback")

        coverage[str(case["tail"])] += 1
        coverage[str(case["branch"])] += 1
        if dps != 50:
            coverage["nondefault_dps"] += 1
        if float(alpha).is_integer():
            coverage["integer"] += 1
        elif float(alpha - 0.5).is_integer():
            coverage["half"] += 1
        else:
            coverage["noninteger"] += 1

        mp.dps = dps
        mp.prec = dps
        python_gamma_prob = blitzgsea.gammacdf(x_value, alpha, beta, dps=dps)
        python_survival_prob = 1 - python_gamma_prob
        pos_ratio_value = float(case["pos_ratio"])
        if case["branch"] == "pos":
            combined = min(python_gamma_prob * pos_ratio_value + 1 - pos_ratio_value, 1)
            python_prob_two_tailed = min(mp.mpf("0.5"), 1 - combined)
            python_nes = -blitzgsea.invcdf(1 - min(1, python_prob_two_tailed))
        else:
            combined = min(
                python_gamma_prob - (python_gamma_prob * pos_ratio_value) + pos_ratio_value, 1
            )
            python_prob_two_tailed = min(mp.mpf("0.5"), 1 - combined)
            if python_prob_two_tailed == mp.mpf("0.5"):
                python_prob_two_tailed -= python_gamma_prob
            python_nes = -blitzgsea.invcdf(min(1, python_prob_two_tailed))
        python_pval = min(mp.mpf(1), 2 * python_prob_two_tailed)

        if python_pval == 0 and not np.isfinite(float(python_nes)):
            coverage["python_underflow_zero_pval"] += 1
            gamma_prob = python_gamma_prob
            survival_prob = python_survival_prob
            prob_two_tailed = python_prob_two_tailed
            pval = python_pval
            nes = python_nes
        else:
            mp.dps = dps
            with mp.extradps(mp.dps):
                x_mpf = mp.mpf(x_value)
                alpha_mpf = mp.mpf(alpha)
                beta_mpf = mp.mpf(beta)
                pos_ratio_mpf = mp.mpf(float(case["pos_ratio"]))
                gamma_prob = mp.gammainc(alpha_mpf, 0, x_mpf / beta_mpf, regularized=True)
                survival_prob = 1 - gamma_prob
                if case["branch"] == "pos":
                    combined = min(gamma_prob * pos_ratio_mpf + 1 - pos_ratio_mpf, 1)
                    prob_two_tailed = min(mp.mpf("0.5"), 1 - combined)
                    nes = -blitzgsea.invcdf(1 - min(1, prob_two_tailed))
                else:
                    combined = min(gamma_prob - (gamma_prob * pos_ratio_mpf) + pos_ratio_mpf, 1)
                    prob_two_tailed = min(mp.mpf("0.5"), 1 - combined)
                    if prob_two_tailed == mp.mpf("0.5"):
                        prob_two_tailed -= gamma_prob
                    nes = -blitzgsea.invcdf(min(1, prob_two_tailed))
                pval = min(mp.mpf(1), 2 * prob_two_tailed)

        rows.append(
            {
                **case,
                "z": x_value / beta,
                "raw_gamma_prob": float(raw_gamma_prob),
                "fallback_used": fallback_used,
                "gamma_prob": float(gamma_prob),
                "survival_prob": float(survival_prob),
                "prob_two_tailed": float(prob_two_tailed),
                "pval": float(pval),
                "nes": float(nes),
            }
        )

    if fallback_count == 0:
        raise SystemExit("tail fallback trace did not include any fallback-triggering rows")
    missing = [name for name, count in coverage.items() if count == 0]
    if missing:
        raise SystemExit(f"tail fallback trace is missing coverage for: {', '.join(missing)}")

    header = [
        "case",
        "branch",
        "x",
        "alpha",
        "beta",
        "z",
        "pos_ratio",
        "deep_accuracy",
        "raw_gamma_prob",
        "fallback_used",
        "gamma_prob",
        "survival_prob",
        "prob_two_tailed",
        "pval",
        "nes",
    ]
    TAIL_TRACE.write_text(
        "\t".join(header)
        + "\n"
        + "".join(
            "\t".join(
                str(row[key])
                if isinstance(row[key], str)
                else str(row[key]).lower()
                if isinstance(row[key], (bool, np.bool_))
                else repr(float(row[key]))
                if isinstance(row[key], float)
                else str(row[key])
                for key in header
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    versions = check_environment()
    OUT.mkdir(parents=True, exist_ok=True)

    signature, library, _, _ = build_inputs()
    write_fixture("synthetic", signature, library, emit_trace=True)

    edge_signature, edge_library = build_edge_inputs()
    write_fixture("edgecases", edge_signature, edge_library)

    publication_signature, publication_library, publication_metadata = load_publication_inputs()
    write_fixture("publication_fgsea", publication_signature, publication_library)
    write_tail_fallback_trace()

    (OUT / "versions.json").write_text(
        json.dumps(
            {
                "PYTHONHASHSEED": "0",
                "multiprocessing_reference": "deterministic processes=4 imap chunksize=1 schedule",
                "publication_fixture": publication_metadata,
                **versions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
