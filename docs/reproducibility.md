# Reproducibility Guide

This project keeps parity and validation workflows close to the codebase.

## Validation Sources

The main validation layers are:

- unit and integration tests in `crates/rsfgsea/tests`
- GPU-focused tests in `crates/rsfgsea/tests`
- ad hoc and semi-structured helpers in `scripts/`

## Scripts Overview

[`scripts/extract_mapping.R`](../scripts/extract_mapping.R)

- exports gene mapping JSON files from an `.rda` source
- use it when local data-prep scripts need mapping files regenerated

[`scripts/prepare_data.py`](../scripts/prepare_data.py)

- builds ranked lists from bladder correlation data
- output goes to `crates/rsfgsea/tests/data/muscle_comparison`

[`scripts/prepare_muscle_data.py`](../scripts/prepare_muscle_data.py)

- computes per-gene Spearman correlations from muscle expression data
- emits ranked lists into `crates/rsfgsea/tests/data/muscle_comparison`

[`scripts/run_fgsea_comparison.R`](../scripts/run_fgsea_comparison.R)

- runs R `fgsea` over the generated ranked lists
- writes a combined R reference result table

[`scripts/test_single_gene.R`](../scripts/test_single_gene.R)

- quick manual R-side sanity check for one ranked list
- useful when debugging overlap or pathway-loading issues

[`scripts/generate_blitz_large_reference.py`](../scripts/generate_blitz_large_reference.py)

- generates repeated full-precision `blitzgsea 1.3.54` results and optional fixed-schedule traces
- validates the pinned Python package/thread environment and repeated output hashes

[`scripts/compare_blitz_precision.py`](../scripts/compare_blitz_precision.py)

- compares full-precision Blitz outputs by absolute error, ULP distance, finite class, ordering, size, and leading edge

[`scripts/bench_blitz_ab.py`](../scripts/bench_blitz_ab.py)

- runs a CPU-pinned, paired, alternating baseline/candidate Blitz benchmark
- records raw timings, hashes, bootstrap intervals, and a strict acceptance gate

## Practical Workflow

When you need to validate behavior against R:

1. generate or refresh ranked lists
2. run the R comparison scripts
3. run the Rust and Python paths on the same inputs
4. compare result files
5. only then interpret statistical differences

## Benchmark Reference

Benchmarked on **AMD Ryzen 9 7950X3D**. Times are **median of 5 runs** after one warmup run.

### Local Optimization Benchmark

This is the fast Criterion benchmark used for Rust-core optimization. It is
synthetic but real-shaped: 10k ranked genes, overlapping pathways, pathway sizes
in the 15-500 range, smooth score tails, and seeded enriched regions.

Run:

```bash
cargo bench -p rsfgsea --bench gsea_bench
```

Latest local Criterion snapshot, release bench mode, median of 10 samples:

| Benchmark | Workload | Median |
| :--- | :--- | ---: |
| `calculate_es_10k_100` | ES kernel, 10k genes / 100 hits | `294 ns` |
| `representative_simple` | 10k genes / 1k pathways / 10k permutations | `2.282 s` |
| `representative_multilevel` | 10k genes / 1k pathways / `nPermSimple=1000` | `3.438 s` |

For larger thread-scaling checks, run the opt-in 5k-pathway matrix:

```bash
for t in 1 2 4 8 16; do
  RAYON_NUM_THREADS=$t RSFGSEA_THREAD_MATRIX_BENCH=1 \
    cargo bench -p rsfgsea --bench gsea_bench
done
```

Use `RSFGSEA_PERM_HEAVY_BENCH=1` for the 100k-permutation simple-mode profile,
and `RSFGSEA_HEAVY_BENCH=1` for the larger 20k-gene / 15k-pathway profile.

### Native Blitz Optimization Benchmark

Blitz-specific Criterion groups are opt-in because default blitz calibration is
slow enough to distort routine benchmark runs:

```bash
RSFGSEA_BLITZ_BENCH=1 cargo bench -p rsfgsea --bench gsea_bench
```

For file-backed Python/Rust comparison on the positive DESeq2 stress workload,
use:

```bash
scripts/bench_blitz_speed.py --reps 1 --json
```

For a change-acceptance benchmark, preserve release binaries built with the
same toolchain and flags, then use the paired harness. It pins the process to
the requested CPUs, alternates `baseline/candidate` execution order, retains
all raw timings and output hashes, and bootstraps paired ratios with a fixed
seed:

```bash
scripts/bench_blitz_ab.py \
  --baseline-bin target/release/rsfgsea-baseline \
  --candidate-bin target/release/rsfgsea-candidate \
  --reps 30 \
  --warmups 2 \
  --cpu-list 8,10,12,14 \
  --bootstrap-resamples 200000 \
  --equivalence-margin-pct 1 \
  --output data/derived/blitz_precision/lung_vs_muscle_go_bp/speed_ab.json
```

The gate requires the candidate/baseline geometric-mean ratio to be at most
`1.0` and the 95% upper confidence bound to be at most `1.01`, for both core
compute time and end-to-end wall time. It also reports within-binary output
determinism and exits nonzero if the combined acceptance gate fails. The
current 30-pair full-cold result against commit `3aeb6ba` passed: compute ratio
`0.99324` (change `-0.676%`, 95% ratio CI `0.97925–1.00735`) and wall ratio
`0.99949` (change `-0.051%`, 95% ratio CI `0.99408–1.00480`). Every measured
output hash was deterministic within its binary. This establishes no speed
loss under the declared 1% equivalence gate on this machine; it is not a
general cross-machine speed claim.

The three canonical Python reference runs took `15.650 s`, `16.335 s`, and
`15.469 s`. Those values document reference-generation cost only; the paired
Rust gate above is the acceptance evidence for this implementation change.

Native blitz also has an in-process null-model cache for repeated identical
library calls. The CLI leaves it off by default because one-shot subprocess runs
cannot reuse process memory; Criterion reports that path as
`blitz_full_memory_cache`.

For local workstation builds, `target-cpu=native` is worth testing:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release -p rsfgsea
```

Keep `target-cpu=native` builds out of portable release artifacts unless you
intentionally want a binary specialized to the build host. Rebenchmark them
with the paired gate rather than relying on an older single-run screen.

The canonical reference environment is locked by
[`reference/blitz/uv.lock`](../reference/blitz/uv.lock): Python `3.12.9`,
blitzgsea `1.3.54`, NumPy `2.5.1`, SciPy `1.18.0`, pandas `3.0.3`,
statsmodels `0.14.6`, and mpmath `1.4.1`. Generate the full-precision reference
and optional fixed-schedule trace with:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --project reference/blitz --frozen \
  python scripts/generate_blitz_large_reference.py \
  --reps 3 \
  --fixed-schedule-trace
```

The generator refuses a dependency-version or thread-environment mismatch and
requires all repeated result hashes to match. Compare a round-trip-precision
CLI output with one generated reference using:

```bash
scripts/compare_blitz_precision.py \
  --reference data/derived/blitz_precision/lung_vs_muscle_go_bp/python_blitzgsea_1_3_54.rep1.tsv \
  --observed data/derived/blitz_precision/lung_vs_muscle_go_bp/rsfgsea_candidate.tsv \
  --output data/derived/blitz_precision/lung_vs_muscle_go_bp/rsfgsea_candidate.comparison.json
```

Generated references and reports live under ignored `data/derived/`; commit the
generators and focused regression fixtures, not the multi-megabyte local audit
outputs. The audited input contained 63,904 ranks and 5,343 source GMT
pathways, of which 5,324 were scored. Input SHA-256 values were
`54d7e5b11c6ccdffc3dc289ed8f4cfc8e11594d093fe0050cf4e587400bb794f`
for the ranks and
`0fc02458765cfce6c9348dce9f7a9397c6caf7c09ba318f468135d7ac60342ee`
for the GMT; all three canonical four-process Python runs produced
`8216f5197730653e59dd98b4f3af29b4637e7b53ae560f58a2be06f8c541da69`.

The current comparison reports 5,324 shared pathways, no size or leading-edge
set mismatches, and finite maximum absolute differences of ES `4.4e-16`, NES
`3.3e-15`, p-value `1.8e-15`, and FDR `2.7e-15`. Updating the interpolation
operation order to SciPy 1.18's stable de Boor weights reduced the immediate
pre-change NES maximum of `1.508e-8` by about 4.53 million× and the FDR maximum
by about 3.19×. This is specifically Blitz reference parity: extreme-tail
context rounding follows the locked upstream finite-precision result and is
not presented as improved mathematical truth. The exact current audit is
[`docs/evidence/blitz-latest.json`](./evidence/blitz-latest.json).

### R fgsea Comparison Benchmark

Benchmark setup:

- ranked list: `data/pearson_symbols.rnk` (356 genes)
- small pathways: `data/h.all.v2025.1.Hs.symbols.gmt` (50 total, 37 passing size filters)
- large pathways: `data/Human_GO_AllPathways_noPFOCR_with_GO_iea_March_01_2024_symbol_renamed.gmt` (29,705 total, 5,582 passing size filters)
- size filters: `minSize=1`, `maxSize=5000`
- Rust timing source: CLI `GSEA_COMP_TIME_MS` in release mode
- R timing source: `system.time(...)["elapsed"]`
- R multicore runs: `BiocParallel::MulticoreParam(workers=16|32)`

### Multilevel Mode

Parameters: `eps=1e-50`, `sampleSize=101`, `nPermSimple=1000`.

| Workload | Rust 1 worker (ms) | R 1 worker (ms) | Rust 16 workers (ms) | R 16 workers (ms) | Rust 32 workers (ms) | R 32 workers (ms) | Rust scale vs 1w | R scale vs 1w | Rust vs R (1w) | Rust vs R (16w) | Rust vs R (32w) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| Small (50 pathways) | 2 | 42 | 3 | 75 | 3 | 72 | 16w: 0.67x, 32w: 0.67x | 16w: 0.56x, 32w: 0.58x | 21.00x | 25.00x | 24.00x |
| Large (29,705 pathways) | 282 | 1,158 | 105 | 977 | 106 | 1,030 | 16w: 2.69x, 32w: 2.66x | 16w: 1.19x, 32w: 1.12x | 4.11x | 9.30x | 9.72x |

### Simple Mode

Parameters: `nPermSimple=1,000,000` for the small workload and `10,000` for the large workload.

| Workload | Rust 1 worker (ms) | R 1 worker (ms) | Rust 16 workers (ms) | R 16 workers (ms) | Rust 32 workers (ms) | R 32 workers (ms) | Rust scale vs 1w | R scale vs 1w | Rust vs R (1w) | Rust vs R (16w) | Rust vs R (32w) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| Small (50 pathways, 1M perms) | 720 | 2,597 | 722 | 423 | 726 | 564 | 16w: 1.00x, 32w: 0.99x | 16w: 6.14x, 32w: 4.60x | 3.61x | 0.59x | 0.78x |
| Large (29,705 pathways, 10k perms) | 962 | 2,864 | 674 | 798 | 683 | 799 | 16w: 1.43x, 32w: 1.41x | 16w: 3.59x, 32w: 3.58x | 2.98x | 1.18x | 1.17x |

### Real-Data Memory Snapshot

On the committed muscle-comparison validation workload
(`h.all.v2025.1.Hs.symbols.gmt`, 12 `.rnk` files, `nPermSimple=1000`,
`sampleSize=101`, `seed=42`, `minSize=15`, `maxSize=500`, `eps=1e-10`),
Rust release validation used `81 MB` peak RSS and completed in `0.21 s`.
The matched R `fgseaMultilevel` run used `329 MB` peak RSS and completed in
`2.56 s`. That is about `4.1x` lower peak memory for Rust on this real-data
check.

### Rust Thread-Scaling Sweep

Additional Rust-only sweep across `1/2/4/8/16/32` workers, median of 3 runs after warmup.

| Workload | 1w (ms) | 2w (ms) | 4w (ms) | 8w (ms) | 16w (ms) | 32w (ms) | Best scaling vs 1w |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Multilevel / Small | 2 | 2 | 2 | 2 | 3 | 3 | 1.00x (1-8w) |
| Multilevel / Large | 273 | 171 | 127 | 105 | 101 | 100 | 2.73x (32w) |
| Simple / Small (1M perms) | 707 | 704 | 707 | 684 | 706 | 708 | 1.03x (8w) |
| Simple / Large (10k perms) | 909 | 676 | 653 | 644 | 653 | 660 | 1.41x (8w) |

## Precision Reference

### CPU Parity vs R

Classic multilevel and Blitz are now reported on the same large input:
`data/deseq2_positive_ranks/lung_vs_muscle.rnk` (63,904 ranks) and
`data/GO_Biological_Process_2025.gmt` (5,343 source pathways; 5,324 scored).
The classic reference used R `4.4.3`, fgsea `1.37.2`, seed `42`,
`nPermSimple=1000`, `sampleSize=101`, `minSize=5`, `maxSize=4000`,
`eps=1e-10`, `scoreType="std"`, and `gseaParam=1`.
The shared input makes the audit scales comparable, but each mode still targets
its own reference implementation; the error maxima do not compare statistical
quality between classic fgsea and Blitz.

| Mode | Metric | Exact | Mean | Median | P95 | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Multilevel | `abs(ES)` | `5,324` | `0` | `0` | `0` | `0` |
| Multilevel | `abs(NES)` | `5,324` | `0` | `0` | `0` | `0` |
| Multilevel | `abs(pval)` | `3,809` | `4.1e-18` | `0` | `3.1e-17` | `8.3e-17` |
| Multilevel | `abs(padj)` | `2,316` | `3.8e-17` | `5.2e-18` | `1.2e-16` | `2.8e-16` |

There were no missing pathways, size mismatches, leading-edge mismatches, or
finite-class mismatches. Release outputs from one and 16 workers were
byte-identical. The exact audit metadata is
[`docs/evidence/classic-latest.json`](./evidence/classic-latest.json).

### GPU Parity vs R

There is no current committed GPU parity table. The old `nPermSimple=1000`
table was removed because it mostly captured Monte Carlo noise in p-values and
FDR rather than useful GPU-vs-R behavior.

For new GPU-vs-R comparisons:

- use `nPermSimple=100000` as the practical baseline
- use `nPermSimple=10000` only for smoke checks
- use `nPermSimple=1000000` for final tail/stress comparisons when runtime allows
- keep `sampleSize`, `eps`, `scoreType`, `gseaParam`, size filters, seeds, and
  input files identical on both sides
- record GPU model, driver, backend, and WSL2 D3D12 environment variables if used

The GPU path has fixed overhead and benefits most from larger work batches:
many tested pathways, larger pathway collections, and higher simple-stage
permutation counts. Small example datasets are useful for checking that the
adapter works, but they are too small to show the intended GPU advantage.

## What To Record

When saving parity or benchmark outputs, record:

- input files
- mode
- permutation counts
- `sampleSize`
- `scoreType`
- `gseaParam`
- seed
- whether GPU was used

Without that context, result diffs are hard to interpret later.

## Suggested Cleanup Direction

The scripts are useful, but they are still local workflow tools rather than a polished pipeline.

If this project grows, the next step should be:

- one documented reproducibility entrypoint
- one place for required input paths
- one report format for parity summaries

That would make the current script collection easier to use in CI or by other contributors.
