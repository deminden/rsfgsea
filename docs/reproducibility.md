# Reproducibility Guide

This project keeps parity and validation workflows close to the codebase.

## Validation Sources

The main validation layers are:

- unit and integration tests in `crates/rsfgsea/tests`
- GPU-focused tests in `crates/rsfgsea/tests`
- ad hoc and semi-structured helpers in `scripts/`

## Scripts Overview

[`scripts/extract_mapping.R`](/home/den/bio/rsfgsea/scripts/extract_mapping.R)

- exports gene mapping JSON files from an `.rda` source
- use it when local data-prep scripts need mapping files regenerated

[`scripts/prepare_data.py`](/home/den/bio/rsfgsea/scripts/prepare_data.py)

- builds ranked lists from bladder correlation data
- output goes to `crates/rsfgsea/tests/data/muscle_comparison`

[`scripts/prepare_muscle_data.py`](/home/den/bio/rsfgsea/scripts/prepare_muscle_data.py)

- computes per-gene Spearman correlations from muscle expression data
- emits ranked lists into `crates/rsfgsea/tests/data/muscle_comparison`

[`scripts/run_fgsea_comparison.R`](/home/den/bio/rsfgsea/scripts/run_fgsea_comparison.R)

- runs R `fgsea` over the generated ranked lists
- writes a combined R reference result table

[`scripts/test_single_gene.R`](/home/den/bio/rsfgsea/scripts/test_single_gene.R)

- quick manual R-side sanity check for one ranked list
- useful when debugging overlap or pathway-loading issues

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

Examples-folder snapshot: `data/Folder_with_examples`, seed `42`, matched pathways `n=746` across `22` files.

| Mode | Metric | Mean | Median | P95 | Max |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Multilevel | `abs(ES)` | `2.535e-09` | `2.583e-09` | `4.723e-09` | `4.988e-09` |
| Multilevel | `abs(NES)` | `2.555e-09` | `2.619e-09` | `4.707e-09` | `4.983e-09` |
| Multilevel | `abs(pval)` | `2.543e-09` | `2.602e-09` | `4.745e-09` | `4.975e-09` |
| Multilevel | `abs(padj)` | `2.607e-09` | `2.183e-09` | `4.542e-09` | `4.965e-09` |
| Simple | `abs(ES)` | `2.535e-09` | `2.583e-09` | `4.723e-09` | `4.988e-09` |
| Simple | `abs(NES)` | `2.555e-09` | `2.619e-09` | `4.707e-09` | `4.983e-09` |
| Simple | `abs(pval)` | `2.534e-09` | `2.577e-09` | `4.745e-09` | `4.975e-09` |
| Simple | `abs(padj)` | `2.605e-09` | `2.183e-09` | `4.542e-09` | `4.965e-09` |

Notes:

- p-value NaN mismatch count was `0` in both modes on this run
- current CPU parity path is thread-count invariant for fixed settings

### GPU Parity vs R

GPU parity was evaluated against R `fgseaMultilevel` using seeds `[11, 23, 42]`, `nPermSimple=1000`, `sampleSize=101`, `eps=1e-50`.

| Metric | Mean | Median | P95 | Max |
| :--- | ---: | ---: | ---: | ---: |
| `abs(ES)` | `2.535e-09` | `2.531e-09` | `4.736e-09` | `4.998e-09` |
| `abs(NES)` | `1.842e-02` | `1.245e-02` | `5.827e-02` | `1.238e-01` |
| `abs(pval)` | `1.548e-02` | `1.199e-02` | `3.996e-02` | `5.007e-01` |
| `abs(padj)` | `1.248e-02` | `5.101e-03` | `5.784e-02` | `2.458e-01` |

Relative p-value difference (`|p_r - p_gpu| / max(|p_r|, 1e-12)`):

- mean: `4.15%`
- median: `2.37%`
- p95: `13.36%`
- max: `67.39%`
- `<1%`: `26.4%` of pathways
- `<10%`: `90.9%` of pathways

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
