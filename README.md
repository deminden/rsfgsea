# High-Performance GSEA Analysis

High-performance Rust implementation of preranked Gene Set Enrichment Analysis (GSEA). Designed as a drop-in, optimized alternative to the R `fgsea` package, implementing the same robust statistical method with significantly improved speed.

## Features

- **fgsea-Compatible Statistics**: Reproduces fgsea-style simple and multilevel workflows with NES, adjusted p-values, and `log2err`; current CPU multilevel parity vs R is near floating-point noise (max abs diff about `5e-9`, see [Precision vs R](#precision-vs-r) and `crates/rsfgsea/tests/r_validation.rs`).
- **Fast Core Algorithms**: Uses \(O(k)\) ES kernels and size-group batching to avoid redundant work; on large 1-worker benchmark workloads, `rsfgsea` is about **3.0x-4.3x faster** than R `fgsea` in this repo's current benchmark setup.
- **Deterministic + High-Throughput RNG Paths**: R-compatible MT19937-based paths are used for parity-sensitive execution, with optimized RNG/shuffle paths available in GPU-oriented flows.
- **Hybrid CPU/GPU Engine (Experimental)**: WebGPU accelerates large simple-stage screening/null generation, while multilevel refinement uses the parity-focused CPU kernel.

## Usage

### As a Binary

```bash
# Build
git clone https://github.com/deminden/rsfgsea
cd rsfgsea
cargo build --workspace --release

# Minimal run (defaults: mode=fgsea, nPermSimple=1000, seed=42)
./target/release/rsfgsea \
    --ranks data/pearson_symbols.rnk \
    --gmt data/h.all.v2025.1.Hs.symbols.gmt \
    --output results.tsv

# Full parameter example
./target/release/rsfgsea \
    --ranks data/pearson_symbols.rnk \
    --gmt data/Human_GO_AllPathways_noPFOCR_with_GO_iea_March_01_2024_symbol_renamed.gmt \
    --mode fgsea \
    --nPermSimple 1000 \
    --minSize 1 \
    --maxSize 5000 \
    --scoreType std \
    --gseaParam 1 \
    --eps 1e-50 \
    --sampleSize 101 \
    --nproc 0 \
    --output results.tsv
```

### As a Crate

Add to `Cargo.toml`:
```toml
[dependencies]
rsfgsea = { git = "https://github.com/deminden/rsfgsea" }
```

Use in code:
```rust
use rsfgsea::prelude::*;

let ranks = RankedList::new(genes, scores);
let pathways = read_gmt("pathways.gmt")?;

// Wrapper-style API (closest to R fgsea semantics).
let results = fgsea(
    &ranks,
    &pathways.pathways,
    None,   // nperm; Some(N) forces simple mode
    1000,   // nPermSimple
    42,     // seed
    1,      // minSize
    ranks.len() - 1,
    1e-50,  // eps
    ScoreType::Std,
    1.0,    // gseaParam
);

// Explicit wrapper sampleSize configuration.
let results = fgsea_with_sample_size(
    &ranks,
    &pathways.pathways,
    None,
    1000,
    42,
    1,
    ranks.len() - 1,
    1e-50,
    ScoreType::Std,
    1.0,
    101,
);
```
### Python Extension

The Python extension lives in `crates/rsfgseapy` and is built with `maturin`.

Current status:
- `run_gsea_py(...)` exposes CPU and hybrid-GPU execution.
- CPU mode supports `fgsea`, `multilevel`, and `simple`.
- GPU mode supports:
  - `mode="fgsea"` wrapper semantics
  - `mode="simple"` for GPU simple-only execution
  - `mode="multilevel"` for GPU screening with CPU multilevel refinement

```bash
# Build
git clone https://github.com/deminden/rsfgsea
cd rsfgsea
cargo build --workspace --release

# Install Python extension
cd crates/rsfgseapy
#pip install maturin # if you don't have maturin installed
maturin develop --release

# Optional: run Python binding tests
pytest tests
```

Usage example:
```python
import rsfgseapy

# Minimal wrapper-style run for light users.
results = rsfgseapy.run_gsea_py(
    ranks={"GENE_A": 2.0, "GENE_B": 1.0, "GENE_C": -1.0, "GENE_D": -2.0},
    gmt_path="pathways.gmt",
)

for res in results:
    print(res["pathway"], res["pval"])
```

Full example:

```python
import rsfgseapy

results = rsfgseapy.run_gsea_py(
    ranks={"GENE_A": 10.5, "GENE_B": 8.4, ...},
    gmt_path="pathways.gmt",
    mode="fgsea",
    gpu=False,
    nPermSimple=1000,
    nperm=None,
    nproc=0,
    minSize=1,
    maxSize=None,
    eps=1e-50,
    sampleSize=101,
    scoreType="std",
    gseaParam=1.0
)
for res in results:
    print(f"Pathway: {res['pathway']}, NES: {res['nes']}, p-val: {res['pval']}")
```

How to think about `nPermSimple` vs `nperm`:

- `nPermSimple` is the normal simple-stage permutation count used by wrapper mode.
- `nperm` is an explicit override that forces wrapper mode into fixed-permutation simple behavior.
- Leave `nperm=None` unless you intentionally want simple mode.
- Change `nPermSimple` when you want to tune the default wrapper-stage screening budget.

Default fgsea-style parameters in this project interfaces:
- `mode=fgsea`
- `gpu=False`
- `nPermSimple=1000`
- `nperm=None` (if set, wrapper mode switches to simple permutations)
- `minSize=1`
- `maxSize=length(stats)-1` (computed automatically if omitted)
- `eps=1e-50`
- `sampleSize=101`
- `scoreType="std"`
- `gseaParam=1.0`
- `nproc=0`


#### GPU Support
To enable GPU acceleration, build with the `gpu` feature:
```bash
cargo build --release --features gpu
```

Current status:
- `run_gsea_gpu(...)` is a hybrid path: GPU is used for simple-stage null generation / ES screening, and CPU is still used for multilevel refinement.
- The main `rsfgsea` CLI can call the hybrid GPU path with `--gpu` when built with `--features gpu`.
- The current CLI GPU path supports wrapper-style `--mode fgsea`, including `--nperm` to force simple-only execution and custom `--sampleSize` / `--eps` for the CPU multilevel refinement stage.
- `--gpu` still rejects `--mode simple` and `--mode multilevel`; those mode names are not wired separately yet.
- `run_gsea_gpu(...)` requires a usable non-CPU WebGPU adapter. Software adapters such as `llvmpipe` are rejected.

Use the hybrid GPU runner in Rust code:
```rust
let results = rsfgsea::algo::run_gsea_gpu(
    &ranks, 
    &pathways, 
    1000,           // initial simple permutations
    42,             // seed
    15, 500,        // size limits
    ScoreType::Std, 
    1.0             // gsea_param
)?;
```

**Hardware Selection (Environment Variables):**
- `WGPU_BACKEND=vulkan`: recommended first on Linux/WSL2 to prefer stable native Vulkan adapters.
- `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`: on WSL2, helps pick the discrete GPU when D3D12 translation is used.
- `RSFGSEA_GPU_ALLOW_GL=1`: opt-in fallback only; GL-translated adapters can be unstable on some Mesa/WSL stacks.

## Documentation

Project documentation is split into compact guides in [`docs/`](./docs/README.md):

- [`docs/cli.md`](./docs/cli.md)
- [`docs/python.md`](./docs/python.md)
- [`docs/algorithms.md`](./docs/algorithms.md)
- [`docs/development.md`](./docs/development.md)
- [`docs/reproducibility.md`](./docs/reproducibility.md)

## Input Format

**Ranked List (`.rnk`)**:
Tab-separated file with gene names in the first column and correlation scores in the second.
```
GENE1   12.34
GENE2   8.90
```

**GMT File (`.gmt`)**:
Standard Gene Matrix Transposed format.
```
PATHWAY_A  description  GENE1  GENE2  GENE3
PATHWAY_B  description  GENE4  GENE5
```

## Performance Comparison (Computation Only)

Benchmarked on **AMD Ryzen 9 7950X3D**. Times are **median of 5 runs** (after one warmup run).

**Benchmark setup**:
- Ranked list: `data/pearson_symbols.rnk` (356 genes)
- Small pathways: `data/h.all.v2025.1.Hs.symbols.gmt` (50 total, 37 passing size filters)
- Large pathways: `data/Human_GO_AllPathways_noPFOCR_with_GO_iea_March_01_2024_symbol_renamed.gmt` (29,705 total, 5,582 passing size filters)
- Size filters: `minSize=1`, `maxSize=5000`
- Rust timing source: CLI compute timer (`GSEA_COMP_TIME_MS`) in release mode
- R timing source: `system.time(...)["elapsed"]` around `fgsea` calls
- R multicore modes: `BiocParallel::MulticoreParam(workers=16|32)` passed as `BPPARAM`

### Modes

- **Multilevel mode** (`run_gsea` / R `fgseaMultilevel`): adaptive multilevel Monte Carlo for very small p-values.
- **Simple mode** (`run_gsea_simple` / R `fgseaSimple`): fixed permutation sampling (`nPermSimple`).
- These modes have different compute structure, so multicore scaling is expected to differ between them.

### Benchmark Results

#### Multilevel Mode
Parameters: `eps=1e-50`, `sampleSize=101` (R), `nPermSimple=1000` (rsfgsea simple stage).

| Workload | Rust 1 worker (ms) | R 1 worker (ms) | Rust 16 workers (ms) | R 16 workers (ms) | Rust 32 workers (ms) | R 32 workers (ms) | Rust scale vs 1w | R scale vs 1w | Rust vs R (1w) | Rust vs R (16w) | Rust vs R (32w) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| Small (50 pathways) | 2 | 43 | 3 | 69 | 4 | 72 | 16w: 0.67x, 32w: 0.50x | 16w: 0.62x, 32w: 0.60x | 21.50x | 23.00x | 18.00x |
| Large (29,705 pathways) | 276 | 1,190 | 104 | 939 | 116 | 1,030 | 16w: 2.65x, 32w: 2.38x | 16w: 1.27x, 32w: 1.16x | 4.31x | 9.03x | 8.88x |

#### Simple Mode
Parameters: `nPermSimple=1,000,000` (small), `nPermSimple=10,000` (large).

| Workload | Rust 1 worker (ms) | R 1 worker (ms) | Rust 16 workers (ms) | R 16 workers (ms) | Rust 32 workers (ms) | R 32 workers (ms) | Rust scale vs 1w | R scale vs 1w | Rust vs R (1w) | Rust vs R (16w) | Rust vs R (32w) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| Small (50 pathways, 1M perms) | 814 | 2,544 | 829 | 406 | 847 | 468 | 16w: 0.98x, 32w: 0.96x | 16w: 6.27x, 32w: 5.44x | 3.13x | 0.49x | 0.55x |
| Large (29,705 pathways, 10k perms) | 948 | 2,875 | 687 | 774 | 797 | 829 | 16w: 1.38x, 32w: 1.19x | 16w: 3.71x, 32w: 3.47x | 3.03x | 1.13x | 1.04x |

#### Rust Thread-Scaling Sweep (Many-Variant View)
Additional Rust-only sweep (same workloads/settings) across `1/2/4/8/16/32` workers (median of 3 runs after warmup):

| Workload | 1w (ms) | 2w (ms) | 4w (ms) | 8w (ms) | 16w (ms) | 32w (ms) | Best scaling vs 1w |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Multilevel / Small | 2 | 2 | 2 | 2 | 3 | 4 | 1.00x (2w) |
| Multilevel / Large | 280 | 175 | 126 | 108 | 106 | 118 | 2.64x (16w) |
| Simple / Small (1M perms) | 850 | 845 | 827 | 849 | 826 | 808 | 1.05x (32w) |
| Simple / Large (10k perms) | 959 | 698 | 686 | 728 | 715 | 800 | 1.40x (4w) |

- This parity-preserving engine keeps outputs aligned across thread counts (`nproc`-invariant).
- Multicore gain is strongest in large multilevel and large simple workloads; small workloads remain overhead-dominated.
- Rust remains faster than R in multilevel mode at all measured worker counts.
- In simple mode, Rust is faster at 1 worker and on the large workload at 16/32 workers; R remains faster on the small 1M-permutation case at 16/32 workers.

## Precision vs R

`rsfgsea` aims for feature and numerical parity with R's `fgsea` package.
- **Validation protocol**: parity tests against R reference outputs are implemented in `crates/rsfgsea/tests/r_validation.rs`.
- **Primary metrics**: max/mean absolute differences for ES, NES, p-value, and adjusted p-value on matched pathways.
- **Examples-folder snapshot** (`data/Folder_with_examples`, 23 files, seed `42`, `nPermSimple=1000`):
  - Multilevel mode vs R `fgseaMultilevel`: max `|ES|` diff `4.988e-09`, max `|NES|` diff `4.983e-09`, max `|pval|` diff `4.975e-09`, max `|padj|` diff `4.965e-09`.
  - Simple mode vs R `fgseaSimple`: max `|ES|` diff `4.988e-09`, max `|NES|` diff `4.983e-09`, max `|pval|` diff `4.975e-09`, max `|padj|` diff `4.965e-09`.
- **Distribution-level parity stats** (examples folder, seed `42`, matched pathways `n=746` across `22` files, absolute differences):

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

- **Finite-value coverage**: p-value NaN mismatch count was `0` in both modes on this run.
- **Interpretation**: in this parity configuration and snapshot, ES/NES/p-value agreement is at floating-point-noise scale.
- **Thread invariance**: with fixed seed/settings, outputs are invariant across `nproc` values in the current CPU parity path.

This section describes the CPU parity path. GPU parity is materially looser at present and is summarized separately in [GPU Accuracy vs R](#gpu-accuracy-vs-r).

### Parity Mode

Use these settings when you want the closest behavior to R `fgsea`:

- CLI:
  - `--mode fgsea` (default): fgsea-style wrapper behavior.
  - `--nPermSimple 1000`: simple stage size used before multilevel refinement.
  - `--seed <int>`: fix seed for reproducible Monte Carlo runs.
  - `--scoreType std|pos|neg`: match the R score mode.
  - `--nproc <N>`: allowed; parity path is thread-count invariant in current implementation.
- To force simple-only comparison (like `fgseaSimple`), use:
  - `--mode simple --nperm <N>`

Speed-oriented paths:
- CPU throughput: increase workers with `--nproc <N>`.
- GPU throughput: use the GPU API (`run_gsea_gpu`) or GPU benchmark/verify binaries built with `--features gpu`.

Examples:
```bash
# R-aligned default (wrapper mode)
./target/release/rsfgsea --mode fgsea --nPermSimple 1000 --seed 42 ...

# Force simple-only mode for direct fgseaSimple-style comparison
./target/release/rsfgsea --mode simple --nperm 100000 --seed 42 ...
```

### GPU Accuracy vs R

Unlike the CPU parity path above, the current hybrid GPU path does not match R at floating-point-noise scale. It uses GPU simple-stage screening/null generation plus CPU multilevel refinement, and its parity characteristics should be interpreted separately from the CPU results.

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

Artifacts:
- `reports/gpu_vs_r/gpu_vs_r_report.md`
- `reports/gpu_vs_r/gpu_vs_r_summary.json`

## Contributing

Contributions are very welcome! 
If you’d like to help improve `rsfgsea`, feel free to open an issue to discuss ideas, report bugs, or request features.

Pull requests are encouraged — especially for:
- performance improvements
- correctness / numerical stability fixes
- additional tests (including cross-validation vs R `fgsea`)
- documentation, examples, and benchmarking

### Development notes

- Please run the full required verification sequence before submitting:
  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  cargo test --workspace --all-features
  ```

## License

MIT License.
