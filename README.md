# High-Performance GSEA Analysis

High-performance Rust implementation of preranked Gene Set Enrichment Analysis (GSEA). Designed as a drop-in, optimized alternative to the R `fgsea` package, implementing the same robust statistical method with significantly improved speed.

## Features

- **Full fgsea logic**: Implements multilevel splitting Monte Carlo for accurate p-value estimation (down to `1e-100` and beyond), Normalized Enrichment Scores (NES), and `log2err`.
- **GPU Acceleration**: WebGPU implementation that combines fast screening with high-precision multilevel refinement.
- **High Efficiency**: Uses $O(k)$ algorithms for Enrichment Score calculation, avoiding redundant scans. **5x faster** than R `fgsea` (multilevel) at scale on CPU, and significantly faster on GPU.
- **Optimized sampling**: Simulates permutations using high-speed non-crypto random number generators (`SmallRng`) and Fisher-Yates shuffling.

## Usage

### As a Binary

```bash
# Build
git clone https://github.com/deminden/rsfgsea
cd rsfgsea
cargo build --workspace --release

# Run GSEA
./target/release/rsfgsea \
    --ranks data/ranks.rnk \
    --gmt data/pathways.gmt \
    --mode fgsea \
    --nPermSimple 1000 \
    # optionally force simple mode like original fgsea(..., nperm=...)
    # --nperm 10000 \
    --minSize 1 \
    --maxSize 355 \
    --scoreType std \
    --gseaParam 1 \
    --eps 1e-50 \
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

let results = run_gsea(
    &ranks, 
    &pathways.pathways, 
    1000,   // nPermSimple
    42,     // seed
    1,      // minSize
    ranks.len() - 1, // maxSize
    1e-50,  // eps
    ScoreType::Std, 
    1.0     // gseaParam
);
```

#### GPU Support
To enable GPU acceleration, build with the `gpu` feature:
```bash
cargo build --release --features gpu
```

Use the GPU-specific runner in your code:
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

### Python Extension

The Python extension lives in `crates/rsfgseapy` and is built with `maturin`.

```bash
# Build
git clone https://github.com/deminden/rsfgsea
cd rsfgsea
cargo build --workspace --release

# Install Python extension
cd crates/rsfgseapy
#pip install maturin # if you don't have maturin installed
maturin develop --release
```

Usage example:
```python
import rsfgseapy

# Prepare inputs
ranks = {"GENE_A": 10.5, "GENE_B": 8.4, ...}
gmt_path = "pathways.gmt"

# Run GSEA
results = rsfgseapy.run_gsea_py(
    ranks=ranks,
    gmt_path=gmt_path,
    nPermSimple=1000,
    nproc=0,
    minSize=1,
    maxSize=None,
    eps=1e-50,
    scoreType="std",
    gseaParam=1.0
)

# Access results
for res in results:
    print(f"Pathway: {res['pathway']}, NES: {res['nes']}, p-val: {res['pval']}")
```

Default fgsea-style parameters in this project interfaces:
- `mode=fgsea`
- `nPermSimple=1000`
- `nperm=None` (if set, wrapper mode switches to simple permutations)
- `minSize=1`
- `maxSize=length(stats)-1` (computed automatically if omitted)
- `eps=1e-50`
- `scoreType="std"`
- `gseaParam=1.0`
- `nproc=0`

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

Benchmarked on **AMD Ryzen 9 7950X3D**. Times are **median of 3 runs** (after one warmup run).

**Benchmark setup**:
- Ranked list: `data/pearson_symbols.rnk` (356 genes)
- Small pathways: `data/h.all.v2025.1.Hs.symbols.gmt` (50 total, 37 passing size filters)
- Large pathways: `data/Human_GO_AllPathways_noPFOCR_with_GO_iea_March_01_2024_symbol_renamed.gmt` (29,705 total, 5,582 passing size filters)
- Size filters: `minSize=1`, `maxSize=5000`
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
| Small (50 pathways) | 6 | 1,233 | 8 | 1,340 | 10 | 1,345 | 16w: 0.75x, 32w: 0.60x | 16w: 0.92x, 32w: 0.92x | 205.5x | 167.5x | 134.5x |
| Large (29,705 pathways) | 574 | 3,216 | 560 | 3,585 | 549 | 3,455 | 16w: 1.03x, 32w: 1.05x | 16w: 0.90x, 32w: 0.93x | 5.6x | 6.4x | 6.3x |

#### Simple Mode
Parameters: `nPermSimple=1,000,000` (small), `nPermSimple=10,000` (large).

| Workload | Rust 1 worker (ms) | R 1 worker (ms) | Rust 16 workers (ms) | R 16 workers (ms) | Rust 32 workers (ms) | R 32 workers (ms) | Rust scale vs 1w | R scale vs 1w | Rust vs R (1w) | Rust vs R (16w) | Rust vs R (32w) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| Small (50 pathways) | 856 | 3,798 | 817 | 1,670 | 843 | 1,780 | 16w: 1.05x, 32w: 1.02x | 16w: 2.27x, 32w: 2.13x | 4.4x | 2.0x | 2.1x |
| Large (29,705 pathways) | 1,289 | 5,249 | 1,229 | 3,408 | 1,227 | 3,313 | 16w: 1.05x, 32w: 1.05x | 16w: 1.54x, 32w: 1.58x | 4.1x | 2.8x | 2.7x |

- R shows clear multicore gains in **Simple** mode on this dataset, but no gain in **Multilevel** mode.

## Precision vs R

`rsfgsea` aims for feature and numerical parity with R's `fgsea` package.
- **Validation protocol**: parity tests against R reference outputs are implemented in `crates/rsfgsea/tests/r_validation.rs`.
- **Primary metrics**: max/mean absolute differences for ES, NES, p-value, and adjusted p-value on matched pathways.
- **Examples-folder snapshot** (`data/Folder_with_examples`, 23 files, seed `42`, `nPermSimple=1000`):
  - Multilevel mode vs R `fgseaMultilevel`: max `|ES|` diff `4.998e-09`, max `|NES|` diff `4.995e-09`, max `|pval|` diff `1.195e-02`, max `|padj|` diff `2.331e-01`.
  - Simple mode vs R `fgseaSimple`: max `|ES|` diff `4.998e-09`, max `|NES|` diff `4.995e-09`, max `|pval|` diff `4.985e-09`, max `|padj|` diff `4.965e-09`.
- **Distribution-level parity stats** (examples folder, seed `42`, matched pathways `n=746` across `22` files, absolute differences):

| Mode | Metric | Mean | Median | P95 | Max |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Multilevel | `abs(ES)` | `2.535e-09` | `2.531e-09` | `4.735e-09` | `4.998e-09` |
| Multilevel | `abs(NES)` | `2.473e-09` | `2.483e-09` | `4.693e-09` | `4.995e-09` |
| Multilevel | `abs(pval)` | `3.270e-05` | `2.682e-09` | `4.785e-09` | `1.195e-02` |
| Multilevel | `abs(padj)` | `9.252e-04` | `2.183e-09` | `4.316e-09` | `2.331e-01` |
| Simple | `abs(ES)` | `2.535e-09` | `2.531e-09` | `4.735e-09` | `4.998e-09` |
| Simple | `abs(NES)` | `2.473e-09` | `2.483e-09` | `4.693e-09` | `4.995e-09` |
| Simple | `abs(pval)` | `2.551e-09` | `2.662e-09` | `4.755e-09` | `4.985e-09` |
| Simple | `abs(padj)` | `2.535e-09` | `2.183e-09` | `4.316e-09` | `4.965e-09` |

- **Finite-value coverage**: p-value NaN mismatch count was `0` in both modes on this run.
- **Interpretation**: ES/NES agreement is typically near floating-point precision; p-value differences are expected to remain stochastic because both implementations use Monte Carlo sampling.
- **Scope**: parity here means statistical agreement under the same method/settings, not bitwise identity of every output field.

### Parity Mode

Use these settings when you want the closest behavior to R `fgsea`:

- CLI:
  - `--mode fgsea` (default): fgsea-style wrapper behavior.
  - `--nPermSimple 1000`: simple stage size used before multilevel refinement.
  - `--seed <int>`: fix seed for reproducible Monte Carlo runs.
  - `--scoreType std|pos|neg`: match the R score mode.
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

GPU parity was evaluated on `data/Folder_with_examples` (23 files) against R `fgseaMultilevel` using seeds `[11, 23, 42]`, `nPermSimple=1000`, `sampleSize=101`, `eps=1e-50` (66 file-seed runs, 2,238 matched pathways total).

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

- Please run formatting and linting before submitting:
  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets --all-features
  cargo test --workspace --all-features

## License

MIT License.
