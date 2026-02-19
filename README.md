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
- `MESA_D3D12_DEFAULT_ADAPTER_NAME`: On WSL2, use `NVIDIA` to force selection of your discrete GPU over integrated graphics.

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

### Modes (Important)

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

### Reading the Table

- Compare rows **within the same mode** first (Multilevel vs Multilevel, Simple vs Simple).
- R shows clear multicore gains in **Simple** mode on this dataset, but little gain in **Multilevel** mode.
- Results are machine- and configuration-dependent.

## Precision vs R

`rsfgsea` aims for feature and numerical parity with R's `fgsea` package.
- **Enrichment Scores (ES)**: Matches R `fgsea` behavior within floating-point tolerances.
- **P-values / NES**: validated against R reference outputs with parity tests in `crates/rsfgsea/tests/r_validation.rs`.
- **Current parity snapshot** (`n_perm=5000`, test dataset): mean relative p-value difference is about `3.10%` (distribution mostly `<10%`).
- **Important**: statistical parity is strong, but results are not guaranteed to be bitwise-identical to R in all cases.

### Parity Mode

Default execution favors R-aligned behavior in the simple permutation batch path (sequential RNG/order compatibility).  
Optimized parallel kernels are kept in the codebase as alternatives for speed-oriented modes.

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
