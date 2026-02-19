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
    --nperm 1000 \
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
    1000,   // permutations
    42,     // seed
    15,     // min_size
    500,    // max_size
    1e-10,  // eps
    ScoreType::Std, 
    1.0     // gsea_param
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
    n_perm=1000,
    min_size=15,
    max_size=500,
    eps=1e-10
)

# Access results
for res in results:
    print(f"Pathway: {res['pathway']}, NES: {res['nes']}, p-val: {res['pval']}")
```

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

Benchmarked on **AMD Ryzen 9 7950X3D (16 cores, 32 threads)**. Times exclude I/O and are **median of 3 runs**.

Inputs:
- Ranked list: `data/pearson_symbols.rnk` (356 genes)
- Small pathways: `data/h.all.v2025.1.Hs.symbols.gmt` (50 total, 37 passing size filters)
- Large pathways: `data/Human_GO_AllPathways_noPFOCR_with_GO_iea_March_01_2024_symbol_renamed.gmt` (29,705 total, 5,582 passing size filters)
- Size filters: `minSize=1`, `maxSize=5000`

### 1. Multilevel GSEA
*Parameters: `eps=1e-50`, `sampleSize=101` (R), `nperm=1000` (rsfgsea simple stage).*

| Pathways | Implementation | 1 Thread (ms) | 32 Threads (ms) | Speedup (32T) |
| :--- | :--- | :--- | :--- | :--- |
| **50** (Small) | **rsfgsea** | **2** | **2** | **1.0x** |
| | R `fgseaMultilevel` | 156 | 168 | 0.9x |
| **29,705** (Large) | **rsfgsea** | **258** | **268** | **1.0x** |
| | R `fgseaMultilevel` | 963 | 966 | 1.0x |

### 2. Simple GSEA
*Parameters: `nperm=1,000,000` (small), `nperm=10,000` (large).*

| Pathways | Variant | Implementation | 1 Thread (ms) | 32 Threads (ms) | Speedup (32T) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **50** | 1M Perms | **rsfgsea** | **822** | **819** | **1.0x** |
| | | R `fgseaSimple` | 3,827 | 1,476 | 2.6x |
| **29,705** | 10k Perms | **rsfgsea** | **979** | **1,013** | **1.0x** |
| | | R `fgseaSimple` | 3,086 | 884 | 3.5x |

**Note**: these are current repo results on this machine/configuration; thread scaling behavior depends on the active execution path and workload shape.

## Precision vs R

`rsfgsea` aims for feature and numerical parity with R's `fgsea` package.
- **Enrichment Scores (ES)**: Matches R `fgsea` behavior within floating-point tolerances.
- **P-values / NES**: validated against R reference outputs with the parity tests in `crates/rsfgsea/tests/r_validation.rs`.

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
