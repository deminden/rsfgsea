# High-Performance GSEA Analysis

High-performance Rust implementation of preranked Gene Set Enrichment Analysis (GSEA). Designed as a drop-in, optimized alternative to the R `fgsea` package, implementing the same robust statistical method with significantly improved speed.

## Features

- **Full fgsea logic**: Implements multilevel splitting Monte Carlo for accurate p-value estimation (down to `1e-50`), Normalized Enrichment Scores (NES), and `log2err`.
- **High Efficiency**: Uses $O(k)$ algorithms for Enrichment Score calculation, avoiding redundant scans. **5x faster** than R `fgsea` (multilevel) at scale.
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

Benchmarked on **AMD Ryzen 9 7950X3D (16 cores, 32 threads)**. Times exclude I/O.

### 1. Multilevel GSEA
*Parameters: `eps=1e-50`, `nPermSimple=1000`. Dataset: 356 genes.*

| Pathways | Implementation | 1 Thread (ms) | 32 Threads (ms) | Speedup (32T) |
| :--- | :--- | :--- | :--- | :--- |
| **50** (Small) | **rsfgsea** | **8.5 ms** | **1.97 ms** | **30.9x** |
| | R `fgsea` | 45.2 ms | 60.90 ms | 1.0x |
| **29,705** (Large) | **rsfgsea** | **358.2 ms** | **21.3 ms** | **49.8x** |
| | R `fgsea` | 1,080.8 ms | 1,060.9 ms | 1.0x |

### 2. Simple GSEA
*Parameters: Fixed permutations. Dataset: 356 genes.*

| Pathways | Variant | Implementation | 1 Thread (ms) | 32 Threads (ms) | Speedup (32T) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **50** | 1M Perms | **rsfgsea** | **560.4 ms** | **85.09 ms** | **7.9x** |
| | | R `fgseaSimple` | 4,205.0 ms | 670.11 ms | 1.0x |
| **29,705** | 10k Perms | **rsfgsea** | **2,850.1 ms** | **143.6 ms** | **6.8x** |
| | | R `fgseaSimple` | 3,920.5 ms | 982.7 ms | 1.0x |

**Note**: `rsfgsea` scales efficiently with thread count, whereas R's `fgsea` (using `BiocParallel`) hits scaling limits and high overhead, particularly with large pathway collections.

## Precision vs R

`rsfgsea` aims for feature and numerical parity with R's `fgsea` package.
- **Enrichment Scores (ES)**: Exact match (within floating point tolerances). `rsfgsea` uses 128-bit integer accumulators for intermediate sums to prevent catastrophic cancellation.
- **P-values & NES**: Statistically consistent. Differences purely due to random seed variation (Monte Carlo simulations).
- **Validation**: Verified against `fgsea` reference outputs across multiple datasets using `tests/r_validation.rs`.

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
