# rsfgseaR

`rsfgseaR` provides an R interface to `rsfgsea`, the Rust implementation of
decor, classic fgsea-compatible, and native blitz preranked Gene Set Enrichment
Analysis workflows.

Current public entrypoints:

- `fgsea(..., method = "decor")` for CPU redundancy-aware decor execution; set `nperm` for fixed-permutation simple runs, or omit it/use `mode = "multilevel"` for decor multilevel refinement
- `fgsea()` for classic wrapper-style execution
- `fgseaSimple()` for fixed-permutation simple mode
- `fgseaMultilevel()` for multilevel refinement
- `fgsea(..., mode = "blitz")` for native blitzGSEA-compatible execution
- `readRanks()` for ranked-list files
- `gmtPathways()` / `writeGmtPathways()` for GMT conversion

GPU note:

- default builds are CPU-only to keep the package CRAN-friendly
- `fgsea(..., gpu = TRUE)` uses the same hybrid GPU path as the Rust CLI when the package is built with `RSFGSEAR_ENABLE_GPU=1`
- GPU is currently supported only for wrapper mode, not `fgseaSimple()` or `fgseaMultilevel()`
- an actual compatible GPU/driver/runtime is still required at execution time
- for CPU/GPU or R/GPU comparisons, use `nPermSimple = 100000L` as a practical baseline; use `10000L` only as a smoke tier and `1000000L` for final tail/stress checks when runtime allows
- on WSL2, if `nvidia-smi` works but `gpu = TRUE` reports a `llvmpipe` adapter, start R with `GALLIUM_DRIVER=d3d12` and `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`

Example WSL2 launch:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
R
```

Performance, precision, and reproducibility evidence is owned by:

- <https://github.com/deminden/rsfgsea/blob/main/docs/reproducibility.md>

Decor example:

```r
library(rsfgseaR)

stats <- c(TP53 = 3.1, MYC = 2.8, ACTB = -1.2)
pathways <- list(PW_A = c("TP53", "MYC"), PW_B = c("ACTB"))

res <- fgsea(
  pathways = pathways,
  stats = stats,
  method = "decor",
  mode = "simple",
  nperm = 10000L,
  decor.cache = "cache/pathways.decor.tsv",
  decor.expression = "data/expression.tsv"
)
print(res)
```

Classic minimal example:

```r
library(rsfgseaR)

stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
pathways <- list(
  PW_A = c("g1", "g2"),
  PW_B = c("g3", "g4")
)

res <- fgsea(pathways = pathways, stats = stats)
print(res)
```

`pathways` can be either:

- a named list of character vectors
- a path to a GMT file on disk

`stats` can be either:

- a named numeric vector
- a path to a ranked-list file

Classic CLI-style file workflow:

```r
res <- fgsea(
  pathways = "pathways.gmt",
  stats = "ranks.rnk",
  mode = "simple",
  nPermSimple = 100000L,
  output = "results.tsv"
)
```

Blitz mode:

```r
res <- fgsea(
  pathways = pathways,
  stats = stats,
  mode = "blitz"
)
```

Blitz mode uses blitz defaults when arguments are omitted (`minSize = 5`, `maxSize = 4000`, seed `0`, four calibration workers) and returns `NA` for `log2err`. `blitz.signature.cache = TRUE` reuses native blitz null-model fits for repeated identical calls in the same R process.

Plot example:

```r
plotEnrichment(
  pathway = c("g1", "g2"),
  stats = c(g1 = 2, g2 = 1, g3 = -1, g4 = -2),
  output = "enrichment.png",
  pathwayName = "PW_A",
  dpi = 300L,
  title = "PW_A"
)
```

For multi-pathway summaries:

```r
plotGseaTable(
  pathways = list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4")),
  stats = c(g1 = 2, g2 = 1, g3 = -1, g4 = -2),
  fgseaRes = data.frame(
    pathway = c("PW_A", "PW_B"),
    nes = c(1.5, -1.4),
    pval = c(0.01, 0.03),
    padj = c(0.02, 0.05)
  ),
  output = "table.png",
  dpi = 300L
)
```

All plotting parameters are available in the R wrapper; the examples above keep
only the most common overrides visible.

For the full cross-interface plotting guide, see:

- <https://github.com/deminden/rsfgsea/blob/main/docs/plotting.md>

Current status:

- package installs and runs locally with `R CMD INSTALL`
- `testthat` coverage exists for the main public wrappers
- GPU support is available in local opt-in builds and disabled by default for CRAN-style builds
- release and CRAN polish are still pending

Project repository:

- <https://github.com/deminden/rsfgsea>
