# rsfgseaR

`rsfgseaR` provides an R interface to `rsfgsea`, the Rust implementation of
fgsea-compatible preranked Gene Set Enrichment Analysis.

Current public entrypoints:

- `fgsea()` for wrapper-style execution
- `fgseaSimple()` for fixed-permutation simple mode
- `fgseaMultilevel()` for multilevel refinement
- `readRanks()` for ranked-list files
- `gmtPathways()` / `writeGmtPathways()` for GMT conversion

GPU note:

- default builds are CPU-only to keep the package CRAN-friendly
- `fgsea(..., gpu = TRUE)` uses the same hybrid GPU path as the Rust CLI when the package is built with `RSFGSEAR_ENABLE_GPU=1`
- GPU is currently supported only for wrapper mode, not `fgseaSimple()` or `fgseaMultilevel()`
- an actual compatible GPU/driver/runtime is still required at execution time

Performance snapshot:

- representative Rust-core Criterion benchmark, simple: `2.282 s` for 10k genes, 1k pathways, 10k permutations
- representative Rust-core Criterion benchmark, multilevel: `3.438 s` for 10k genes, 1k pathways, `nPermSimple=1000`
- file-backed comparison, multilevel large workload, 16 workers: Rust `105 ms` vs R `977 ms` (`9.3x` faster)
- file-backed comparison, simple large workload, 16 workers: Rust `674 ms` vs R `798 ms` (`1.18x` faster)
- real muscle-comparison validation workload: Rust `81 MB` peak RSS vs R `329 MB` peak RSS (`4.1x` lower)

Full benchmark setup, thread-scaling tables, and parity notes are in:

- <https://github.com/deminden/rsfgsea/blob/main/docs/reproducibility.md>

Minimal example:

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

CLI-style file workflow:

```r
res <- fgsea(
  pathways = "pathways.gmt",
  stats = "ranks.rnk",
  mode = "simple",
  nPermSimple = 1000L,
  output = "results.tsv"
)
```

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
