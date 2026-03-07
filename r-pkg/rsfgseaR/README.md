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

Current status:

- package installs and runs locally with `R CMD INSTALL`
- `testthat` coverage exists for the main public wrappers
- GPU support is available in local opt-in builds and disabled by default for CRAN-style builds
- release and CRAN polish are still pending

Project repository:

- <https://github.com/deminden/rsfgsea>
