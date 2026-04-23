# rsfgsea

High-performance Rust implementation of preranked Gene Set Enrichment Analysis (GSEA), designed as a drop-in, optimized alternative to the R `fgsea` package.

What it focuses on:

- fgsea-compatible simple and multilevel workflows
- library and CLI use from one Rust crate
- deterministic parity-focused CPU path
- optional hybrid GPU acceleration for large simple-stage screening

Headline results from the main project benchmarks:

- local representative Criterion benchmark, simple: `2.411 s` for 10k genes, 1k pathways, 10k permutations
- local representative Criterion benchmark, multilevel: `2.739 s` for 10k genes, 1k pathways, `nPermSimple=1000`
- multilevel, small workload, 1 worker: `2 ms` vs R `43 ms` (`21.5x` faster)
- multilevel, large workload, 16 workers: `109 ms` vs R `934 ms` (`8.6x` faster)
- simple, small workload, 1 worker: `812 ms` vs R `2528 ms` (`3.1x` faster)
- simple, large workload, 16 workers: `739 ms` vs R `835 ms` (`1.13x` faster)

Run the local optimization benchmark with:

```bash
cargo bench -p rsfgsea --bench gsea_bench
```

Current CPU multilevel parity vs R is near floating-point noise in this repo's validation setup:

- max `|ES|` diff: `4.988e-09`
- max `|NES|` diff: `4.983e-09`
- max `|pval|` diff: `4.975e-09`
- max `|padj|` diff: `4.965e-09`

## Install

Library:

```toml
[dependencies]
rsfgsea = "0.3.3"
```

CLI:

```bash
cargo install rsfgsea
```

## Minimal Rust Example

```rust
use rsfgsea::prelude::*;

let ranks = RankedList::new(
    vec!["g1".into(), "g2".into(), "g3".into(), "g4".into()],
    vec![2.0, 1.0, -1.0, -2.0],
);

let pathways = vec![
    Pathway {
        name: "PW_A".into(),
        description: None,
        genes: vec!["g1".into(), "g2".into()],
    },
];

let results = fgsea(
    &ranks,
    &pathways,
    None,
    1000,
    42,
    1,
    ranks.len() - 1,
    1e-50,
    ScoreType::Std,
    1.0,
);

println!("{}", results[0].pathway_name);
```

## CLI

```bash
rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --output results.tsv
```

For deeper usage, benchmarks, parity notes, and reproducibility details, see the main repository:
<https://github.com/deminden/rsfgsea>
