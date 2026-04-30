# CLI Guide

The `rsfgsea` binary runs preranked gene set enrichment from a ranked list and a GMT file.

## When To Use It

Use the CLI when you already have:

- a preranked list in `.rnk`-style tabular form
- a pathway collection in GMT format
- a shell-based workflow, CI job, or reproducible batch process

## Build

```bash
cargo build --release
```

For hybrid GPU support:

```bash
cargo build --release --features gpu
```

The binary will be at `target/release/rsfgsea`.

Or install using:

```bash
# Install from crates.io
cargo install rsfgsea
```

## Inputs

Ranked list:

- whitespace-separated
- first column: gene identifier
- second column: numeric score
- duplicate genes are rejected
- non-finite scores are rejected
- malformed rows are rejected

GMT:

- tab-separated
- first column: pathway name
- second column: description
- remaining columns: genes
- malformed rows are rejected

## Modes

`--mode fgsea`

- wrapper mode
- behaves like fgsea's standard interface
- uses simple screening first
- uses multilevel refinement unless `--nperm` forces simple mode

`--mode simple`

- fixed-permutation simple mode only

`--mode multilevel`

- explicit multilevel workflow

## Important Arguments

`--method`

- `classic`: default fgsea-compatible method
- `decor`: experimental redundancy-aware method

`--nPermSimple`

- simple-stage permutation count
- used directly in `simple` mode
- used as the initial screening count in wrapper and multilevel flows

`--nperm`

- wrapper override
- in `--mode fgsea`, this forces simple-mode execution

`--sampleSize`

- multilevel sample size
- default is `101`

`--scoreType`

- `std`: two-sided behavior
- `pos`: positive enrichment only
- `neg`: negative enrichment only

`--gseaParam`

- weighting exponent applied to ranking magnitudes

`--nproc`

- Rayon worker count
- `0` means default threadpool behavior

`--gpu`

- enables the hybrid GPU path
- currently only supported with `--mode fgsea`

Decor options:

- `--decor-cache`: path to the decor cache containing redundancy scores
- `--decor-expression`: normalized tab-separated expression matrix used to build the cache
- `--decor-alpha`: non-negative redundancy penalty strength, default `23`
- `--decor-cache-mode`: `auto`, `reuse`, or `rebuild`
- `--decor-correlation`: `pearson` or `spearman`; only Pearson is currently implemented
- `--decor-redundancy`: `positive_mean` or `abs_mean`
- `--decor-expression-format`: `auto` or `tsv`; CSV is parsed as an option but not implemented yet
- `--decor-expression-has-header`: `true` or `false`, default `true`

## Typical Commands

Wrapper mode:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --output results.tsv
```

Simple mode:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --mode simple \
  --nPermSimple 10000 \
  --output results.tsv
```

Wrapper mode forced to simple:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --mode fgsea \
  --nperm 10000 \
  --output results.tsv
```

Experimental decor first run:

```bash
rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --output results.decor.tsv \
  --method decor \
  --mode simple \
  --nperm 10000 \
  --decor-cache cache/pathways.decor.tsv \
  --decor-expression data/expression.tsv \
  --decor-alpha 23
```

Experimental decor reuse run:

```bash
rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --output results.decor.tsv \
  --method decor \
  --mode simple \
  --nperm 10000 \
  --decor-cache cache/pathways.decor.tsv
```

Hybrid GPU mode:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --gpu \
  --mode fgsea \
  --output results.tsv
```

Full parameter example with the installed binary:

```bash
rsfgsea \
    --ranks data/pearson_symbols.rnk \
    --gmt data/h.all.v2025.1.Hs.symbols.gmt \
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

## Output

The CLI writes a TSV with these columns:

- `pathway`
- `size`
- `es`
- `nes`
- `pval`
- `padj`
- `log2err`
- `leading_edge`

`leading_edge` is written as a comma-separated gene list.

For decor runs, `es`, `nes`, `pval`, `padj`, `log2err`, and `leading_edge`
refer to the decor statistic and its conditional redundancy-profile null.

## GPU Notes

The GPU path is hybrid, not fully GPU-native:

- GPU: simple-stage null generation and screening
- CPU: multilevel refinement when needed

The binary rejects `--gpu` with `--mode simple` or `--mode multilevel`.

Useful environment variables:

- `WGPU_BACKEND=vulkan`
- `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`
- `RSFGSEA_GPU_ALLOW_GL=1`
