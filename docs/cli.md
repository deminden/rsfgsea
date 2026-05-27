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

## Methods And Modes

`--method decor`

- redundancy-aware preranked GSEA method
- first-class path for expression-correlated pathway genes
- uses CPU fixed-permutation simple mode when `--nperm` is provided
- uses decor multilevel refinement in wrapper mode when `--nperm` is omitted, or in explicit `--mode multilevel`
- supports calibrated presets and the `--decor-stringency` ladder

`--method classic`

- default fgsea-compatible method
- second public track for classic wrapper, simple, and multilevel workflows

`--mode fgsea`

- classic wrapper mode
- behaves like fgsea's standard interface
- uses simple screening first
- uses multilevel refinement unless `--nperm` forces simple mode

`--mode simple`

- fixed-permutation simple mode only

`--mode multilevel`

- explicit multilevel workflow

`--mode blitz`

- third public track: native Rust blitzGSEA-compatible workflow
- uses blitz defaults when mode-specific arguments are omitted: `minSize=5`, `maxSize=4000`, `seed=0`, `nPermSimple=1000`, `--blitz-anchors 40`, and four calibration workers
- rejects incompatible fgsea options: `--gpu`, `--method decor`, `--nperm`, `--scoreType` other than `std`, and `--gseaParam` other than `1`
- reports blitz p-values in `pval`, BH/FDR in `padj`, and `NA` for `log2err`

## Important Arguments

`--method`

- `decor`: redundancy-aware preranked GSEA method
- `classic`: default fgsea-compatible method

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
- in blitz mode, `0` uses the blitz reference default of four calibration workers

`--gpu`

- enables the hybrid GPU path
- currently only supported with `--mode fgsea`

Decor options:

- `--decor-cache`: path to the decor cache containing redundancy scores
- `--decor-expression`: normalized tab-separated expression matrix used to build the cache
- `--decor-preset`: `sensitive`, `balanced`, `specific`, or `strict`; default `balanced`
- `--decor-stringency`: optional numeric 0-100 convenience control that autoswitches calibrated presets
- `--decor-cache-mode`: `auto`, `reuse`, or `rebuild`
- `--decor-expression-format`: `auto` or `tsv`; CSV is parsed as an option but not implemented yet
- `--decor-expression-has-header`: `true` or `false`, default `true`
- `--decor-tail-reliability adaptive`: experimental opt-in reliability pass for decor multilevel lower-tail results

Decor presets:

- `sensitive`: raw-rational, `alpha=22`
- `balanced`: threshold-rational, `tau=0.04`, `alpha=60`; the default held-out-validated balanced preset
- `specific`: threshold-rational, `tau=0.05`, `alpha=65`
- `strict`: exp-scaled, target median penalty `0.10`

Stringency is a preset ladder, not a continuous formula interpolation: `0 <= x < 35` resolves to `sensitive`, `35 <= x < 65` to `balanced`, `65 <= x < 85` to `specific`, and `85 <= x <= 100` to `strict`.

The CLI prints the resolved preset, formula, and parameters for reproducibility.

Decor multilevel lower-tail caveat:

- `log2err` should be inspected for very small decor multilevel p-values, especially when conclusions depend on the exact ordering of the strongest pathways.
- Very broad GO terms can be expensive to refine because decor multilevel samples pathway-specific penalized hit profiles rather than the size-only classic fgsea null.
- `--decor-tail-reliability adaptive` can be used as an explicit final-analysis check for low-tail decor results, but it is not the default because a small number of large triggered pathways can dominate runtime.
- For broad ontology collections, consider whether very large terms are scientifically useful before increasing `sampleSize` or enabling adaptive reliability for final runs.

Blitz options:

- `--blitz-anchors`: number of calibration anchors, default `40`
- `--blitz-symmetric`: use one symmetric positive/negative null fit
- `--blitz-no-center`: disable signature centering
- `--blitz-signature-cache`: enable the in-process blitz null-model cache for repeated embedded CLI-style calls; ordinary one-shot CLI runs leave it off
- `--blitz-accuracy`: normal-tail accuracy setting retained for blitz compatibility metadata, default `40`
- `--blitz-deep-accuracy`: deep-tail accuracy setting retained for blitz compatibility metadata, default `50`

## Typical Commands

Decor first run:

```bash
rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --output results.decor.tsv \
  --method decor \
  --mode simple \
  --nperm 10000 \
  --decor-cache cache/pathways.decor.tsv \
  --decor-expression data/expression.tsv
```

Decor with a selected preset:

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
  --decor-preset specific
```

Decor with easy stringency:

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
  --decor-stringency 75
```

Decor reuse run:

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

Classic wrapper mode:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --output results.tsv
```

Classic simple mode:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --mode simple \
  --nPermSimple 10000 \
  --output results.tsv
```

Classic wrapper forced to simple:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --mode fgsea \
  --nperm 10000 \
  --output results.tsv
```

Hybrid GPU mode:

```bash
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --gpu \
  --mode fgsea \
  --nPermSimple 100000 \
  --output results.tsv
```

WSL2 with CUDA visible but WebGPU selecting `llvmpipe`:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
./target/release/rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --gpu \
  --mode fgsea \
  --nPermSimple 100000 \
  --output results.tsv
```

Blitz mode:

```bash
rsfgsea \
  --ranks data/example.rnk \
  --gmt data/pathways.gmt \
  --output results.blitz.tsv \
  --mode blitz
```

Full parameter example with the installed binary and a comparison-friendly
simple-stage budget:

```bash
rsfgsea \
    --ranks data/pearson_symbols.rnk \
    --gmt data/h.all.v2025.1.Hs.symbols.gmt \
    --mode fgsea \
    --nPermSimple 100000 \
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
refer to the decor statistic.

## GPU Notes

The GPU path is hybrid, not fully GPU-native:

- GPU: simple-stage null generation and screening
- CPU: multilevel refinement when needed

For CPU/GPU or R/GPU result comparisons, use `--nPermSimple 100000` as a
practical baseline. Use `10000` only as a smoke tier, and use `1000000` for
final tail/stress checks when runtime allows. The default `1000` is better
treated as a quick execution check because p-value and FDR comparisons are
dominated by Monte Carlo noise at that scale.

The binary rejects `--gpu` with `--mode simple` or `--mode multilevel`.

Useful environment variables:

- `WGPU_BACKEND=vulkan`
- `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`
- `RSFGSEA_GPU_ALLOW_GL=1`
- `GALLIUM_DRIVER=d3d12`

For WSL2, first check whether the graphics stack is using software rendering:

```bash
glxinfo -B | grep -E 'OpenGL renderer|Accelerated'
vulkaninfo --summary | grep -E 'deviceName|deviceType|driverName'
```

If those commands report `llvmpipe` even though `nvidia-smi` sees the GPU,
run `rsfgsea` with:

```bash
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

This forces Mesa's D3D12-backed OpenGL path and asks WSL to choose the NVIDIA
adapter. `rsfgsea` treats `GALLIUM_DRIVER=d3d12` as an intentional GL fallback.
If you are debugging older builds, also try:

```bash
WGPU_BACKEND=gl RSFGSEA_GPU_ALLOW_GL=1
```
