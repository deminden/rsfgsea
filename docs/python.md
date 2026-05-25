# Python Guide

The Python extension lives in `crates/rsfgseapy` and exposes the same Rust backend for decor, classic fgsea-compatible, and native blitz workflows.

## Build And Install

```bash
cd crates/rsfgseapy
maturin develop --release
```

For GPU-enabled builds:

```bash
maturin develop --release --features gpu
```

## Main Entry Point

Use `rsfgseapy.run_gsea_py(...)`.

Inputs:

- `ranks`: Python mapping of `gene -> score`
- `gmt_path`: path to a GMT file

Important options:

- `method="decor" | "classic"`
- `decor_cache=None`
- `decor_expression=None`
- `decor_preset=None` (defaults to `balanced`)
- `decor_stringency=None`
- `decor_cache_mode="auto" | "reuse" | "rebuild"`
- `decor_correlation="pearson"`
- `decor_redundancy="positive_mean" | "abs_mean"`
- `mode="fgsea" | "simple" | "multilevel" | "blitz"`
- `gpu=False`
- `nPermSimple=1000`
- `seed=None`
- `nperm=None`
- `minSize=None` (`1` for fgsea/simple/multilevel, `5` for blitz)
- `maxSize=None`
- `eps=1e-50`
- `sampleSize=101`
- `scoreType="std" | "pos" | "neg"`
- `gseaParam=1.0`
- `nproc=0`
- `blitz_anchors=40`
- `blitz_symmetric=False`
- `blitz_center=True`
- `blitz_accuracy=40`
- `blitz_deep_accuracy=50`
- `blitz_signature_cache=True`

## Decor Example

Decor is CPU-only. It uses fixed-permutation simple runs when `nperm` is set,
and decor multilevel refinement when `mode="multilevel"` or wrapper mode omits
`nperm`.

```python
import rsfgseapy

res = rsfgseapy.run_gsea_py(
    ranks={"TP53": 3.1, "MYC": 2.8, "ACTB": -1.2},
    gmt_path="data/pathways.gmt",
    method="decor",
    mode="simple",
    nperm=10000,
    decor_cache="cache/pathways.decor.tsv",
    decor_expression="data/expression.tsv",
)
```

Use a named preset to change the specificity/sensitivity tradeoff:

```python
res = rsfgseapy.run_gsea_py(
    ranks={"TP53": 3.1, "MYC": 2.8, "ACTB": -1.2},
    gmt_path="data/pathways.gmt",
    method="decor",
    mode="simple",
    nperm=10000,
    decor_cache="cache/pathways.decor.tsv",
    decor_expression="data/expression.tsv",
    decor_preset="specific",
)
```

Or use the high-level stringency ladder when you want one easy knob:

```python
res = rsfgseapy.run_gsea_py(
    ranks={"TP53": 3.1, "MYC": 2.8, "ACTB": -1.2},
    gmt_path="data/pathways.gmt",
    method="decor",
    mode="simple",
    nperm=10000,
    decor_cache="cache/pathways.decor.tsv",
    decor_expression="data/expression.tsv",
    decor_stringency=75,
)
```

The expression matrix is tab-separated with genes in rows and samples in columns. Values should already be normalized or transformed as appropriate for correlation analysis.

Supported decor presets are `sensitive`, `balanced`, `specific`, and `strict`. The default `balanced` preset resolves to threshold-rational decor with `tau=0.04` and `alpha=60`. Stringency autoswitches calibrated presets: `0 <= x < 35` is `sensitive`, `35 <= x < 65` is `balanced`, `65 <= x < 85` is `specific`, and `85 <= x <= 100` is `strict`.

## Classic Minimal Example

Wrapper mode with defaults is the closest match to the standard R `fgsea` interface.

```python
import rsfgseapy

results = rsfgseapy.run_gsea_py(
    ranks={"GENE_A": 2.0, "GENE_B": 1.0, "GENE_C": -1.0, "GENE_D": -2.0},
    gmt_path="data/pathways.gmt",
)

for row in results:
    print(row["pathway"], row["pval"])
```

## Classic Full Example

```python
import rsfgseapy

ranks = {
    "GENE_A": 3.2,
    "GENE_B": 1.7,
    "GENE_C": -2.4,
}

results = rsfgseapy.run_gsea_py(
    ranks=ranks,
    gmt_path="data/pathways.gmt",
    mode="fgsea",
    gpu=False,
    nPermSimple=100000,
    seed=None,
    nperm=None,
    minSize=1,
    maxSize=None,
    eps=1e-50,
    sampleSize=101,
    scoreType="std",
    gseaParam=1.0,
    nproc=0,
)

for row in results:
    print(row["pathway"], row["nes"], row["pval"])
```

## Blitz Example

Blitz mode is a native Rust implementation of the `blitzgsea.gsea()` workflow. It keeps the rsfgsea result shape: `pval` is the blitz p-value, `padj` is BH/FDR, and `log2err` is `None`.

```python
res = rsfgseapy.run_gsea_py(
    ranks={"TP53": 3.1, "MYC": 2.8, "ACTB": -1.2, "GATA3": -2.0, "ESR1": 1.5},
    gmt_path="data/pathways.gmt",
    mode="blitz",
)
```

Blitz mode keeps an in-process null-model cache enabled by default for repeated identical calls. Set `blitz_signature_cache=False` to force cold calibration. It rejects `gpu=True`, `method="decor"`, `nperm`, `scoreType` values other than `"std"`, and `gseaParam` values other than `1.0`.

## `nPermSimple` vs `nperm`

This is the main point that confuses new users.

`nPermSimple`

- the normal simple-stage permutation count
- used by wrapper mode before any multilevel refinement

`nperm`

- explicit fixed-permutation override
- if you set `nperm` in wrapper mode, wrapper mode stops being adaptive and behaves like simple mode

Use this rule:

- leave `nperm=None` for normal fgsea-style wrapper behavior
- leave `seed=None` for a fresh random run, or set `seed=<int>` for reproducibility
- tune `nPermSimple` if you want a different wrapper screening budget
- set `nperm` only when you deliberately want simple mode

## Result Shape

Each result is returned as a dictionary with:

- `pathway`
- `size`
- `es`
- `nes`
- `pval`
- `padj`
- `log2err`
- `leading_edge`

`leading_edge` is a Python list of genes.

## Mode Semantics

`mode="fgsea"`

- wrapper behavior
- uses multilevel refinement unless `nperm` is provided

`mode="simple"`

- simple permutations only

`mode="multilevel"`

- explicit multilevel workflow

## GPU Notes

The Python `gpu=True` path is hybrid:

- GPU for simple-stage screening
- CPU for multilevel refinement

For CPU/GPU or R/GPU result comparisons, use `nPermSimple=100000` as a
practical baseline. Use `10000` only as a smoke tier, and use `1000000` for
final tail/stress checks when runtime allows. The default `1000` is a fast
screening budget and can make p-value and FDR differences look worse than they
are.

Current restriction:

- `gpu=True` is intended for wrapper-style execution
- if the extension is built without the `gpu` feature, `gpu=True` raises a runtime error

The Python binding uses the same WebGPU adapter selection as the Rust CLI. On
WSL2, if `nvidia-smi` sees the GPU but `gpu=True` fails with a `llvmpipe`
adapter error, set the D3D12 fallback variables before starting Python:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

For older builds or deeper debugging, also try:

```bash
export WGPU_BACKEND=gl
export RSFGSEA_GPU_ALLOW_GL=1
```
