# Python Guide

The Python extension lives in `crates/rsfgseapy` and exposes the same fgsea-compatible workflows as the Rust library.

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

- `mode="fgsea" | "simple" | "multilevel"`
- `gpu=False`
- `nPermSimple=1000`
- `nperm=None`
- `minSize=1`
- `maxSize=None`
- `eps=1e-50`
- `sampleSize=101`
- `scoreType="std" | "pos" | "neg"`
- `gseaParam=1.0`
- `nproc=0`

## Example

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
    nPermSimple=1000,
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

Current restriction:

- `gpu=True` is intended for wrapper-style execution
- if the extension is built without the `gpu` feature, `gpu=True` raises a runtime error
