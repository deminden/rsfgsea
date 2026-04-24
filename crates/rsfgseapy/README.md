# rsfgseapy

Python bindings for `rsfgsea`, a Rust implementation of preranked fgsea-compatible gene set enrichment analysis.

## What It Exposes

The package currently exposes one public entrypoint:

- `run_gsea_py(...)`

The API intentionally keeps fgsea-style parameter names and execution modes:

- `mode="fgsea"`
- `mode="simple"`
- `mode="multilevel"`
- `nPermSimple`
- `seed`
- `nperm`
- `minSize`
- `maxSize`
- `sampleSize`
- `scoreType`
- `gseaParam`

## Installation

From PyPI:

```bash
pip install rsfgseapy
```

From a repository:

```bash
git clone https://github.com/deminden/rsfgsea
cd rsfgsea
cd crates/rsfgseapy
maturin develop --release
```

## Input Shape

`ranks`

- Python mapping of `gene -> score`
- values must be finite numeric scores

`gmt_path`

- path to a GMT file

## Performance Snapshot

The Python package calls the same Rust backend as the CLI and R wrapper.
Current local benchmark snapshots from the main repository, measured on an AMD
Ryzen 7950X3D:

- representative Criterion benchmark, simple: `2.201 s` for 10k genes, 1k pathways, 10k permutations
- representative Criterion benchmark, multilevel: `2.540 s` for 10k genes, 1k pathways, `nPermSimple=1000`
- file-backed comparison, multilevel large workload, 16 workers: Rust `99 ms` vs R `1039 ms` (`10.5x` faster)
- file-backed comparison, simple large workload, 16 workers: Rust `653 ms` vs R `916 ms` (`1.40x` faster)
- real muscle-comparison validation workload: Rust `81 MB` peak RSS vs R `328 MB` peak RSS (`4.0x` lower)

Full benchmark setup, thread-scaling tables, and parity notes are in:

- https://github.com/deminden/rsfgsea/blob/main/docs/reproducibility.md

## Minimal Example

For most users, wrapper mode with defaults is the right starting point.

```python
import rsfgseapy

results = rsfgseapy.run_gsea_py(
    ranks={"GENE_A": 2.0, "GENE_B": 1.0, "GENE_C": -1.0, "GENE_D": -2.0},
    gmt_path="pathways.gmt",
)

for row in results:
    print(row["pathway"], row["pval"])
```

## Full Example

```python
import rsfgseapy

ranks = {
    "GENE_A": 3.2,
    "GENE_B": 1.7,
    "GENE_C": -2.4,
    "GENE_D": -3.1,
}

results = rsfgseapy.run_gsea_py(
    ranks=ranks,
    gmt_path="pathways.gmt",
    mode="fgsea",
    gpu=False,
    nPermSimple=1000,
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

## Plotting

```python
import rsfgseapy

rsfgseapy.write_enrichment_plot_png_py(
    ranks={"GENE_A": 2.0, "GENE_B": 1.0, "GENE_C": -1.0, "GENE_D": -2.0},
    pathway_genes=["GENE_A", "GENE_B"],
    output_path="enrichment.png",
    pathway_name="PW_A",
    dpi=300,
    title="PW_A",
)
```

For multi-pathway summaries:

```python
import rsfgseapy

rsfgseapy.write_gsea_table_plot_png_py(
    ranks={"GENE_A": 2.0, "GENE_B": 1.0, "GENE_C": -1.0, "GENE_D": -2.0},
    pathways=[("PW_A", ["GENE_A", "GENE_B"]), ("PW_B", ["GENE_C", "GENE_D"])],
    results=[
        {"pathway": "PW_A", "nes": 1.5, "pval": 0.01, "padj": 0.02},
        {"pathway": "PW_B", "nes": -1.4, "pval": 0.03, "padj": 0.05},
    ],
    output_path="table.png",
    dpi=300,
)
```

All plotting parameters are available in the Python API; the examples above keep
only the most common publication-oriented overrides visible.

For the full cross-interface plotting guide, see:

- https://github.com/deminden/rsfgsea/blob/main/docs/plotting.md

## `nPermSimple` vs `nperm`

These two names come from fgsea and they are not interchangeable.

`nPermSimple`

- the normal simple-stage permutation count
- used by default in wrapper mode
- tune this when you want a different wrapper screening budget

`nperm`

- explicit fixed-permutation override
- in wrapper mode, setting `nperm` forces simple-mode execution instead of multilevel refinement
- leave this as `None` unless you intentionally want simple mode

Practical rule:

- leave `seed=None` for a fresh random run, or set `seed=<int>` for reproducibility
- light users: keep `nperm=None`
- use `nPermSimple` to tune the default wrapper behavior
- only set `nperm` when you deliberately want fixed-permutation simple execution

## Returned Results

Each result row is a dictionary with:

- `pathway`
- `size`
- `es`
- `nes`
- `pval`
- `padj`
- `log2err`
- `leading_edge`

`leading_edge` is returned as a Python list of genes.

## GPU Support

`gpu=True` enables the hybrid GPU path when the extension is built with the `gpu` feature.

Current behavior:

- GPU accelerates simple-stage screening
- CPU performs parity-focused multilevel refinement

If the extension is built without GPU support, `gpu=True` raises a runtime error.

## Supported Python Versions

The package metadata currently targets Python 3.8 and newer.

## Project Links

- Repository: https://github.com/deminden/rsfgsea
- Main project docs: https://github.com/deminden/rsfgsea/tree/main/docs
- Rust crate: https://crates.io/crates/rsfgsea
