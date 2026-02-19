# GPU vs R fgsea Parity Report

Dataset: `data/Folder_with_examples` (23 files), seeds=[11, 23, 42], nPermSimple=1000.

R reference: `fgseaMultilevel(sampleSize=101, eps=1e-50, nPermSimple=1000)`.

Matched pathways total: **2238** across **66** file-seed runs.

## Absolute Differences

| Metric | Mean | Median | P95 | Max |
| :--- | ---: | ---: | ---: | ---: |
| `|ES|` | 2.535e-09 | 2.531e-09 | 4.736e-09 | 4.998e-09 |
| `|NES|` | 1.842e-02 | 1.245e-02 | 5.827e-02 | 1.238e-01 |
| `|pval|` | 1.548e-02 | 1.199e-02 | 3.996e-02 | 5.007e-01 |
| `|padj|` | 1.248e-02 | 5.101e-03 | 5.784e-02 | 2.458e-01 |

## Relative p-value Differences

- Mean: `4.150%`
- Median: `2.365%`
- P95: `13.360%`
- Max: `67.391%`
- Fraction `<1%`: `26.4%`
- Fraction `<10%`: `90.9%`
