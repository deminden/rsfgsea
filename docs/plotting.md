# Plotting Guide

`rsfgsea` can write single-pathway enrichment plots and multi-pathway GSEA
table plots as PNG from the CLI, Python, and R wrappers.

Current scope:

- single enrichment plot per call
- multi-pathway GSEA table plot per call
- PNG output
- publication-oriented physical sizing via inches + DPI
- optional transparent background

## CLI

Minimal example:

```bash
rsfgsea-plot-enrichment \
    --ranks data/pearson_symbols.rnk \
    --gmt data/h.all.v2025.1.Hs.symbols.gmt \
    --pathway HALLMARK_APOPTOSIS \
    --output enrichment.png \
    --dpi 300 \
    --title "HALLMARK_APOPTOSIS"
```

Example enrichment plot:

![Enrichment plot example](images/HADHB_GTEX_muscle_go_table_multilevel_Pearson_top5000_15_500_cell_adhesion_enrichment.png)

Useful plotting options:

- `--dpi`
- `--width-in`
- `--height-in`
- `--transparent-background`
- `--title`
- `--scoreType`
- `--gseaParam`

Table-plot CLI example:

```bash
rsfgsea-plot-gsea-table \
    --ranks data/derived/LEF1_top300_abs_pearson_symbols.rnk \
    --gmt data/Human_GO_AllPathways_noPFOCR_with_GO_iea_March_01_2024_symbol_renamed.gmt \
    --pathway 'GOLGI APPARATUS%GOCC%GO:0005794' \
    --pathway 'HYDROLASE ACTIVITY%GOMF%GO:0016787' \
    --output table.png \
    --dpi 300
```

Example table plot:

![GSEA table plot example](images/HADHB_GTEX_muscle_go_table_multilevel_Pearson_top5000_15_500_top10_table.png)

## Python

Minimal example:

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

Available plotting arguments:

- `pathway_name`
- `scoreType`
- `gseaParam`
- `width_inches`
- `height_inches`
- `dpi`
- `transparent_background`
- `title`

Table-plot example:

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

Available table-plot arguments:

- `gseaParam`
- `width_inches`
- `height_inches`
- `dpi`
- `transparent_background`

## R

Minimal example:

```r
rsfgseaR::plotEnrichment(
  pathway = c("g1", "g2"),
  stats = c(g1 = 2, g2 = 1, g3 = -1, g4 = -2),
  output = "enrichment.png",
  pathwayName = "PW_A",
  dpi = 300L,
  title = "PW_A"
)
```

Available plotting arguments:

- `pathwayName`
- `scoreType`
- `gseaParam`
- `width_inches`
- `height_inches`
- `dpi`
- `transparent_background`
- `title`

Table-plot example:

```r
rsfgseaR::plotGseaTable(
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

Available table-plot arguments:

- `gseaParam`
- `width_inches`
- `height_inches`
- `dpi`
- `transparent_background`

## Defaults

Current plot defaults:

- width: `4.5 in`
- height: `3.2 in`
- dpi: `300`
- background: white
- title: not drawn unless explicitly provided

Current table-plot defaults:

- preferred width hint: `5.6 in`
- height: derived from row count and layout content unless explicitly set
- dpi: `300`
- background: white
- final canvas width: derived from the rendered table width plus margins

## Notes

- DPI affects generated pixel dimensions, not just PNG metadata.
- `transparent_background = TRUE` writes an RGBA PNG.
- Plotting is opt-in and does not change the main enrichment result path when no
  plot is requested.
- For table plots, names without `%` are displayed unchanged except that `_` is
  replaced by spaces.
- For table plots, names with `%...` suffixes are displayed without that suffix
  to keep the table readable.
