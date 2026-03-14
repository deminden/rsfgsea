# Plotting Guide

`rsfgsea` can write single-pathway enrichment plots as PNG from the CLI, Python,
and R wrappers.

Current scope:

- single enrichment plot per call
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

Useful plotting options:

- `--dpi`
- `--width-in`
- `--height-in`
- `--transparent-background`
- `--title`
- `--scoreType`
- `--gseaParam`

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

## Defaults

Current plot defaults:

- width: `4.5 in`
- height: `3.2 in`
- dpi: `300`
- background: white
- title: not drawn unless explicitly provided

## Notes

- DPI affects generated pixel dimensions, not just PNG metadata.
- `transparent_background = TRUE` writes an RGBA PNG.
- Plotting is opt-in and does not change the main enrichment result path when no
  plot is requested.
