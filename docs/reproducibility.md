# Reproducibility Guide

This project keeps parity and validation workflows close to the codebase.

## Validation Sources

The main validation layers are:

- unit and integration tests in `crates/rsfgsea/tests`
- GPU-focused tests in `crates/rsfgsea-gpu/tests`
- generated reports in `reports/`
- ad hoc and semi-structured helpers in `scripts/`

## Scripts Overview

[`scripts/extract_mapping.R`](/home/den/bio/rsfgsea/scripts/extract_mapping.R)

- exports gene mapping JSON files from an `.rda` source
- use it when local data-prep scripts need mapping files regenerated

[`scripts/prepare_data.py`](/home/den/bio/rsfgsea/scripts/prepare_data.py)

- builds ranked lists from bladder correlation data
- output goes to `tests/data/muscle_comparison`

[`scripts/prepare_muscle_data.py`](/home/den/bio/rsfgsea/scripts/prepare_muscle_data.py)

- computes per-gene Spearman correlations from muscle expression data
- emits ranked lists for downstream fgsea comparison

[`scripts/run_fgsea_comparison.R`](/home/den/bio/rsfgsea/scripts/run_fgsea_comparison.R)

- runs R `fgsea` over the generated ranked lists
- writes a combined R reference result table

[`scripts/test_single_gene.R`](/home/den/bio/rsfgsea/scripts/test_single_gene.R)

- quick manual R-side sanity check for one ranked list
- useful when debugging overlap or pathway-loading issues

[`scripts/compare_folder_examples.py`](/home/den/bio/rsfgsea/scripts/compare_folder_examples.py)

- compares Rust CLI, Python bindings, and R fgsea on a folder of examples
- writes a parity report for those examples

## Practical Workflow

When you need to validate behavior against R:

1. generate or refresh ranked lists
2. run the R comparison scripts
3. run the Rust and Python paths on the same inputs
4. compare result files or generated reports
5. only then interpret statistical differences

## What To Record

When saving parity or benchmark outputs, record:

- input files
- mode
- permutation counts
- `sampleSize`
- `scoreType`
- `gseaParam`
- seed
- whether GPU was used

Without that context, result diffs are hard to interpret later.

## Suggested Cleanup Direction

The scripts are useful, but they are still local workflow tools rather than a polished pipeline.

If this project grows, the next step should be:

- one documented reproducibility entrypoint
- one place for required input paths
- one report format for parity summaries

That would make the current script collection easier to use in CI or by other contributors.
