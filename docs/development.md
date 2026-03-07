# Development Guide

## Workspace Layout

Main crates:

- `crates/rsfgsea`: core algorithms, public Rust API, and CLI binaries
- `crates/rsfgseapy`: Python bindings

Supporting directories:

- `scripts/`: local data-prep and parity helpers
- `tests/`: integration-level validation assets
- `reports/`: generated comparison reports
- `results/`: benchmark and parity outputs

## What To Keep Stable

Treat these as user-facing contracts:

- ranked-list parsing rules
- GMT parsing rules
- result column names and meanings
- wrapper/simple/multilevel mode semantics
- GPU wrapper restrictions

## Required Checks

From repo root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Do not commit or push with any of these failing.

## Editing Guidance

Good changes usually:

- preserve fgsea-compatible behavior
- reduce duplicated logic instead of adding parallel implementations
- keep library output free of unconditional `println!` noise
- add tests when parser behavior, routing, or statistical semantics change

## Where To Put New Documentation

- user-facing usage: `docs/cli.md` or `docs/python.md`
- statistical behavior: `docs/algorithms.md`
- reproducibility workflows and scripts: `docs/reproducibility.md`
- contributor workflow and project structure: `docs/development.md`

Avoid hiding project knowledge only in commit history or inline comments.

## Release Checklist

For each release:

1. bump crate and Python package versions consistently
2. rerun the full required checks
3. verify README and docs commands still match the current API
4. verify any changed mode semantics are reflected in `docs/algorithms.md`
5. verify script descriptions and reproducibility notes still point to the right files
