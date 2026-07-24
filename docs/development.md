# Development Guide

## Workspace Layout

Main crates:

- `crates/rsfgsea`: core algorithms, public Rust API, and CLI binaries
- `crates/rsfgseapy`: Python bindings

Supporting directories:

- `scripts/`: local data-prep and parity helpers
- `tests/`: integration-level validation assets
- `results/`: benchmark and parity outputs
- `r-pkg/`: R package sources
- `reference/blitz/`: exact Python/Blitz reference project and lockfile
- `docs/evidence/`: small, reviewable machine-readable audit summaries

R package maintenance:

- `scripts/sync_r_vendor.sh`: refreshes the vendored Rust core used by `r-pkg/rsfgseaR`
- `r-pkg/rsfgseaR/cleanup`: removes generated R package build artifacts before packaging/checks

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
./r-pkg/rsfgseaR/cleanup
./scripts/sync_r_vendor.sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
R CMD build r-pkg/rsfgseaR
R CMD check rsfgseaR_<version>.tar.gz --no-manual
```

Remove the built tarball and `rsfgseaR.Rcheck/` after validation. Do not commit
or push with any of these failing.

## Branch Workflow

Use `dev` as the active development branch.

- commit and push day-to-day work on `dev`
- let GitHub CI run on `dev`
- only merge or fast-forward `main` after `dev` is green

Avoid making routine feature commits directly on `main`.

Before running `R CMD check`, make sure the package tree does not contain built
artifacts such as:

- `r-pkg/rsfgseaR/src/entrypoint.o`
- `r-pkg/rsfgseaR/src/rsfgseaR.so`
- `r-pkg/rsfgseaR/src/rust/target/`

These can be cleaned with:

```bash
./r-pkg/rsfgseaR/cleanup
```

If you edit Rust sources after running `./scripts/sync_r_vendor.sh`, run
`./scripts/sync_r_vendor.sh` again before committing so the vendored R Rust core
stays in sync with `crates/rsfgsea`.

During normal iteration, prefer finalizing the Rust-side implementation first.
Do not repeatedly resync vendored R Rust sources or wrapper-facing generated
artifacts after every small Rust edit. Do that once the Rust changes are ready,
then run the full required verification sequence before committing.

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
- current numeric audit records: `docs/evidence/`
- contributor workflow and project structure: `docs/development.md`

The root and package READMEs should summarize and link; they should not copy
large benchmark tables. Run `python3 scripts/check_docs.py` when changing the
locked Blitz stack or its evidence. Avoid hiding project knowledge only in
commit history or inline comments.

## Release Checklist

For each release:

1. bump crate and Python package versions consistently
2. rerun the full required checks
3. verify README and docs commands still match the current API
4. verify any changed mode semantics are reflected in `docs/algorithms.md`
5. verify script descriptions and reproducibility notes still point to the right files
