# AGENTS.md

## Pre-commit Requirements (Mandatory)

Before every commit and before every push, run all checks below from repo root and ensure they pass.

1. `./r-pkg/rsfgseaR/cleanup`
2. `./scripts/sync_r_vendor.sh`
3. `cargo fmt --all -- --check`
4. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
5. `cargo test --workspace --all-features`
6. `R CMD check r-pkg/rsfgseaR --no-manual`

## Rules

- Do not commit if any check fails.
- Do not commit if the vendored R Rust core under `r-pkg/rsfgseaR/src/rust/vendor/rsfgsea` is out of sync with `crates/rsfgsea`.
- Prevent R package artifact failures by running `./r-pkg/rsfgseaR/cleanup` before verification and before packaging.
- Do not commit if R package build artifacts are still present after cleanup, especially in `r-pkg/rsfgseaR/src` or `r-pkg/rsfgseaR/src/rust/target`.
- If a check fails, fix the issue, then rerun the full sequence.
- If tests fail due to environment/hardware constraints (for example GPU-only runtime tests), report this clearly in the commit/PR notes.

## Optional local speed-up (not for final verification)

During development you may run targeted checks, but final commit/push still requires the full mandatory sequence above.
