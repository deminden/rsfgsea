# AGENTS.md

## Pre-commit Requirements (Mandatory)

Before every commit and before every push, run all checks below from repo root and ensure they pass.

1. `./scripts/sync_r_vendor.sh`
2. `cargo fmt --all -- --check`
3. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
4. `cargo test --workspace --all-features`

## Rules

- Do not commit if any check fails.
- Do not commit if the vendored R Rust core under `r-pkg/rsfgseaR/src/rust/vendor/rsfgsea` is out of sync with `crates/rsfgsea`.
- If a check fails, fix the issue, then rerun the full sequence.
- If tests fail due to environment/hardware constraints (for example GPU-only runtime tests), report this clearly in the commit/PR notes.

## Optional local speed-up (not for final verification)

During development you may run targeted checks, but final commit/push still requires the full mandatory sequence above.
