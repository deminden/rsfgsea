# AGENTS.md

## Pre-commit Requirements (Mandatory)

Before every commit and before every push, run all checks below from repo root and ensure they pass.

1. `./r-pkg/rsfgseaR/cleanup`
2. `./scripts/sync_r_vendor.sh`
3. `cargo fmt --all -- --check`
4. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
5. `cargo test --workspace --all-features`
6. `R CMD build r-pkg/rsfgseaR`
7. `R CMD check rsfgseaR_<version>.tar.gz --no-manual`
8. Remove the built tarball and `rsfgseaR.Rcheck/` after validation

## Rules

- Do routine development work on `dev`, not directly on `main`.
- Only update `main` after `dev` has been pushed and GitHub CI has passed on `dev`.
- Prevent R package artifact failures by running `./r-pkg/rsfgseaR/cleanup` before verification and before packaging.
- During iterative Rust-side development, do not spend time resyncing the vendored R Rust core or updating wrapper-facing vendored files after every edit.
- Finalize the Rust-side implementation first, then rerun `./scripts/sync_r_vendor.sh` and update wrapper-facing generated/vendor artifacts before the final verification sequence and commit.
- If you edit Rust after running `./scripts/sync_r_vendor.sh`, rerun it before committing.
- The vendored R Rust core under `r-pkg/rsfgseaR/src/rust/vendor/rsfgsea` must stay in sync with `crates/rsfgsea`.
- After sync, the R package source tree should contain `r-pkg/rsfgseaR/src/rust/vendor/rsfgsea` and `r-pkg/rsfgseaR/src/rust/vendor.tar.xz`, but not a live `r-pkg/rsfgseaR/src/vendor` tree or `r-pkg/rsfgseaR/src/.cargo`.
- Do not commit if required checks fail or if cleanup leaves build artifacts in `r-pkg/rsfgseaR/src` or `r-pkg/rsfgseaR/src/rust/target`.
- Built R source tarballs such as `rsfgseaR_*.tar.gz` and local `rsfgseaR.Rcheck/` directories are packaging artifacts and must not be committed.
- If a check fails, fix it and rerun the full mandatory sequence.
- If tests fail due to environment/hardware constraints (for example GPU-only runtime tests), report this clearly in the commit/PR notes.

## Optional local speed-up (not for final verification)

During development you may run targeted checks, but final commit/push still requires the full mandatory sequence above.

## Optional local speed-up details

If you need a quicker local signal during iteration, a directory-based check such as
`R CMD check r-pkg/rsfgseaR --no-manual` is acceptable as a temporary shortcut, but it
does not replace the mandatory tarball-based validation above because it bypasses
`.Rbuildignore` filtering and can produce packaging notes that are not relevant to the
built source package.
