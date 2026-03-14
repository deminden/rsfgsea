#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_root="$repo_root/crates/rsfgsea"
package_src_root="$repo_root/r-pkg/rsfgseaR/src"
rust_root="$package_src_root/rust"
vendor_root="$rust_root/vendor/rsfgsea"
vendor_tarball="$rust_root/vendor.tar.xz"
vendor_config="$rust_root/vendor-config.toml"
expanded_vendor_root="$package_src_root/vendor"

if [[ ! -d "$src_root" ]]; then
  echo "Source crate not found: $src_root" >&2
  exit 1
fi

rm -rf "$vendor_root"
mkdir -p "$vendor_root"

cp "$src_root/Cargo.toml" "$vendor_root/Cargo.toml"
cp "$src_root/README.md" "$vendor_root/README.md"
cp -r "$src_root/src" "$vendor_root/src"
if [[ -d "$src_root/assets" ]]; then
  cp -r "$src_root/assets" "$vendor_root/assets"
fi

python - <<'PY'
from pathlib import Path

manifest_path = Path("r-pkg/rsfgseaR/src/rust/vendor/rsfgsea/Cargo.toml")
lines = manifest_path.read_text().splitlines()

out = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    if stripped.startswith("autobins ="):
        i += 1
        continue

    if stripped == "[dev-dependencies]":
        i += 1
        while i < len(lines) and not lines[i].startswith("["):
            i += 1
        continue

    if stripped.startswith("[[bin]]") or stripped.startswith("[[bench]]"):
        i += 1
        while i < len(lines) and not lines[i].startswith("["):
            i += 1
        continue

    out.append(line)
    i += 1

manifest_path.write_text("\n".join(out) + "\n")
PY

cargo generate-lockfile --manifest-path "$rust_root/Cargo.toml"

tmp_vendor_dir="$(mktemp -d)"
tmp_vendor_config="$(mktemp)"

cleanup() {
  rm -rf "$tmp_vendor_dir"
  rm -f "$tmp_vendor_config"
}
trap cleanup EXIT

cargo vendor --locked --manifest-path "$rust_root/Cargo.toml" "$tmp_vendor_dir" > "$tmp_vendor_config"

rm -rf "$expanded_vendor_root"
mv "$tmp_vendor_dir" "$expanded_vendor_root"

python - "$tmp_vendor_config" "$vendor_config" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
patched_lines = []
for line in source.splitlines():
    if line.strip().startswith("directory = "):
        patched_lines.append('directory = "vendor"')
    else:
        patched_lines.append(line)
patched = "\n".join(patched_lines) + "\n"
Path(sys.argv[2]).write_text(patched)
PY

rm -f "$vendor_tarball"
tar -cJf "$vendor_tarball" -C "$package_src_root" vendor
rm -rf "$expanded_vendor_root"

echo "Synced vendored rsfgsea into $vendor_root"
echo "Updated offline Cargo bundle at $vendor_tarball"
