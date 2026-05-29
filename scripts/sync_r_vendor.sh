#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_root="$repo_root/crates/rsfgsea"
package_src_root="$repo_root/r-pkg/rsfgseaR/src"
rust_root="$package_src_root/rust"
vendor_root="$rust_root/vendor/rsfgsea"
vendor_tarball="$rust_root/vendor.tar.xz"
python_bin="${PYTHON:-}"

if [[ ! -d "$src_root" ]]; then
  echo "Source crate not found: $src_root" >&2
  exit 1
fi

if [[ -z "$python_bin" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  elif command -v python >/dev/null 2>&1; then
    python_bin="python"
  else
    echo "Neither python3 nor python was found on PATH" >&2
    exit 1
  fi
fi

rm -rf "$vendor_root"
mkdir -p "$vendor_root"

cp "$src_root/Cargo.toml" "$vendor_root/Cargo.toml"
cp "$src_root/README.md" "$vendor_root/README.md"
cp -r "$src_root/src" "$vendor_root/src"
if [[ -d "$src_root/assets" ]]; then
  cp -r "$src_root/assets" "$vendor_root/assets"
fi

"$python_bin" - <<'PY'
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

cleanup() {
  rm -rf "$tmp_vendor_dir"
}
trap cleanup EXIT

cargo vendor --locked --manifest-path "$rust_root/Cargo.toml" "$tmp_vendor_dir" >/dev/null

rm -f "$vendor_tarball"
"$python_bin" - "$tmp_vendor_dir" "$vendor_tarball" <<'PY'
import os
import sys
import tarfile
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])


def reset_metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


with tarfile.open(out, "w:xz") as tar:
    for path in sorted(src.rglob("*")):
        arcname = Path(".") / path.relative_to(src)
        tar.add(path, arcname=os.fspath(arcname), recursive=False, filter=reset_metadata)
PY

# Keep the R package source tree free of unpacked Cargo vendor directories.
rm -rf "$package_src_root/vendor" "$package_src_root/.cargo"
rm -f "$rust_root/vendor-config.toml"

echo "Synced vendored rsfgsea into $vendor_root"
echo "Updated offline Cargo bundle at $vendor_tarball"
