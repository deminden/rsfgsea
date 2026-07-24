#!/usr/bin/env python3
"""Fail when versioned Blitz documentation drifts from its locked evidence."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "blitz-latest.json"
REFERENCE_PROJECT = ROOT / "reference" / "blitz" / "pyproject.toml"
STALE_TEXT = ("NumPy `2.4.0`", "SciPy `1.16.3`", "pandas `2.3.3`")


def exact_dependencies(project: dict[str, object]) -> dict[str, str]:
    dependencies = project["project"]["dependencies"]
    result = {}
    for dependency in dependencies:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^; ]+)", dependency)
        if match is None:
            raise SystemExit(f"reference dependency is not exact: {dependency}")
        result[match.group(1)] = match.group(2)
    return result


def documentation_files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "crates" / "rsfgseapy" / "README.md",
        ROOT / "r-pkg" / "rsfgseaR" / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    project = tomllib.loads(REFERENCE_PROJECT.read_text(encoding="utf-8"))
    expected_python = project["project"]["requires-python"].removeprefix("==")
    expected_packages = exact_dependencies(project)
    if evidence["reference"]["python"] != expected_python:
        raise SystemExit("Blitz evidence Python version differs from pyproject.toml")
    if evidence["reference"]["packages"] != expected_packages:
        raise SystemExit("Blitz evidence packages differ from pyproject.toml")

    failures = []
    for path in documentation_files():
        text = path.read_text(encoding="utf-8")
        for stale in STALE_TEXT:
            if stale in text:
                failures.append(f"{path.relative_to(ROOT)} contains stale {stale!r}")
        if "/home/den/" in text:
            failures.append(f"{path.relative_to(ROOT)} contains a machine-local path")
    if failures:
        raise SystemExit("\n".join(failures))

    print("documentation versions and paths match the locked Blitz evidence")


if __name__ == "__main__":
    main()
