"""Validate and describe the locked Python Blitz reference environment."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import multiprocessing
import platform
import re
import tomllib
from pathlib import Path


THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
EXACT_DEPENDENCY = re.compile(r"^([A-Za-z0-9_.-]+)==([^; ]+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_project(repo_root: Path) -> Path:
    return repo_root / "reference" / "blitz"


def expected_profile(repo_root: Path) -> tuple[str, dict[str, str]]:
    project = reference_project(repo_root)
    config = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    python_spec = config["project"]["requires-python"]
    if not python_spec.startswith("=="):
        raise RuntimeError(f"reference Python must be exact, observed {python_spec!r}")
    packages: dict[str, str] = {}
    for requirement in config["project"]["dependencies"]:
        match = EXACT_DEPENDENCY.fullmatch(requirement)
        if match is None:
            raise RuntimeError(f"reference dependency must be exact: {requirement!r}")
        packages[match.group(1)] = match.group(2)
    return python_spec[2:], packages


def validate_reference_environment(
    repo_root: Path,
    *,
    require_thread_limits: bool = True,
) -> dict[str, object]:
    project = reference_project(repo_root)
    expected_python, expected_packages = expected_profile(repo_root)
    observed_python = platform.python_version()
    if observed_python != expected_python:
        raise SystemExit(
            f"reference Python mismatch: expected {expected_python}, observed {observed_python}"
        )

    observed_packages = {
        package: metadata.version(package) for package in expected_packages
    }
    mismatches = {
        package: {"expected": expected_packages[package], "observed": observed}
        for package, observed in observed_packages.items()
        if observed != expected_packages[package]
    }
    if mismatches:
        raise SystemExit(f"locked reference package mismatch: {mismatches}")

    import os

    thread_environment = {
        variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT
    }
    if require_thread_limits:
        thread_mismatches = {
            variable: value
            for variable, value in thread_environment.items()
            if value != "1"
        }
        if thread_mismatches:
            raise SystemExit(
                "Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1; "
                f"observed {thread_mismatches}"
            )
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 for Blitz reference generation")

    lock = project / "uv.lock"
    if not lock.is_file():
        raise SystemExit(f"missing locked reference environment: {lock}")

    return {
        "python": observed_python,
        "packages": observed_packages,
        "lock_sha256": sha256(lock),
        "platform": platform.platform(),
        "multiprocessing_start_method": multiprocessing.get_start_method(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "thread_environment": thread_environment,
    }
