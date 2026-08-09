"""Reproducibility manifest: git SHA, seeds, dependency versions.

Writes ``manifest.json`` alongside experiment artefacts so every run
is independently reproducible.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime


def get_git_sha(repo_path: str = ".") -> str | None:
    """Return the current git HEAD SHA, or None if not in a repo."""
    try:
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def get_git_branch(repo_path: str = ".") -> str | None:
    """Return the current git branch name."""
    try:
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def get_dependency_versions(packages: list[str] | None = None) -> dict[str, str]:
    """Return installed versions of key packages."""
    if packages is None:
        packages = [
            "torch",
            "numpy",
            "sklearn",
            "networkx",
            "transformers",
            "pandas",
            "scipy",
        ]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            for attr in ("__version__", "version"):
                v = getattr(mod, attr, None)
                if v:
                    versions[pkg] = str(v)
                    break
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def write_manifest(
    output_dir: str,
    method: str,
    dataset_name: str,
    seed: int,
    extra: dict | None = None,
) -> str:
    """Write ``manifest.json`` and return its path.

    Args:
        output_dir: Directory to write the manifest to.
        method: Training method (e.g. ``"tabular_gbdt"``).
        dataset_name: Dataset identifier.
        seed: Random seed.
        extra: Additional key-value pairs to include.

    Returns:
        Path to the written file.
    """
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": get_git_sha(),
        "git_branch": get_git_branch(),
        "method": method,
        "dataset": dataset_name,
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "dependencies": get_dependency_versions(),
    }
    if extra:
        manifest["extra"] = extra

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return path
