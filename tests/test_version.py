"""The version comes from ``project.version`` in pyproject, with a metadata fallback."""

import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import hmc

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_comes_from_pyproject():
    """``__version__`` is the pyproject's ``project.version``."""
    expected = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert hmc.__version__ == expected


def test_version_falls_back_to_metadata(monkeypatch, tmp_path):
    """Without the pyproject, the installed package metadata wins."""
    monkeypatch.setattr(hmc, "_PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(hmc, "version", lambda _: "9.9.9")
    assert hmc._read_version() == "9.9.9"


def test_version_without_metadata(monkeypatch, tmp_path):
    """Without pyproject and metadata, the version is 0.0.0."""

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(hmc, "_PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(hmc, "version", missing)
    assert hmc._read_version() == "0.0.0"
