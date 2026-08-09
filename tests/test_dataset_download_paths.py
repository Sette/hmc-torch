"""Tests for dataset download output layout."""

from pathlib import Path

from hmc.datasets import download_all
from hmc.datasets.wos.download_wos import _resolve_output_dir


def test_wos_download_resolves_data_root_to_wos_subdir():
    assert _resolve_output_dir("./data") == Path("data/wos")


def test_wos_download_keeps_explicit_wos_dir():
    assert _resolve_output_dir("./data/wos") == Path("data/wos")


def test_download_all_uses_dataset_subdirectories(monkeypatch):
    calls = []

    def fake_run_module(module_name, output_dir):
        calls.append((module_name, output_dir))
        return True

    monkeypatch.setattr(download_all, "_run_module", fake_run_module)
    monkeypatch.setattr(
        "sys.argv",
        ["download_all", "--output_dir", "./data", "--groups", "arxiv,wos"],
    )

    download_all.main()

    assert calls == [
        ("hmc.datasets.arxiv.download_arxiv", "data/arxiv"),
        ("hmc.datasets.wos.download_wos", "data/wos"),
    ]
