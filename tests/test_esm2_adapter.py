"""ESM-2 adapter for the experiment scripts: features swap and protocol-epoch redirection."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import esm2_features  # pylint: disable=wrong-import-position
import run_protocol  # pylint: disable=wrong-import-position
import run_sr_gae_sota  # pylint: disable=wrong-import-position


def _protein_lm_json(path, epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = [{"features": f, "objective": "mc", "seed": s, "best_epoch": e}
            for f, e in (("esm", epochs), ("orig", 999), ("esm+orig", 777)) for s in (0, 42)]
    path.write_text(json.dumps({"dataset": "seq_FUN", "protocol": "p", "runs": runs}))


def test_write_protocol_esm2_keeps_only_esm_runs(tmp_path):
    _protein_lm_json(tmp_path / "src" / "tag" / "seq_FUN.json", epochs=40)
    (written,) = esm2_features.write_protocol_esm2("tag", str(tmp_path / "out"), str(tmp_path / "src"))
    rec = json.loads(open(written, encoding="utf-8").read())
    assert {r["features"] for r in rec["runs"]} == {"esm"}
    assert {r["best_epoch"] for r in rec["runs"]} == {40}


def test_sota_reads_epochs_from_the_redirected_protocol_dir(tmp_path, monkeypatch):
    """Without redirection ``seq_FUN`` would silently reuse the epochs of the benchmark features."""
    _protein_lm_json(tmp_path / "src" / "tag" / "seq_FUN.json", epochs=40)
    esm2_features.write_protocol_esm2("tag", str(tmp_path / "out"), str(tmp_path / "src"))
    monkeypatch.setattr(run_sr_gae_sota, "PROTOCOL_DIR", str(tmp_path / "out"))
    epochs = run_sr_gae_sota._protocol_best_epochs("seq_FUN", "mc")  # pylint: disable=protected-access
    assert {r["best_epoch"] for r in epochs.values()} == {40}
    monkeypatch.setattr(run_sr_gae_sota, "PROTOCOL_DIR", "output/protocol")
    if os.path.exists("output/protocol/seq_FUN.json"):
        bench = run_sr_gae_sota._protocol_best_epochs("seq_FUN", "mc")  # pylint: disable=protected-access
        assert {r["best_epoch"] for r in bench.values()} != {40}


def test_install_swaps_features_only_for_sequence_datasets(tmp_path, monkeypatch):
    calls = []

    def fake_original(name):
        calls.append(name)
        return {"x": [np.zeros((2, 3)), np.zeros((1, 3)), np.zeros((1, 3))], "y": "labels"}

    class FakeEncoder:
        def __init__(self, *_a, **_k):
            pass

        def encode(self, sequences):
            return np.array([[len(s), 1.0] for s in sequences], dtype=np.float32)

    ids = tmp_path / "seq_FUN_ids.tsv"
    ids.write_text("split\trow\torf\tdist\tsecond\tunique\n"
                   "train\t0\tYA\t0\t9\t1\ntrain\t1\t\tinf\tinf\t0\nvalid\t0\tYA\t0\t9\t1\ntest\t0\tYA\t0\t9\t1\n")
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">YA Chr I, x\nMKTAYIAKQR\n")
    monkeypatch.setattr(run_protocol, "load_dataset", fake_original)
    monkeypatch.setattr(esm2_features, "PROTEOME", str(fasta))
    monkeypatch.setattr(esm2_features, "IDS_DIR", str(tmp_path))
    monkeypatch.setattr("hmc.features.protein.ProteinFeatureEncoder", FakeEncoder)
    esm2_features.install()
    data = run_protocol.load_dataset("seq_FUN")
    assert [x.shape for x in data["x"]] == [(2, 3), (1, 3), (1, 3)]  # 2 embedding dims + missing flag
    assert data["features"] == "esm2" and data["y"] == "labels"
    np.testing.assert_array_equal(data["x"][0][:, -1], [0, 1])
    assert run_protocol.load_dataset("cellcycle_FUN")["x"][0].shape == (2, 3)  # untouched
    assert "features" not in run_protocol.load_dataset("cellcycle_FUN")
    assert calls == ["seq_FUN", "cellcycle_FUN", "cellcycle_FUN"]


def test_launcher_requires_a_protected_out_dir(monkeypatch):
    import run_with_esm2  # pylint: disable=import-outside-toplevel

    for argv in (["x.py"], ["x.py", "--out_dir", "output/sr_gae_paired"]):
        monkeypatch.setattr(sys, "argv", ["run_with_esm2.py", *argv])
        with pytest.raises(SystemExit):
            run_with_esm2.main()
