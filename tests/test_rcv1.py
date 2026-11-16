"""Unit tests for RCV1HierarchyManager and RCV1Manager (Spec 12)."""

import json

import numpy as np

from hmc.datasets.rcv1.dataset_rcv1 import RCV1HierarchyManager
from hmc.datasets.rcv1.manager import RCV1Manager


def test_rcv1_hierarchy_includes_test_only_topics(tmp_path):
    """Verify Spec 12 §3.1 and §3.2: ECAT/E31/E312 and GCAT/GMIL are included and 90/10 split works."""
    train_docs = [
        {"token": f"doc {i} market trade", "label": ["CCAT/C15/C151", "ECAT/E31"]}
        for i in range(20)
    ]
    test_docs = [
        {"token": "test doc military", "label": ["GCAT/GMIL", "ECAT/E31/E312"]}
    ]
    (tmp_path / "rcv1_train.json").write_text(
        "\n".join(json.dumps(d) for d in train_docs) + "\n", encoding="utf-8"
    )
    (tmp_path / "rcv1_test.json").write_text(
        "\n".join(json.dumps(d) for d in test_docs) + "\n", encoding="utf-8"
    )
    mgr = RCV1Manager(str(tmp_path), load_features=False)
    tr, va, te = mgr.get_datasets()
    assert len(tr.y) == 18
    assert len(va.y) == 2
    assert len(te.y) == 1
    assert "ECAT/E31/E312" in mgr.nodes_idx
    assert "GCAT/GMIL" in mgr.nodes_idx
    assert te.y[0, mgr.nodes_idx["ECAT/E31/E312"]] == 1.0
    assert te.y[0, mgr.nodes_idx["GCAT/GMIL"]] == 1.0
    assert isinstance(tr.texts, list) and len(tr.texts) == 18


def test_rcv1_hierarchy_manager_prefix_expansion():
    h = RCV1HierarchyManager.from_topic_paths([["CCAT/C15/C151", "GCAT/GPOL"]])
    yg, yl = h.get_labels(["CCAT/C15/C151"])
    assert yg[h.nodes_idx["root"]] == 1.0
    assert yg[h.nodes_idx["CCAT"]] == 1.0
    assert yg[h.nodes_idx["CCAT/C15"]] == 1.0
    assert yg[h.nodes_idx["CCAT/C15/C151"]] == 1.0
    assert yg[h.nodes_idx["GCAT"]] == 0.0
    assert len(yl) == 4
    assert np.all(np.isfinite(yg))
