"""Tests for the label-embedding heads and the GO ontology parser."""

import numpy as np
import pytest
import torch

from hmc.models.hierarchical.label_embedding import (
    MODES,
    LabelEmbeddingHead,
    go_term_texts,
    load_go_obo,
    normalized_adjacency,
)

OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: mitochondrion inheritance
def: "The distribution of mitochondria into daughter cells." [GOC:mcc]
alt_id: GO:0000002

[Term]
id: GO:0000003
name: obsolete reproduction
def: "OBSOLETE. The production of new individuals." [GOC:go]

[Typedef]
id: part_of
name: part of
"""


@pytest.fixture(name="obo_path")
def fixture_obo_path(tmp_path):
    path = tmp_path / "go.obo"
    path.write_text(OBO, encoding="utf-8")
    return str(path)


def test_load_go_obo_keeps_alt_ids_and_strips_obsolete(obo_path):
    texts = load_go_obo(obo_path)
    assert texts["GO:0000001"].startswith("mitochondrion inheritance: The distribution")
    assert texts["GO:0000002"] == texts["GO:0000001"]
    assert texts["GO:0000003"].startswith("reproduction:")
    assert "part_of" not in texts


def test_go_term_texts_maps_dataset_names(obo_path):
    assert go_term_texts(["root", "GO0000002"], obo_path) == [
        "root",
        load_go_obo(obo_path)["GO:0000001"],
    ]


def test_normalized_adjacency_is_symmetric():
    adj = normalized_adjacency(np.array([[1, 0], [2, 0]]), 3).to_dense()
    assert torch.allclose(adj, adj.T)
    assert adj[0, 0] > 0 and adj[1, 2] == 0


@pytest.mark.parametrize("mode", MODES)
def test_heads_start_identical_to_free(mode):
    adjacency = normalized_adjacency(np.array([[1, 0], [2, 0]]), 3)
    text = torch.randn(3, 4)
    torch.manual_seed(0)
    free = LabelEmbeddingHead(5, 8, 3, mode="free")
    torch.manual_seed(0)
    head = LabelEmbeddingHead(
        5, 8, 3, mode=mode, adjacency=adjacency, text_embeddings=text
    )
    # Zero-initialized gates: the knowledge branches start switched off.
    assert torch.allclose(head.label_embeddings(), free.label_embeddings())
    x = torch.randn(2, 5)
    out = head.eval()(x)
    assert out.shape == (2, 3) and torch.all((out > 0) & (out < 1))


def test_missing_inputs_are_rejected():
    with pytest.raises(ValueError):
        LabelEmbeddingHead(5, 8, 3, mode="graph")
    with pytest.raises(ValueError):
        LabelEmbeddingHead(5, 8, 3, mode="text")
