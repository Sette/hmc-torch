"""Tests for the label-embedding heads and the GO ontology parser."""

import numpy as np
import pytest
import torch

from hmc.models.hierarchical.graph_autoencoder import pretrain_label_gae
from hmc.models.hierarchical.label_embedding import (
    MODES,
    LabelEmbeddingHead,
    go_term_texts,
    load_go_obo,
    normalized_adjacency,
    pretrain_two_stage,
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


@pytest.mark.parametrize("mode", [m for m in MODES if "gae" not in m])
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
    with pytest.raises(ValueError):
        LabelEmbeddingHead(5, 8, 3, mode="gae")


@pytest.mark.parametrize("mode", ["gae", "gae+free"])
def test_gae_heads_score_by_scaled_cosine(mode):
    feats = torch.randn(3, 4)
    head = LabelEmbeddingHead(5, 8, 3, mode=mode, label_features=feats).eval()
    x = torch.randn(2, 5)
    h = torch.nn.functional.normalize(head.encoder(x), dim=1)
    emb = torch.nn.functional.normalize(head.label_embeddings(), dim=1)
    expected = torch.sigmoid(10.0 * (h @ emb.T) + head.bias)
    assert torch.allclose(head(x), expected, atol=1e-6)


def test_gae_init_starts_from_the_projected_prior():
    feats = torch.randn(3, 4)
    head = LabelEmbeddingHead(5, 8, 3, mode="gae-init", label_features=feats)
    prior = head.label_proj(feats).detach()
    cos = torch.nn.functional.cosine_similarity(head.class_vectors, prior, dim=1)
    assert torch.allclose(cos, torch.ones(3), atol=1e-5)
    assert float(head.extra_loss()) == 0.0
    x = torch.randn(2, 5)
    expected = torch.sigmoid(head.encoder.eval()(x) @ head.class_vectors.T + head.bias)
    assert torch.allclose(head.eval()(x), expected)


def test_gae_reg_penalizes_distance_to_the_prior():
    feats = torch.randn(3, 4)
    head = LabelEmbeddingHead(
        5, 8, 3, mode="gae-reg", label_features=feats, prior_weight=0.5
    )
    gap = head.class_vectors - head.label_proj(feats)
    assert torch.allclose(head.extra_loss(), 0.5 * gap.pow(2).sum(1).mean())
    head.extra_loss().backward()
    assert (
        head.class_vectors.grad is not None and head.label_proj.weight.grad is not None
    )


def test_two_stage_regresses_label_embeddings_then_freezes_the_encoder():
    torch.manual_seed(0)
    feats = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    y = np.eye(3, dtype=np.float32)[np.arange(60) % 3]
    x = (y @ np.random.default_rng(0).normal(size=(3, 5)) * 3).astype(np.float32)
    head = LabelEmbeddingHead(
        5, 16, 3, mode="gae-2stage", label_features=feats, dropout=0.0
    )
    pretrain_two_stage(head, x, y, x, y, lr=1e-2, max_epochs=100)
    head.eval()
    with torch.no_grad():
        cos = head.embed(torch.from_numpy(x)) @ feats.T
    # Each sample lands closest to the embedding of its own label.
    assert (cos.argmax(1).numpy() == y.argmax(1)).mean() > 0.9
    trainable = {n for n, p in head.named_parameters() if p.requires_grad}
    assert trainable == {"bias", "log_scale", "class_vectors"}
    assert not head.train().encoder.training


def test_gae_reconstructs_tree_edges():
    # Binary tree of depth 3 (15 nodes), child -> parent.
    edges = np.array([[i, (i - 1) // 2] for i in range(1, 15)])
    depth = np.floor(np.log2(np.arange(15) + 1))
    z = torch.from_numpy(pretrain_label_gae(edges, 15, dim=16, epochs=300, depth=depth))
    assert z.shape == (15, 16)
    scores = z @ z.T
    pos = scores[edges[:, 0], edges[:, 1]]
    linked = {(int(a), int(b)) for a, b in edges} | {(int(b), int(a)) for a, b in edges}
    neg = torch.stack(
        [
            scores[i, j]
            for i in range(15)
            for j in range(15)
            if i != j and (i, j) not in linked
        ]
    )
    auc = (pos[:, None] > neg[None, :]).float().mean()
    assert auc > 0.9
