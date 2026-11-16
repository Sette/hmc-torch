"""HPT prompt model, its losses and the macro-F1 used to compare with HPT."""

import networkx as nx
import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

from hmc.models.global_classifier.constraint.utils import get_constr_out
from hmc.models.global_classifier.hpt import (
    HPTPromptModel,
    build_edge_index,
    hpt_objective,
    mc_logits,
    zmlce,
)
from hmc.models.hierarchical.sparse_r import BlockDiagonalR
from hmc.pipeline.hpt.main import HPTConfig, selection_score
from hmc.utils.metrics.multilabel import macro_f1

WORDS = ["science", "computer", "machine", "learning", "biology", "cell", "the", "a"]

# Columns: 0 root, 1 cs, 2 bio, 3 ml (cs), 4 cell (bio).  child -> parent edges.
EDGES = [(1, 0), (2, 0), (3, 1), (4, 2)]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    vocab = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)


def _model(tokenizer, direction="both", use_mlm=True):
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    # Label space (no root): 0 cs, 1 bio, 2 ml, 3 cell.
    return HPTPromptModel(
        BertForMaskedLM(cfg),
        tokenizer,
        label_texts=["computer science", None, "machine learning", "cell biology"],
        label_codes=["cs", "bio", "ml", "cell"],
        label_depth=[0, 0, 1, 1],
        edges=np.array([[0, 2], [1, 3]]),
        graph_direction=direction,
        use_mlm=use_mlm,
    )


def _sparse_r():
    g = nx.DiGraph()
    g.add_nodes_from(range(5))
    g.add_edges_from(EDGES)
    return BlockDiagonalR(g, column_index={i: i for i in range(5)})


def _dense_r():
    r = torch.eye(5, dtype=torch.float64)
    for child, anc in [(1, 0), (2, 0), (3, 1), (4, 2), (3, 0), (4, 0)]:
        r[anc, child] = 1  # pipeline layout: row = ancestor
    return r.unsqueeze(0)


def test_prompt_has_one_pred_slot_per_level_and_truncates(tokenizer):
    model = _model(tokenizer)
    rows = model.prompt_rows(["machine learning " * 40, "cell"], max_length=24)
    assert len(rows[0]) == 24
    assert rows[0][-len(model.prompt_ids) :] == model.prompt_ids
    batch = model.collate(rows)
    assert (batch["input_ids"] == model.pred_id).sum(1).tolist() == [2, 2]
    assert batch["attention_mask"][1].sum() == len(rows[1])


def test_forward_shapes_and_mlm_loss(tokenizer):
    model = _model(tokenizer)
    batch = model.collate(model.prompt_rows(["the machine learning"] * 3, 32))
    model.train()
    logits, mlm_loss = model(**batch)
    assert logits.shape == (3, 4)
    assert mlm_loss is not None and torch.isfinite(mlm_loss)
    model.eval()
    _, mlm_loss = model(**batch)
    assert mlm_loss is None


def test_unnamed_label_starts_as_mean_of_children(tokenizer):
    model = _model(tokenizer)
    assert torch.allclose(model.label_emb[1], model.label_emb[3])  # bio has one child


def test_graph_direction_controls_level_inputs(tokenizer):
    down = build_edge_index(np.array([[0, 2]]), [0, 0, 1], "down")
    # Level nodes (3, 4) are never targets of HPT's one-way edges.
    assert not set(down[1].tolist()) & {3, 4}
    both = build_edge_index(np.array([[0, 2]]), [0, 0, 1], "both")
    assert {3, 4} <= set(both[1].tolist())

    model = _model(tokenizer, "both")
    nodes = model.graph_nodes()
    model.label_emb.data[2] += 1.0  # changing a level-1 label moves level token 1
    assert not torch.allclose(model.graph_nodes()[5], nodes[5])
    assert model.mlm.get_input_embeddings().num_embeddings == tokenizer.vocab_size


def test_zmlce_matches_closed_form():
    z = torch.tensor([[2.0, -1.0, 0.5], [-0.3, 0.7, -2.0]])
    y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    expected = []
    for zi, yi in zip(z, y):
        neg = torch.log1p(torch.exp(zi[yi == 0]).sum())
        pos = torch.log1p(torch.exp(-zi[yi == 1]).sum())
        expected.append(neg + pos)
    assert torch.allclose(zmlce(z, y), torch.stack(expected).mean())


def test_mc_on_logits_matches_mcloss_on_probabilities():
    torch.manual_seed(1)
    z = torch.randn(6, 5, dtype=torch.float64)
    y = torch.tensor(
        [[1, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 0, 0]] * 2, dtype=torch.float64
    )
    p = torch.sigmoid(z)
    r = _dense_r()
    mcloss_out = (1 - y) * get_constr_out(p, r) + y * get_constr_out(y * p, r)
    assert torch.allclose(
        torch.sigmoid(mc_logits(z, y, _sparse_r()).double()), mcloss_out, atol=1e-6
    )


def test_objectives_are_differentiable():
    torch.manual_seed(2)
    z = torch.randn(4, 5, requires_grad=True)
    y = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 0, 1]] * 2, dtype=torch.float32)
    cols = torch.arange(1, 5)
    anc, desc = torch.tensor([0, 0, 1, 2, 0, 0]), torch.tensor([1, 2, 3, 4, 3, 4])
    for kind in ("none", "mc", "hinge"):
        loss = hpt_objective(kind, z, y, cols, _sparse_r(), anc, desc)
        (grad,) = torch.autograd.grad(loss, z)
        assert (
            torch.isfinite(loss) and torch.isfinite(grad).all() and grad.abs().sum() > 0
        )
    with pytest.raises(ValueError):
        hpt_objective("bce", z, y, cols, _sparse_r(), anc, desc)


def test_macro_f1_counts_absent_labels_as_zero():
    y = np.array([[1, 0, 0], [0, 1, 0]])
    s = np.array([[0.9, 0.1, 0.1], [0.2, 0.8, 0.1]])
    assert macro_f1(y, s, 0.5) == pytest.approx(2 / 3)


def test_selection_score_hpt_raw_uses_raw_macro_f1():
    y = np.array([[1, 0], [1, 0], [0, 1]])
    raw = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    rec = np.zeros_like(raw)
    mask = np.array([True, True])
    assert selection_score("hpt_raw", y, raw, rec, mask, 0.5) == pytest.approx(1.0)


def test_selection_score_default_uses_reconciled_micro_f1():
    y = np.array([[1, 0], [0, 1]])
    raw = np.zeros((2, 2))
    rec = np.array([[0.9, 0.1], [0.1, 0.9]])
    mask = np.array([True, True])
    assert selection_score("rec_val_micro", y, raw, rec, mask, 0.5) == pytest.approx(
        1.0
    )
    assert selection_score("rec_val_micro", y, raw, rec, mask, 0.95) == pytest.approx(
        0.0
    )


def test_hpt_config_selection_values():
    assert HPTConfig().selection == "rec_val_micro"
    assert HPTConfig(selection="hpt_raw").selection == "hpt_raw"


def test_spec10_m1_sentence_and_random_label_encoders(tokenizer):
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    for enc in ("sentence", "random"):
        mod = HPTPromptModel(
            BertForMaskedLM(cfg),
            tokenizer,
            label_texts=["computer science", None, "computer science > machine learning", "biology > cell"],
            label_codes=["cs", "bio", "ml", "cell"],
            label_depth=[0, 0, 1, 1],
            edges=np.array([[0, 2], [1, 3]]),
            label_encoder=enc,
        )
        assert mod.label_emb.shape == (4, 16)
        assert torch.isfinite(mod.label_emb).all()
        batch = mod.collate(mod.prompt_rows(["the machine learning"] * 2, 32))
        logits, _ = mod(**batch)
        assert logits.shape == (2, 4)


def test_spec10_m2_multi_token_prompts_and_crossattn_head(tokenizer):
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    mod = HPTPromptModel(
        BertForMaskedLM(cfg),
        tokenizer,
        label_texts=["computer science", "biology", "machine learning", "cell biology"],
        label_codes=["cs", "bio", "ml", "cell"],
        label_depth=[0, 0, 1, 1],
        edges=np.array([[0, 2], [1, 3]]),
        label_encoder="sentence",
        label_prompt_tokens=4,
        score_head="crossattn",
    )
    # 2 levels * (4 virtual prompt tokens + 1 [PRED] slot) + [SEP] = 11 prompt ids
    assert len(mod.prompt_ids) == 11
    batch = mod.collate(mod.prompt_rows(["science computer", "cell biology"], 32))
    assert (batch["input_ids"] == mod.pred_id).sum(1).tolist() == [2, 2]
    mod.train()
    logits, mlm_loss = mod(**batch)
    assert logits.shape == (2, 4)
    loss = logits.sum() + (mlm_loss if mlm_loss is not None else 0.0)
    loss.backward()
    assert mod.prompt_token_offset.grad is not None
    assert mod.prompt_token_offset.grad.abs().sum() > 0
