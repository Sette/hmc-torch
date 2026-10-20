"""Tests of the Surj reproduction (run from the repository root: ``uv run pytest surj_repro/tests``)."""

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from config import Recipe, load_recipe  # noqa: E402
from data import ancestor_pairs, leaves  # noqa: E402
from gae import figure2, train_gae  # noqa: E402
from metrics import hypotheses, micro_ap  # noqa: E402
from model import ChmcnnNet, Constraint, SurjNet  # noqa: E402

# root(0) <- A(1) <- B(2), C(3); B <- D(4), E(5); C <- F(6). a[i, j] = 1: j is i's parent.
A = np.zeros((7, 7))
for child, parent in [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3)]:
    A[child, parent] = 1
HAS_DATA = os.path.exists("C-HMCNN/HMC_data")


def test_ancestor_pairs_and_leaves():
    anc, desc = ancestor_pairs(A)
    assert len(anc) == 14
    assert set(leaves(A)) == {4, 5, 6}


def test_gae_deterministic_and_structural():
    z1, z2 = train_gae(A, epochs=100, seed=1), train_gae(A, epochs=100, seed=1)
    np.testing.assert_array_equal(z1, z2)
    d = figure2(A, z1, max_len=4)
    assert d[1] < d[4]  # neighbours closer than distant pairs (Surj Fig. 2)


def test_surjnet_onehot_and_shapes():
    net = SurjNet(5, 8, 7, None, dropout=0.0).eval()
    x = torch.randn(3, 5)
    h = torch.nn.functional.normalize(net.fc[-1](torch.relu(net.fc[1](torch.relu(net.fc[0](x))))),
                                      dim=1)
    torch.testing.assert_close(net.similarities(x), h)
    assert SurjNet(5, 8, 7, torch.randn(7, 3)).eval()(x).shape == (3, 7)
    with pytest.raises(ValueError):
        SurjNet(5, 8, 7, torch.randn(6, 3))


def test_chmcnn_net_matches_their_init():
    from compare_chmcnn_loader import ConstrainedFFNNModel  # pylint: disable=import-outside-toplevel

    torch.manual_seed(0)
    theirs = ConstrainedFFNNModel(5, 8, 7, {"num_layers": 3, "dropout": 0.7, "non_lin": "relu"},
                                  None)
    torch.manual_seed(0)
    ours = ChmcnnNet(5, 8, 7)
    for a, b in zip(theirs.parameters(), ours.parameters()):
        torch.testing.assert_close(a, b)


def test_constraint_dense_equals_sparse():
    s = torch.rand(9, 7)
    y = (torch.rand(9, 7) > 0.5).double()
    dense, sparse = Constraint(A, "cpu", sparse=False), Constraint(A, "cpu", sparse=True)
    torch.testing.assert_close(dense(s), sparse(s))
    torch.testing.assert_close(dense.mcloss_output(s, y), sparse.mcloss_output(s, y))
    out = dense(s).numpy()
    anc, desc = ancestor_pairs(A)
    assert np.all(out[:, anc] >= out[:, desc])


def test_metrics():
    y = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=float)
    s = np.array([[0.9, 0.8, 0.1], [0.9, 0.2, 0.7], [0.9, 0.1, 0.3]])
    mask = np.array([False, True, True])
    assert micro_ap(y, s, mask) == pytest.approx(1.0)
    h = hypotheses(y, s, mask)
    assert h["micro_ap_with_roots"] == pytest.approx(1.0)
    assert set(h) == {"micro_ap", "micro_ap_with_roots", "micro_trapezoid", "weighted_ap",
                      "macro_ap", "samples_ap"}


def test_recipes_load():
    for name in ("chmcnn_fixed", "L0_chmcnn", "L3_surj", "L5_surj_mc"):
        assert isinstance(load_recipe(name), Recipe)
    assert load_recipe("L3_surj", {"batch": "32"}).batch == 32
    with pytest.raises(KeyError):
        load_recipe("L3_surj", {"nope": "1"})


@pytest.mark.skipif(not HAS_DATA, reason="C-HMCNN data not available")
def test_rerun_is_bit_identical():
    from data import load  # pylint: disable=import-outside-toplevel
    from train import Trainer  # pylint: disable=import-outside-toplevel

    data = load("eisen_FUN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recipe = Recipe(net="surj", epochs="select_norefit", max_epochs=2, batch=32)
    z = train_gae(data["a"], seed=0, device=device)
    runs = [Trainer(data, recipe, device, z).run(0, "eisen_FUN") for _ in range(2)]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    assert runs[0][0]["test_ap"] == runs[1][0]["test_ap"]
