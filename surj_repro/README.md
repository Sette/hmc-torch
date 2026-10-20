# Surj reproduction

A standalone, reproducible reimplementation of **Surj** (Yang & Howe, *Ontological Learning for
Fast, Accurate, and Robust Hierarchical Multi-label Classification*, WWW'22 companion; paper
in `docs/`). Its code was announced ("will be published on github") but never released.

This folder is independent from the `hmc` package. It borrows C-HMCNN's parser (through
`experiments/compare_chmcnn_loader.py`), plus `BlockDiagonalR` and
`global_hierarchy_violation` from `src/`. The package's own version (`--method surj`,
`experiments/run_surj.py`) evaluates Surj under our paper's validation-only protocol.

## Layout

| File | Role |
|---|---|
| `data.py` | C-HMCNN's parser and preprocessing; `_others` validation split; label subsetting (evolving) and subsampling (data size) |
| `gae.py` | Kipf & Welling graph autoencoder, ported from `tkipf/gae`; `python surj_repro/gae.py cellcycle_FUN` checks Fig. 2 |
| `model.py` | `ChmcnnNet` (C-HMCNN's network), `SurjNet` (Surj's head, GAE or one-hot labels), `Constraint` (C-HMCNN's MCM/MCLoss) |
| `config.py`, `recipes.json` | every training knob as a named recipe |
| `train.py` | one recipe × datasets × seeds → `results/<run>/<dataset>/seed<k>.{json,npz}` |
| `metrics.py` | C-HMCNN's micro AP plus the alternative metrics used as hypotheses |
| `hpsearch.py` | validation-only hyperparameter search → recipes `selected` / `selected_onehot` |
| `summarize.py` | `results.md` from the saved files only |
| `run_all.sh` | stages `sanity ladder hpsearch final extras summary` |
| `tests/` | unit tests, including a bit-identical rerun |

## Reproducing

From the repository root. The data and parser come from C-HMCNN's repository at the commit
used here (unzip its `HMC_data` archive in place if the clone has it zipped):

```bash
git clone https://github.com/EGiunchiglia/C-HMCNN.git && git -C C-HMCNN checkout e887855ab4b3b8f22e4fafc80a0a33c1d1ac0041
uv run pytest surj_repro/tests -q
surj_repro/run_all.sh sanity ladder hpsearch final extras summary   # P=3 parallel processes by default
```

Every result JSON stores the recipe, the git commit (plus a dirty flag), torch/CUDA/GPU
versions, the MD5 of the ARFF files, epochs and timings. The NPZ next to it holds the test
scores, so `summarize.py` and any new metric can run without retraining (the NPZs, ~300 MB,
are not versioned; `run_all.sh` regenerates them). Seeds use C-HMCNN's
scheme plus `torch.use_deterministic_algorithms`; a rerun on the same machine gives identical
scores (`tests/test_repro.py::test_rerun_is_bit_identical`).

## What the paper fixes

| Paper | Here |
|---|---|
| "we adopt the code provided by Giunchiglia et al. (C-HMCNN) for dataset pre-processing and evaluation" (Sec. 4.4) | C-HMCNN's `utils/parser.py`, unmodified; mean imputation + StandardScaler fit on train+val, as in `C-HMCNN/main.py` |
| roots removed from evaluation; micro AU(PRC) | the parser's `to_eval`; `average_precision_score(..., average="micro")` |
| GAE of Kipf & Welling, `Â ReLU(Â S W0) W1`, `σ(ZZᵀ)`, no node features | `gae.py` with `tkipf/gae` defaults (32 → 16, Adam 0.01, 200 epochs), S = I |
| FC layers → cosine with the node embeddings → another FC layer → sigmoid, BCE (Sec. 3.3) | `SurjNet` |
| ablation: same network with one-hot labels (Tab. 5) | `net: onehot` (Z = I) |
| lr 1e-3; hyperparameters chosen on validation and identical across datasets | `hpsearch.py` |
| average of 10 trials | seeds 0–9 |
| Tab. 1: `_others` split 70/30 into train/validation | 30% of the training file held out with seed 0 |

## What it leaves open

Batch size, epochs/stopping, dropout, weight decay, D, the FC sizes actually used, and whether
C-HMCNN's hierarchy constraint (MCM/MCLoss) stayed in the adopted code. The paper reports
**zero** hierarchy violations on all 20 datasets, which an unconstrained sigmoid head cannot
produce. These are recipe fields, settled on validation by `hpsearch.py`, never on test.

## Method of the investigation

1. **Sanity.** C-HMCNN's network through our training loop must reproduce its published
   numbers, first with its hard-coded epochs and then with our early stopping + refit. Until
   this passes, any gap could be the harness's fault.
2. **Ladder.** From C-HMCNN to Surj one component at a time (`recipes.json`, `L0`…`L5`):
   output head, constraint, GAE embeddings, learning rate. This locates which change costs or
   gains accuracy.
3. **Hyperparameter search** on validation, as the paper describes.
4. **Final runs** with the selected recipe: Tables 2–5 and Fig. 5.
5. **Metric hypotheses** (diagnostic only): roots kept, trapezoidal PR area, weighted, macro
   and per-sample AP, recomputed from the saved scores. An explanation of the published numbers
   must fit C-HMCNN (ratio ≈ 1) and Surj on every dataset at once.

## Pilot (before this setup; `results/pilot_v0/`)

Guessed recipe (lr 1e-3, batch 4, dropout 0.7, D 16), 3 seeds:

| dataset | Surj paper | ours surj | onehot paper | ours onehot | C-HMCNN |
|---|---|---|---|---|---|
| eisen_FUN | 0.392 | 0.263 | 0.281 | 0.160 | 0.306 |
| cellcycle_FUN | 0.269 | 0.221 | 0.211 | 0.155 | 0.255 |

* The one-hot head lands far below C-HMCNN's network on identical data and stops at epoch
  4–9, so that recipe is broken. This is what motivated the ladder.
* MCM + MCLoss (eisen, seed 0) gives zero violations but the same AP (0.266).
* Keeping the root columns adds +0.08–0.14, but not in the paper's per-dataset pattern.

## Scope of the runs

* **Tables 2 and 5:** run with the `selected` recipe and its one-hot partner. FunCat uses 10
  seeds; the `_others` and GO use 3 seeds. At batch 4, ImCLEF (10k samples) and GO (4k labels)
  cost ~45 min per seed, and 3 seeds are enough to measure the gap.
* **Not run:** the `extras` stage (evolving ontology, Table 4; data size, Fig. 5). It is
  implemented (`train.py --evolve`, `--frac`) but was not needed once the main table did not
  reproduce.

## Results and conclusion

See `results.md` (generated) and `analysis.md`.

* **FunCat, 10 seeds:** our Surj averages 0.232, against 0.312 reported and 0.264 for
  C-HMCNN. It is below C-HMCNN on every dataset.
* **GO and `_others`, 3 seeds:**
  * GO: 0.352, against 0.468 reported and 0.420 for C-HMCNN.
  * `_others`: below both on every dataset; diatoms collapses (0.279).
* **Metrics:** no alternative metric maps both C-HMCNN and Surj to their published numbers.
* **Verdict:** the reported results are not reproducible from the paper. The code was never
  released.
