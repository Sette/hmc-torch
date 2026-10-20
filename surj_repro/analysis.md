# Verdict: Surj's reported numbers do not reproduce

All numbers below come from `results/` and are tabulated in `results.md` (`summarize.py`).

## 1. The harness is sound

* **C-HMCNN reproduces.** Its network, trained through this harness on C-HMCNN's own parser
  (imported unmodified) with its published epochs, gives its published FunCat numbers within
  0.003 AUPRC: cellcycle 0.256 vs 0.255, derisi 0.196 vs 0.195, eisen 0.309 vs 0.306.
* **The metric is the same.** Micro AP with the roots removed maps C-HMCNN to 1.00–1.01× its
  published values.
* **Runs are deterministic.** A rerun on the same machine is bit-identical
  (`tests/test_repro.py`).

## 2. Surj, as described, stays below C-HMCNN

* **Recipe.** Every hyperparameter the paper leaves open was chosen on FunCat validation
  (`hpsearch.py`; 25 configurations, test never scored), as the paper says it did. The search
  selected lr 1e-3, batch 4, dropout 0.7, hidden 512, D 16, and C-HMCNN's MCM + MCLoss.
  Without MCM, the model makes 10⁴–10⁵ hierarchy violations per dataset, while Surj reports
  zero.
* **The search space is flat.** Validation AP over the 25 configurations spans only
  0.187–0.217.
* **FunCat test AUPRC, 10 seeds:**

  | | Surj reported | Surj ours | One-hot reported | One-hot ours | C-HMCNN |
  |---|---|---|---|---|---|
  | **Average over 8 datasets** | 0.312 | 0.232 | 0.227 | 0.156 | 0.264 |

  Our Surj is below C-HMCNN on every dataset; per-dataset numbers are in `results.md` §4.
* **The ontology embeddings do help.** They add +0.076 over the one-hot head, close to the
  +0.085 that Surj reports. But the cosine head starts so far below C-HMCNN's plain output
  layer that the embeddings only make up part of the difference. The one-hot head sits at
  ≈0.155 on all datasets, i.e. close to label frequency.

**GO and `_others`, same recipe, 3 seeds:**

* **GO:** our Surj averages 0.352 (0.308–0.396), against 0.468 reported and 0.420 for C-HMCNN.
  The one-hot head sits at ≈0.34 throughout.
* **`_others`:**

  | Dataset | Ours | Surj reported | C-HMCNN |
  |---|---|---|---|
  | enron | 0.713 | 0.743 | 0.756 |
  | imclef07a | 0.914 | 0.943 | 0.956 |
  | imclef07d | 0.904 | 0.917 | 0.927 |
  | diatoms | 0.279 | 0.772 | 0.758 |

* **Diatoms collapses.** The FunCat-selected recipe (batch 4, dropout 0.7) does not transfer to
  its 399 labels. The paper says hyperparameters were kept identical across datasets.

## 3. No evaluation slip explains the reported numbers

Ratios of our Surj (FunCat, 10 seeds) and of C-HMCNN to their published numbers, recomputed
from the saved test scores:

| Metric | Surj ours / reported | C-HMCNN ours / published |
|---|---|---|
| micro AP, roots removed (C-HMCNN's) | 0.75 (0.67–0.84) | 1.01 |
| micro AP, roots kept | 1.12 (0.92–1.36) | 1.45 |
| per-sample AP | 1.09 (0.94–1.21) | 1.40 |
| trapezoidal PR area | 0.75 | 1.00 |
| weighted AP | 0.72 | 0.92 |

* Keeping the roots, or averaging per sample, brings our Surj close to its reported average.
  The per-dataset ratios then scatter from 0.92 to 1.36, which does not match the pattern the
  paper reports.
* The same change inflates C-HMCNN by about 1.4×. If Surj had been evaluated that way and
  C-HMCNN had not, its Table 2 would compare two different metrics.
* No single metric maps both models to their published figures.

## 4. Inconsistencies inside the Surj paper

* **Table 4 repeats Table 2.** Its "V1" column, for an ontology with 20% of the leaves removed,
  repeats the full-ontology numbers of Table 2 exactly.
* **Table 4 repeats Table 5.** Its baseline column repeats the one-hot numbers of Table 5.
* **The one-hot baseline is weak.** Surj's own one-hot baseline (0.227 average) sits well below
  C-HMCNN (0.264). Yet the two share their preprocessing, their evaluation and, per the paper,
  their constraint.

## 5. Conclusion

* **Why the numbers cannot be verified.** The code was never released. A reimplementation that
  follows the paper, starts from the reference pipeline it names (and reproduces that pipeline's
  published numbers), and picks the open hyperparameters on validation does not approach the
  reported results.
* **How we report them.** The hmc-paper lists Surj's numbers as reported but unverifiable, in a
  separate column, outside all rankings and statistical tests, and points to this folder.
