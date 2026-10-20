#!/usr/bin/env python3
"""Tables of ``surj_repro/results/`` -> ``surj_repro/results.md``.

Reads only the saved per-seed JSON (and, for the metric hypotheses, the NPZ scores), so the
tables can be regenerated anywhere without retraining.

Usage (from the repository root):  PYTHONPATH=src python surj_repro/summarize.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from data import CHMCNN_PUBLISHED  # noqa: E402

RESULTS = os.path.join(HERE, "results")
PARITY = os.path.join(ROOT, "output", "chmcnn_parity", "results.json")

# Surj, Table 2 (AU(PRC), mean of 10 trials).
SURJ = {
    "cellcycle_FUN": 0.269, "derisi_FUN": 0.231, "eisen_FUN": 0.392, "expr_FUN": 0.382,
    "gasch1_FUN": 0.369, "gasch2_FUN": 0.273, "seq_FUN": 0.341, "spo_FUN": 0.241,
    "cellcycle_GO": 0.460, "derisi_GO": 0.445, "eisen_GO": 0.487, "expr_GO": 0.477,
    "gasch1_GO": 0.481, "gasch2_GO": 0.473, "seq_GO": 0.478, "spo_GO": 0.439,
    "enron_others": 0.743, "diatoms_others": 0.772,
    "imclef07a_others": 0.943, "imclef07d_others": 0.917,
}
# Table 5, "w/o ontology learning".
SURJ_ONEHOT = {
    "cellcycle_FUN": 0.211, "derisi_FUN": 0.209, "eisen_FUN": 0.281, "expr_FUN": 0.223,
    "gasch1_FUN": 0.256, "gasch2_FUN": 0.208, "seq_FUN": 0.218, "spo_FUN": 0.209,
}
# Table 4: V1 data + V2 ontology (Surj, baseline).
SURJ_V2 = {
    "cellcycle_FUN": (0.213, 0.137), "derisi_FUN": (0.144, 0.114), "eisen_FUN": (0.277, 0.179),
    "expr_FUN": (0.288, 0.149), "gasch1_FUN": (0.271, 0.175), "gasch2_FUN": (0.213, 0.118),
    "seq_FUN": (0.242, 0.146), "spo_FUN": (0.207, 0.129),
}
# Table 3: Surj total training time (s), 32 CPU cores.
SURJ_TIME = {
    "cellcycle_FUN": 46.1, "derisi_FUN": 44.4, "eisen_FUN": 24.7, "expr_FUN": 56.1,
    "gasch1_FUN": 57.1, "gasch2_FUN": 59.2, "seq_FUN": 61.5, "spo_FUN": 44.6,
}
ORDER = list(SURJ)
LADDER = ["L0_chmcnn", "L1_onehot_mc", "L2_onehot", "L3_surj", "L4_onehot_lr3", "L4_surj_lr3",
          "L5_surj_mc"]
METRICS = ["micro_ap", "micro_ap_with_roots", "micro_trapezoid", "weighted_ap", "macro_ap",
           "samples_ap"]


def load_run(run: str) -> dict:
    """{dataset: [seed records]} of one results folder."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, run, "*", "seed*.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        out.setdefault(rec["dataset"], []).append(rec)
    return out


def values(recs, key="test_ap", setting="full"):
    vals = []
    for r in recs or []:
        res = r["settings"].get(setting, {})
        v = res.get("metrics", {}).get(key) if key in METRICS else res.get(key)
        if v is not None:
            vals.append(v)
    return vals


def fmt(vals, std=True):
    if not vals:
        return "—"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if std and len(vals) > 1 \
        else f"{np.mean(vals):.3f}"


def ordered(names):
    return sorted(names, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER))


def sanity():
    parity = json.load(open(PARITY, encoding="utf-8")) if os.path.exists(PARITY) else {}
    fixed, l0 = load_run("chmcnn_fixed"), load_run("L0_chmcnn")
    if not fixed and not l0:
        return []
    lines = ["## 1. Harness sanity: C-HMCNN through our code", "",
             "| dataset | C-HMCNN published | parity run (their loop) | ours, published epochs |"
             " ours, early stopping + refit | epochs chosen |", "|---|---|---|---|---|---|"]
    for ds in ordered(set(fixed) | set(l0)):
        par = [v["theirs"] for v in parity.get(ds, {}).values()]
        lines.append(f"| {ds} | {CHMCNN_PUBLISHED.get(ds, '—')} | {fmt(par)} | "
                     f"{fmt(values(fixed.get(ds)))} | {fmt(values(l0.get(ds)))} | "
                     f"{fmt(values(l0.get(ds), 'epochs'), False)} |")
    return lines + [""]


def ladder():
    runs = {r: load_run(r) for r in LADDER}
    runs = {r: v for r, v in runs.items() if v}
    if not runs:
        return []
    names = ordered({ds for v in runs.values() for ds in v})
    lines = ["## 2. Ladder from C-HMCNN to Surj (test micro AP, mean of 3 seeds)", "",
             "| dataset | Surj paper | onehot paper | " + " | ".join(runs) + " |",
             "|---|---|---|" + "---|" * len(runs)]
    for ds in names:
        lines.append(f"| {ds} | {SURJ.get(ds, '—')} | {SURJ_ONEHOT.get(ds, '—')} | " + " | ".join(
            fmt(values(v.get(ds)), False) for v in runs.values()) + " |")
    lines.append("| **mean** | {:.3f} | {:.3f} | ".format(
        np.mean([SURJ[d] for d in names]), np.mean([SURJ_ONEHOT.get(d, np.nan) for d in names]))
        + " | ".join(f"{np.mean([np.mean(values(v[d])) for d in names if d in v]):.3f}"
                     for v in runs.values()) + " |")
    lines.append("| GHV (mean) | 0 | — | " + " | ".join(
        f"{np.mean([x['violations'] for d in v for x in values(v[d], 'ghv')]):.0f}"
        for v in runs.values()) + " |")
    lines.append("| epochs (mean) | | | " + " | ".join(
        f"{np.mean([x for d in v for x in values(v[d], 'epochs')]):.0f}"
        for v in runs.values()) + " |")
    return lines + [""]


def hpsearch():
    path = os.path.join(RESULTS, "hpsearch", "selection.json")
    if not os.path.exists(path):
        return []
    sel = json.load(open(path, encoding="utf-8"))
    lines = ["## 3. Hyperparameter search (FunCat validation AP, seed 0; test never scored)", "",
             f"Base recipe `{sel['base']}`.", "", "| stage | config | mean val AP |", "|---|---|---|"]
    for e in sel["log"]:
        lines.append(f"| {e['stage']} | `{e['config']}` | {e['mean_val_ap']:.4f} |")
    return lines + ["", f"Selected: `{sel['best']}`", ""]


def final():
    s, o = load_run("selected"), load_run("selected_onehot")
    if not s:
        return []
    lines = ["## 4. Final: `selected` recipe, 10 seeds (Tables 2, 5 and 3)", "",
             "| dataset | Surj paper | ours surj | onehot paper | ours onehot | C-HMCNN | "
             "GHV surj | time surj (s) | time paper (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for ds in ordered(set(s) | set(o)):
        ghv = [x["violations"] for x in values(s.get(ds), "ghv")]
        t = [r["settings"]["full"].get("time_select_s", 0) + r["settings"]["full"]["time_train_s"]
             + r["gae_time_s"].get("full", 0) for r in s.get(ds, [])]
        lines.append(
            f"| {ds} | {SURJ.get(ds, '—')} | {fmt(values(s.get(ds)))} | "
            f"{SURJ_ONEHOT.get(ds, '—')} | {fmt(values(o.get(ds)))} | "
            f"{CHMCNN_PUBLISHED.get(ds, '—')} | {np.mean(ghv) if ghv else '—'} | "
            f"{np.mean(t) if t else '—':.0f} | {SURJ_TIME.get(ds, '—')} |")
    return lines + [""]


def metric_hypotheses():
    """Each metric, for C-HMCNN (L0) and Surj (selected, else L4), against the published."""
    ref = load_run("L0_chmcnn") or load_run("chmcnn_fixed")
    surj_name = "selected" if load_run("selected") else "L4_surj_lr3"
    surj = load_run(surj_name)
    if not ref or not surj:
        return []
    lines = ["## 5. Metric hypotheses (diagnostic): ratio ours / published", "",
             "A metric explains the paper only if it maps C-HMCNN to ~1.00 **and** our Surj to "
             "~1.00 on every dataset (including `_others`, where Surj claims no gain).", "",
             "| metric | C-HMCNN ratio (mean, range) | Surj (`" + surj_name +
             "`) ratio (mean, range) |", "|---|---|---|"]
    for m in METRICS:
        cells = []
        for runs, pub in ((ref, CHMCNN_PUBLISHED), (surj, SURJ)):
            r = [np.mean(values(runs[d], m)) / pub[d] for d in runs if d in pub and values(runs[d], m)]
            cells.append(f"{np.mean(r):.2f} ({min(r):.2f}–{max(r):.2f})" if r else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines + [""]


def evolving():
    runs = {h: load_run(f"{h}__evolve") for h in ("selected", "selected_onehot")}
    if not runs["selected"]:
        return []
    lines = ["## 6. Evolving ontology (Table 4)", "",
             "| dataset | V1 surj | V2 surj | Δ | V1 onehot | V2 onehot | Δ | paper surj | "
             "paper baseline |", "|---|---|---|---|---|---|---|---|---|"]
    for ds in ordered(runs["selected"]):
        row = [ds]
        for h in runs.values():
            v1, v2 = values(h.get(ds), setting="V1"), values(h.get(ds), setting="V2")
            d = f"{100 * (np.mean(v2) / np.mean(v1) - 1):+.1f}%" if v1 and v2 else "—"
            row += [fmt(v1, False), fmt(v2, False), d]
        p = SURJ_V2.get(ds)
        row += [f"{SURJ[ds]} → {p[0]}", f"{SURJ_ONEHOT[ds]} → {p[1]}"] if p else ["—", "—"]
        lines.append("| " + " | ".join(row) + " |")
    return lines + [""]


def datasize():
    fracs = ["0.2", "0.4", "0.6", "0.8"]
    runs = {f: load_run(f"selected__frac-{f}") for f in fracs}
    full = load_run("selected")
    if not any(runs.values()):
        return []
    lines = ["## 7. Data size on GO (Fig. 5)", "",
             "| dataset | " + " | ".join(f"{f}" for f in fracs) + " | 1.0 | C-HMCNN (full) |",
             "|---|" + "---|" * (len(fracs) + 2)]
    for ds in ordered({d for v in runs.values() for d in v}):
        lines.append(f"| {ds} | " + " | ".join(fmt(values(runs[f].get(ds)), False) for f in fracs)
                     + f" | {fmt(values(full.get(ds)), False)} | {CHMCNN_PUBLISHED.get(ds, '—')} |")
    return lines + [""]


def main():
    lines = ["# Surj reproduction: results", "",
             "Generated by `surj_repro/summarize.py` from `surj_repro/results/`; setup in "
             "`README.md`. Test micro AP over the evaluated labels (C-HMCNN's metric).", ""]
    for part in (sanity, ladder, hpsearch, final, metric_hypotheses, evolving, datasize):
        lines += part()
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
