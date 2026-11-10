#!/usr/bin/env python3
"""Aggregate ``output/protocol/*.json`` into the paper's numbers and figures.

Reads the per-dataset records written by ``run_protocol.py`` and
``run_wos_e2e.py`` and writes, under ``--paper_dir``:

  * ``data/protocol_summary.json`` -- every number quoted in the paper;
  * ``figures/fig_decomposition.pdf`` -- train-time vs inference-time effect;
  * ``figures/fig_structure.pdf`` -- reconciliation gain vs hierarchy shape;
  * ``figures/fig_go_sparse.pdf`` -- sparse reconciliation on GO;
  * ``figures/fig_per_level.pdf`` -- F1 by hierarchy depth;
  * ``figures/fig_cd_diagram.pdf`` -- Friedman + Nemenyi on FunCat.

It also prints the table rows used in ``main.tex``.

Usage:
    python experiments/make_paper_artifacts.py --paper_dir docs/hmc-paper
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 9

FUNCAT = ["cellcycle", "church", "derisi", "eisen", "expr", "gasch1", "gasch2", "pheno", "seq", "spo"]
GO = ["cellcycle", "derisi", "eisen", "expr", "gasch1", "gasch2", "pheno", "seq", "spo"]
OTHERS = ["enron", "diatoms", "imclef07a", "imclef07d"]
TEXT = ["arxiv", "wos", "aapd", "eurlex"]

# Published AUPRC (Surj: Yang & Howe WWW'22; C-HMCNN: Giunchiglia & Lukasiewicz
# NeurIPS'20, Table 2; HMCN-F: Wehrmann et al. ICML'18, Table 4).
LIT_FUN = {
    "cellcycle": (0.269, 0.255, 0.252), "derisi": (0.231, 0.195, 0.193),
    "eisen": (0.392, 0.306, 0.298), "expr": (0.382, 0.302, 0.301),
    "gasch1": (0.369, 0.286, 0.284), "gasch2": (0.273, 0.258, 0.254),
    "seq": (0.341, 0.292, 0.291), "spo": (0.241, 0.215, 0.211),
}
LIT_GO = {
    "cellcycle": (0.460, 0.413, 0.400), "derisi": (0.445, 0.370, 0.369),
    "eisen": (0.487, 0.455, 0.440), "expr": (0.477, 0.447, 0.452),
    "gasch1": (0.481, 0.436, 0.428), "gasch2": (0.473, 0.414, 0.465),
    "seq": (0.478, 0.446, 0.447), "spo": (0.439, 0.382, 0.376),
}
# C-HMCNN and HMCN-F on the cross-domain datasets (AUPRC).
LIT_OTHERS = {
    "enron": (0.756, 0.724), "diatoms": (0.758, 0.530),
    "imclef07a": (0.956, 0.950), "imclef07d": (0.927, 0.920),
}
ARMS = [("bce", "raw"), ("bce", "rec"), ("hinge", "raw"), ("hinge", "rec"), ("mc", "raw"), ("mc", "rec")]
ARM_LABELS = {
    ("bce", "raw"): "BCE", ("bce", "rec"): "BCE + rec.",
    ("hinge", "raw"): "hinge", ("hinge", "rec"): "hinge + rec.",
    ("mc", "raw"): "MCLoss", ("mc", "rec"): "MCLoss + rec.",
}
NEMENYI_Q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def load(out_dir):
    recs = {}
    for path in glob.glob(os.path.join(out_dir, "*.json")):
        name = os.path.basename(path)[:-5]
        if name.startswith("wos_e2e_seed") or name == "structure":
            continue
        with open(path) as f:
            recs[name] = json.load(f)
    e2e = []
    for path in sorted(glob.glob(os.path.join(out_dir, "wos_e2e_seed*.json"))):
        with open(path) as f:
            e2e.append(json.load(f))
    return recs, e2e


def arm_values(rec, obj, mode, metric):
    return np.array([r["eval"][mode][metric] for r in rec["runs"] if r["objective"] == obj])


def summarize(rec):
    out = {k: rec[k] for k in ("n_train", "n_valid", "n_test", "n_features", "n_nodes",
                               "n_eval", "n_internal_eval", "n_edges", "max_depth")}
    out["arms"] = {}
    for obj, mode in ARMS:
        vals = {}
        for metric in ("f1", "auprc", "precision", "recall"):
            v = arm_values(rec, obj, mode, metric)
            if len(v):
                vals[metric] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
        if vals:
            out["arms"][f"{obj}_{mode}"] = vals
    # Paired deltas (same seed) of reconciliation for the BCE model.
    for metric in ("f1", "auprc"):
        raw, rec_ = arm_values(rec, "bce", "raw", metric), arm_values(rec, "bce", "rec", metric)
        if len(raw):
            d = rec_ - raw
            out[f"delta_rec_{metric}"] = {"mean": float(d.mean()), "std": float(d.std())}
    lv = {}
    for r in rec["runs"]:
        if r["objective"] == "bce":
            for k, v in r["eval"]["rec"]["per_level_f1"].items():
                lv.setdefault(int(k), []).append(v)
    out["per_level_f1"] = {k: float(np.mean(v)) for k, v in sorted(lv.items())}
    out["train_time_s"] = float(np.mean([r["train_time_s"] for r in rec["runs"] if r["objective"] == "bce"]))
    if "gbdt" in rec:
        g = rec["gbdt"]
        out["gbdt"] = {"f1": g["rec"]["f1"], "auprc": g["rec"]["auprc"],
                       "precision": g["rec"]["precision"], "recall": g["rec"]["recall"],
                       "raw_f1": g["raw"]["f1"], "raw_auprc": g["raw"]["auprc"],
                       "trainonly_auprc": g["trainonly_auprc"], "platt_auprc": g["platt_auprc"],
                       "train_time_s": g["train_time_s"]}
    return out


def m(s, arm, metric="f1"):
    return s["arms"][arm][metric]["mean"]


def cd_diagram(names, avg_ranks, cd, path, title):
    """Critical-difference diagram in the usual Demsar layout."""
    order = list(np.argsort(avg_ranks))
    k = len(names)
    half = (k + 1) // 2
    ranks_sorted = [avg_ranks[i] for i in order]
    # Maximal groups of methods that are not significantly different.
    groups = []
    for i in range(k):
        j = max(j for j in range(i, k) if ranks_sorted[j] - ranks_sorted[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    bar_gap, label_gap = 0.28, 0.5
    label_top = -(0.35 + bar_gap * len(groups))
    fig, ax = plt.subplots(figsize=(6.4, 1.2 + 0.3 * (len(groups) + half)))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ax.set_ylim(label_top - label_gap * half - 0.2, 1.5)
    ax.axis("off")

    def x(r):
        return hi + lo - r  # rank 1 on the right

    ax.plot([x(lo), x(hi)], [0, 0], color="k", lw=0.8)
    for r in range(lo, hi + 1):
        ax.plot([x(r), x(r)], [0, 0.12], color="k", lw=0.8)
        ax.text(x(r), 0.22, str(r), ha="center", va="bottom", fontsize=7.5)
    ax.plot([x(lo), x(lo + cd)], [1.1, 1.1], color="#c0392b", lw=1.6)
    ax.text((x(lo) + x(lo + cd)) / 2, 1.18, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=7, color="#c0392b")
    for n, (i, j) in enumerate(groups):
        y = -0.3 - bar_gap * n
        ax.plot([x(ranks_sorted[i]) + 0.04, x(ranks_sorted[j]) - 0.04], [y, y], color="#2c3e50", lw=2.2,
                solid_capstyle="butt")
    for side, idxs in (("right", order[:half]), ("left", order[half:][::-1])):
        for i, idx in enumerate(idxs):
            y = label_top - label_gap * i
            end_x = x(lo) + 0.1 if side == "right" else x(hi) - 0.1
            ax.plot([x(avg_ranks[idx]), x(avg_ranks[idx]), end_x], [0, y, y], color="k", lw=0.6, zorder=0)
            ax.text(end_x + (0.05 if side == "right" else -0.05), y, f"{names[idx]} ({avg_ranks[idx]:.2f})",
                    ha="left" if side == "right" else "right", va="center", fontsize=7.5)
    ax.set_title(title, fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="output/protocol")
    ap.add_argument("--paper_dir", default="docs/hmc-paper")
    args = ap.parse_args()
    recs, e2e = load(args.out_dir)
    fig_dir = os.path.join(args.paper_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.join(args.paper_dir, "data"), exist_ok=True)

    S = {name: summarize(rec) for name, rec in sorted(recs.items())}
    structure_path = os.path.join(args.out_dir, "structure.json")
    if os.path.exists(structure_path):
        with open(structure_path) as f:
            for name, st in json.load(f).items():
                if name in S:
                    S[name]["structure"] = st
    summary = {"datasets": S}

    # --- WOS end-to-end -----------------------------------------------------
    if e2e:
        summary["wos_e2e"] = {
            mode: {metric: {"mean": float(np.mean([r[mode][metric] for r in e2e])),
                            "std": float(np.std([r[mode][metric] for r in e2e])),
                            "values": [r[mode][metric] for r in e2e]}
                   for metric in ("f1", "auprc", "precision", "recall")}
            for mode in ("raw", "rec")}
        summary["wos_e2e"]["seeds"] = [r["seed"] for r in e2e]
        summary["wos_e2e"]["best_epochs"] = [r["best_epoch"] for r in e2e]
        summary["wos_e2e"]["train_time_s"] = float(np.mean([r["train_time_s"] for r in e2e]))

    # --- Statistics on FunCat -----------------------------------------------
    fun = [d for d in LIT_FUN if f"{d}_FUN" in S and "gbdt" in S[f"{d}_FUN"]]
    if len(fun) == len(LIT_FUN):
        methods = ["Surj", "C-HMCNN", "HMCN-F", "GBDT", "MLP+R", "MCLoss (ours)"]
        table = np.array([[*LIT_FUN[d], S[f"{d}_FUN"]["gbdt"]["auprc"],
                           m(S[f"{d}_FUN"], "bce_rec", "auprc"),
                           m(S[f"{d}_FUN"], "mc_rec", "auprc")] for d in fun])
        chi2, p = friedmanchisquare(*table.T)
        ranks = np.array([len(methods) + 1 - (np.argsort(np.argsort(row)) + 1) for row in table], float)
        avg = ranks.mean(axis=0)
        k, n = len(methods), len(fun)
        cd = NEMENYI_Q[k] * np.sqrt(k * (k + 1) / (6 * n))
        pairs = [(methods[i], methods[j], float(abs(avg[i] - avg[j])))
                 for i in range(k) for j in range(i + 1, k) if abs(avg[i] - avg[j]) >= cd]
        summary["funcat_stats"] = {"methods": methods, "chi2": float(chi2), "p": float(p),
                                   "avg_ranks": dict(zip(methods, map(float, avg))),
                                   "cd": float(cd), "significant_pairs": pairs}
        cd_diagram(methods, avg, cd, os.path.join(fig_dir, "fig_cd_diagram.pdf"),
                   f"FunCat AUPRC, {n} datasets (Friedman p = {p:.1e}, Nemenyi α = 0.05)")

    # Paired Wilcoxon: reconciliation vs raw (BCE model), per-dataset means.
    rows = [(n, s["delta_rec_f1"]["mean"], s["delta_rec_auprc"]["mean"]) for n, s in S.items()
            if "delta_rec_f1" in s]
    if len(rows) >= 6:
        df1 = np.array([r[1] for r in rows])
        dau = np.array([r[2] for r in rows])
        summary["reconciliation_wilcoxon"] = {
            "n_datasets": len(rows),
            "f1": {"median": float(np.median(df1)), "positive": int((df1 > 0).sum()),
                   "negative": int((df1 < 0).sum()),
                   "p": float(wilcoxon(df1).pvalue) if np.any(df1) else 1.0},
            "auprc": {"median": float(np.median(dau)), "positive": int((dau > 0).sum()),
                      "negative": int((dau < 0).sum()),
                      "p": float(wilcoxon(dau).pvalue) if np.any(dau) else 1.0},
        }

    with open(os.path.join(args.paper_dir, "data", "protocol_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # --- Figures ----------------------------------------------------------------
    families = [("Text", [t for t in TEXT if t in S]),
                ("FunCat", [f"{d}_FUN" for d in FUNCAT if f"{d}_FUN" in S]),
                ("GO", [f"{d}_GO" for d in GO if f"{d}_GO" in S]),
                ("Other", [f"{d}_others" for d in OTHERS if f"{d}_others" in S])]
    families = [(f, ds) for f, ds in families if ds]

    # Decomposition: delta of each arm vs plain BCE (raw), per family.
    if families:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
        arms = ARMS[1:]
        colors = ["#2471a3", "#f5b041", "#ca6f1e", "#58d68d", "#1e8449"]
        width = 0.8 / len(arms)
        for ax, metric, title in zip(axes, ("f1", "auprc"), ("Micro-F1", "AUPRC")):
            for a, (arm, c) in enumerate(zip(arms, colors)):
                for fi, (fam, ds) in enumerate(families):
                    deltas = [m(S[d], f"{arm[0]}_{arm[1]}", metric) - m(S[d], "bce_raw", metric) for d in ds]
                    x = fi + (a - (len(arms) - 1) / 2) * width
                    ax.bar(x, np.mean(deltas), width, color=c, label=ARM_LABELS[arm] if fi == 0 else None)
                    ax.scatter([x] * len(deltas), deltas, s=4, color="k", zorder=3, lw=0)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(len(families)))
            ax.set_xticklabels([f"{f}\n(n={len(d)})" for f, d in families])
            ax.set_title(f"{title}: change vs. plain BCE")
        axes[0].set_ylabel("absolute difference")
        axes[1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.savefig(os.path.join(fig_dir, "fig_decomposition.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Structure: reconciliation gain vs how much room the hierarchy leaves.
    fam_color = {"Text": "#1a5276", "FunCat": "#f39c12", "GO": "#27ae60", "Other": "#8e44ad"}

    def fam_of(n):
        return "FunCat" if n.endswith("_FUN") else "GO" if n.endswith("_GO") else \
            "Other" if n.endswith("_others") else "Text"
    from scipy.stats import spearmanr  # pylint: disable=import-outside-toplevel

    pts = {n: s_ for n, s_ in S.items() if "delta_rec_auprc" in s_ and "changed_share" in s_.get("structure", {})}
    for s_ in pts.values():
        st = s_["structure"]
        # Average raise per evaluated score: raises happen only on internal nodes.
        st["mean_raise"] = st["changed_share"] * st["mean_change"] * st["internal_label_share"]
    if pts:
        panels = [("internal_positive_share", "positive labels on internal nodes", False),
                  ("mean_raise", "average raise per evaluated score", True)]
        summary["structure_spearman"] = {}
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, (key, label, logx) in zip(axes, panels):
            xs = [pts[n]["structure"][key] for n in pts]
            ys = [pts[n]["delta_rec_auprc"]["mean"] for n in pts]
            rho, pval = spearmanr(xs, ys)
            summary["structure_spearman"][key] = {"rho": float(rho), "p": float(pval), "n": len(xs)}
            for fam, c in fam_color.items():
                sel = [(pts[n]["structure"][key], pts[n]["delta_rec_auprc"]["mean"], n)
                       for n in pts if fam_of(n) == fam]
                if sel:
                    ax.scatter([p[0] for p in sel], [p[1] for p in sel], color=c, s=16, label=fam)
                    for x_, y_, n in sel:
                        if fam in ("Text", "Other"):
                            ax.annotate(n.replace("_others", ""), (x_, y_), fontsize=6,
                                        xytext=(3, 2), textcoords="offset points")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel(label)
            ax.set_title(f"Spearman ρ = {rho:.2f} (p = {pval:.3f})", fontsize=8)
            if logx:
                ax.set_xscale("log")
        axes[0].set_ylabel("Δ AUPRC from reconciliation")
        axes[1].legend(fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_structure.pdf"), bbox_inches="tight")
        plt.close(fig)
        with open(os.path.join(args.paper_dir, "data", "protocol_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)

    # GO: reconciliation gain per dataset.
    go = [(d, S[f"{d}_GO"]) for d in GO if f"{d}_GO" in S]
    if go:
        fig, ax = plt.subplots(figsize=(6.2, 2.3))
        x = np.arange(len(go))
        for off, metric, c, lab in ((-0.2, "auprc", "#27ae60", "Δ AUPRC"), (0.2, "f1", "#7f8c8d", "Δ Micro-F1")):
            mu = [s[f"delta_rec_{metric}"]["mean"] for _, s in go]
            sd = [s[f"delta_rec_{metric}"]["std"] for _, s in go]
            ax.bar(x + off, mu, 0.4, yerr=sd, color=c, label=lab, capsize=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([d for d, _ in go], rotation=30)
        ax.set_ylabel("reconciled − raw")
        ax.legend(fontsize=7, frameon=False)
        fig.savefig(os.path.join(fig_dir, "fig_go_sparse.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Per-level F1.
    sel = [n for n in ("arxiv", "wos", "cellcycle_FUN", "cellcycle_GO") if n in S]
    if sel:
        fig, axes = plt.subplots(1, len(sel), figsize=(1.8 * len(sel), 2.2))
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, sel):
            lv = S[n]["per_level_f1"]
            ax.plot(list(lv.keys()), list(lv.values()), marker="o", ms=3, color="#1a5276")
            ax.set_title(n.replace("_", " "), fontsize=8)
            ax.set_xlabel("depth")
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Micro-F1")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_per_level.pdf"), bbox_inches="tight")
        plt.close(fig)

    # --- LaTeX tables (\input by main.tex) ---------------------------------------
    tab_dir = os.path.join(args.paper_dir, "tables")
    os.makedirs(tab_dir, exist_ok=True)

    def cell(v, sd=None, bold=False, digits=3):
        txt = f"{v:.{digits}f}"
        if sd is not None:
            txt += rf"{{\scriptsize$\pm${sd:.{digits}f}}}"
        return rf"\textbf{{{txt}}}" if bold else txt

    def write(name, lines):
        with open(os.path.join(tab_dir, name), "w") as f:
            f.write("\n".join(lines) + "\n")

    def pretty(n):
        return n.replace("_others", "").replace("_", r"\_") if n not in TEXT else \
            {"arxiv": "ArXiv", "wos": "WOS", "aapd": "AAPD", "eurlex": "EUR-Lex 57K"}[n]

    # Datasets.
    lines = []
    for fam, ds in families:
        lines.append(rf"\midrule\multicolumn{{8}}{{l}}{{\textit{{{fam}}}}} \\")
        for n in ds:
            s_ = S[n]
            lines.append(f"{pretty(n)} & {s_['n_train']:,} & {s_['n_valid']:,} & {s_['n_test']:,} & "
                         f"{s_['n_features']:,} & {s_['n_eval']:,} & {s_['n_internal_eval']:,} & "
                         f"{s_['max_depth']} \\\\")
    write("datasets.tex", lines)

    # Decomposition by family (mean over datasets of the seed means).
    for metric, fname in (("auprc", "decomp_auprc.tex"), ("f1", "decomp_f1.tex")):
        lines = []
        for fam, ds in families:
            vals = [np.mean([m(S[d], f"{o}_{md}", metric) for d in ds]) for o, md in ARMS]
            best = int(np.argmax(vals))
            lines.append(f"{fam} ({len(ds)}) & " + " & ".join(
                cell(v, bold=i == best) for i, v in enumerate(vals)) + r" \\")
        write(fname, lines)

    # Per-dataset decomposition (appendix).
    lines = []
    for fam, ds in families:
        lines.append(rf"\midrule\multicolumn{{7}}{{l}}{{\textit{{{fam}}}}} \\")
        for n in ds:
            vals = [m(S[n], f"{o}_{md}", "auprc") for o, md in ARMS]
            best = int(np.argmax(vals))
            lines.append(f"{pretty(n)} & " + " & ".join(
                cell(v, bold=i == best) for i, v in enumerate(vals)) + r" \\")
    write("decomp_per_dataset.tex", lines)

    # FunCat / GO against published results.
    for fam_key, lit, suffix, fname, with_gbdt in (
        ("FUN", LIT_FUN, "_FUN", "funcat.tex", True), ("GO", LIT_GO, "_GO", "go.tex", False)):
        lines, cols = [], []
        names = FUNCAT if fam_key == "FUN" else GO
        for d in names:
            n = d + suffix
            if n not in S:
                continue
            ours = [S[n]["gbdt"]["auprc"]] if with_gbdt else []
            ours += [m(S[n], "bce_rec", "auprc"), m(S[n], "mc_rec", "auprc")]
            row = list(lit.get(d, (None, None, None))) + ours
            if d in lit:
                cols.append(row)
            best = max(v for v in row if v is not None)
            lines.append(f"{d} & " + " & ".join("--" if v is None else cell(v, bold=v == best)
                                                  for v in row) + r" \\")
        if cols:
            avg = np.mean(np.array(cols, dtype=float), axis=0)
            lines.append(r"\midrule")
            lines.append(rf"\textit{{Avg ({len(cols)} common)}} & " + " & ".join(
                cell(v, bold=v == avg.max()) for v in avg) + r" \\")
        write(fname, lines)

    # Cross-domain against published results.
    lines = []
    for d in OTHERS:
        n = f"{d}_others"
        if n not in S:
            continue
        a = S[n]["arms"]
        row = [*LIT_OTHERS[d], a["bce_rec"]["auprc"]["mean"], a["mc_rec"]["auprc"]["mean"]]
        best = max(row)
        lines.append(f"{d} & {S[n]['n_eval']} & " + " & ".join(cell(v, bold=v == best) for v in row)
                     + f" & {cell(a['bce_rec']['f1']['mean'])} \\\\")
    write("others.tex", lines)

    # Text results (MLP on frozen SPECTER2).
    lines = []
    for n in TEXT:
        if n not in S:
            continue
        a = S[n]["arms"]
        lines.append(f"{pretty(n)} & {S[n]['n_internal_eval']}/{S[n]['n_eval']} & "
                     f"{cell(a['bce_raw']['f1']['mean'], a['bce_raw']['f1']['std'])} & "
                     f"{cell(a['bce_rec']['f1']['mean'], a['bce_rec']['f1']['std'])} & "
                     f"{cell(a['mc_rec']['f1']['mean'], a['mc_rec']['f1']['std'])} & "
                     f"{cell(a['bce_rec']['auprc']['mean'], a['bce_rec']['auprc']['std'])} \\\\")
    write("text.tex", lines)

    # GBDT vs MLP on FunCat (precision/recall) and calibration.
    fun_names = [f"{d}_FUN" for d in FUNCAT if f"{d}_FUN" in S and "gbdt" in S[f"{d}_FUN"]]
    if fun_names:
        g = {k: np.mean([S[n]["gbdt"][k] for n in fun_names])
             for k in ("precision", "recall", "f1", "auprc", "trainonly_auprc", "platt_auprc")}
        mlp = {k: np.mean([m(S[n], "bce_rec", k) for n in fun_names])
               for k in ("precision", "recall", "f1", "auprc")}
        mc = {k: np.mean([m(S[n], "mc_rec", k) for n in fun_names])
              for k in ("precision", "recall", "f1", "auprc")}
        write("pr.tex", [
            f"MLP, BCE + rec. & {cell(mlp['precision'])} & {cell(mlp['recall'])} & {cell(mlp['f1'])} & {cell(mlp['auprc'])} \\\\",
            f"MLP, MCLoss + rec. & {cell(mc['precision'])} & {cell(mc['recall'])} & {cell(mc['f1'])} & {cell(mc['auprc'])} \\\\",
            f"GBDT + rec. & {cell(g['precision'])} & {cell(g['recall'])} & {cell(g['f1'])} & {cell(g['auprc'])} \\\\",
        ])
        summary["funcat_means"] = {"gbdt": g, "mlp_bce_rec": mlp, "mlp_mc_rec": mc,
                                   "n_datasets": len(fun_names)}
        with open(os.path.join(args.paper_dir, "data", "protocol_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)

    # Label-knowledge heads (run_label_embeddings.py).
    le_dir = os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "label_embeddings")
    le = {}
    for path in sorted(glob.glob(os.path.join(le_dir, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        heads = {}
        for r in rec["runs"]:
            heads.setdefault(r["head"], []).append(r["eval"]["rec"]["auprc"])
        le[rec["dataset"]] = {h: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                              for h, v in heads.items()}
    if le:
        summary["label_embeddings"] = le
        lines = []
        head_order = ["free", "graph", "text", "text+graph"]
        for fam, suffix, names_ in (("FunCat", "_FUN", FUNCAT), ("GO", "_GO", GO)):
            rows = [(d, le[d + suffix]) for d in names_ if d + suffix in le]
            if not rows:
                continue
            lines.append(rf"\midrule\multicolumn{{5}}{{l}}{{\textit{{{fam}}}}} \\")
            for d, h in rows:
                vals = [h[k]["mean"] if k in h else None for k in head_order]
                best = max(v for v in vals if v is not None)
                lines.append(f"{d} & " + " & ".join(
                    "--" if v is None else cell(v, bold=abs(v - best) < 5e-4) for v in vals) + r" \\")
            avg = [np.mean([h[k]["mean"] for _, h in rows]) if all(k in h for _, h in rows) else None
                   for k in head_order]
            lines.append(rf"\textit{{Avg}} & " + " & ".join("--" if v is None else cell(v) for v in avg) + r" \\")
        write("label_embeddings.tex", lines)
        with open(os.path.join(args.paper_dir, "data", "protocol_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)

    # --- Console tables ---------------------------------------------------------
    def fmt(s, arm, metric):
        a = s["arms"][arm][metric]
        return f"{a['mean']:.4f}±{a['std']:.4f}"

    print("\n# Decomposition (Micro-F1 / AUPRC), mean ± std over seeds")
    for n, s in S.items():
        cells = "  ".join(f"{a[0]}_{a[1]}={fmt(s, f'{a[0]}_{a[1]}', 'f1')}/{fmt(s, f'{a[0]}_{a[1]}', 'auprc')}"
                          for a in ARMS if f"{a[0]}_{a[1]}" in s["arms"])
        print(f"{n:<18} int/eval={s['n_internal_eval']}/{s['n_eval']}  {cells}")
    for n, s in S.items():
        if "gbdt" in s:
            g = s["gbdt"]
            print(f"GBDT {n:<16} F1={g['f1']:.4f} AUPRC={g['auprc']:.4f} P={g['precision']:.3f} "
                  f"R={g['recall']:.3f} trainonly={g['trainonly_auprc']:.4f} platt={g['platt_auprc']:.4f}")
    for key in ("funcat_stats", "reconciliation_wilcoxon", "wos_e2e"):
        if key in summary:
            print(f"\n# {key}\n{json.dumps(summary[key], indent=1)}")


if __name__ == "__main__":
    main()
