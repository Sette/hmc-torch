#!/usr/bin/env bash
# Surj reproduction battery, by stage. Usage: surj_repro/run_all.sh STAGE [STAGE...]
#   sanity   C-HMCNN through our harness (published epochs, then early stopping + refit)
#   ladder   rungs L1..L5 from C-HMCNN to Surj (8 FunCat, 3 seeds)
#   hpsearch validation-only grid for Surj (hpsearch.py)
#   final    recipe `selected` (written by hpsearch.py) on the 20 datasets, 10 seeds, both heads
#   extras   evolving ontology (FunCat), data size (GO), with `selected`
#   summary  results.md / analysis.md from the saved JSON/NPZ
# Every run skips seeds already on disk, so stages resume. P=<n> sets parallel processes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
P=${P:-3}
FUN="cellcycle_FUN derisi_FUN eisen_FUN expr_FUN gasch1_FUN gasch2_FUN seq_FUN spo_FUN"
GO="cellcycle_GO derisi_GO eisen_GO expr_GO gasch1_GO gasch2_GO seq_GO spo_GO"
OTH="enron_others diatoms_others imclef07a_others imclef07d_others"
LOGS=surj_repro/logs
mkdir -p "$LOGS"

# run RECIPE SEEDS "DATASETS" [extra train.py args...]: one process per dataset, P at a time
run() {
  local recipe=$1 seeds=$2 datasets=$3; shift 3
  printf '%s\n' $datasets | xargs -P "$P" -I{} sh -c \
    "uv run python surj_repro/train.py --recipe $recipe --seeds $seeds --datasets {} $* \
       >> $LOGS/$recipe$(printf '%s' "$*" | tr ' =' '__').{}.log 2>&1 || echo FAILED $recipe {}"
}

for stage in "$@"; do
  case $stage in
    sanity)
      run chmcnn_fixed 0,1,2 "$FUN"
      run L0_chmcnn 0,1,2 "$FUN" ;;
    ladder)
      for r in L1_onehot_mc L2_onehot L3_surj L4_surj_lr3 L4_onehot_lr3 L5_surj_mc; do
        run "$r" 0,1,2 "$FUN"
      done ;;
    hpsearch)
      uv run python surj_repro/hpsearch.py ;;
    final)
      # 10 seeds on FunCat (the paper's protocol); 3 on _others and GO, where batch 4 on
      # 10k (ImCLEF) or 4k-label (GO) problems costs ~45 min per seed.
      for r in selected selected_onehot; do run "$r" 0,1,2,3,4,5,6,7,8,9 "$FUN"; done
      for r in selected selected_onehot; do run "$r" 0,1,2 "$OTH"; done
      for r in selected selected_onehot; do P=2 run "$r" 0,1,2 "$GO"; done ;;
    extras)
      run selected 0,1,2,3,4,5,6,7,8,9 "$FUN" --evolve
      run selected_onehot 0,1,2,3,4,5,6,7,8,9 "$FUN" --evolve
      for f in 0.8 0.6 0.4 0.2; do P=1 run selected 0,1,2 "$GO" --frac "$f"; done ;;
    summary)
      uv run python surj_repro/summarize.py ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
