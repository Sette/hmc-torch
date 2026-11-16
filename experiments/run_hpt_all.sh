#!/usr/bin/env bash
# HPT + R-matrix battery (experiments/run_hpt.py) on every text dataset.
#
# Per dataset: the screen (7 configurations, seed 0, <=10 epochs, validation
# only), then 3 seeds of HPT as published (bert/none/down) and 3 seeds of the
# screen's best configuration.  Finished runs are skipped, so the script can be
# restarted.  RTX 3060: ~17 min per WOS epoch; the whole battery takes days.
#
# Usage: PYTHONPATH=src bash experiments/run_hpt_all.sh [dataset ...]
set -euo pipefail
export PYTHONPATH=src
datasets=("${@:-wos aapd arxiv eurlex}")
for ds in ${datasets[@]}; do
  mkdir -p output/hpt/logs
  log="output/hpt/logs/${ds}.log"
  python experiments/run_hpt.py --dataset "$ds" --stage screen 2>&1 | tee -a "$log"
  python experiments/run_hpt.py --dataset "$ds" --stage final \
    --backbone bert --loss none --graph down 2>&1 | tee -a "$log"
  python experiments/run_hpt.py --dataset "$ds" --stage final --auto 2>&1 | tee -a "$log"
done
