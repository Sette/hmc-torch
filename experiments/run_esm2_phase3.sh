#!/usr/bin/env bash
# Spec 30: every other project method on frozen ESM-2 features (seq_FUN, seq_GO), in series.
# Each script skips outputs that already exist, so the queue can be restarted.
# Usage: PYTHONPATH=src experiments/run_esm2_phase3.sh [pid-to-wait-for]
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
DS=seq_FUN,seq_GO
OUT=output/protein_lm_methods
RUN="uv run python experiments/run_with_esm2.py"

if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

step() { echo; echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }

# 1. MLP global with the hinge objective (BCE and MCLoss already in output/protein_lm)
step uv run python experiments/run_protein_lm.py --datasets $DS --variants esm \
  --objectives hinge --out_dir $OUT/hinge
# 2. H-MLCL + H-KNN with SR-GAE and random prototypes (MCLoss trunk, then BCE trunk)
step $RUN experiments/run_sr_gae_sota.py --datasets $DS --out_dir $OUT/sr_gae_paired_mc
HMC_SOTA_OBJECTIVE=bce step $RUN experiments/run_sr_gae_sota.py --datasets $DS --out_dir $OUT/sr_gae_paired_bce
# 3. DepthCal on the blend
step $RUN experiments/run_sr_gae_depth.py --datasets $DS --seeds 0,42,123 --out_dir $OUT/sr_gae_depth
# 4. label-embedding heads (free, graph, GAE, GAE+free; text heads on GO)
for obj in bce mc; do
  step $RUN experiments/run_label_embeddings.py --datasets $DS --objective $obj --out_dir $OUT/label_embeddings
done
# 5. autoencoders: one variant per family
for v in gae_hybrid@64 feat_dae label_ae@1; do
  step $RUN experiments/run_autoencoder.py --stage final --variant "$v" --datasets $DS --out_dir $OUT/autoencoder
done
echo; echo "=== $(date '+%F %T') PHASE3_DONE"
