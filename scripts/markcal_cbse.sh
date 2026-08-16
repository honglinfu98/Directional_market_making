#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N markcal
#$ -l h_rt=4:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-9
set -o pipefail

# MARK-CALIBRATION DIAGNOSTIC (markcal): eval-only, no training.
#
# Decomposes the E_p_model[w] ~= 0.77 deficit from the typed-kick probe
# (doc 4.15) into head miscalibration (teacher-forced, real states) vs
# free-rollout drift (model states), on the SHIPPED lgm-kf2 mark checkpoints.
# The typed assembly pins E[w]=1 under the EMPIRICAL mark law but the ground
# fires under the MODEL's law; if those disagree the ground under-fires at
# baseline, which is the kappa ~= 1.3 every typed arm needed.
#
# No kappa calibration here on purpose: the mark head is a rate-neutral
# softmax, and kf2's own kappa is 1.014-1.019, so the uncalibrated rollout is
# the right measurement for mark PROPORTIONS.
REPO="${REPO:-$HOME/simulation}"
COINS=(btc eth sol); COIN=${COINS[$(( (SGE_TASK_ID-1)/3 ))]}
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/ma_cbse/$COIN"
SEED=$(( (SGE_TASK_ID-1)%3 + 1 ))
TAG="lgm-kf2-s${SEED}"

hostname; date
cd "$REPO" || exit 1
[ -d "$DATA" ] || { echo "SAN_NOT_VISIBLE $DATA"; exit 1; }
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4

B="$ROOT/$TAG"
CKPT="$B/train/best_model.pt"
TYPED="$REPO/kirchner_typed_${COIN}.json"
OUT="$B/markcal_${TAG}.json"
[ -s "$CKPT" ]  || { echo "missing checkpoint $CKPT"; exit 1; }
[ -s "$TYPED" ] || { echo "missing typed fit $TYPED"; exit 1; }

echo "START $(date) COIN=$COIN TAG=$TAG"
python3 -u scripts/mark_calibration_probe.py \
  --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" \
  --cache-dir "$CACHE" --typed-json "$TYPED" --label "$TAG" --output "$OUT" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 64 --device cuda \
  --tf-batches 200 --max-real-windows 4096 \
  --rollout-duration 600 --rollout-sequences 32 --rollout-seed 1
RC=$?

# verify by ARTIFACT, not rc (house rule 8)
if [ "$RC" -eq 0 ] && [ -s "$OUT" ]; then
  echo "DONE $(date) STATUS=0 COIN=$COIN TAG=$TAG OUT=$OUT"
else
  echo "DONE $(date) STATUS=1 COIN=$COIN TAG=$TAG rc=$RC"
  exit 1
fi
