#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N lgm_rk
#$ -l h_rt=12:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-9
set -o pipefail

# STAGE A of the c(x)Q architecture: typed kicks with REGRESSION-identified
# magnitudes.
#
# M[g,h] factors exactly as  M = c_h * Q[g,h]  with c_h = sum_g M[g,h] the net
# offspring (MAGNITUDE) and Q[.,h] the offspring type mix (TRANSITION).  This
# arm deploys the c half only: w <- c, into the ground's existing kick buffer.
# The Q half (a logit residual on the mark head) is Stage B and needs new
# decoder code; running A alone first gives the ablation that separates
# magnitude impact from transition impact.
#
# vs doc 4.15: identical machinery, but w comes from the JOINT REGRESSION
# (controls for all groups' lagged counts) rather than an unconditional
# covariance proxy.  MO drops from 122-458x to 1.0x (SOL) / 2.4x (ETH) /
# 3.2-6.3x (BTC), and E[w^2]/E[w]^2 falls from ~20 to 1.008/1.014/1.032.
#
# PRE-REGISTERED PREDICTIONS (this is the test):
#   1. Fano moves < 5% vs kf2 at every scale       (n unchanged; only E[w^2] shifts)
#   2. kappa stays ~1.0-1.05                       (4.15 needed ~1.3)
#   3. calibration succeeds on all 9 arms          (4.15: channel kicks failed everywhere)
# Failure of (1) or (2) falsifies the confounding explanation of 4.15.
REPO="${REPO:-$HOME/simulation}"
COINS=(btc eth sol); COIN=${COINS[$(( (SGE_TASK_ID-1)/3 ))]}
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/ma_cbse/$COIN"
SEED=$(( (SGE_TASK_ID-1)%3 + 1 ))
TAG="lgm-rk-s${SEED}"
DONOR="$ROOT/lgm-w4k48-s${SEED}/train/best_model.pt"
SAMPLER=inversion
SF_CAL="--calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 --calibrate-final-tol 0.15"

# shipped per-asset branching, unchanged -- only the kick table differs
case "$COIN" in
  sol) NTARGET=0.99   ;;
  btc) NTARGET=0.9925 ;;
  eth) NTARGET=0.995  ;;
esac

hostname; date
cd "$REPO" || exit 1
[ -d "$DATA" ] || { echo "SAN_NOT_VISIBLE $DATA"; exit 1; }
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4

B="$ROOT/$TAG"; mkdir -p "$B/train"
CKPT="$B/train/best_model.pt"
GROUND="$REPO/kirchner_ground_${COIN}.json"
TYPED="$REPO/regkicks_${COIN}.json"
for f in "$DONOR" "$GROUND" "$TYPED"; do
  [ -s "$f" ] || { echo "missing $f"; exit 1; }
done
rm -rf "$B"/sf_r*
ML="$B/master.log"; : > "$ML"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) COIN=$COIN TAG=$TAG n=$NTARGET host=$(hostname)"

log "ASSEMBLE $(date)"
python3 -u assemble_typed_ckpt.py --donor "$DONOR" --ground "$GROUND" \
  --typed "$TYPED" --variant group --n "$NTARGET" --out "$CKPT" 2>&1 | tee -a "$ML"
{ [ "${PIPESTATUS[0]}" -eq 0 ] && [ -s "$CKPT" ]; } || fail assemble 1

log "RHO $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$ML"
import sys, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); cfg = ck["config"]
m = create_volume_set_mtpp(cfg.get("num_channels", 62), cfg, torch.device("cpu"), use_volume=cfg.get("use_volume", False))
m.load_state_dict(ck["model_state_dict"])
d = m.decoder
import torch.nn.functional as F
ew = float((d.p_bar * F.softplus(d.kick_raw)).sum()) if getattr(d, "use_kicks", False) else 1.0
print("RHO n=%.4f pinned_rate=%.4f E[w]=%.4f betas=%s" % (
      d.closed_form_rho(), float(d.target_rate), ew,
      [round(float(b), 4) for b in d._betas()]))
PY

log "GENUINE-STREAMING $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 64 --device cuda --label "$TAG" \
  --streaming --dt-horizon 60 --dt-grid-points 32 --output "$B/genuine_${TAG}.json" 2>&1 | tail -20 | tee -a "$ML"
[ -s "$B/genuine_${TAG}.json" ] || fail genuine 1

for R in 1 2 3; do
  log "SF $(date) rollout_seed=$R"
  mkdir -p "$B/sf_r$R"
  python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
    --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/sf_r$R" --device cuda --sampler "$SAMPLER" \
    --context-mode carried $SF_CAL --match-durations \
    --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
    --rollout-seed "$R" --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf_r$R.log" 2>&1
  SF_RC=$?
  grep -E "CONTEXT_MODE|CALIBRAT" "$B/sf_r$R.log" | tee -a "$ML"
  { [ "$SF_RC" -eq 0 ] && [ -s "$B/sf_r$R/stylized_facts_${TAG}.json" ]; } \
    || { tail -25 "$B/sf_r$R.log" | tee -a "$ML"; fail "sf_r$R" "$SF_RC"; }
done

# composition check on the SAME checkpoint -- does typed excitation move the
# mark marginals that markcal found broken (MO at ~40% of real)?
log "MARKCAL $(date)"
python3 -u scripts/mark_calibration_probe.py --checkpoint "$CKPT" --data-dir "$DATA" \
  --max-files "$MAXFILES" --cache-dir "$CACHE" --typed-json "$TYPED" --label "$TAG" \
  --output "$B/markcal_${TAG}.json" --seq-length "$SEQ" --stride "$STRIDE" \
  --batch-size 64 --device cuda --tf-batches 200 --max-real-windows 4096 \
  --rollout-duration 600 --rollout-sequences 32 --rollout-seed 1 2>&1 | tail -8 | tee -a "$ML"

log "DONE $(date) STATUS=0 COIN=$COIN TAG=$TAG BASE=$B"
