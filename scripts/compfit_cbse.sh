#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N compfit
#$ -l h_rt=4:00:00
#$ -l tmem=24G
#$ -l h_vmem=24G
#$ -l h=!hoots-207-1*
#$ -t 1-3
set -o pipefail

# OPTION-B STAGE 1: fit the typed COMPOSITION matrix per asset (CPU only).
#
# Fits Phi[target group, source group, lag cell] by regressing per-group binned
# counts on lagged per-group counts (8 responses, same lin-log design matrix as
# the shipped kirchner_fit_typed, which used only ONE response = total counts
# and so could never recover source->target structure).
#
# Unlike the Option-3 ground fit this deliberately keeps the UNCONSTRAINED (OLS)
# solution with its inhibitory mass: Option B uses Phi only for the group
# COMPOSITION p_g = [lambda_g]^+ / sum_g' [lambda_g']^+, which needs lambda_g
# non-negative pointwise (a floor at sampling time) and never needs the
# branching representation.  So the antisymmetric MO structure
#   MO_b -> LO_b +0.678,  MO_a -> LO_b -0.423
# survives here, where NNLS would have flipped its sign.
#
# The rate stays kf2's blind ground, so Fano / pin / transplant are untouched
# by construction.
REPO="${REPO:-$HOME/simulation}"
COINS=(sol eth btc); COIN=${COINS[$((SGE_TASK_ID-1))]}
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
OUT="$REPO/experiments/compfit"

hostname; date
cd "$REPO" || exit 1
[ -d "$DATA" ] || { echo "SAN_NOT_VISIBLE $DATA"; exit 1; }
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO:$REPO/scripts" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
mkdir -p "$OUT"

echo "START $(date) COIN=$COIN"
python3 -u scripts/typed_matrix_probe.py \
  --data-dir "$DATA" --label "comp_${COIN}" --fit-only \
  --out "$OUT/comp_${COIN}.json"
RC=$?

if [ "$RC" -eq 0 ] && [ -s "$OUT/comp_${COIN}.json" ]; then
  echo "DONE $(date) STATUS=0 COIN=$COIN OUT=$OUT/comp_${COIN}.json"
else
  echo "DONE $(date) STATUS=1 COIN=$COIN rc=$RC"
  exit 1
fi
