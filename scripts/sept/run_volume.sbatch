#!/bin/bash
#SBATCH --job-name=sept_seg
#SBATCH --partition=amp48,amp20
#SBATCH --qos=high-memory-privileged
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%3
#SBATCH --output=/data/psyas17/SeptCTData/logs/seg_%a_%j.log
# One array task per .vol in /data/psyas17/SeptCTData/volumes.txt:
#   convert -> RootSeg-LSTM -> Radicle -> projections (both models)
set -euo pipefail
ROOT=/data/psyas17/SeptCTData
APP=/home/psyas17/root-annotator
PY=/home/psyas17/roots/.venv/bin/python
VOL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $ROOT/volumes.txt)
NAME=$(basename "$VOL" .vol)
NPY=$ROOT/volumes/$NAME.npy
cd $APP
echo "== $NAME  ($(date))"
[ -f "$NPY" ] || $PY scripts/sept/vol_to_npy.py "$VOL" "$NPY"
LSTM=$ROOT/results/lstm/${NAME}_rootseg_pred.npy
[ -f "$LSTM" ] || $PY -m app.cli --input "$NPY" --out-dir $ROOT/results/lstm
RAD=$ROOT/results/radicle/${NAME}_radicle_mask.npy
[ -f "$RAD" ] || $PY scripts/sept/radicle_lean.py "$NPY" $ROOT/results/radicle
for spec in "$LSTM lstm" "$RAD radicle"; do
  set -- $spec
  $PY scripts/sept/make_projections.py "$1" "$2" $APP/temp/visualisations $ROOT/visualisations
done
echo "== done $NAME ($(date))"
