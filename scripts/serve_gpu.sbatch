#!/bin/bash
#SBATCH --job-name=root_annotator_web
#SBATCH --partition=general
#SBATCH --account=elevated
#SBATCH --qos=core
#SBATCH --constraint=turing|ampere|ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/home/psyas17/root_annotator_web_%j.out
#
# Default is the general partition (usually idle GPUs); the constraint skips the old Pascal (1080 Ti)
# card on havok, which this PyTorch build cannot run on. For ada24 pass:
#   --partition=ada24 --account=cvl --qos=cvl     (it only starts when an ada24 GPU is free)
# Runs the Root Annotator web UI on a GPU node. Submit, then read the log for the node name and
# the ssh tunnel command:
#   sbatch scripts/serve_gpu.sbatch ; tail -f ~/root_annotator_web_<jobid>.out
set -euo pipefail
PORT="${PORT:-8765}"
cd /home/psyas17/root-annotator
echo "node: $(hostname)"
echo "on your laptop:  ssh -L ${PORT}:$(hostname):${PORT} psyas17@<cluster login host>   then open http://localhost:${PORT}"
exec /home/psyas17/roots/.venv/bin/python main.py --web --host 0.0.0.0 --port "$PORT"
