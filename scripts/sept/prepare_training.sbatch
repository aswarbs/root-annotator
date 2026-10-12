#!/bin/bash
#SBATCH --job-name=sept_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/data/psyas17/SeptCTData/train/logs/prep_%a_%j.log
set -euo pipefail
cd /home/psyas17/root-annotator
/home/psyas17/roots/.venv/bin/python scripts/sept/prepare_training.py "$SLURM_ARRAY_TASK_ID"
