#!/bin/bash
#SBATCH --job-name=rootseg_sept
#SBATCH --partition=amp48,amp20
#SBATCH --account=elevated
#SBATCH --qos=core
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/psyas17/SeptCTData/train/logs/train_%j.log
# Retrain RootSeg-LSTM on the original data + Sept CT volumes (Radicle labels), with intensity
# augmentation. The rootseg-lstm repo must stay on branch sept-radicle-labels while this runs.
set -euo pipefail
T=/data/psyas17/SeptCTData/train
PY=/home/psyas17/roots/.venv/bin/python
# merge the original normalisation stats and root index with the Sept ones (idempotent)
$PY - <<'PYEOF'
import glob, json, os
T = "/data/psyas17/SeptCTData/train"
stats = json.load(open("/data/psyas17/world_model/norm_stats_alldata.json"))
for f in glob.glob(f"{T}/stats/*.json"):
    stats[os.path.splitext(os.path.basename(f))[0]] = json.load(open(f))
json.dump(stats, open(f"{T}/norm_stats.json", "w"), indent=1)
print(len(stats), "volumes in norm_stats.json")
PYEOF
cp -n /data/psyas17/world_model/root_index/*.npz $T/root_index/
echo "root index files: $(ls $T/root_index | wc -l)"
cd /home/psyas17/rootseg-lstm
git branch --show-current
$PY train.py --config configs/sept.yaml
