#!/bin/bash
#$ -N multiagent
#$ -cwd
#$ -S /bin/bash
#$ -l h_rt=48:00:00
#$ -j y

# Usage: qsub submit_multiagent.sh <experiment config> <run id> [output root] [manifest]
bwrap --die-with-parent --unshare-net --ro-bind / / true \
  || { echo "bubblewrap sandbox unavailable on $(hostname)"; exit 1; }
PYTHONPATH=src exec "${PYTHON:-python}" -u run_multiagent.py \
  --experiment "${1:?Provide an experiment config}" \
  --manifest "${4:-benchmark_manifest.json}" \
  --output-root "${3:-experiment_runs}" \
  --run-id "${2:?Provide a run id}"
