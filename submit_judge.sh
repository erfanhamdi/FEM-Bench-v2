#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -l h_rt=12:00:00
#$ -j y

# Usage: qsub submit_judge.sh --config PATH --models MODEL ... --tasks TASK ...
# Preview locally: bash submit_judge.sh --config PATH ... --dry-run
# Keep the original positional config form working.
if [[ $# -gt 0 && "$1" != -* ]]; then
    config="$1"
    shift
    set -- --config "$config" "$@"
fi
exec .venv/bin/python -u run_judge.py "$@"
