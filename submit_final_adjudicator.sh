#!/bin/bash
#$ -N adjudicate_astra_v5
#$ -cwd
#$ -S /bin/bash
#$ -l h_rt=12:00:00
#$ -j y

exec .venv/bin/python -u run_adjudicator.py "$@"
