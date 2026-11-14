# FEM-Bench

FEM-Bench evaluates language models on finite element implementation and test
generation. It supports single-pass evaluation, SimpleAgent, OpenHands,
LLM-as-judge analysis, final adjudication, and an exploratory coder–tester harness.

The release includes 33 legacy tasks. Experiment data and new held-out tasks are
excluded. See the [reproducibility guide](docs/REPRODUCIBILITY.md) for paper
settings and the private assets needed to reproduce its results.

## Install

Use Python 3.12 and run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Add your provider credentials to `.env` or your shell environment. Model settings
and credential variable names are in [configs/models.json](configs/models.json).

- **SimpleAgent and coder–tester:** Linux and Bubblewrap (`bwrap`).
- **OpenHands:** Linux, Singularity, and a built container image. See
  [runtime setup](docs/REPRODUCIBILITY.md#runtime-requirements).

## Run the benchmark

The [example configs](configs/experiments) select one task, model, and trial;
they are not the paper's full experimental settings. For your own runs, copy a
config into `configs/experiments/local/` and pass it with `--experiment PATH`.

```bash
# Single-pass implementation and test generation
python run_pipeline.py
python run_pipeline.py --experiment configs/experiments/onepass_test.yaml

# Agent harnesses
python run_simpleagent.py --run-id simpleagent-example
python run_openhands.py --run-id openhands-example
python run_multiagent.py --run-id coder-tester-example
```

These commands make paid model requests. Add `--dry-run` to save and inspect the
configuration without model calls. OpenHands requires its image even for a dry run.

Results and configuration snapshots go to `experiment_runs/`. Each benchmark
invocation creates a new run; use a new `--run-id` when repeating an experiment.

## LLM-as-judge

Prepare evidence from completed, evaluated, incorrect SimpleAgent/OpenHands
trials, then preview and run the judges:

```bash
python prepare_judge_inputs.py \
  experiment_runs/simpleagent-example experiment_runs/openhands-example
python run_judge.py --dry-run
python run_judge.py
```

The [judge config](configs/experiments/llm_as_judge.yaml) selects the codebook,
prompt, and three judges. Results go to `judge_runs/`; valid saved judgments are
skipped on resume. Use a new experiment ID when changing inputs or settings.

## Final adjudication

After all three judges finish, prepare adjudication inputs without model calls:

```bash
python run_adjudicator.py --prepare-only --agent-models claude_haiku_4_5_aws
```

Remove `--prepare-only` to run the final judge on disagreements. Replace the
example agent alias with all evaluated agent aliases in a fixed order. Valid
saved adjudications are skipped; use a new `--output-dir` if inputs change.
Use `--help` on any runner for additional options.

## Tests

```bash
python -m pytest -q
```

Tests make no paid model calls. Linux/Bubblewrap tests are skipped when unavailable;
OpenHands tests mock the container process. Live provider and container execution
require separate checks.

## License

[MIT](LICENSE).
