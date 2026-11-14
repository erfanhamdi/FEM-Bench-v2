"""Run one configured benchmark matrix."""

import argparse
import json
from pathlib import Path

from fem_bench.execution import execute_trial, selected_task_entries
from fem_bench.experiment import (
    ConfigError,
    initialize_experiment_run,
    resolve_experiment,
    verify_run_identity,
)
from fem_bench.run_audit import audit_experiment_run, format_audit_report


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(harness: str, default_config: str, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Run FEM-Bench with {harness}.")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / default_config,
    )
    parser.add_argument(
        "--models-config", type=Path, default=REPO_ROOT / "configs/models.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "benchmark_manifest.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "experiment_runs"
    )
    parser.add_argument("--run-id", help="Name for the new run directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save and display the resolved setup without model calls",
    )
    args = parser.parse_args(argv)

    resolved = resolve_experiment(
        args.experiment, args.models_config, args.manifest, REPO_ROOT
    )
    if resolved.config["harness"] != harness:
        raise ConfigError(f"This runner requires harness: {harness}")
    run = initialize_experiment_run(
        resolved, args.manifest, args.output_root, REPO_ROOT, args.run_id
    )
    verify_run_identity(run, expected=resolved, harness=harness)
    print(f"Run: {run.run_dir}")
    print(
        json.dumps(
            {
                key: run.resolved[key]
                for key in (
                    "track",
                    "harness",
                    "prompt_set_id",
                    "model_keys",
                    "task_ids",
                    "trial_numbers",
                )
            },
            indent=2,
        )
    )
    if args.dry_run:
        print("Dry run: no model calls.")
        return 0

    for model_key in run.resolved["model_keys"]:
        for task in selected_task_entries(run):
            for trial_number in run.resolved["trial_numbers"]:
                print(
                    f"Running {model_key}/{task['task_id']}/trial_{trial_number:03d}",
                    flush=True,
                )
                execute_trial(run, model_key, task, trial_number, REPO_ROOT)

    report = audit_experiment_run(run)
    print(format_audit_report(report))
    return 0 if report["status"] == "ok" else 1
