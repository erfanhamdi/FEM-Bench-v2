"""Check that the saved experiment matrix completed and was evaluated."""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from fem_bench.experiment import ExperimentRun, load_json, verify_run_identity
from fem_bench.trial import trial_directory, trial_id


def audit_experiment_run(run: ExperimentRun) -> dict[str, Any]:
    verify_run_identity(run)
    resolved = run.resolved
    expected = {
        trial_directory(run, model, task, number) / "result.json": (model, task, number)
        for model, task, number in product(
            resolved["model_keys"], resolved["task_ids"], resolved["trial_numbers"]
        )
    }
    observed = set((run.run_dir / "trials").glob("*/*/*/result.json"))
    issues = []
    outcomes = Counter()
    clean_trials = 0
    for path, (model, task, number) in expected.items():
        problems = []
        if path not in observed:
            problems.append(("missing_trial", "The expected result.json is missing."))
        else:
            try:
                result = load_json(path)
                identity = {
                    "experiment_run_id": resolved["experiment_run_id"],
                    "trial_id": trial_id(
                        resolved["experiment_run_id"], model, task, number
                    ),
                    "trial_number": number,
                    "track": resolved["track"],
                    "harness": resolved["harness"],
                }
                if (
                    any(
                        result["identity"].get(key) != value
                        for key, value in identity.items()
                    )
                    or result["model"]["model_key"] != model
                    or result["task"]["task_id"] != task
                    or result["prompting"]["prompt_set_id"] != resolved["prompt_set_id"]
                ):
                    problems.append(
                        (
                            "identity_mismatch",
                            "Result identity differs from the resolved experiment.",
                        )
                    )
                if result["execution"]["status"] != "completed":
                    problems.append(
                        ("execution_failure", "Execution did not complete.")
                    )
                if result["evaluation"]["evaluated"] is not True:
                    problems.append(
                        ("unevaluated_trial", "The result was not evaluated.")
                    )
                elif resolved["track"] == "implementation":
                    outcome = (
                        "matched"
                        if result["evaluation"]["function_correctness"] == 1
                        else "not_matched"
                    )
                    outcomes[outcome] += 1
                else:
                    outcomes["test_generation_evaluated"] += 1
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                problems.append(("corrupt_trial", f"Cannot read the result: {exc}"))
        if not problems:
            clean_trials += 1
        for kind, message in problems:
            issues.append(
                {
                    "kind": kind,
                    "message": message,
                    "trial_path": str(path.parent),
                    "model_key": model,
                    "task_id": task,
                    "trial_number": number,
                }
            )
    for path in sorted(observed - expected.keys()):
        issues.append(
            {
                "kind": "unexpected_trial",
                "message": "Result is outside the resolved experiment matrix.",
                "trial_path": str(path.parent),
            }
        )
    return {
        "run_dir": str(run.run_dir),
        "experiment_run_id": resolved["experiment_run_id"],
        "track": resolved["track"],
        "harness": resolved["harness"],
        "status": "ok" if not issues else "anomalies_found",
        "summary": {
            "expected_trials": (
                len(resolved["model_keys"])
                * len(resolved["task_ids"])
                * len(resolved["trial_numbers"])
            ),
            "observed_results": len(observed),
            "clean_trials": clean_trials,
            "issue_count": len(issues),
            "issue_counts": dict(Counter(issue["kind"] for issue in issues)),
            "outcomes": dict(outcomes),
        },
        "issues": issues,
    }


def write_audit_report(report: dict[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def format_audit_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Run status: {report['status']}; {summary['clean_trials']}/{summary['expected_trials']} trials completed and evaluated."
    ]
    lines.extend(
        f"{issue['trial_path']}: {issue['message']}" for issue in report["issues"]
    )
    return "\n".join(lines)
