"""Trial records and scores derived from evaluator results."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fem_bench.experiment import ExperimentRun
from fem_bench.version import SCHEMA_VERSION


class TrialError(ValueError):
    """A trial path or evaluator result is invalid."""


class EvaluatorResultError(TrialError):
    """The evaluator artifact has an unexpected shape."""


def _safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise TrialError(f"Invalid trial path component: {value!r}")
    return value


def trial_id(
    experiment_run_id: str, model_key: str, task_id: str, trial_number: int
) -> str:
    return "__".join(
        (experiment_run_id, model_key, task_id, f"trial_{trial_number:03d}")
    )


def trial_directory(
    run: ExperimentRun, model_key: str, task_id: str, trial_number: int
) -> Path:
    return (
        run.run_dir
        / "trials"
        / _safe_component(model_key)
        / _safe_component(task_id)
        / f"trial_{trial_number:03d}"
    )


def normalize_usage(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    source = usage or {}
    normalized = {
        key: source.get(key)
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
            "cost_usd",
        )
    }
    if normalized["cache_read_tokens"] is None:
        normalized["cache_read_tokens"] = normalized["cached_tokens"]
    if normalized["cached_tokens"] is None:
        normalized["cached_tokens"] = normalized["cache_read_tokens"]
    if normalized["total_tokens"] is None:
        input_tokens = normalized["input_tokens"]
        output_tokens = normalized["output_tokens"]
        if input_tokens is not None and output_tokens is not None:
            normalized["total_tokens"] = input_tokens + output_tokens
    return normalized


def _parse_pairs(value: Any, name: str) -> list[tuple[str, bool]]:
    if not isinstance(value, list):
        raise EvaluatorResultError(f"Evaluator {name} must be a list")
    parsed = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], bool)
        ):
            raise EvaluatorResultError(f"Malformed evaluator {name}[{index}]: {item!r}")
        parsed.append((item[0], item[1]))
    return parsed


def _test_generation_summary(
    tests_run: bool,
    reference_pairs: list[tuple[str, bool]],
    failure_pairs: list[tuple[str, bool]],
) -> tuple[int, dict[str, Any]]:
    reference_by_name: dict[str, list[bool]] = defaultdict(list)
    failure_by_name: dict[str, list[bool]] = defaultdict(list)
    for name, passed in reference_pairs:
        reference_by_name[name].append(passed)
    for name, detected in failure_pairs:
        failure_by_name[name].append(detected)

    per_test = []
    joint_total = 0
    joint_passed = 0
    for name in sorted(set(reference_by_name) | set(failure_by_name)):
        ref_values = reference_by_name.get(name, [])
        failure_values = failure_by_name.get(name, [])
        has_joint_checks = bool(ref_values and failure_values)
        joint_success = bool(
            has_joint_checks and all(ref_values) and all(failure_values)
        )
        if has_joint_checks:
            joint_total += 1
            joint_passed += int(joint_success)
        per_test.append(
            {
                "test_name": name,
                "reference_checks_total": len(ref_values),
                "reference_checks_passed": sum(ref_values),
                "expected_failure_checks_total": len(failure_values),
                "expected_failures_detected": sum(failure_values),
                "joint_success": int(joint_success) if has_joint_checks else None,
            }
        )

    all_reference_passed = bool(reference_pairs) and all(
        passed for _, passed in reference_pairs
    )
    all_expected_failures_detected = bool(failure_pairs) and all(
        detected for _, detected in failure_pairs
    )
    # This is deliberately derived only from the evaluator artifact. It retains
    # every repeated expected-failure check instead of collapsing test names.
    joint_test_success = int(
        tests_run and all_reference_passed and all_expected_failures_detected
    )
    summary = {
        "tests_run": tests_run,
        "reference_checks": {
            "total": len(reference_pairs),
            "passed": sum(passed for _, passed in reference_pairs),
            "all_passed": all_reference_passed,
        },
        "expected_failure_checks": {
            "total": len(failure_pairs),
            "detected": sum(detected for _, detected in failure_pairs),
            "all_detected": all_expected_failures_detected,
        },
        "joint_tests": {
            "total": joint_total,
            "passed": joint_passed,
            "rate": joint_passed / joint_total if joint_total else 0.0,
            "all_passed": bool(joint_total) and joint_passed == joint_total,
        },
        "suite_success": bool(joint_test_success),
        "per_test": per_test,
    }
    return joint_test_success, summary


def parse_evaluator_result(
    evaluator_result: dict[str, Any], track: str
) -> dict[str, Any]:
    """Normalize function correctness and joint test success from evaluator output."""
    if not isinstance(evaluator_result, dict):
        raise EvaluatorResultError("Evaluator result must be an object")
    if track == "implementation":
        match = evaluator_result.get("matches_reference")
        if not isinstance(match, bool):
            raise EvaluatorResultError(
                "Implementation evaluator result lacks boolean matches_reference"
            )
        return {
            "evaluated": True,
            "function_correctness": int(match),
            "joint_test_success": None,
            "joint_test_success_rate": None,
            "joint_tests_passed": None,
            "joint_tests_total": None,
            "score_source": "evaluator",
            "test_generation_summary": None,
            "original_evaluator_result": evaluator_result,
        }
    if track != "test_generation":
        raise EvaluatorResultError(f"Unknown evaluation track: {track!r}")
    tests_run = evaluator_result.get("tests_run")
    if not isinstance(tests_run, bool):
        raise EvaluatorResultError("Test evaluator result lacks boolean tests_run")
    test_results = evaluator_result.get("test_results", {})
    if not isinstance(test_results, dict):
        raise EvaluatorResultError("Evaluator test_results must be an object")
    references = _parse_pairs(test_results.get("reference_pass", []), "reference_pass")
    failures = _parse_pairs(test_results.get("failure_fail", []), "failure_fail")
    joint, summary = _test_generation_summary(tests_run, references, failures)
    joint_tests = summary["joint_tests"]
    return {
        "evaluated": True,
        "function_correctness": None,
        "joint_test_success": joint,
        "joint_test_success_rate": joint_tests["rate"],
        "joint_tests_passed": joint_tests["passed"],
        "joint_tests_total": joint_tests["total"],
        "score_source": "evaluator",
        "test_generation_summary": summary,
        "original_evaluator_result": evaluator_result,
    }


def protocol_failure_evaluation(track: str) -> dict[str, Any]:
    if track == "implementation":
        function_correctness, joint = 0, None
        joint_rate, joint_passed, joint_total = None, None, None
    elif track == "test_generation":
        function_correctness, joint = None, 0
        joint_rate, joint_passed, joint_total = 0.0, 0, 0
    else:
        raise TrialError(f"Unknown track: {track!r}")
    return {
        "evaluated": False,
        "function_correctness": function_correctness,
        "joint_test_success": joint,
        "joint_test_success_rate": joint_rate,
        "joint_tests_passed": joint_passed,
        "joint_tests_total": joint_total,
        "score_source": "protocol_failure",
        "test_generation_summary": None,
        "original_evaluator_result": None,
    }


def make_trial_result(
    *,
    run: ExperimentRun,
    model_key: str,
    task: dict[str, Any],
    trial_number: int,
    outcome: dict[str, Any],
    evaluation: dict[str, Any],
    artifacts: dict[str, str | None],
) -> dict[str, Any]:
    model = run.models_snapshot["models"][model_key]
    harness_config = run.resolved["harness_config"]
    if harness_config.get("agent") == "openhands":
        harness_config = harness_config["agent_config"]
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "experiment_name": run.resolved["experiment_name"],
            "experiment_run_id": run.resolved["experiment_run_id"],
            "trial_id": trial_id(
                run.resolved["experiment_run_id"],
                model_key,
                task["task_id"],
                trial_number,
            ),
            "trial_number": trial_number,
            "track": run.resolved["track"],
            "harness": run.resolved["harness"],
            "created_at": outcome["finished_at"],
        },
        "task": {
            "task_id": task["task_id"],
            "split": task["split"],
            "benchmark_version": run.manifest["benchmark_version"],
        },
        "model": {
            "model_key": model_key,
            "display_name": model["display_name"],
            "provider": model["provider"],
            "model_id": model["model_id"],
            "transport": model["adapter"],
            "reasoning": model["request"]["reasoning"],
            "effective_parameters": outcome["effective_parameters"],
        },
        "prompting": {"prompt_set_id": run.resolved["prompt_set_id"]},
        "execution": {
            "status": outcome["status"],
            "termination_reason": outcome["termination_reason"],
            "started_at": outcome["started_at"],
            "finished_at": outcome["finished_at"],
            "wall_time_seconds": outcome["wall_time_seconds"],
            "model_calls": outcome["model_calls"],
            "code_executions": outcome["code_executions"],
            "token_budget": harness_config.get("total_token_budget"),
        },
        "evaluation": evaluation,
        "usage": normalize_usage(outcome["usage"]),
        "artifacts": artifacts,
    }


def write_trial_result(trial_dir: Path | str, result: dict[str, Any]) -> Path:
    path = Path(trial_dir) / "result.json"
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return path
