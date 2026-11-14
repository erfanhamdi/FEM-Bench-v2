"""Run each model-task trial once and evaluate its submission."""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fem_bench.benchmark_manifest import task_source_path
from fem_bench.experiment import (
    ExperimentRun,
    ensure_model_is_runnable,
)
from fem_bench.evaluate_output import (
    evaluate_function_output_match,
    evaluate_task_tests,
    extract_function_name,
    load_function_from_code,
)
from fem_bench.task_loader import load_task_from_info
from fem_bench.task_to_prompt import task_to_code_prompt, task_to_test_prompt
from fem_bench.trial import (
    make_trial_result,
    parse_evaluator_result,
    protocol_failure_evaluation,
    trial_directory,
    write_trial_result,
)
from llm_api.clean_utils import clean_and_extract_function, extract_test_functions
from llm_api.configured_client import (
    ConfiguredModelClient,
    write_raw_response,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _load_benchmark_task_path(path: Path, entry: dict[str, Any]):
    module_name = f"fem_bench_v2_task_{entry['task_id']}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return load_task_from_info(module.task_info)


def load_benchmark_task(repo_root: Path | str, entry: dict[str, Any]):
    return _load_benchmark_task_path(task_source_path(repo_root, entry), entry)


def build_run_benchmark_prompt(
    run: ExperimentRun,
    repo_root: Path | str,
    entry: dict[str, Any],
    track: str,
) -> tuple[Any, str]:
    """Build a prompt from the run's task and template snapshots."""
    task = _load_benchmark_task_path(
        run.run_dir / "benchmark_tasks" / entry["split"] / f"{entry['task_id']}.py",
        entry,
    )
    templates = str(run.run_dir / "prompt_set" / "prompt_templates")
    prompt = (
        task_to_code_prompt(task, templates, "code_prompt.j2")
        if track == "implementation"
        else task_to_test_prompt(task, templates, "test_prompt.j2")
    )
    return task, prompt


def build_sandbox_preamble(task) -> str:
    return "\n\n".join([*task.required_imports, *task.fcn_dependency_code])


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return float(value.real)
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _clean_onepass_output(raw: str, track: str) -> str:
    if track == "implementation":
        try:
            return clean_and_extract_function(raw)
        except (SyntaxError, ValueError):
            # Malformed model output still receives an evaluator score.
            return raw
    tests = extract_test_functions(raw)
    return "\n\n".join(tests.values())


def runtime_model_config(run: ExperimentRun, model_key: str) -> dict[str, Any]:
    """Apply the experiment's immutable request settings to a model config."""
    model = copy.deepcopy(run.models_snapshot["models"][model_key])
    model["request"]["generation"].update(
        run.resolved["effective_generation_by_model"][model_key]
    )
    model["prompt_caching"] = copy.deepcopy(run.resolved["prompt_caching"])
    return model


def run_onepass_attempt(
    run: ExperimentRun,
    model_key: str,
    task_entry: dict[str, Any],
    attempt_dir: Path,
    repo_root: Path | str,
    client_factory: Callable[
        [dict[str, Any]], ConfiguredModelClient
    ] = ConfiguredModelClient,
) -> dict[str, Any]:
    model = runtime_model_config(run, model_key)
    ensure_model_is_runnable(model)
    _, prompt = build_run_benchmark_prompt(
        run, repo_root, task_entry, run.resolved["track"]
    )
    raw_dir = attempt_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    started = time.monotonic()
    response = client_factory(model).call(
        [{"role": "user", "content": prompt}],
        system_prompt=run.prompt_manifest.get("system_prompt"),
    )
    finished_at = now_iso()
    (raw_dir / "completion.txt").write_text(response.text, encoding="utf-8")
    write_raw_response(raw_dir / "provider_response.json", response)
    submission = _clean_onepass_output(response.text, run.resolved["track"])
    submission_path = attempt_dir / "submission.py"
    submission_path.write_text(submission, encoding="utf-8")
    return {
        "status": "completed",
        "termination_reason": "submitted",
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": time.monotonic() - started,
        "model_calls": 1,
        "code_executions": 0,
        "usage": response.usage,
        "effective_parameters": response.effective_parameters,
        "submission": "submission.py",
        "trajectory": None,
        "stdout": None,
        "stderr": None,
    }


def _load_simpleagent_prompts(run: ExperimentRun) -> dict[str, str]:
    path = next(
        item["path"]
        for item in run.prompt_manifest["files"]
        if item["path"].endswith("/prompts.json")
    )
    return json.loads((run.run_dir / "prompt_set" / path).read_text(encoding="utf-8"))


def _finalize_simpleagent_attempt(
    result: dict[str, Any], attempt_dir: Path, started_at: str
) -> dict[str, Any]:
    trajectory = next(attempt_dir.glob("*_trajectory_*.json"))
    trajectory.rename(attempt_dir / "trajectory.json")

    stdout = "".join(record.get("stdout", "") for record in result["execution_records"])
    stderr = "".join(record.get("stderr", "") for record in result["execution_records"])
    (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    exit_status = result["exit_status"]
    if exit_status == "submitted":
        status, reason = "completed", "submitted"
        (attempt_dir / "submission.py").write_text(
            result["submission"], encoding="utf-8"
        )
        submission = "submission.py"
    else:
        budget_reasons = {
            "token_budget_exceeded": "token_budget",
            "wall_time_exceeded": "wall_time_limit",
            "max_iterations_reached": "max_iterations",
            "max_rounds_reached": "max_rounds",
        }
        if exit_status in budget_reasons:
            status, reason = "budget_exceeded", budget_reasons[exit_status]
        else:
            status, reason = "model_error", "agent_terminated"
        submission = None
    return {
        "status": status,
        "termination_reason": reason,
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_time_seconds": result["wall_time_seconds"],
        "model_calls": result["n_model_calls"],
        "code_executions": result["n_executions"],
        "usage": result["usage"],
        "effective_parameters": result["effective_parameters"],
        "submission": submission,
        "trajectory": "trajectory.json",
        "stdout": "stdout.txt",
        "stderr": "stderr.txt",
    }


def run_simpleagent_attempt(
    run: ExperimentRun,
    model_key: str,
    task_entry: dict[str, Any],
    attempt_dir: Path,
    repo_root: Path | str,
    client_factory: Callable[
        [dict[str, Any]], ConfiguredModelClient
    ] = ConfiguredModelClient,
) -> dict[str, Any]:
    import simpleagent

    model = runtime_model_config(run, model_key)
    ensure_model_is_runnable(model)
    task, prompt = build_run_benchmark_prompt(
        run, repo_root, task_entry, "implementation"
    )
    prompts = _load_simpleagent_prompts(run)
    started_at = now_iso()
    result = simpleagent.run_agent(
        prompt,
        task_entry["task_id"],
        build_sandbox_preamble(task),
        model_config=model,
        harness_config=run.resolved["harness_config"],
        output_dir=attempt_dir,
        prompts=prompts,
        model_client=client_factory(model),
    )
    return _finalize_simpleagent_attempt(result, attempt_dir, started_at)


def run_multiagent_attempt(
    run: ExperimentRun,
    model_key: str,
    task_entry: dict[str, Any],
    attempt_dir: Path,
    repo_root: Path | str,
    client_factory: Callable[
        [dict[str, Any]], ConfiguredModelClient
    ] = ConfiguredModelClient,
) -> dict[str, Any]:
    import multiagent

    model = runtime_model_config(run, model_key)
    ensure_model_is_runnable(model)
    tester = copy.deepcopy(run.models_snapshot["tester_model"])
    tester["prompt_caching"] = copy.deepcopy(run.resolved["prompt_caching"])
    ensure_model_is_runnable(tester)
    task, prompt = build_run_benchmark_prompt(
        run, repo_root, task_entry, "implementation"
    )
    started_at = now_iso()
    result = multiagent.run_agent(
        prompt,
        task_entry["task_id"],
        build_sandbox_preamble(task),
        model_config=model,
        harness_config=run.resolved["harness_config"],
        output_dir=attempt_dir,
        prompts=_load_simpleagent_prompts(run),
        model_client=client_factory(model),
        tester_model_config=tester,
        tester_client=client_factory(tester),
    )
    return _finalize_simpleagent_attempt(result, attempt_dir, started_at)


def _evaluate_code(task, source: str, track: str) -> dict[str, Any]:
    implementation = track == "implementation"
    result = {
        "task_id": task.task_id,
        "llm_name": "canonical",
        **(
            {"matches_reference": False, "test_results": []}
            if implementation
            else {"tests_run": False, "test_results": {}}
        ),
    }
    try:
        nodes = [
            node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
        ]
        if implementation:
            nodes = nodes[:1]
        functions = {node.name: ast.unparse(node) for node in nodes}
    except SyntaxError as exc:
        result["error"] = (
            f"SyntaxError in {'function' if implementation else 'test'} code: {exc}"
        )
        return result
    if not implementation:
        functions = {
            name: code for name, code in functions.items() if name.startswith("test_")
        }
    if not functions:
        result["error"] = (
            "No function definition found."
            if implementation
            else "No test functions found."
        )
        return result

    reference = load_function_from_code(
        task.main_fcn_code, task.required_imports, task.fcn_dependency_code
    )
    try:
        if implementation:
            generated = load_function_from_code(
                next(iter(functions.values())),
                task.required_imports,
                task.fcn_dependency_code,
            )
            result["matches_reference"], result["test_results"] = (
                evaluate_function_output_match(
                    reference,
                    generated,
                    task.reference_verification_inputs,
                    scale_invariant_output_indices=task.scale_invariant_output_indices,
                )
            )
        else:
            expected_failures = {
                extract_function_name(case["test_code"]): case["expected_failures"]
                for case in task.test_cases
            }
            task.test_cases = [
                {
                    "test_code": code,
                    "expected_failures": expected_failures.get(name, []),
                }
                for name, code in functions.items()
            ]
            result["test_results"] = evaluate_task_tests(task, reference)
            result["tests_run"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def evaluate_submission(
    repo_root: Path | str,
    task_entry: dict[str, Any],
    track: str,
    submission_path: Path,
    evaluator_dir: Path,
    *,
    run: ExperimentRun | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    """Evaluate one submission with the benchmark's numerical checks."""
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    task = (
        _load_benchmark_task_path(
            run.run_dir
            / "benchmark_tasks"
            / task_entry["split"]
            / f"{task_entry['task_id']}.py",
            task_entry,
        )
        if run is not None
        else load_benchmark_task(repo_root, task_entry)
    )
    stdout_path = evaluator_dir / "stdout.txt"
    stderr_path = evaluator_dir / "stderr.txt"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        result = _evaluate_code(
            task, submission_path.read_text(encoding="utf-8"), track
        )
    result_path = evaluator_dir / "results.json"
    _write_json(result_path, result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return parse_evaluator_result(result, track), result_path, stdout_path, stderr_path


def execute_trial(
    run: ExperimentRun,
    model_key: str,
    task_entry: dict[str, Any],
    trial_number: int,
    repo_root: Path | str,
    *,
    client_factory: Callable[
        [dict[str, Any]], ConfiguredModelClient
    ] = ConfiguredModelClient,
) -> tuple[str, dict[str, Any]]:
    """Run one trial in a new directory, preserving any previous output."""
    directory = trial_directory(run, model_key, task_entry["task_id"], trial_number)
    directory.mkdir(parents=True)
    harness = run.resolved["harness"]
    if harness == "onepass":
        runner = run_onepass_attempt
    elif harness == "simpleagent":
        runner = run_simpleagent_attempt
    elif harness == "multiagent":
        runner = run_multiagent_attempt
    else:
        from fem_bench.agents.runner import run_openhands_attempt

        runner = run_openhands_attempt

    stage = "generation"
    try:
        outcome = runner(
            run, model_key, task_entry, directory, repo_root, client_factory
        )
        _write_json(directory / "execution.json", outcome)
        evaluation = protocol_failure_evaluation(run.resolved["track"])
        evaluator_path = None
        if outcome["status"] == "completed":
            stage = "evaluation"
            evaluation, evaluator_path, _, _ = evaluate_submission(
                repo_root,
                task_entry,
                run.resolved["track"],
                directory / outcome["submission"],
                directory / "evaluator",
                run=run,
            )
    except Exception as exc:
        _write_json(
            directory / "error.json",
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise

    artifacts = {
        key: outcome[key] for key in ("submission", "trajectory", "stdout", "stderr")
    }
    artifacts["evaluator_result"] = (
        evaluator_path.relative_to(directory).as_posix() if evaluator_path else None
    )
    artifacts["raw_logs"] = "raw" if harness == "onepass" else "."
    result = make_trial_result(
        run=run,
        model_key=model_key,
        task=task_entry,
        trial_number=trial_number,
        outcome=outcome,
        evaluation=evaluation,
        artifacts=artifacts,
    )
    write_trial_result(directory, result)
    return "completed", result


def selected_task_entries(run: ExperimentRun) -> list[dict[str, Any]]:
    by_id = {entry["task_id"]: entry for entry in run.manifest["tasks"]}
    return [by_id[task_id] for task_id in run.resolved["task_ids"]]
