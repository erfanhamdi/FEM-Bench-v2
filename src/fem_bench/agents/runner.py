"""Run OpenHands and collect its submission."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from fem_bench.agents.base import redact_credentials
from fem_bench.agents.openhands import OpenHandsWrapper
from fem_bench.execution import (
    build_run_benchmark_prompt,
    now_iso,
    runtime_model_config,
)
from fem_bench.experiment import ConfigError, ExperimentRun, ensure_model_is_runnable


def _load_instructions(run: ExperimentRun) -> str:
    matches = [
        item["path"]
        for item in run.prompt_manifest.get("files", [])
        if item.get("path", "").endswith("/instructions.md")
    ]
    if len(matches) != 1:
        raise ConfigError("OpenHands prompt set must contain one instructions.md")
    return (run.run_dir / "prompt_set" / matches[0]).read_text(encoding="utf-8")


def run_openhands_attempt(
    run: ExperimentRun,
    model_key: str,
    task_entry: dict[str, Any],
    attempt_dir: Path,
    repo_root: Path | str,
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    """Run the configured wrapper and return a normal attempt outcome."""

    del client_factory
    model = runtime_model_config(run, model_key)
    ensure_model_is_runnable(model)
    _, benchmark_prompt = build_run_benchmark_prompt(
        run, repo_root, task_entry, "implementation"
    )
    prompt = f"{_load_instructions(run).rstrip()}\n\n{benchmark_prompt}"

    harness_config = run.resolved["harness_config"]
    agent_name = harness_config["agent"]
    agent_config = dict(harness_config.get("agent_config", {}))

    workspace = attempt_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "TASK.md").write_text(prompt, encoding="utf-8")

    started_at = now_iso()
    started = time.monotonic()
    agent = OpenHandsWrapper(agent_name, model, agent_config)
    response = agent.generate_solution(prompt, workspace, artifact_dir=attempt_dir)

    stdout_path = attempt_dir / "agent_output.txt"
    stderr_path = None
    if response.error:
        stderr_path = attempt_dir / "agent_error.txt"
        stderr_path.write_text(
            redact_credentials(response.error, model), encoding="utf-8"
        )

    trajectory_path = None
    if response.trajectory is not None:
        trajectory_path = attempt_dir / "trajectory.json"
        trajectory_path.write_text(
            redact_credentials(
                json.dumps(response.trajectory, indent=2, default=str), model
            )
            + "\n",
            encoding="utf-8",
        )

    submission_path = workspace / "submission.py"
    submission = None
    if response.success and submission_path.is_file():
        shutil.copyfile(submission_path, attempt_dir / "submission.py")
        submission = "submission.py"
        status = "completed"
        reason = "submitted"
    elif response.termination_reason in {
        "wall_time_limit",
        "token_budget",
        "max_iterations",
    }:
        status = "budget_exceeded"
        reason = response.termination_reason
    else:
        status = "model_error"
        reason = (
            "missing_submission" if response.success else response.termination_reason
        )

    return {
        "status": status,
        "termination_reason": reason,
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_time_seconds": time.monotonic() - started,
        "model_calls": response.model_calls,
        "code_executions": response.code_executions,
        "usage": response.usage,
        "effective_parameters": {
            "agent": agent_name,
            "model": agent.model.name,
            "prompt_caching": model["prompt_caching"],
            **response.effective_parameters,
        },
        "submission": submission,
        "trajectory": trajectory_path.name if trajectory_path else None,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name if stderr_path else None,
    }
