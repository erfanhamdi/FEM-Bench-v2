import json
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from fem_bench import execution
from fem_bench.evaluate_output import evaluate_function_output_match
from fem_bench.experiment import (
    ConfigError,
    initialize_experiment_run,
    load_experiment_run,
    resolve_experiment,
    verify_run_identity,
)
from fem_bench.run_audit import audit_experiment_run
from fem_bench.trial import parse_evaluator_result

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark_manifest.json"
TASK = json.loads(MANIFEST.read_text())["tasks"][0]


def experiment_files(directory, track="implementation", harness="onepass"):
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "models.json"
    model_path.write_text(
        json.dumps(
            {
                "models": {
                    "fake": {
                        "display_name": "Offline model",
                        "provider": "aws",
                        "adapter": "aws_bedrock_converse",
                        "model_id": "offline.fake",
                        "request": {
                            "reasoning": {"enabled": False, "parameters": {}},
                            "generation": {"max_output_tokens": 512},
                        },
                    }
                }
            }
        )
    )
    harness_config = {}
    if harness == "simpleagent":
        harness_config = {"network_access": False}
    elif harness == "multiagent":
        harness_config = {
            "network_access": False, "tester_model": "fake", "max_coder_runs_per_round": 3
        }
    elif harness == "openhands":
        image = directory / "offline.sif"
        image.write_bytes(b"offline image")
        harness_config = {
            "agent": "openhands",
            "agent_config": {
                "runtime": "singularity",
                "image": str(image),
                "total_token_budget": 100,
            },
        }
    config_path = directory / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "offline"},
                "track": track,
                "harness": harness,
                "benchmark": {"split": TASK["split"], "task_ids": [TASK["task_id"]]},
                "models": ["fake"],
                "trials": {"numbers": [2, 4]},
                "prompt_set": (
                    f"onepass_{track}_v1" if harness == "onepass" else {
                        "simpleagent": "simpleagent_v1",
                        "multiagent": "multiagent_v2",
                        "openhands": "openhands_v2",
                    }[harness]
                ),
                "generation": {"max_output_tokens": 512},
                "prompt_caching": {"enabled": False},
                "harness_config": harness_config,
            }
        )
    )
    return config_path, model_path


def test_numerical_comparison_respects_tolerance_shape_and_eigenvector_scale():
    reference = lambda: (3.0, np.array([1.0, -2.0, 4.0]))
    for output, expected in [
        ((3.0 + 1e-9, np.array([-2.0, 4.0, -8.0])), True),
        ((4.0, np.array([1.0, -2.0, 4.0])), False),
        ((3.0, np.array([1.0, 2.0, 4.0])), False),
        ((3.0, np.zeros(3)), False),
        ((3.0, np.ones((3, 1))), False),
    ]:
        matched, _ = evaluate_function_output_match(
            reference, lambda: output, [[]], scale_invariant_output_indices=[1]
        )
        assert matched == expected


@pytest.mark.parametrize("track", ["implementation", "test_generation"])
def test_reference_submission_scores_one_and_wrong_submission_scores_zero(
    tmp_path, track
):
    task = execution.load_benchmark_task(ROOT, TASK)
    if track == "implementation":
        submissions = [
            task.main_fcn_code,
            "def wrong(*args, **kwargs):\n    return None",
        ]
        metric = "function_correctness"
    else:
        submissions = [
            "\n\n".join(case["test_code"] for case in task.test_cases),
            "def test_nothing(fcn):\n    pass",
        ]
        metric = "joint_test_success"
    for index, code in enumerate(submissions):
        submission = tmp_path / f"submission_{index}.py"
        submission.write_text(code)
        score, artifact, _, _ = execution.evaluate_submission(
            ROOT, TASK, track, submission, tmp_path / f"evaluation_{index}"
        )
        assert score["evaluated"] is True
        assert score[metric] == 1 - index
        assert score["original_evaluator_result"] == json.loads(artifact.read_text())


def test_test_generation_score_counts_every_fault_check():
    score = parse_evaluator_result(
        {
            "tests_run": True,
            "test_results": {
                "reference_pass": [["test_a", True], ["test_b", True]],
                "failure_fail": [["test_a", True], ["test_a", False], ["test_b", True]],
            },
        },
        "test_generation",
    )
    assert score["joint_test_success"] == 0
    assert score["joint_test_success_rate"] == 0.5
    assert score["test_generation_summary"]["expected_failure_checks"]["total"] == 3


def test_snapshots_keep_tracks_distinct_and_reject_wrong_identity(tmp_path):
    snapshots = []
    for track in ("implementation", "test_generation"):
        config_path, models_path = experiment_files(tmp_path / track, track)
        resolved = resolve_experiment(config_path, models_path, MANIFEST, ROOT)
        run = initialize_experiment_run(
            resolved, MANIFEST, tmp_path / "runs", ROOT, run_id=track
        )
        snapshot = json.loads((run.run_dir / "resolved_experiment.json").read_text())
        assert snapshot["track"] == track
        assert snapshot["prompt_set_id"] == f"onepass_{track}_v1"
        assert snapshot["model_keys"] == ["fake"]
        assert snapshot["task_ids"] == [TASK["task_id"]]
        assert snapshot["trial_numbers"] == [2, 4]
        verify_run_identity(
            load_experiment_run(run.run_dir), track=track, harness="onepass"
        )
        with pytest.raises(ConfigError):
            verify_run_identity(
                run,
                track=(
                    "test_generation" if track == "implementation" else "implementation"
                ),
            )
        config = yaml.safe_load(config_path.read_text())
        config["prompt_set"] = (
            "onepass_test_generation_v1"
            if track == "implementation"
            else "onepass_implementation_v1"
        )
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ConfigError):
            resolve_experiment(config_path, models_path, MANIFEST, ROOT)
        snapshots.append(snapshot)
    assert snapshots[0]["track"] != snapshots[1]["track"]
    assert snapshots[0]["prompt_set_id"] != snapshots[1]["prompt_set_id"]


@pytest.mark.parametrize(
    "harness", ["onepass", "simpleagent", "multiagent", "openhands"]
)
def test_runner_dispatch_and_canonical_completion(tmp_path, monkeypatch, harness):
    from fem_bench import cli
    from fem_bench.agents import runner

    config_path, models_path = experiment_files(tmp_path, harness=harness)
    calls = []

    def offline_attempt(
        selected, run, model_key, task, directory, repo_root, client_factory
    ):
        calls.append(selected)
        (directory / "submission.py").write_text(
            "def wrong(*args, **kwargs):\n    return None"
        )
        return {
            "status": "completed",
            "termination_reason": "submitted",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "wall_time_seconds": 1,
            "model_calls": 1,
            "code_executions": 0,
            "usage": {},
            "effective_parameters": {},
            "submission": "submission.py",
            "trajectory": None,
            "stdout": None,
            "stderr": None,
        }

    monkeypatch.setattr(
        execution, "run_onepass_attempt", partial(offline_attempt, "onepass")
    )
    monkeypatch.setattr(
        execution, "run_simpleagent_attempt", partial(offline_attempt, "simpleagent")
    )
    monkeypatch.setattr(
        execution, "run_multiagent_attempt", partial(offline_attempt, "multiagent")
    )
    monkeypatch.setattr(
        runner, "run_openhands_attempt", partial(offline_attempt, "openhands")
    )
    options = [
        "--models-config",
        str(models_path),
        "--manifest",
        str(MANIFEST),
        "--output-root",
        str(tmp_path / "runs"),
    ]
    wrong_harness = "simpleagent" if harness == "onepass" else "onepass"
    with pytest.raises(ConfigError):
        cli.main(wrong_harness, str(config_path), options)
    assert (
        cli.main(harness, str(config_path), options + ["--run-id", "dry", "--dry-run"])
        == 0
    )
    assert calls == []
    assert not list((tmp_path / "runs" / "dry").rglob("result.json"))
    assert cli.main(harness, str(config_path), options + ["--run-id", "smoke"]) == 0
    assert calls == [harness, harness]
    run_dir = tmp_path / "runs" / "smoke"
    run = load_experiment_run(run_dir)
    results = sorted(run_dir.glob("trials/*/*/*/result.json"))
    assert len(results) == 2
    assert audit_experiment_run(run)["status"] == "ok"
    for path in results:
        result = json.loads(path.read_text())
        assert result["evaluation"]["function_correctness"] == 0
        assert result["evaluation"]["evaluated"] is True
    if harness != "onepass":
        return
    original = json.loads(results[0].read_text())
    for section, key, wrong_value in [
        ("identity", "track", "test_generation"),
        ("execution", "status", "model_error"),
        ("evaluation", "evaluated", False),
    ]:
        changed = json.loads(json.dumps(original))
        changed[section][key] = wrong_value
        results[0].write_text(json.dumps(changed))
        assert audit_experiment_run(run)["status"] == "anomalies_found"
    results[0].write_text(json.dumps(original))
    extra = run_dir / "trials" / "extra" / TASK["task_id"] / "trial_002" / "result.json"
    extra.parent.mkdir(parents=True)
    extra.write_text(json.dumps(original))
    assert audit_experiment_run(run)["status"] == "anomalies_found"
    extra.unlink()
    results[0].unlink()
    assert audit_experiment_run(run)["status"] == "anomalies_found"


@pytest.mark.parametrize(
    "stop, stored_harness",
    [("submitted", "openhands"), ("token_budget", "openhands"),
     ("wall_time_limit", "openhands"), ("parser_error", "openhands"),
     ("submitted", "archived_wrapper")]
)
def test_openhands_subprocess_preserves_usage_limits_and_redacts_artifacts(
    tmp_path, monkeypatch, stop, stored_harness
):
    from fem_bench.agents import openhands
    from fem_bench.agents.openhands_cli import TOKEN_BUDGET_MARKER

    config_path, models_path = experiment_files(tmp_path, harness="openhands")
    resolved = resolve_experiment(config_path, models_path, MANIFEST, ROOT)
    run = initialize_experiment_run(resolved, MANIFEST, tmp_path / "runs", ROOT)
    run.resolved["harness"] = stored_harness
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "offline-secret")
    monkeypatch.setenv("HOST_ONLY_SECRET", "must-not-pass")
    monkeypatch.setattr(
        openhands,
        "validate_container_image",
        lambda config: ("singularity", Path(config["image"])),
    )

    def container(command, *, cwd, env, **kwargs):
        assert "HOST_ONLY_SECRET" not in env
        assert env["SINGULARITYENV_AWS_SECRET_ACCESS_KEY"] == "offline-secret"
        bind = next(value for value in command if value.endswith(":/state:rw"))
        state = Path(bind.removesuffix(":/state:rw"))
        (state / "metrics.json").write_text(
            json.dumps(
                {
                    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                    "model_calls": 2,
                    "accounting_complete": True,
                }
            )
        )
        (cwd / "submission.py").write_text(
            "def wrong(*args, **kwargs):\n    return None"
        )
        events = [
            {"kind": "MessageEvent", "llm_message": {"role": "user", "content": []}},
            {
                "kind": "ActionEvent",
                "tool_name": "terminal",
                "action": {"command": "echo offline-secret"},
                "reasoning_content": "Native reasoning.",
                "thinking_blocks": [{"type": "thinking", "thinking": "Check it."}],
                "responses_reasoning_item": {
                    "summary": [
                        {"type": "summary_text", "text": "Check offline-secret."},
                        {"type": "summary_text", "text": "Then test."},
                    ],
                    "encrypted_content": "opaque-reasoning",
                },
            },
            {
                "kind": "ActionEvent",
                "tool_name": "terminal",
                "action": None,
                "responses_reasoning_item": None,
            },
            {
                "kind": "ActionEvent",
                "tool_name": "terminal",
                "action": {"command": "echo hello"},
                "responses_reasoning_item": {"summary": "Plain summary."},
            },
            {
                "kind": "ActionEvent",
                "tool_name": "terminal",
                "action": {"command": "echo world"},
                "responses_reasoning_item": {
                    "summary": [
                        "First.",
                        {"type": "summary_text", "text": "Second."},
                        None,
                        7,
                        {"text": None},
                    ]
                },
            },
            {
                "kind": "ActionEvent",
                "tool_name": "terminal",
                "action": "invalid action",
                "responses_reasoning_item": "invalid reasoning item",
            },
            {
                "kind": "ActionEvent",
                "tool_name": "terminal",
                "action": {"command": "input", "is_input": True},
                "responses_reasoning_item": {
                    "summary": {"type": "summary_text", "text": "Single block."}
                },
            },
            {
                "kind": "ActionEvent",
                "responses_reasoning_item": {"summary": None},
            },
        ]
        output = "\n".join(json.dumps(event) for event in events)
        if stop == "wall_time_limit":
            raise subprocess.TimeoutExpired(command, 1, output=output.encode())
        return SimpleNamespace(
            returncode=0,
            stdout=output,
            stderr=TOKEN_BUDGET_MARKER if stop == "token_budget" else "",
        )

    monkeypatch.setattr(openhands.subprocess, "run", container)
    trial = run.run_dir / "trials" / "fake" / TASK["task_id"] / "trial_002"
    if stop == "parser_error":

        def fail_parsing(output):
            raise RuntimeError("Parser failed after the model finished")

        monkeypatch.setattr(
            openhands.OpenHandsWrapper, "_parse_events", staticmethod(fail_parsing)
        )
        with pytest.raises(RuntimeError, match="Parser failed"):
            execution.execute_trial(run, "fake", TASK, 2, ROOT)
        assert (trial / "workspace/submission.py").is_file()
        raw = (trial / "agent_output.txt").read_text()
        assert "Plain summary." in raw
        assert "offline-secret" not in raw
        assert "[REDACTED]" in raw
        assert (
            json.loads((trial / "agent_metrics.json").read_text())["usage"][
                "total_tokens"
            ]
            == 7
        )
        assert json.loads((trial / "agent_process.json").read_text())["returncode"] == 0
        return
    _, result = execution.execute_trial(run, "fake", TASK, 2, ROOT)
    assert result["identity"]["harness"] == stored_harness
    assert result["execution"]["token_budget"] == 100
    assert result["execution"]["termination_reason"] == stop
    assert result["execution"]["status"] == (
        "completed" if stop == "submitted" else "budget_exceeded"
    )
    assert result["execution"]["model_calls"] == 2
    assert result["execution"]["code_executions"] == 3
    assert result["usage"]["total_tokens"] == 7
    assert result["evaluation"]["evaluated"] is (stop == "submitted")
    assert (
        json.loads((trial / "agent_metrics.json").read_text())["usage"]["total_tokens"]
        == 7
    )
    process = json.loads((trial / "agent_process.json").read_text())
    assert process["timed_out"] is (stop == "wall_time_limit")
    assert process["returncode"] == (-1 if stop == "wall_time_limit" else 0)
    for artifact in ("agent_output.txt", "trajectory.json"):
        text = (trial / artifact).read_text()
        assert "offline-secret" not in text
        assert "[REDACTED]" in text
    trajectory = json.loads((trial / "trajectory.json").read_text())
    assert trajectory["reasoning_records"][:2] == [
        {
            "event_index": 1,
            "summary_text": "Check [REDACTED].\nThen test.",
            "summary_available": True,
            "reasoning_content_available": True,
            "thinking_blocks_available": True,
            "encrypted_reasoning_available": True,
        },
        {
            "event_index": 2,
            "summary_text": "",
            "summary_available": False,
            "reasoning_content_available": False,
            "thinking_blocks_available": False,
            "encrypted_reasoning_available": False,
        },
    ]
    assert [item["summary_text"] for item in trajectory["reasoning_records"][2:]] == [
        "Plain summary.",
        "First.\nSecond.",
        "",
        "Single block.",
        "",
    ]
    assert trajectory["events"][1]["reasoning_content"] == "Native reasoning."
    assert trajectory["events"][1]["thinking_blocks"][0]["thinking"] == "Check it."
    assert (
        trajectory["events"][1]["responses_reasoning_item"]["encrypted_content"]
        == "opaque-reasoning"
    )


def test_container_keeps_host_files_and_unapproved_environment_out(
    tmp_path, monkeypatch
):
    from fem_bench.agents.singularity_runtime import (
        OpenHandsContainerError,
        singularity_exec_command,
        singularity_subprocess_environment,
    )

    state, workspace = tmp_path / "state", tmp_path / "workspace"
    state.mkdir()
    workspace.mkdir()
    command = singularity_exec_command(
        "singularity", tmp_path / "image.sif", workspace, state, ["python"]
    )
    assert {"--containall", "--cleanenv", "--no-home"}.issubset(command)
    assert command[command.index("--no-mount") + 1] == "hostfs,bind-paths,home,cwd"
    assert [
        command[index + 1] for index, item in enumerate(command) if item == "--bind"
    ] == [
        f"{workspace}:/workspace:rw",
        f"{state}:/state:rw",
        f"{state / 'resolv.conf'}:/etc/resolv.conf:ro",
    ]
    monkeypatch.setenv("HOST_ONLY_SECRET", "must-not-pass")
    assert singularity_subprocess_environment({"LLM_API_KEY": "test"}) == {
        "SINGULARITYENV_LLM_API_KEY": "test"
    }
    with pytest.raises(OpenHandsContainerError, match="unapproved"):
        singularity_subprocess_environment({"HOST_ONLY_SECRET": "must-not-pass"})
