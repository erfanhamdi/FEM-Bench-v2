import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare_judge_inputs as prepare


@pytest.mark.parametrize("harness", ["simpleagent", "openhands", "archived_wrapper"])
def test_prepare_from_frozen_reference_and_refuse_changed_evidence(tmp_path, monkeypatch, harness):
    run_dir = tmp_path / "run"
    reference = run_dir / "benchmark_tasks/legacy/task.py"
    reference.parent.mkdir(parents=True)
    reference.write_text("def task():\n    return 42\n")
    trial = run_dir / "trials/model/task/trial_001"
    trial.mkdir(parents=True)
    if harness == "simpleagent":
        trajectory = {"messages": [
            {"role": "user", "content": [{"text": "Implement task."}]},
            {"role": "assistant", "content": [{"text": "THOUGHT: Done.\n```submit\npass\n```"}]},
        ], "reasoning_summaries": [""]}
    else:
        trajectory = {"events": [
            {"kind": "MessageEvent", "llm_message": {"role": "user", "content": [
                {"type": "text", "text": "Implement task."}]}},
            {"kind": "MessageEvent", "llm_message": {"role": "assistant", "content": [
                {"type": "text", "text": "Done."}]}},
        ]}
    (trial / "trajectory.json").write_text(json.dumps(trajectory))
    (trial / "submission.py").write_text("def task():\n    return 0\n")
    result = {"execution": {"status": "completed"}, "task": {"task_id": "task"},
              "evaluation": {"evaluated": True, "function_correctness": 0,
                             "original_evaluator_result": {"test_results": [
                                 {"test_case": 1, "inputs": [], "match": False}]}},
              "artifacts": {"trajectory": "trajectory.json", "submission": "submission.py"}}
    (trial / "result.json").write_text(json.dumps(result))
    run = SimpleNamespace(run_dir=run_dir, resolved={"harness": harness, "track": "implementation"},
                          manifest={"tasks": [{"task_id": "task", "split": "legacy"}]})
    if harness == "archived_wrapper":
        run.resolved["harness_config"] = {"agent": "openhands"}
    monkeypatch.setattr(prepare, "load_experiment_run", lambda path: run)
    monkeypatch.setattr(prepare, "audit_experiment_run", lambda run: {"issues": []})
    output = tmp_path / "packages"
    prepare.main([str(run_dir), "--output-dir", str(output)])
    paths = prepare.prepare_run(run_dir, output)
    assert len(paths) == 1
    assert paths[0].relative_to(output).parts == (
        "simpleagent" if harness == "simpleagent" else "openhands", "model", "task", "trial_001", "input_package.json")
    package = json.loads(paths[0].read_text())
    assert package["reference_code"] == "def task():\n    return 42"
    assert package["initial_prompt"] == "Implement task."
    assert package["evaluation"]["test_results"][0]["verification_inputs"] == []
    result["evaluation"]["function_correctness"] = 1
    (trial / "result.json").write_text(json.dumps(result))
    assert prepare.prepare_run(run_dir, output) == []
    result["evaluation"]["function_correctness"] = 0
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "submission.py").write_text("def task():\n    return 1\n")
    with pytest.raises(ValueError, match="Different evidence"):
        prepare.prepare_run(run_dir, output)
    assert json.loads(paths[0].read_text()) == package
    result["artifacts"]["submission"] = "../outside.py"
    (trial / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="outside trial directory"):
        prepare.prepare_run(run_dir, output)


def test_prepare_real_run_and_evaluated_trial(tmp_path, monkeypatch):
    from fem_bench import execution
    from fem_bench.experiment import initialize_experiment_run, resolve_experiment

    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "benchmark_manifest.json"
    resolved = resolve_experiment(repo / "configs/experiments/simpleagent.yaml",
                                  repo / "configs/models.json", manifest, repo)
    run = initialize_experiment_run(resolved, manifest, tmp_path, repo, "offline-agent")

    def fake_attempt(run, model_key, task, attempt_dir, repo_root, client_factory):
        (attempt_dir / "submission.py").write_text(
            f"def {task['task_id']}(*args, **kwargs):\n    return None\n")
        (attempt_dir / "trajectory.json").write_text(json.dumps({
            "messages": [{"role": "user", "content": [{"text": "Synthetic public-task check."}]}],
            "reasoning_summaries": [],
        }))
        return {"status": "completed", "termination_reason": "submitted",
                "started_at": "2026-11-14T00:00:00Z", "finished_at": "2026-11-14T00:00:01Z",
                "wall_time_seconds": 1, "model_calls": 1, "code_executions": 0,
                "usage": {}, "effective_parameters": {}, "submission": "submission.py",
                "trajectory": "trajectory.json", "stdout": None, "stderr": None}

    monkeypatch.setattr(execution, "run_simpleagent_attempt", fake_attempt)
    _, result = execution.execute_trial(run, run.resolved["model_keys"][0], resolved.tasks[0],
                                        run.resolved["trial_numbers"][0], repo)
    assert result["evaluation"]["evaluated"] is True
    assert result["evaluation"]["function_correctness"] == 0
    paths = prepare.prepare_run(run.run_dir, tmp_path / "packages")
    assert len(paths) == 1
    package = json.loads(paths[0].read_text())
    assert package["initial_prompt"] == "Synthetic public-task check."
    assert package["reference_code"].startswith("def " + resolved.tasks[0]["task_id"])
    assert package["evaluation"]["function_correctness"] == 0
    assert package["evaluation"]["test_results"]
