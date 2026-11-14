"""One small check for judge package preparation."""

import importlib.util
import json
from pathlib import Path

import yaml
import pytest


def test_extract_judgment_repairs_only_invalid_escapes():
    from run_judge import extract_judgment

    valid = r'{"rationale": "line\n\"quote\" \\alpha \u03b1 \/", "flag_for_review": false}'
    assert extract_judgment("```json\n" + valid + "\n```") == json.loads(valid)
    malformed = r'{"rationale": "\sigma + \epsilon; \\sigma; line\n\"quote\" \u03b1", "flag_for_review": false}'
    assert extract_judgment("```json\n" + malformed + "\n```") == {
        "rationale": '\\sigma + \\epsilon; \\sigma; line\n"quote" α',
        "flag_for_review": False,
    }
    for broken in ('{"codes": [}', '{"rationale": "unfinished', r'{"rationale": "\sigma",}'):
        with pytest.raises(json.JSONDecodeError):
            extract_judgment(broken)
    for empty in ("", "   ", "I cannot provide a judgment.", None):
        with pytest.raises(ValueError, match="no JSON object"):
            extract_judgment(empty)


def test_judge_package(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "run_judge", Path(__file__).resolve().parents[1] / "run_judge.py"
    )
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    reference = "def example_task():\n    return 42"
    (tmp_path / "example_task.py").write_text(
        "import numpy as np\n\ndef helper():\n    return 0\n\n" + reference + "\n"
    )
    simple = {
        "messages": [
            {"role": "user", "content": [{"text": "Implement f."}]},
            {"role": "assistant", "content": [{"text": "THOUGHT: Check f.\n```python\nf()\n```"}]},
            {"role": "user", "content": [{"text": "NameError"}]},
        ],
        "reasoning_summaries": ["Need a check."],
    }
    openhands = {
        "reasoning_records": [],
        "events": [
            {"kind": "MessageEvent", "llm_message": {"role": "user", "content": [{"type": "text", "text": "Implement f."}]}},
            {"kind": "ActionEvent", "reasoning_content": "Need a check.", "thought": [{"type": "text", "text": "Check f."}], "tool_call": {"name": "terminal", "arguments": '{"command": "f()", "security_risk": "LOW"}'}},
            {"kind": "ObservationEvent", "observation": {"content": [{"type": "text", "text": "NameError"}]}},
            {"kind": "MessageEvent", "llm_message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}},
            {"kind": "MessageEvent", "llm_message": {"role": "user", "content": [{"type": "text", "text": "Use a tool."}]}},
        ],
    }
    for cleaner, raw in (
        (judge.clean_simpleagent_trajectory, simple),
        (judge.clean_openhands_trajectory, openhands),
    ):
        prompt, entries = cleaner(raw)
        package = judge.prepare_input_package(prompt, entries, Path("example_task"), {
            "function_correctness": 0, "joint_tests_total": 2,
            "test_generation_summary": None,
            "original_evaluator_result": {"llm_name": "model", "test_results": [{
                "test_case": 1, "match": False, "error": None,
                "generated_output": [1], "inputs": [2], "reference_output": [3],
            }]},
        }, tmp_path, "def example_task():\n    return 0\n")
        assert package["submitted_code"] == "def example_task():\n    return 0\n"
        assert package["reference_code"] == reference
        assert prompt == "Implement f."
        assert entries[0]["thought"] == "Need a check.\n\nCheck f."
        assert "f()" in entries[0]["action"]
        assert "security_risk" not in entries[0]["action"]
        assert entries[1] == {"thought": "", "action": "", "observation": "NameError"}
        if raw is openhands:
            assert entries[2] == {"thought": "Done.", "action": "", "observation": ""}
            assert entries[3] == {"thought": "", "action": "", "observation": "Use a tool."}
        assert package["evaluation"] == {
            "function_correctness": 0,
            "original_evaluator_result": {},
            "test_results": [{
                "test_case": 1, "verification_inputs": [2],
                "test_results": {"match": False, "error": None},
            }],
        }

    assert judge.clean_openhands_trajectory({"events": openhands["events"]}) == (
        judge.clean_openhands_trajectory(openhands)
    )
    openhands["events"][1].update(reasoning_content=None, thinking_blocks=[])
    openhands["reasoning_records"] = [
        {"event_index": 1, "summary_text": "Provider summary."}
    ]
    _, entries = judge.clean_openhands_trajectory(openhands)
    assert entries[0]["thought"] == "Provider summary.\n\nCheck f."


def test_openhands_rejected_tool_call_keeps_error():
    from run_judge import clean_openhands_trajectory

    error = "Error validating tool 'file_editor': Failed to provide security_risk field."
    trajectory = {"events": [
        {"kind": "MessageEvent", "llm_message": {
            "role": "user", "content": [{"type": "text", "text": "Fix f."}],
        }},
        {"kind": "ActionEvent", "reasoning_content": "Edit f.", "thought": [],
         "tool_call": {"name": "file_editor", "arguments": '{"command": "str_replace"}'}},
        {"kind": "AgentErrorEvent", "tool_name": "file_editor", "error": error},
        {"kind": "MessageEvent", "llm_message": {
            "role": "assistant", "content": [{"type": "text", "text": "Retry."}],
        }},
    ]}
    prompt, entries = clean_openhands_trajectory(trajectory)
    assert prompt == "Fix f."
    assert entries == [
        {"thought": "Edit f.", "action": 'file_editor\n{"command": "str_replace"}', "observation": ""},
        {"thought": "", "action": "", "observation": "file_editor\n" + error},
        {"thought": "Retry.", "action": "", "observation": ""},
    ]


def test_trial_selection(tmp_path, monkeypatch):
    import run_judge as judge

    monkeypatch.setattr(judge, "__file__", str(tmp_path / "run_judge.py"))
    selected_packages = []
    monkeypatch.setattr(judge, "run_judge", lambda config, packages, codebook: selected_packages.extend(packages))
    (tmp_path / "llm-as-judge").mkdir()
    codebook_path = tmp_path / judge.DEFAULT_CODEBOOK
    codebook_path.write_text("code,instruction\nA,Read the trajectory\n")
    assert json.loads(judge.load_codebook(codebook_path)) == [{"code": "A", "instruction": "Read the trajectory"}]
    config_path = tmp_path / "configs/experiments/llm_as_judge.yaml"
    config_path.parent.mkdir(parents=True)
    trial_dirs = ["llm-as-judge/simpleagent/model/task/trial_001", "llm-as-judge/openhands/model/task/trial_002"]
    for directory in trial_dirs:
        trial = tmp_path / directory
        trial.mkdir(parents=True)
        (trial / "input_package.json").write_text("{}")
    for selection, expected in ([trial_dirs[1]], 1), ([], 2):
        selected_packages.clear()
        config_path.write_text(yaml.safe_dump({
            "input_dir": "llm-as-judge", "trial_dirs": selection,
            "codebook_version": "v2",
        }))
        judge.main([])
        assert len(selected_packages) == expected
        if selection:
            assert selected_packages[0].parent.name == "trial_002"


@pytest.mark.parametrize("use_defaults", [False, True])
def test_configured_codebook_and_system_prompt(tmp_path, monkeypatch, use_defaults):
    import run_judge as judge
    from types import SimpleNamespace

    monkeypatch.setattr(judge, "__file__", str(tmp_path / "run_judge.py"))
    codebook_path = tmp_path / (judge.DEFAULT_CODEBOOK if use_defaults else "new.csv")
    prompt_path = tmp_path / (judge.DEFAULT_SYSTEM_PROMPT if use_defaults else "prompt.txt")
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook_path.write_text("\ufeffCode ID,Description\nVF-01,New definition\n")
    prompt_path.write_text("New instructions\n{CODEBOOK}")
    package = tmp_path / "inputs/simpleagent/model/task/trial_001/input_package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"trajectory": []}')
    config = {
        "experiment": {"id": "fresh"}, "codebook_version": "new-v1",
        "models": ["judge"], "generation": {}, "input_dir": "inputs", "trial_dirs": [],
    }
    if not use_defaults:
        config.update(codebook="new.csv", system_prompt="prompt.txt")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(judge, "load_models_catalog", lambda path: {})
    monkeypatch.setattr(judge, "resolve_models", lambda *args: {"models": {"judge": {"request": {"generation": {}}}}})
    monkeypatch.setattr(judge, "resolve_generation_by_model", lambda *args: {"judge": {}})
    calls = []

    def call(messages, *, system_prompt):
        calls.append(system_prompt)
        assert messages == [{"role": "user", "content": package.read_text()}]
        assert system_prompt.startswith("New instructions\n")
        assert json.loads(system_prompt.split("\n", 1)[1]) == [{"Code ID": "VF-01", "Description": "New definition"}]
        return SimpleNamespace(text='{"codes": [], "flag_for_review": true}')

    monkeypatch.setattr(judge, "ConfiguredModelClient", lambda model: SimpleNamespace(call=call))
    judge.main(["--config", "config.yaml"])
    judge.main(["--config", "config.yaml"])
    assert len(calls) == 1
    output = tmp_path / "judge_runs/fresh/new-v1/judge/simpleagent/model/task/trial_001/judge_eval.json"
    assert json.loads(output.read_text()) == {"codes": [], "flag_for_review": True}


@pytest.mark.parametrize("bad_response", [
    "",
    "No judgment available.",
    '{"codes": [',
    '{}',
    '{"codes": [], "flag_for_review": "false"}',
    '{"codes": [{"code_id": "UNKNOWN", "rationale": "Evidence"}], "flag_for_review": false}',
    '{"codes": [{"code_id": "TC", "rationale": ""}], "flag_for_review": false}',
    '{"codes": [{"code_id": "TC", "rationale": "Evidence"}, {"code_id": "TC", "rationale": "Duplicate"}], "flag_for_review": false}',
    TimeoutError("Provider timed out"),
])
def test_judge_retries_continues_and_resumes(tmp_path, monkeypatch, bad_response):
    import run_judge as judge
    from types import SimpleNamespace

    monkeypatch.setattr(judge, "__file__", str(tmp_path / "run_judge.py"))
    monkeypatch.setattr(judge.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(judge, "load_models_catalog", lambda path: {})
    monkeypatch.setattr(judge, "resolve_models", lambda *args: {"models": {"judge": {"request": {"generation": {}}}}})
    monkeypatch.setattr(judge, "resolve_generation_by_model", lambda *args: {"judge": {}})
    (tmp_path / "prompt.txt").write_text("Judge with {CODEBOOK}")
    config = {
        "experiment": {"id": "retry"}, "codebook_version": "v2",
        "system_prompt": "prompt.txt", "models": ["judge"],
        "generation": {}, "input_dir": "inputs",
    }
    packages = []
    outputs = []
    for index in range(4):
        relative = Path(f"simpleagent/model/task/trial_{index:03d}")
        package = tmp_path / "inputs" / relative / "input_package.json"
        package.parent.mkdir(parents=True)
        package.write_text(str(index))
        packages.append(package)
        outputs.append(tmp_path / "judge_runs/retry/v2/judge" / relative / "judge_eval.json")
    good = '{"codes": [{"code_id": "TC", "rationale": "Evidence"}], "flag_for_review": false}'
    # A valid saved result is skipped; an interrupted/corrupt result is regenerated.
    for output, content in zip(outputs[2:], (good, '{"codes":')):
        output.parent.mkdir(parents=True)
        output.write_text(content)
    responses = iter([bad_response] * 4 + [good, good])
    calls = []

    def call(messages, **kwargs):
        calls.append(messages[0]["content"])
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response, stop_reason="end_turn", usage={}, raw_response={"text": response})

    monkeypatch.setattr(judge, "ConfiguredModelClient", lambda model: SimpleNamespace(call=call))
    codebook = '[{"Code ID": "TC"}]'
    assert judge.run_judge(config, packages, codebook) == 1
    assert calls == ["0", "0", "0", "1", "1", "3"]
    assert not outputs[0].exists()
    assert all(json.loads(output.read_text()) == json.loads(good) for output in outputs[1:])
    errors_path = outputs[0].with_name("judge_errors.jsonl")
    records = [json.loads(line) for line in errors_path.read_text().splitlines()]
    assert [record["attempt"] for record in records] == [1, 2, 3]
    if isinstance(bad_response, Exception):
        assert records[0]["response_text"] is None
        assert records[0]["error"] == "TimeoutError: Provider timed out"
    else:
        assert records[0]["response_text"] == bad_response
        assert records[0]["raw_response"] == {"text": bad_response}
        assert records[0]["stop_reason"] == "end_turn"
    original_errors = errors_path.read_text()
    responses = iter([good])
    calls.clear()
    assert judge.run_judge(config, packages, codebook) == 0
    assert calls == ["0"]
    assert errors_path.read_text() == original_errors
    assert json.loads(outputs[0].read_text()) == json.loads(good)
    assert not list(tmp_path.rglob("*.tmp"))


def test_judge_main_reports_incomplete_batch(tmp_path, monkeypatch):
    import run_judge as judge

    monkeypatch.setattr(judge, "__file__", str(tmp_path / "run_judge.py"))
    (tmp_path / "book.json").write_text("[]")
    package = tmp_path / "inputs/simpleagent/model/task/trial_001/input_package.json"
    package.parent.mkdir(parents=True)
    package.write_text("{}")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "codebook": "book.json", "input_dir": "inputs", "trial_dirs": [],
    }))
    monkeypatch.setattr(judge, "run_judge", lambda *args: 1)
    with pytest.raises(SystemExit) as error:
        judge.main(["--config", "config.yaml"])
    assert error.value.code == 1


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("selection", ["tasks", "trial-dirs"])
def test_judge_cli_selection_and_overrides(tmp_path, monkeypatch, capsys, selection, dry_run):
    import run_judge as judge

    monkeypatch.setattr(judge, "__file__", str(tmp_path / "run_judge.py"))
    (tmp_path / "custom.csv").write_text("Code ID,Code\nVF-01,Custom definition\n")
    (tmp_path / "custom-prompt.txt").write_text("Custom prompt {CODEBOOK}")
    folders = ["inputs/simpleagent/model/task_a/trial_001",
               "inputs/openhands/model/task_a/trial_002",
               "inputs/simpleagent/model/task_b/trial_001"]
    for folder in folders:
        path = tmp_path / folder
        path.mkdir(parents=True)
        (path / "input_package.json").write_text("{}")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "experiment": {"id": "old"}, "codebook_version": "old",
        "models": ["old_judge"], "codebook": "missing.csv", "system_prompt": "missing.txt",
        "input_dir": "inputs", "trial_dirs": [folders[2]],
        "generation": {"max_output_tokens": "provider_max"},
    }))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.json").write_text(json.dumps({"models": {
        key: {"provider": "example", "request": {"generation": {"max_output_tokens": 100}}}
        for key in ("judge_a", "judge_b")
    }}))
    calls = []
    monkeypatch.setattr(judge, "run_judge", lambda *args: calls.append(args))
    args = ["--config", "config.yaml", "--models", "judge_a", "judge_b",
            "--codebook", "custom.csv", "--system-prompt", "custom-prompt.txt",
            "--run-id", "fresh", "--codebook-version", "custom"]
    args += ["--tasks", "task_a"] if selection == "tasks" else ["--trial-dirs", folders[1]]
    judge.main(args + (["--dry-run"] if dry_run else []))
    expected = folders[:2] if selection == "tasks" else [folders[1]]
    if dry_run:
        assert not calls
        output = capsys.readouterr().out
        assert "Judges: judge_a, judge_b" in output
        assert "judge_runs/fresh/custom" in output
        for folder in expected:
            assert folder.removeprefix("inputs/") in output
        assert "task_b" not in output
        assert not (tmp_path / "judge_runs").exists()
    else:
        config, packages, codebook = calls[0]
        assert config["models"] == ["judge_a", "judge_b"]
        assert config["experiment"]["id"] == "fresh"
        assert config["codebook_version"] == "custom"
        assert config["system_prompt"] == "custom-prompt.txt"
        assert json.loads(codebook) == [{"Code ID": "VF-01", "Code": "Custom definition"}]
        assert set(packages) == {tmp_path / folder / "input_package.json" for folder in expected}
    calls.clear()
    with pytest.raises(SystemExit) as error:
        judge.main(["--config", "config.yaml", "--codebook", "custom.csv",
                    "--tasks", "task_a", "typo"])
    assert error.value.code == 2
    assert not calls


@pytest.mark.parametrize("config_args", [["config with spaces.yaml"], ["--config", "config with spaces.yaml"]])
def test_submit_judge_forwards_options(tmp_path, config_args):
    import subprocess

    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    options = ["--models", "judge_a", "judge_b", "--tasks", "task_a", "--dry-run"]
    result = subprocess.run(
        ["bash", str(Path(__file__).resolve().parents[1] / "submit_judge.sh"), *config_args, *options],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert result.stdout.splitlines() == [
        "-u", "run_judge.py", "--config", "config with spaces.yaml", *options,
    ]
