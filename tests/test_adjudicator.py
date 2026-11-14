import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_adjudicator as adjudicator


def test_v5_adjudication_sources_retry_and_resume(tmp_path, monkeypatch):
    repo = adjudicator.REPO_ROOT
    monkeypatch.setattr(adjudicator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(adjudicator.time, "sleep", lambda seconds: None)
    for relative in (adjudicator.CODEBOOK, adjudicator.PROMPT, Path("configs/models.json")):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    case = "openhands/kimi_k3_together/task/trial_001"
    folder = tmp_path / "llm-as-judge" / case
    folder.mkdir(parents=True)
    package = {"trajectory": ["evidence"], "submitted_code": "code", "reference_code": "reference"}
    (folder / "input_package.json").write_text(json.dumps(package))
    for judge, code in zip(adjudicator.JUDGES, ["VF-01", "VF-02", "VF-01"]):
        path = tmp_path / adjudicator.RUN_DIR / judge / case / "judge_eval.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"codes": [{"code_id": code, "rationale": "evidence"}],
                                    "flag_for_review": False}))
    results = adjudicator.load_results()
    agreements, disagreements = adjudicator.check_agreement(results)
    assert not agreements and list(disagreements) == [case]
    path = adjudicator.input_package_builder(case, results[case])
    prepared = json.loads(path.read_text())
    assert prepared["input_package"] == package
    assert set(prepared["judgments"]) == {"judge_1", "judge_2", "judge_3"}
    assert list(prepared["judgments"].values()) == list(results[case].values())
    calls = []
    valid = {"codes": [{"code_id": "VF-02", "rationale": "specific evidence"}], "flag_for_review": False}
    def call(messages, system_prompt):
        calls.append(messages)
        assert "{CODEBOOK}" not in system_prompt and "there are no tiers" in system_prompt
        return SimpleNamespace(text=json.dumps({} if len(calls) == 1 else valid), usage={"input_tokens": 42})
    monkeypatch.setattr(adjudicator, "ConfiguredModelClient", lambda model: SimpleNamespace(call=call))
    assert adjudicator.run_final_judge([path]) == 0
    assert len(calls) == 2
    assert json.loads(path.with_name("final_judgment.json").read_text()) == valid
    assert adjudicator.run_final_judge([path]) == 0 and len(calls) == 2
    assert json.loads(path.with_name("usage.json").read_text()) == {"input_tokens": 42}
    # An exact-case rerun has its own destination and preserves the old result.
    original = path.with_name("final_judgment.json").read_bytes()
    adjudicator.main(["--cases", case, "--output-dir", "rerun", "--prepare-only"])
    manifest = json.loads((tmp_path / "rerun/manifest.json").read_text())
    assert manifest["selected_cases"] == [case]
    rerun_package = next((tmp_path / "rerun").rglob("input_package.json"))
    assert json.loads(rerun_package.read_text()) == prepared
    assert not rerun_package.with_name("final_judgment.json").exists()
    assert path.with_name("final_judgment.json").read_bytes() == original
    with pytest.raises(SystemExit):
        adjudicator.main(["--cases", "missing", "--output-dir", "bad-rerun", "--prepare-only"])
    assert not (tmp_path / "bad-rerun").exists()
    # Missing source judges must fail before any adjudication is submitted.
    (tmp_path / adjudicator.RUN_DIR / next(iter(adjudicator.JUDGES)) / case / "judge_eval.json").unlink()
    with pytest.raises(ValueError, match="missing 1"):
        adjudicator.load_results()


def test_custom_package_root_source_run_and_anonymous_models(tmp_path, monkeypatch):
    repo = adjudicator.REPO_ROOT
    monkeypatch.setattr(adjudicator, "REPO_ROOT", tmp_path)
    for relative in (adjudicator.CODEBOOK, adjudicator.PROMPT):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    case = "simpleagent/example_agent/task/trial_001"
    package = tmp_path / "private-inputs" / case / "input_package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"trajectory": []}')
    for judge, code in zip(adjudicator.JUDGES, ["VF-01", "VF-02", "VF-01"]):
        path = tmp_path / "private-judgments" / judge / case / "judge_eval.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"codes": [{"code_id": code, "rationale": "evidence"}],
                                    "flag_for_review": False}))
    args = ["--source-run", "private-judgments", "--input-dir", "private-inputs", "--prepare-only"]
    with pytest.raises(SystemExit):
        adjudicator.main(args)
    assert not (tmp_path / "private-judgments/final_adjudicator").exists()
    adjudicator.main(args + ["--agent-models", "example_agent"])
    root = tmp_path / "private-judgments/final_adjudicator"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["agent_models"] == {"example_agent": "model_1"}
    assert manifest["source_run"] == "private-judgments"
    assert manifest["input_dir"] == "private-inputs"
    prepared = json.loads((root / "task/simpleagent/model_1/trial_001/input_package.json").read_text())
    assert prepared["case_id"] == "simpleagent/model_1/task/trial_001"
    assert prepared["input_package"] == {"trajectory": []}
    adjudicator.main(args + ["--agent-models", "example_agent"])
