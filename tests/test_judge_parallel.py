"""Check disjoint assignments and safe resumption without scheduler/API calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import submit_judge_parallel as parallel


@pytest.mark.parametrize("use_defaults", [False, True])
def test_parallel_submission_resumes_and_partitions(tmp_path, monkeypatch, use_defaults):
    monkeypatch.setattr(parallel, "__file__", str(tmp_path / "submit_judge_parallel.py"))
    monkeypatch.setattr(parallel.run_judge, "__file__", str(tmp_path / "run_judge.py"))
    packages = []
    for index in range(7):
        folder = tmp_path / f"inputs/openhands/agent/task/trial_{index:03d}"
        folder.mkdir(parents=True)
        (folder / "input_package.json").write_text("{}")
        packages.append(folder)
    root = tmp_path / "judge_runs/test/v5"
    for index, content in [(0, '{"codes": [], "flag_for_review": false}'), (1, '{}')]:
        path = root / "judge" / packages[index].relative_to(tmp_path / "inputs") / "judge_eval.json"
        path.parent.mkdir(parents=True)
        path.write_text(content)
    codebook_path = tmp_path / (parallel.run_judge.DEFAULT_CODEBOOK if use_defaults else "book.csv")
    prompt_path = tmp_path / (parallel.run_judge.DEFAULT_SYSTEM_PROMPT if use_defaults else "prompt.txt")
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook_path.write_text("Code ID,Code\nVF-01,Example\n")
    prompt_path.write_text("Judge using {CODEBOOK}")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.json").write_text(json.dumps({"models": {
        "judge": {"provider": "test", "request": {"generation": {"max_output_tokens": 100}}}
    }}))
    config = {"experiment": {"id": "test"}, "codebook_version": "v5",
              "models": ["judge"], "generation": {"max_output_tokens": "provider_max"},
              "input_dir": "inputs"}
    if not use_defaults:
        config.update(codebook="book.csv", system_prompt="prompt.txt")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    for jobs in (1, 4, 20):
        shards = parallel.split_pending(tmp_path, config, jobs)
        assigned = [folder for shard in shards for folder in shard]
        assert sorted(assigned) == packages[1:]  # Retry malformed result, skip valid result.
        assert len(assigned) == len(set(assigned))
        assert all(shards) and len(shards) <= jobs
    selected = {**config, "trial_dirs": [str(packages[0]), str(packages[3]), str(packages[3])]}
    assert parallel.split_pending(tmp_path, selected, 4) == [[packages[3]]]
    with pytest.raises(ValueError):
        parallel.split_pending(tmp_path, config, 0)

    calls = []
    def submit(command, **kwargs):
        worker = yaml.safe_load(Path(command[-1]).read_text())
        calls.append(worker)
        return SimpleNamespace(stdout=str(100 + len(calls)))
    monkeypatch.setattr(parallel.subprocess, "run", submit)
    parallel.main(["--config", str(path), "--jobs", "4", "--dry-run"])
    assert not calls and not (root / "parallel_submissions").exists()
    parallel.main(["--config", str(path), "--jobs", "4", "--no-dry-run"])
    assert len(calls) == 4
    assert sorted(folder for worker in calls for folder in worker["trial_dirs"]) == list(map(str, packages[1:]))
    for worker in calls:
        assert worker["models"] == ["judge"] and worker["experiment"]["id"] == "test"
        assert json.loads(Path(worker["codebook"]).read_text())[0]["Code ID"] == "VF-01"
        assert Path(worker["system_prompt"]).read_text() == "Judge using {CODEBOOK}"
    records = next((root / "parallel_submissions").glob("*/jobs.jsonl"))
    assert [json.loads(line)["job_id"] for line in records.read_text().splitlines()] == ["101", "102", "103", "104"]
