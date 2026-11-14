"""Exercise subset submission without contacting the scheduler or providers."""

import json
from pathlib import Path

import pytest
import yaml

import submit_judge_subset as subset


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(subset, "__file__", str(tmp_path / "submit_judge_subset.py"))
    # Tests use fixed defaults, independent of the user's editable settings.
    monkeypatch.setattr(subset, "SETTINGS", {
        **subset.SETTINGS, "models": None, "mode": "qsub", "dry_run": False,
        "codebook_version": "rejudge", "input_dir": "llm-as-judge", "tasks": [], "trial_dirs": [],
    })
    source = tmp_path / "judge_runs/previous/v1/source_judge"
    cases = ["openhands/agent/task_a/trial_001", "openhands/agent/task_a/trial_002",
             "simpleagent/agent/task_b/trial_003"]
    for case, codes in zip(cases, [["VF-01", "VF-02"], ["VF-02"], ["VF-01"]]):
        folder = source / case
        folder.mkdir(parents=True)
        (folder / "judge_eval.json").write_text(json.dumps({
            "codes": [{"code_id": code, "rationale": "Evidence"} for code in codes],
            "flag_for_review": False,
        }))
        package = tmp_path / "llm-as-judge" / case
        package.mkdir(parents=True)
        (package / "input_package.json").write_text("{}")
    (tmp_path / "book with spaces.csv").write_text("Code ID,Code\nNEW,Revised category\n")
    (tmp_path / "prompt.txt").write_text("Rejudge with {CODEBOOK}")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.json").write_text(json.dumps({"models": {
        key: {"provider": "example", "request": {"generation": {"max_output_tokens": 100}}}
        for key in ("source_judge", "other_judge")
    }}))
    calls = []
    monkeypatch.setattr(subset.shutil, "which", lambda name: "/usr/bin/qsub")
    monkeypatch.setattr(subset.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    args = ["--source-run", "judge_runs/previous/v1", "--source-judge", "source_judge",
            "--code", "vf-01", "--codebook", "book with spaces.csv", "--system-prompt", "prompt.txt",
            "--run-id", "fresh"]
    return args, cases, source, calls


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("override_models", [False, True])
@pytest.mark.parametrize("mode", ["qsub", "terminal"])
def test_subset_submission(selection, tmp_path, monkeypatch, capsys, dry_run, override_models, mode):
    args, cases, source, calls = selection
    args += ["--mode", mode]
    if mode == "terminal":
        monkeypatch.setattr(subset.shutil, "which", lambda name: None)
    if override_models:
        args += ["--models", "source_judge", "other_judge"]
    if dry_run:
        args += ["--dry-run"]
    originals = {p: p.read_bytes() for p in source.rglob("judge_eval.json")}
    subset.main(args)
    output = capsys.readouterr().out
    assert "Selected 2 of 3 judgments containing VF-01" in output
    assert cases[0] in output and cases[2] in output and cases[1] not in output
    assert all(p.read_bytes() == content for p, content in originals.items())
    destination = tmp_path / "judge_runs/fresh/rejudge"
    if dry_run:
        assert not destination.exists()
        assert not calls
        return
    config_path = destination / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["models"] == (["source_judge", "other_judge"] if override_models else ["source_judge"])
    assert config["trial_dirs"] == [str(tmp_path / "llm-as-judge" / case) for case in (cases[0], cases[2])]
    assert config["selection"] == {"source_judge_dir": str(source), "code": "VF-01"}
    (tmp_path / "book with spaces.csv").write_text("changed after submission")
    (tmp_path / "prompt.txt").write_text("changed after submission")
    assert json.loads(Path(config["codebook"]).read_text()) == [{"Code ID": "NEW", "Code": "Revised category"}]
    assert Path(config["system_prompt"]).read_text() == "Rejudge with {CODEBOOK}"
    if mode == "qsub":
        command = ["qsub", "-wd", str(tmp_path), "-o", str(destination / "logs"),
                   str(tmp_path / "submit_judge.sh"), "--config", str(config_path)]
    else:
        command = [subset.sys.executable, "-u", str(tmp_path / "run_judge.py"), "--config", str(config_path)]
        assert not (destination / "logs").exists()
    assert calls == [((command,), {"cwd": tmp_path, "check": True})]
    # Confirm that the existing runner accepts the saved config and snapshot files.
    import run_judge
    monkeypatch.setattr(run_judge, "__file__", str(tmp_path / "run_judge.py"))
    run_judge.main(["--config", str(config_path), "--dry-run"])
    with pytest.raises(SystemExit) as error:
        subset.main(args)
    assert error.value.code == 2
    assert len(calls) == 1


@pytest.mark.parametrize("problem", ["no_match", "missing_judge", "malformed", "missing_package",
                                    "bad_prompt", "bad_model", "bad_mode", "no_scheduler"])
def test_subset_invalid_selection_does_not_submit(selection, tmp_path, monkeypatch, problem):
    args, cases, source, calls = selection
    if problem == "no_match":
        args[args.index("--code") + 1] = "NV"
    elif problem == "missing_judge":
        args[args.index("--source-judge") + 1] = "missing"
    elif problem == "malformed":
        (source / cases[0] / "judge_eval.json").write_text("{broken")
    elif problem == "missing_package":
        (tmp_path / "llm-as-judge" / cases[0] / "input_package.json").unlink()
    elif problem == "bad_prompt":
        (tmp_path / "prompt.txt").write_text("No codebook placeholder")
    elif problem == "bad_model":
        args += ["--models", "typo"]
    elif problem == "bad_mode":
        monkeypatch.setitem(subset.SETTINGS, "mode", "typo")
    else:
        monkeypatch.setattr(subset.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as error:
        subset.main(args)
    assert error.value.code == 2
    assert not calls
    assert not (tmp_path / "judge_runs/fresh").exists()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("mode", ["qsub", "terminal"])
def test_editable_settings_without_arguments(selection, tmp_path, monkeypatch, dry_run, mode):
    args, cases, source, calls = selection
    settings = dict(subset.SETTINGS)
    settings.update({args[i][2:].replace("-", "_"): args[i + 1] for i in range(0, len(args), 2)})
    settings.update(models=["other_judge"], dry_run=dry_run, mode=mode)
    monkeypatch.setattr(subset, "SETTINGS", settings)
    subset.main([])
    if dry_run:
        assert not calls
        assert not (tmp_path / "judge_runs/fresh").exists()
    else:
        assert len(calls) == 1
        assert calls[0][0][0][0] == ("qsub" if mode == "qsub" else subset.sys.executable)
        config = yaml.safe_load((tmp_path / "judge_runs/fresh/rejudge/config.yaml").read_text())
        assert config["models"] == ["other_judge"]
        assert config["selection"]["code"] == "VF-01"
        assert len(config["trial_dirs"]) == 2


@pytest.mark.parametrize("mode", ["qsub", "terminal"])
@pytest.mark.parametrize("use_cli", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_explicit_tasks_ignore_previous_labels(selection, tmp_path, monkeypatch, capsys, mode, use_cli, dry_run):
    args, cases, source, calls = selection
    # Include a case with no previous judgment, from another harness/model.
    new_case = "simpleagent/another_agent/task_a/trial_004"
    folder = tmp_path / "llm-as-judge" / new_case
    folder.mkdir(parents=True)
    (folder / "input_package.json").write_text("{}")
    for path in source.rglob("judge_eval.json"):
        path.unlink()
    settings = {**subset.SETTINGS,
                **{args[i][2:].replace("-", "_"): args[i + 1] for i in range(0, len(args), 2)},
                "models": ["other_judge"], "source_run": None, "source_judge": None,
                "code": None, "tasks": [] if use_cli else ["task_a", "task_a"],
                "mode": mode, "dry_run": dry_run}
    monkeypatch.setattr(subset, "SETTINGS", settings)
    subset.main(["--tasks", "task_a", "task_a"] if use_cli else [])
    output = capsys.readouterr().out
    assert "Selected 3 prepared cases for 1 tasks." in output
    assert cases[2] not in output
    destination = tmp_path / "judge_runs/fresh/rejudge"
    if dry_run:
        assert not calls
        assert not destination.exists()
    else:
        assert len(calls) == 1
        assert calls[0][0][0][0] == ("qsub" if mode == "qsub" else subset.sys.executable)
        config = yaml.safe_load((destination / "config.yaml").read_text())
        assert config["selection"] == {"tasks": ["task_a"]}
        assert set(config["trial_dirs"]) == {
            str(tmp_path / "llm-as-judge" / case) for case in [*cases[:2], new_case]
        }
        assert len(config["trial_dirs"]) == 3


@pytest.mark.parametrize("tasks", [["task_a", "typo"], [""], "task_a"])
def test_invalid_tasks_do_not_submit(selection, tmp_path, monkeypatch, tasks):
    args, cases, source, calls = selection
    monkeypatch.setitem(subset.SETTINGS, "tasks", tasks)
    with pytest.raises(SystemExit) as error:
        subset.main(args)
    assert error.value.code == 2
    assert not calls
    assert not (tmp_path / "judge_runs/fresh").exists()


@pytest.mark.parametrize("mode", ["qsub", "terminal"])
@pytest.mark.parametrize("use_cli", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_exact_case_addresses(selection, tmp_path, monkeypatch, capsys, mode, use_cli, dry_run):
    args, cases, source, calls = selection
    folder = tmp_path / "llm-as-judge" / cases[0]
    addresses = [f"llm-as-judge/{cases[0]}", str(folder), str(folder / "input_package.json"),
                 cases[0], f"{cases[0]}/input_package.json"]
    settings = {**subset.SETTINGS,
                **{args[i][2:].replace("-", "_"): args[i + 1] for i in range(0, len(args), 2)},
                "models": ["other_judge"], "source_run": None, "source_judge": None,
                "code": None, "trial_dirs": [] if use_cli else addresses,
                "mode": mode, "dry_run": dry_run}
    monkeypatch.setattr(subset, "SETTINGS", settings)
    subset.main(["--trial-dirs", *addresses] if use_cli else [])
    output = capsys.readouterr().out
    assert "Selected 1 exact prepared cases." in output
    assert cases[1] not in output and cases[2] not in output
    destination = tmp_path / "judge_runs/fresh/rejudge"
    if dry_run:
        assert not calls
        assert not destination.exists()
    else:
        assert len(calls) == 1
        assert calls[0][0][0][0] == ("qsub" if mode == "qsub" else subset.sys.executable)
        config = yaml.safe_load((destination / "config.yaml").read_text())
        assert config["trial_dirs"] == [str(folder)]
        assert config["selection"] == {"trial_dirs": [str(folder)]}


@pytest.mark.parametrize("problem", ["missing", "outside", "task_folder", "conflict", "not_list"])
def test_invalid_exact_case_addresses(selection, tmp_path, monkeypatch, problem):
    args, cases, source, calls = selection
    address = f"llm-as-judge/{cases[0]}"
    if problem == "missing":
        address = address.replace("trial_001", "trial_999")
    elif problem == "outside":
        address = str(tmp_path / "outside/openhands/agent/task_a/trial_001")
    elif problem == "task_folder":
        address = str(Path(address).parent)
    elif problem == "conflict":
        monkeypatch.setitem(subset.SETTINGS, "tasks", ["task_a"])
    monkeypatch.setitem(subset.SETTINGS, "trial_dirs", address if problem == "not_list" else [address])
    with pytest.raises(SystemExit) as error:
        subset.main(args)
    assert error.value.code == 2
    assert not calls
    assert not (tmp_path / "judge_runs/fresh").exists()


@pytest.mark.parametrize("prefix", ["input_root", "repository", "absolute"])
@pytest.mark.parametrize("suffix", ["", "/input_package.json"])
def test_address_without_trial_resolves_only_prepared_trial(selection, tmp_path, prefix, suffix):
    args, cases, source, calls = selection
    task = Path(cases[2]).parent
    if prefix == "repository":
        task = Path("llm-as-judge") / task
    elif prefix == "absolute":
        task = tmp_path / "llm-as-judge" / task
    subset.main([*args, "--trial-dirs", str(task) + suffix])
    config = yaml.safe_load((tmp_path / "judge_runs/fresh/rejudge/config.yaml").read_text())
    assert config["trial_dirs"] == [str(tmp_path / "llm-as-judge" / cases[2])]
    assert len(calls) == 1


def test_short_address_with_multiple_trials_requires_explicit_trial(selection, tmp_path, capsys):
    args, cases, source, calls = selection
    address = str(Path(cases[0]).parent / "input_package.json")
    with pytest.raises(SystemExit) as error:
        subset.main([*args, "--trial-dirs", address])
    assert error.value.code == 2
    assert "Found 2 prepared trials" in capsys.readouterr().err
    assert not calls
    assert not (tmp_path / "judge_runs/fresh").exists()
