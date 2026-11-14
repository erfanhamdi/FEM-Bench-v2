"""Load experiment settings and preserve each run's inputs."""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fem_bench.benchmark_manifest import (
    canonical_json_bytes,
    load_manifest,
    task_entries,
    task_source_path,
)
from fem_bench.version import SCHEMA_VERSION


TRACKS = {"implementation", "test_generation"}
HARNESSES = {"onepass", "simpleagent", "multiagent", "openhands"}
PROVIDER_MAX_OUTPUT_TOKENS = "provider_max"


class ConfigError(ValueError):
    """The experiment settings do not describe a runnable condition."""


@dataclass(frozen=True)
class ResolvedExperiment:
    source_path: Path
    config: dict[str, Any]
    tasks: list[dict[str, Any]]
    trial_numbers: list[int]
    models_snapshot: dict[str, Any]
    effective_generation_by_model: dict[str, dict[str, int]]
    prompt_manifest: dict[str, Any]
    prompt_manifest_path: Path
    prompt_files: dict[str, bytes]
    openhands_container: dict[str, str] | None = None


@dataclass(frozen=True)
class ExperimentRun:
    run_dir: Path
    config: dict[str, Any]
    resolved: dict[str, Any]
    manifest: dict[str, Any]
    models_snapshot: dict[str, Any]
    prompt_manifest: dict[str, Any]
    metadata: dict[str, Any]


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_models_catalog(path: Path | str) -> dict[str, Any]:
    return load_json(path)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _path_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", value) is None:
        raise ConfigError(f"Invalid path identifier: {value!r}")
    return value


def resolve_models(catalog: dict[str, Any], selected_keys: list[str]) -> dict[str, Any]:
    if not selected_keys or len(selected_keys) != len(set(selected_keys)):
        raise ConfigError("models must contain unique model aliases")
    models = {}
    for key in selected_keys:
        _path_identifier(key)
        model = catalog["models"][key]
        defaults = catalog.get("provider_defaults", {}).get(model["provider"], {})
        models[key] = _deep_merge(defaults, model)
        models[key]["model_key"] = key
    return {"schema_version": SCHEMA_VERSION, "models": models}


def resolve_generation_by_model(
    generation: dict[str, Any], models_snapshot: dict[str, Any]
) -> dict[str, dict[str, int]]:
    requested = generation["max_output_tokens"]
    limits = {}
    for key, model in models_snapshot["models"].items():
        ceiling = model["request"]["generation"]["max_output_tokens"]
        limit = ceiling if requested == PROVIDER_MAX_OUTPUT_TOKENS else requested
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 0 < limit <= ceiling
        ):
            raise ConfigError(
                f"max_output_tokens must be between 1 and {ceiling} for {key}"
            )
        limits[key] = {"max_output_tokens": limit}
    return limits


def ensure_model_is_runnable(model: dict[str, Any]) -> None:
    if model.get("metadata", {}).get("placeholder") or "REPLACE_" in model["model_id"]:
        raise ConfigError(f"Fill in the model deployment for {model['model_key']!r}")


def load_experiment_yaml(path: Path | str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def expand_trial_numbers(trials: dict[str, Any]) -> list[int]:
    if ("count" in trials) == ("numbers" in trials):
        raise ConfigError("trials must contain either count or numbers")
    numbers = (
        list(range(1, trials["count"] + 1)) if "count" in trials else trials["numbers"]
    )
    if (
        not numbers
        or any(
            not isinstance(number, int) or isinstance(number, bool) or number < 1
            for number in numbers
        )
        or len(numbers) != len(set(numbers))
    ):
        raise ConfigError("Trial numbers must be unique positive integers")
    return numbers


def _verify_prompt_identity(
    manifest: dict[str, Any], prompt_set_id: str, harness: str, track: str
) -> None:
    expected = {"prompt_set_id": prompt_set_id, "harness": harness, "track": track}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ConfigError(
            f"Prompt set does not match the experiment: expected {expected}"
        )
    if harness == "onepass" and prompt_set_id != f"onepass_{track}_v1":
        raise ConfigError(f"{track} requires prompt_set: onepass_{track}_v1")


def load_prompt_set(
    repo_root: Path | str, prompt_set_id: str, harness: str, track: str
) -> tuple[dict[str, Any], Path, dict[str, bytes]]:
    root = Path(repo_root).resolve()
    manifest_path = (
        root / "prompt_sets" / _path_identifier(prompt_set_id) / "prompt_manifest.json"
    )
    manifest = load_json(manifest_path)
    _verify_prompt_identity(manifest, prompt_set_id, harness, track)
    files = {}
    for item in manifest["files"]:
        relative = Path(item["path"])
        source = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not source.is_relative_to(root)
        ):
            raise ConfigError(
                f"Prompt path must stay inside the repository: {relative}"
            )
        files[relative.as_posix()] = source.read_bytes()
    return manifest, manifest_path, files


def resolve_experiment(
    experiment_path: Path | str,
    models_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
) -> ResolvedExperiment:
    source = Path(experiment_path).resolve()
    config = load_experiment_yaml(source)
    if config["track"] not in TRACKS or config["harness"] not in HARNESSES:
        raise ConfigError("Unknown experiment track or harness")
    if config["harness"] != "onepass" and config["track"] != "implementation":
        raise ConfigError("Agent harnesses support the implementation track only")
    if (
        config["harness"] in {"simpleagent", "multiagent"}
        and config["harness_config"].get("network_access") is not False
    ):
        raise ConfigError(f"{config['harness']} requires network_access: false")
    catalog = load_models_catalog(models_path)
    models = resolve_models(catalog, config["models"])
    if config["harness"] == "multiagent":
        # The tester is fixed across the coder matrix; snapshot it fully resolved.
        tester_key = config["harness_config"]["tester_model"]
        tester = resolve_models(catalog, [tester_key])
        tester["models"][tester_key]["request"]["generation"].update(
            resolve_generation_by_model(config["generation"], tester)[tester_key]
        )
        models["tester_model"] = tester["models"][tester_key]
    tasks = task_entries(load_manifest(manifest_path), **config["benchmark"])
    if not tasks:
        raise ConfigError("The experiment must select at least one task")
    prompt_manifest, prompt_path, prompt_files = load_prompt_set(
        repo_root, config["prompt_set"], config["harness"], config["track"]
    )
    container = None
    if config["harness"] == "openhands":
        if config["harness_config"]["agent"] != "openhands":
            raise ConfigError("The openhands harness requires agent: openhands")
        from fem_bench.agents.singularity_runtime import resolve_image_metadata

        agent_config = config["harness_config"]["agent_config"]
        if agent_config.get("network_access", True) is not True:
            raise ConfigError(
                "OpenHands uses the host network; network_access must be true"
            )
        container = resolve_image_metadata(
            agent_config["image"], repo_root, runtime=agent_config["runtime"]
        )
    return ResolvedExperiment(
        source_path=source,
        config=config,
        tasks=tasks,
        trial_numbers=expand_trial_numbers(config["trials"]),
        models_snapshot=models,
        effective_generation_by_model=resolve_generation_by_model(
            config["generation"], models
        ),
        prompt_manifest=prompt_manifest,
        prompt_manifest_path=prompt_path,
        prompt_files=prompt_files,
        openhands_container=container,
    )


def initialize_experiment_run(
    resolved: ResolvedExperiment,
    manifest_path: Path | str,
    output_root: Path | str,
    repo_root: Path | str,
    run_id: str | None = None,
) -> ExperimentRun:
    created_at = datetime.now().astimezone()
    if run_id is None:
        name = re.sub(
            r"[^A-Za-z0-9_-]+", "-", resolved.config["experiment"]["name"]
        ).strip("-")
        run_id = f"{name}_{created_at:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_root).resolve() / _path_identifier(run_id)
    # Never overwrite a previous run, including an incorrectly configured one.
    run_dir.mkdir(parents=True, exist_ok=False)
    config = resolved.config
    harness_config = copy.deepcopy(config.get("harness_config", {}))
    if resolved.openhands_container is not None:
        harness_config["agent_config"].update(resolved.openhands_container)
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_run_id": run_id,
        "experiment_name": config["experiment"]["name"],
        "track": config["track"],
        "harness": config["harness"],
        "model_keys": list(resolved.models_snapshot["models"]),
        "task_ids": [entry["task_id"] for entry in resolved.tasks],
        "trial_numbers": resolved.trial_numbers,
        "prompt_set_id": config["prompt_set"],
        "generation": config["generation"],
        "effective_generation_by_model": resolved.effective_generation_by_model,
        "prompt_caching": config["prompt_caching"],
        "harness_config": harness_config,
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at.isoformat(),
        "experiment_run_id": run_id,
        "task_count": len(resolved.tasks),
        "model_count": len(record["model_keys"]),
        "statistical_trial_count": (
            len(resolved.tasks)
            * len(record["model_keys"])
            * len(resolved.trial_numbers)
        ),
    }
    snapshots = {
        "experiment_config.yaml": yaml.safe_dump(config, sort_keys=False).encode(
            "utf-8"
        ),
        "models_snapshot.json": canonical_json_bytes(resolved.models_snapshot),
        "benchmark_manifest.json": Path(manifest_path).read_bytes(),
        "prompt_manifest.json": canonical_json_bytes(resolved.prompt_manifest),
        "resolved_experiment.json": canonical_json_bytes(record),
        "experiment_metadata.json": canonical_json_bytes(metadata),
    }
    for relative, content in resolved.prompt_files.items():
        snapshots[f"prompt_set/{relative}"] = content
    for task in resolved.tasks:
        relative = f"benchmark_tasks/{task['split']}/{task['task_id']}.py"
        snapshots[relative] = task_source_path(repo_root, task).read_bytes()
    for relative, content in snapshots.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
        path.chmod(0o444)
    (run_dir / "trials").mkdir()
    run = load_experiment_run(run_dir)
    verify_run_identity(run, expected=resolved)
    return run


def load_experiment_run(run_dir: Path | str) -> ExperimentRun:
    directory = Path(run_dir).resolve()
    run = ExperimentRun(
        run_dir=directory,
        config=load_experiment_yaml(directory / "experiment_config.yaml"),
        resolved=load_json(directory / "resolved_experiment.json"),
        manifest=load_manifest(directory / "benchmark_manifest.json"),
        models_snapshot=load_json(directory / "models_snapshot.json"),
        prompt_manifest=load_json(directory / "prompt_manifest.json"),
        metadata=load_json(directory / "experiment_metadata.json"),
    )
    verify_run_identity(run)
    return run


def verify_run_identity(
    run: ExperimentRun,
    *,
    expected: ResolvedExperiment | None = None,
    track: str | None = None,
    harness: str | None = None,
) -> None:
    """Compare the saved condition with its intended task, model, and trial matrix."""
    config = expected.config if expected is not None else run.config
    tasks = (
        expected.tasks
        if expected is not None
        else task_entries(run.manifest, **config["benchmark"])
    )
    identity = {
        "experiment_name": config["experiment"]["name"],
        "track": track or config["track"],
        "harness": harness or config["harness"],
        "model_keys": config["models"],
        "task_ids": [task["task_id"] for task in tasks],
        "trial_numbers": expand_trial_numbers(config["trials"]),
        "prompt_set_id": config["prompt_set"],
    }
    mismatches = {
        key: {"expected": value, "actual": run.resolved.get(key)}
        for key, value in identity.items()
        if run.resolved.get(key) != value
    }
    if mismatches:
        raise ConfigError(f"Resolved experiment identity mismatch: {mismatches}")
    if set(run.models_snapshot["models"]) != set(identity["model_keys"]):
        raise ConfigError("Resolved model keys differ from the models snapshot")
    _verify_prompt_identity(
        run.prompt_manifest,
        identity["prompt_set_id"],
        identity["harness"],
        identity["track"],
    )
