"""Read and generate the benchmark task catalog."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from fem_bench.version import SCHEMA_VERSION


SPLIT_DIRECTORIES = {"legacy": "tasks", "heldout": "new_tasks"}


class ManifestError(ValueError):
    """A requested task or benchmark split does not exist."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def generate_manifest(repo_root: Path | str) -> dict[str, Any]:
    tasks = [
        {"task_id": path.stem, "split": split}
        for split, directory in SPLIT_DIRECTORIES.items()
        for path in sorted((Path(repo_root) / directory).glob("*.py"))
    ]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ManifestError("Task IDs must be unique across benchmark splits")
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": "2.0",
        "tasks": sorted(tasks, key=lambda task: task["task_id"]),
    }


def load_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_entries(
    manifest: dict[str, Any], split: str = "all", task_ids: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    if split not in {"all", *SPLIT_DIRECTORIES}:
        raise ManifestError(f"Unknown benchmark split: {split!r}")
    eligible = [
        task for task in manifest["tasks"] if split == "all" or task["split"] == split
    ]
    by_id = {task["task_id"]: task for task in eligible}
    requested = list(by_id) if task_ids is None else list(task_ids)
    if len(requested) != len(set(requested)) or len(by_id) != len(eligible):
        raise ManifestError("Task IDs must be unique")
    for task_id in requested:
        if task_id not in by_id:
            raise ManifestError(f"Task {task_id!r} does not exist in split {split!r}")
        if re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", task_id) is None:
            raise ManifestError(f"Invalid task ID: {task_id!r}")
        if by_id[task_id]["split"] not in SPLIT_DIRECTORIES:
            raise ManifestError(f"Invalid task split: {by_id[task_id]['split']!r}")
    return [by_id[task_id] for task_id in requested]


def task_source_path(repo_root: Path | str, task_entry: dict[str, Any]) -> Path:
    return (
        Path(repo_root)
        / SPLIT_DIRECTORIES[task_entry["split"]]
        / f"{task_entry['task_id']}.py"
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Regenerate the benchmark task catalog."
    )
    parser.add_argument(
        "--output", type=Path, default=repo_root / "benchmark_manifest.json"
    )
    args = parser.parse_args()
    manifest = generate_manifest(repo_root)
    args.output.write_bytes(canonical_json_bytes(manifest))
    print(f"Wrote {len(manifest['tasks'])} tasks to {args.output}")
