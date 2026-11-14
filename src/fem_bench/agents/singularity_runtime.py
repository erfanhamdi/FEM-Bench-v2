"""Run OpenHands in an isolated Singularity container."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

HOST_RESOLVER_PATH = Path("/etc/resolv.conf")


class OpenHandsContainerError(RuntimeError):
    """The configured OpenHands container cannot be used safely."""


def resolve_image_metadata(
    image: str, repo_root: Path | str, *, runtime: str = "singularity"
) -> dict[str, str]:
    image_path = (Path(repo_root) / image).resolve()
    if runtime != "singularity":
        raise OpenHandsContainerError("OpenHands requires runtime: singularity")
    if not image_path.is_file():
        raise OpenHandsContainerError(f"OpenHands image does not exist: {image_path}")
    return {"runtime": runtime, "image": str(image_path)}


def validate_container_image(metadata: Mapping[str, Any]) -> tuple[str, Path]:
    if metadata["runtime"] != "singularity":
        raise OpenHandsContainerError("OpenHands requires runtime: singularity")
    runtime_path = shutil.which("singularity")
    if runtime_path is None:
        raise OpenHandsContainerError("OpenHands requires the singularity executable")
    image_path = Path(metadata["image"])
    if not image_path.is_absolute() or not image_path.is_file():
        raise OpenHandsContainerError(
            f"OpenHands image must be an existing absolute path: {image_path}"
        )
    return runtime_path, image_path


def singularity_exec_command(
    runtime_path: str,
    image_path: Path | str,
    workspace: Path | str,
    state: Path | str,
    inner_command: Sequence[str],
) -> list[str]:
    """Build the only permitted OpenHands container invocation."""

    resolver_path = prepare_resolver_config(state)

    return [
        runtime_path,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "hostfs,bind-paths,home,cwd",
        "--bind",
        f"{Path(workspace).resolve()}:/workspace:rw",
        "--bind",
        f"{Path(state).resolve()}:/state:rw",
        "--bind",
        f"{resolver_path}:/etc/resolv.conf:ro",
        "--pwd",
        "/workspace",
        str(Path(image_path).resolve()),
        *inner_command,
    ]


def prepare_resolver_config(state: Path | str) -> Path:
    """Restore DNS inside the container without exposing host directories."""
    resolver_path = Path(state).resolve() / "resolv.conf"
    shutil.copyfile(HOST_RESOLVER_PATH, resolver_path)
    resolver_path.chmod(0o600)
    return resolver_path


_CONTAINER_ENV_ALLOWLIST = frozenset(
    {
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_REGION_NAME",
        "FEM_BENCH_OPENHANDS_CONFIG",
        "FEM_BENCH_OPENHANDS_METRICS_PATH",
        "OPENHANDS_SUPPRESS_BANNER",
        "OPENHANDS_PERSISTENCE_DIR",
        "OPENHANDS_CONVERSATIONS_DIR",
        "OPENHANDS_WORK_DIR",
    }
)


def singularity_subprocess_environment(
    container_environment: Mapping[str, str],
) -> dict[str, str]:
    """Encode an exact environment allowlist for ``singularity --cleanenv``."""

    unknown = sorted(set(container_environment) - _CONTAINER_ENV_ALLOWLIST)
    if unknown:
        raise OpenHandsContainerError(
            f"Refusing to pass unapproved variables to OpenHands: {unknown}"
        )
    return {
        f"SINGULARITYENV_{name}": str(value)
        for name, value in container_environment.items()
    }
