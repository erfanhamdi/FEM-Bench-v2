"""Run OpenHands inside Singularity."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from fem_bench.agents.base import AgentResponse, redact_credentials, resolve_agent_model
from fem_bench.agents.openhands_cli import TOKEN_BUDGET_MARKER
from fem_bench.agents.singularity_runtime import (
    singularity_exec_command,
    singularity_subprocess_environment,
    validate_container_image,
)
from llm_api.configured_client import calculate_usage_cost


def _summary_text(value: Any) -> str:
    """Read summaries represented as text, a single block, or a list of blocks."""
    blocks = value if isinstance(value, list) else [value]
    texts = []
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else block
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


class OpenHandsWrapper:
    """Run the configured model and collect its events and usage."""

    def __init__(self, agent_name: str, model: dict, config: dict):
        self.model_config = model
        self.model = resolve_agent_model(model)
        self.config = config
        self.timeout = config.get("timeout_seconds", 600)
        self.runtime_path, self.image_path = validate_container_image(config)

    def _environment(self) -> dict[str, str]:
        environment = {"LLM_MODEL": self.model.name}
        if self.model.api_key:
            environment["LLM_API_KEY"] = self.model.api_key
        elif self.model_config["adapter"] == "aws_bedrock_converse":
            # The CLI requires a key; the SDK uses the AWS credential chain.
            environment["LLM_API_KEY"] = "bedrock-iam"
        if self.model.base_url:
            environment["LLM_BASE_URL"] = self.model.base_url
        if self.model.region:
            for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_REGION_NAME"):
                environment[name] = self.model.region
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            if value := os.environ.get(name):
                environment[name] = value
        environment["FEM_BENCH_OPENHANDS_CONFIG"] = json.dumps(
            {
                "model_canonical_name": (
                    "openai/gpt-5"
                    if self.model_config["adapter"] == "aws_bedrock_mantle"
                    else None
                ),
                "max_output_tokens": self.model.max_output_tokens,
                "temperature": self.model.temperature,
                "reasoning_effort": self.model.reasoning_effort,
                "reasoning_summary": self.model.reasoning_summary,
                "native_reasoning_parameters": self.model.reasoning_parameters,
                "provider_routing": self.model_config.get("provider_routing"),
                "enable_encrypted_reasoning": "reasoning.encrypted_content"
                in (self.model.response_include or []),
                "parallel_tool_calls": self.config.get(
                    "parallel_tool_calls", self.model.parallel_tool_calls
                ),
                "prompt_caching": self.model_config["prompt_caching"],
                # This SDK's feature table predates the configured Bedrock models.
                "force_prompt_caching": self.model_config["adapter"]
                == "aws_bedrock_converse"
                and self.model_config["prompt_caching"]["enabled"],
                "supports_prompt_cache_options": "gpt-5.6"
                in self.model_config["model_id"],
                "request_timeout_seconds": self.config.get(
                    "request_timeout_seconds", 300
                ),
                "model_retry_attempts": 1,
                "tool_choice": "required",
                "step_limit": self.config.get("step_limit", 100),
                "cost_limit": self.config.get("cost_limit"),
                "total_token_budget": self.config["total_token_budget"],
                "network_access": self.config.get("network_access", True),
                "enable_browser": self.config.get("enable_browser", True),
                "enable_subagents": self.config.get("enable_subagents", True),
            }
        )
        environment.update(
            {
                "OPENHANDS_SUPPRESS_BANNER": "1",
                "OPENHANDS_PERSISTENCE_DIR": "/state",
                "OPENHANDS_CONVERSATIONS_DIR": "/state/conversations",
                "OPENHANDS_WORK_DIR": "/workspace",
                "FEM_BENCH_OPENHANDS_METRICS_PATH": "/state/metrics.json",
            }
        )
        return environment

    def _read_metrics(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"usage": {}, "model_calls": 0, "accounting_complete": False}
        metrics = json.loads(path.read_text())
        metrics["usage"] = calculate_usage_cost(metrics["usage"], self.model_config)
        return metrics

    @staticmethod
    def _parse_events(output: str) -> list[dict]:
        events = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def generate_solution(
        self, prompt: str, workspace: Path, *, artifact_dir: Path | None = None
    ) -> AgentResponse:
        inner_command = [
            "python",
            "-m",
            "fem_bench.agents.openhands_cli",
            "--headless",
            "--json",
            "--override-with-envs",
            "--exit-without-confirmation",
            "-t",
            prompt,
        ]
        started = time.monotonic()
        timed_out = False
        with tempfile.TemporaryDirectory(prefix="fem-bench-openhands-") as state_dir:
            command = singularity_exec_command(
                self.runtime_path,
                self.image_path,
                workspace,
                state_dir,
                inner_command,
            )
            try:
                result = subprocess.run(
                    command,
                    cwd=workspace,
                    env=singularity_subprocess_environment(self._environment()),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                output, error_output = result.stdout, result.stderr
                returncode = result.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                output = (exc.stdout or b"").decode(errors="replace")
                error_output = (exc.stderr or b"").decode(errors="replace")
                returncode = -1
            raw = output + error_output
            metrics_path = Path(state_dir) / "metrics.json"
            if artifact_dir is not None:
                # Preserve evidence before parsing or deleting the temporary state.
                artifacts = {
                    "agent_output.txt": raw,
                    "agent_process.json": json.dumps(
                        {"returncode": returncode, "timed_out": timed_out}
                    )
                    + "\n",
                }
                if metrics_path.is_file():
                    artifacts["agent_metrics.json"] = metrics_path.read_text(
                        encoding="utf-8"
                    )
                for name, content in artifacts.items():
                    (artifact_dir / name).write_text(
                        redact_credentials(content, self.model_config), encoding="utf-8"
                    )
            metrics = self._read_metrics(metrics_path)

        events = self._parse_events(output)
        reasoning_records = []
        for index, event in enumerate(events):
            if event.get("kind") != "ActionEvent":
                continue
            item = event.get("responses_reasoning_item")
            if not isinstance(item, dict):
                item = {}
            summary = _summary_text(item.get("summary"))
            reasoning_records.append(
                {
                    "event_index": index,
                    "summary_text": summary,
                    "summary_available": bool(summary),
                    "reasoning_content_available": bool(event.get("reasoning_content")),
                    "thinking_blocks_available": bool(event.get("thinking_blocks")),
                    "encrypted_reasoning_available": bool(
                        item.get("encrypted_content")
                    ),
                }
            )
        conversation_error = next(
            (
                event
                for event in events
                if event.get("kind") == "ConversationErrorEvent"
            ),
            None,
        )
        reason, error = "submitted", None
        if timed_out:
            reason, error = (
                "wall_time_limit",
                f"OpenHands timed out after {self.timeout} seconds",
            )
        elif TOKEN_BUDGET_MARKER in raw:
            reason, error = "token_budget", "OpenHands reached the token limit"
        elif conversation_error:
            reason, error = "agent_error", str(
                conversation_error.get("detail") or conversation_error.get("code")
            )
        elif returncode:
            reason, error = "agent_error", f"OpenHands exited with status {returncode}"
        return AgentResponse(
            success=reason == "submitted",
            raw_response=raw,
            error=error,
            usage=metrics["usage"],
            model_calls=metrics["model_calls"],
            code_executions=sum(
                event.get("kind") == "ActionEvent"
                and event.get("tool_name") == "terminal"
                and isinstance(event.get("action"), dict)
                and event["action"].get("is_input") is not True
                for event in events
            ),
            trajectory={"events": events, "reasoning_records": reasoning_records},
            termination_reason=reason,
            effective_parameters={
                "latency_seconds": time.monotonic() - started,
                "usage_accounting_complete": metrics["accounting_complete"],
                "reasoning_effort": self.model.reasoning_effort,
                "reasoning_summary": self.model.reasoning_summary,
                "native_reasoning_parameters": self.model.reasoning_parameters,
            },
        )
