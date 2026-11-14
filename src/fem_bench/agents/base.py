"""Model settings and results for OpenHands."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def redact_credentials(text: str, model: dict[str, Any]) -> str:
    """Remove configured provider credentials from saved agent artifacts."""
    names = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        model.get("api_key_env"),
    }
    for name in names:
        if name and (secret := os.environ.get(name)):
            text = text.replace(secret, "[REDACTED]")
    return text


@dataclass(frozen=True)
class AgentModel:
    """Model settings used by OpenHands wrappers."""

    name: str
    api_key: str | None
    base_url: str | None
    max_output_tokens: int | None
    temperature: float | None
    reasoning_effort: str | None
    reasoning_summary: str | None
    reasoning_parameters: dict[str, Any]
    response_include: list[str] | None
    parallel_tool_calls: bool | None
    thinking: dict[str, Any] | None
    region: str | None


@dataclass
class AgentResponse:
    """Common result returned by every OpenHands wrapper."""

    success: bool
    raw_response: str = ""
    error: str | None = None
    usage: dict[str, int | float | None] = field(default_factory=dict)
    trajectory: Any = None
    model_calls: int = 0
    code_executions: int = 0
    termination_reason: str = "agent_error"
    effective_parameters: dict[str, Any] = field(default_factory=dict)


def resolve_agent_model(model: dict[str, Any]) -> AgentModel:
    """Translate one catalog entry to provider-neutral agent settings."""

    adapter = model["adapter"]
    model_id = model["model_id"]
    if adapter == "aws_bedrock_converse":
        name = f"bedrock/{model_id}"
    elif adapter == "aws_bedrock_mantle":
        name = f"openai/{model_id}"
    else:
        prefixes = {
            "openrouter_chat_completions": "openrouter",
            "aws_bedrock_chat_completions": "openai",
            "openai_responses": "openai",
            "openai_chat_completions": "openai",
            "anthropic_messages": "anthropic",
            "google_genai": "gemini",
            "deepseek_chat_completions": "openai",
            "together_openai_compatible": "openai",
        }
        prefix = prefixes.get(adapter)
        name = f"{prefix}/{model_id}" if prefix else model_id

    api_key_env = model.get("api_key_env")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    generation = model.get("request", {}).get("generation", {})
    reasoning = model.get("request", {}).get("reasoning", {})
    parameters = reasoning.get("parameters", {}) if reasoning.get("enabled") else {}
    output_config = parameters.get("output_config", {})
    reasoning_config = parameters.get("reasoning", {})
    thinking_config = parameters.get("thinking_config", {})
    effort = parameters.get("reasoning_effort")
    if effort is None and isinstance(output_config, dict):
        effort = output_config.get("effort")
    if effort is None and isinstance(reasoning_config, dict):
        effort = reasoning_config.get("effort")
    if effort is None and isinstance(thinking_config, dict):
        effort = thinking_config.get("thinking_level")

    summary = None
    if isinstance(reasoning_config, dict):
        summary = reasoning_config.get("summary") or reasoning_config.get(
            "generate_summary"
        )

    max_tokens = generation.get("max_output_tokens", generation.get("max_tokens"))
    return AgentModel(
        name=name,
        api_key=api_key,
        base_url=model.get("base_url"),
        max_output_tokens=max_tokens,
        temperature=generation.get("temperature"),
        reasoning_effort=effort,
        reasoning_summary=summary,
        reasoning_parameters=parameters,
        response_include=parameters.get("include"),
        parallel_tool_calls=generation.get("parallel_tool_calls"),
        thinking=parameters.get("thinking"),
        region=model.get("region"),
    )
