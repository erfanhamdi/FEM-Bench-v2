"""Model requests and response parsing for the configured provider transports."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Existing environment values take precedence over repository credentials.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@dataclass(frozen=True)
class ConfiguredResponse:
    text: str
    stop_reason: str | None
    usage: dict[str, int | float | None]
    raw_usage: dict[str, Any] | None
    effective_parameters: dict[str, Any]
    raw_response: dict[str, Any]
    reasoning_summary: str = ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text", ""), str)
        )
    raise ValueError(f"Unsupported message content: {type(content).__name__}")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {
            "user",
            "assistant",
        }:
            raise ValueError(f"Invalid message at index {index}")
        normalized.append(
            {"role": message["role"], "content": _content_text(message.get("content"))}
        )
    return normalized


def _request_parts(model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = model.get("request", {})
    reasoning = request.get("reasoning", {})
    reasoning_parameters = dict(reasoning.get("parameters", {}))
    generation = {
        key: value
        for key, value in request.get("generation", {}).items()
        if value is not None
    }
    return reasoning_parameters, generation


def _merge_stop_sequences(
    generation: dict[str, Any], stop_sequences: list[str] | None
) -> dict[str, Any]:
    merged = dict(generation)
    if stop_sequences is not None:
        merged["stop_sequences"] = list(stop_sequences)
    return merged


def _prompt_caching_enabled(model: dict[str, Any]) -> bool | None:
    policy = model.get("prompt_caching")
    if policy is None:
        return None
    if not isinstance(policy, dict) or not isinstance(policy.get("enabled"), bool):
        raise ValueError("prompt_caching.enabled must be boolean")
    return policy["enabled"]


def _apply_bedrock_cache_checkpoint(
    request: dict[str, Any], model: dict[str, Any]
) -> None:
    """Add one cache checkpoint at the end of an enabled conversation."""
    if _prompt_caching_enabled(model) is not True or not request["messages"]:
        return
    request["messages"][-1]["content"].append(
        {"cachePoint": {"type": "default", "ttl": "5m"}}
    )


def _record_prompt_caching(
    effective: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    enabled = _prompt_caching_enabled(model)
    if enabled is not None:
        effective["prompt_caching"] = {"enabled": enabled}
    return effective


def build_bedrock_converse_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a Bedrock Converse request without performing network I/O."""
    reasoning, generation = _request_parts(model)
    generation = _merge_stop_sequences(generation, stop_sequences)
    inference: dict[str, Any] = {}
    key_map = {
        "max_output_tokens": "maxTokens",
        "max_tokens": "maxTokens",
        "temperature": "temperature",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    unknown = sorted(set(generation) - set(key_map))
    if unknown:
        raise ValueError(
            f"Unsupported aws_bedrock_converse generation parameters: {unknown}"
        )
    for source, destination in key_map.items():
        if source in generation:
            if destination in inference:
                raise ValueError("Configure only one Bedrock maximum-token field")
            inference[destination] = generation[source]
    request: dict[str, Any] = {
        "modelId": model["model_id"],
        "messages": [
            {"role": item["role"], "content": [{"text": item["content"]}]}
            for item in normalize_messages(messages)
        ],
    }
    if system_prompt is not None:
        request["system"] = [{"text": system_prompt}]
    if inference:
        request["inferenceConfig"] = inference
    if reasoning:
        request["additionalModelRequestFields"] = reasoning
    _apply_bedrock_cache_checkpoint(request, model)
    effective = {
        key: value
        for key, value in (
            ("inferenceConfig", inference or None),
            ("additionalModelRequestFields", reasoning or None),
        )
        if value is not None
    }
    return request, _record_prompt_caching(effective, model)


def build_openai_compatible_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build Chat-Completions parameters for the configured adapter."""
    reasoning, generation = _request_parts(model)
    generation = _merge_stop_sequences(generation, stop_sequences)
    adapter = model["adapter"]
    client_stop_sequences = None
    capabilities = model.get("capabilities", {})
    reasoning_enabled = (
        model.get("request", {}).get("reasoning", {}).get("enabled") is True
    )
    if capabilities.get("server_stop_sequences") is False or (
        reasoning_enabled
        and capabilities.get("server_stop_sequences_with_reasoning") is False
    ):
        client_stop_sequences = generation.pop("stop_sequences", None)
    parameters: dict[str, Any] = {}
    for key, value in generation.items():
        if key == "max_output_tokens":
            destination = (
                "max_completion_tokens"
                if adapter == "openai_chat_completions"
                else "max_tokens"
            )
        elif key == "stop_sequences":
            destination = "stop"
        else:
            destination = key
        if destination in parameters:
            raise ValueError(
                f"Generation fields resolve to duplicate parameter {destination!r}"
            )
        parameters[destination] = value
    overlap = set(parameters) & set(reasoning)
    if overlap:
        raise ValueError(f"Reasoning/generation parameter collision: {sorted(overlap)}")
    parameters.update(reasoning)
    normalized = normalize_messages(messages)
    if adapter == "aws_bedrock_chat_completions":
        # Bedrock rejects this tool-only option on ordinary chat requests.
        parameters.pop("parallel_tool_calls", None)
    if system_prompt is not None:
        normalized.insert(0, {"role": "system", "content": system_prompt})
    request = {"model": model["model_id"], "messages": normalized, **parameters}
    effective = dict(parameters)
    if adapter == "openrouter_chat_completions":
        routing = dict(model["provider_routing"])
        request["extra_body"] = {"provider": routing}
        effective["provider_routing"] = routing
    if client_stop_sequences is not None:
        effective["client_stop_sequences"] = list(client_stop_sequences)
    return request, _record_prompt_caching(effective, model)


def build_bedrock_mantle_responses_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an OpenAI Responses API request for OpenAI or Bedrock Mantle."""
    reasoning, generation = _request_parts(model)
    generation = _merge_stop_sequences(generation, stop_sequences)
    client_stop_sequences = generation.pop("stop_sequences", None)
    parameters: dict[str, Any] = {}
    for key, value in generation.items():
        destination = (
            "max_output_tokens"
            if key in {"max_tokens", "max_completion_tokens"}
            else key
        )
        if destination in parameters:
            raise ValueError(
                f"Generation fields resolve to duplicate parameter {destination!r}"
            )
        parameters[destination] = value
    overlap = set(parameters) & set(reasoning)
    if overlap:
        raise ValueError(f"Reasoning/generation parameter collision: {sorted(overlap)}")
    parameters.update(reasoning)
    caching_enabled = _prompt_caching_enabled(model)
    if caching_enabled is False and any(
        family in model["model_id"] for family in ("gpt-5.6", "gpt-6")
    ):
        # Explicit mode with no breakpoints prevents cache reads and writes.
        parameters["prompt_cache_options"] = {"mode": "explicit"}
    request: dict[str, Any] = {
        "model": model["model_id"],
        "input": normalize_messages(messages),
        **parameters,
    }
    if system_prompt is not None:
        request["instructions"] = system_prompt
    effective = dict(parameters)
    if client_stop_sequences is not None:
        effective["client_stop_sequences"] = list(client_stop_sequences)
    return request, _record_prompt_caching(effective, model)


def build_bedrock_mantle_chat_completions_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build Chat Completions parameters for a Bedrock Mantle model.

    Mantle model entries retain Responses-style reasoning configuration because
    Responses remains their default transport. This builder translates the
    overlapping controls to their Chat Completions names for callers, such as
    SimpleAgent, that require server-side stop sequences.
    """
    reasoning, generation = _request_parts(model)
    generation = _merge_stop_sequences(generation, stop_sequences)
    parameters: dict[str, Any] = {}
    for key, value in generation.items():
        if key in {"max_output_tokens", "max_tokens"}:
            destination = "max_completion_tokens"
        elif key == "stop_sequences":
            destination = "stop"
        else:
            destination = key
        if destination in parameters:
            raise ValueError(
                f"Generation fields resolve to duplicate parameter {destination!r}"
            )
        parameters[destination] = value

    stop = parameters.get("stop")
    if stop is not None:
        stop_values = [stop] if isinstance(stop, str) else stop
        if (
            not isinstance(stop_values, list)
            or not 1 <= len(stop_values) <= 4
            or any(not isinstance(item, str) or not item for item in stop_values)
        ):
            raise ValueError(
                "Chat Completions stop must be one to four non-empty strings"
            )

    chat_reasoning = dict(reasoning)
    responses_reasoning = chat_reasoning.pop("reasoning", None)
    # ``include`` controls Responses output items and has no Chat Completions
    # equivalent. Chat responses expose any provider reasoning fields directly
    # on the assistant message instead.
    chat_reasoning.pop("include", None)
    if responses_reasoning is not None:
        if not isinstance(responses_reasoning, dict):
            raise ValueError(
                "Mantle request.reasoning.parameters.reasoning must be an object"
            )
        unknown = sorted(set(responses_reasoning) - {"effort"})
        if unknown:
            raise ValueError(
                f"Unsupported Mantle Chat Completions reasoning fields: {unknown}"
            )
        if "effort" in responses_reasoning:
            if "reasoning_effort" in chat_reasoning:
                raise ValueError("Configure only one reasoning effort field")
            chat_reasoning["reasoning_effort"] = responses_reasoning["effort"]

    overlap = set(parameters) & set(chat_reasoning)
    if overlap:
        raise ValueError(f"Reasoning/generation parameter collision: {sorted(overlap)}")
    parameters.update(chat_reasoning)
    normalized = normalize_messages(messages)
    if system_prompt is not None:
        normalized.insert(0, {"role": "system", "content": system_prompt})
    request = {"model": model["model_id"], "messages": normalized, **parameters}
    return request, _record_prompt_caching(dict(parameters), model)


def build_anthropic_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasoning, generation = _request_parts(model)
    generation = _merge_stop_sequences(generation, stop_sequences)
    parameters: dict[str, Any] = {}
    for key, value in generation.items():
        destination = "max_tokens" if key == "max_output_tokens" else key
        parameters[destination] = value
    overlap = set(parameters) & set(reasoning)
    if overlap:
        raise ValueError(f"Reasoning/generation parameter collision: {sorted(overlap)}")
    parameters.update(reasoning)
    request: dict[str, Any] = {
        "model": model["model_id"],
        "messages": normalize_messages(messages),
        **parameters,
    }
    if system_prompt is not None:
        request["system"] = system_prompt
    return request, _record_prompt_caching(dict(parameters), model)


def build_google_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasoning, generation = _request_parts(model)
    generation = _merge_stop_sequences(generation, stop_sequences)
    config: dict[str, Any] = {}
    for key, value in generation.items():
        if key == "max_tokens":
            destination = "max_output_tokens"
        else:
            destination = "stop_sequences" if key == "stop_sequences" else key
        config[destination] = value
    overlap = set(config) & set(reasoning)
    if overlap:
        raise ValueError(f"Reasoning/generation parameter collision: {sorted(overlap)}")
    config.update(reasoning)
    if system_prompt is not None:
        config["system_instruction"] = system_prompt
    normalized = normalize_messages(messages)
    contents = [
        {
            "role": "model" if item["role"] == "assistant" else "user",
            "parts": [{"text": item["content"]}],
        }
        for item in normalized
    ]
    return {
        "model": model["model_id"],
        "contents": contents,
        "config": config,
    }, _record_prompt_caching(dict(config), model)


def build_provider_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter = model["adapter"]
    kwargs = {
        "system_prompt": system_prompt,
        "stop_sequences": stop_sequences,
    }
    if adapter == "aws_bedrock_converse":
        return build_bedrock_converse_request(model, messages, **kwargs)
    if adapter == "aws_bedrock_mantle":
        return build_bedrock_mantle_responses_request(model, messages, **kwargs)
    if adapter == "openai_responses":
        return build_bedrock_mantle_responses_request(model, messages, **kwargs)
    if adapter in {
        "openrouter_chat_completions",
        "aws_bedrock_chat_completions",
        "openai_chat_completions",
        "deepseek_chat_completions",
        "together_openai_compatible",
    }:
        return build_openai_compatible_request(model, messages, **kwargs)
    if adapter == "anthropic_messages":
        return build_anthropic_request(model, messages, **kwargs)
    if adapter == "google_genai":
        return build_google_request(model, messages, **kwargs)
    raise ValueError(f"Unsupported configured adapter: {adapter!r}")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return _json_safe(value)
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _json_safe(method())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return {"value": str(value)}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} provider bytes omitted>"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _normalized_usage(adapter: str, usage: Any) -> dict[str, int | float | None]:
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }
    input_excludes_cache = False
    if adapter == "aws_bedrock_converse":
        input_tokens = _field(usage, "inputTokens")
        output_tokens = _field(usage, "outputTokens")
        total_tokens = _field(usage, "totalTokens")
        cache_read_tokens = _field(usage, "cacheReadInputTokens")
        cache_write_tokens = _field(usage, "cacheWriteInputTokens")
        reasoning_tokens = None
        input_excludes_cache = True
    elif adapter in {"aws_bedrock_mantle", "openai_responses"}:
        input_tokens = _field(usage, "input_tokens")
        output_tokens = _field(usage, "output_tokens")
        total_tokens = _field(usage, "total_tokens")
        input_details = _field(usage, "input_tokens_details", {})
        output_details = _field(usage, "output_tokens_details", {})
        cache_read_tokens = _field(input_details, "cached_tokens")
        cache_write_tokens = _field(input_details, "cache_write_tokens")
        reasoning_tokens = _field(output_details, "reasoning_tokens")
    elif adapter == "google_genai":
        input_tokens = _field(usage, "prompt_token_count")
        output_tokens = _field(usage, "candidates_token_count")
        total_tokens = _field(usage, "total_token_count")
        cache_read_tokens = _field(usage, "cached_content_token_count")
        cache_write_tokens = None
        reasoning_tokens = _field(usage, "thoughts_token_count")
    elif adapter == "anthropic_messages":
        input_tokens = _field(usage, "input_tokens")
        output_tokens = _field(usage, "output_tokens")
        total_tokens = None
        cache_read_tokens = _field(usage, "cache_read_input_tokens")
        cache_write_tokens = _field(usage, "cache_creation_input_tokens")
        reasoning_tokens = None
        input_excludes_cache = True
    else:
        input_tokens = _field(usage, "prompt_tokens")
        output_tokens = _field(usage, "completion_tokens")
        total_tokens = _field(usage, "total_tokens")
        prompt_details = _field(usage, "prompt_tokens_details", {})
        completion_details = _field(usage, "completion_tokens_details", {})
        cache_read_tokens = _field(prompt_details, "cached_tokens")
        cache_write_tokens = _field(prompt_details, "cache_write_tokens")
        reasoning_tokens = _field(completion_details, "reasoning_tokens")
    if input_excludes_cache and input_tokens is not None:
        input_tokens += cache_read_tokens or 0
        input_tokens += cache_write_tokens or 0
        total_tokens = None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        # Keep cached_tokens as a backward-compatible cache-read alias.
        "cached_tokens": cache_read_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "total_tokens": total_tokens,
        "cost_usd": None,
    }


def calculate_usage_cost(
    usage: dict[str, int | float | None], model: dict[str, Any]
) -> dict[str, int | float | None]:
    """Calculate token cost with separate cache-read and cache-write rates."""
    pricing = model.get("pricing", {})
    input_price = (
        pricing.get("input_per_million") if isinstance(pricing, dict) else None
    )
    output_price = (
        pricing.get("output_per_million") if isinstance(pricing, dict) else None
    )
    cache_read_price = None
    cache_write_price = None
    if isinstance(pricing, dict):
        cache_read_price = pricing.get(
            "cache_read_input_per_million",
            pricing.get("cached_input_per_million"),
        )
        cache_write_price = pricing.get("cache_write_input_per_million")
    if (
        isinstance(input_price, (int, float))
        and not isinstance(input_price, bool)
        and isinstance(output_price, (int, float))
        and not isinstance(output_price, bool)
        and usage["input_tokens"] is not None
        and usage["output_tokens"] is not None
    ):
        usage = dict(usage)
        cache_read_tokens = usage.get("cache_read_tokens", usage.get("cached_tokens"))
        cache_write_tokens = usage.get("cache_write_tokens")
        cache_read_tokens = (
            cache_read_tokens
            if isinstance(cache_read_tokens, (int, float))
            and not isinstance(cache_read_tokens, bool)
            else 0
        )
        cache_write_tokens = (
            cache_write_tokens
            if isinstance(cache_write_tokens, (int, float))
            and not isinstance(cache_write_tokens, bool)
            else 0
        )
        cache_read_price = (
            cache_read_price
            if isinstance(cache_read_price, (int, float))
            and not isinstance(cache_read_price, bool)
            else input_price
        )
        cache_write_price = (
            cache_write_price
            if isinstance(cache_write_price, (int, float))
            and not isinstance(cache_write_price, bool)
            else input_price
        )
        uncached_tokens = max(
            usage["input_tokens"] - cache_read_tokens - cache_write_tokens,
            0,
        )
        input_cost = (
            uncached_tokens * input_price
            + cache_read_tokens * cache_read_price
            + cache_write_tokens * cache_write_price
        )
        usage["cost_usd"] = (
            input_cost + usage["output_tokens"] * output_price
        ) / 1_000_000
    return usage


def _reasoning_from_blocks(blocks: list[Any]) -> str:
    summaries = []
    for block in blocks:
        reasoning = _field(block, "reasoningContent")
        if reasoning is None:
            reasoning = _field(block, "thinking")
        text = _field(_field(reasoning, "reasoningText", reasoning), "text")
        if isinstance(text, str) and text.strip():
            summaries.append(text.strip())
    return "\n\n".join(summaries)


def _reasoning_from_response_output(items: list[Any]) -> str:
    summaries = []
    for item in items:
        if _field(item, "type") != "reasoning":
            continue
        for field in ("content", "summary"):
            for block in _field(item, field, []) or []:
                text = _field(block, "text")
                if isinstance(text, str) and text.strip():
                    summaries.append(text.strip())
    return "\n\n".join(summaries)


def _apply_client_stop_sequences(
    text: str, stop_sequences: list[str] | None
) -> tuple[str, bool]:
    positions = [
        text.find(sequence)
        for sequence in stop_sequences or []
        if sequence and sequence in text
    ]
    if not positions:
        return text, False
    return text[: min(positions)], True


def _consume_openai_compatible_stream(
    chunks: Any,
) -> tuple[dict[str, Any], str, str | None, Any, str]:
    """Aggregate one OpenAI-compatible text stream and retain every chunk."""
    raw_chunks = []
    text_parts = []
    reasoning_parts = []
    stop_reason = None
    usage_value = None
    received_choice = False

    for chunk in chunks:
        raw_chunks.append(_as_mapping(chunk) or {})
        chunk_usage = _field(chunk, "usage")
        if chunk_usage is not None:
            usage_value = chunk_usage
        choices = _field(chunk, "choices", []) or []
        if not choices:
            continue
        received_choice = True
        choice = choices[0]
        delta = _field(choice, "delta", {}) or {}
        content = _field(delta, "content", "") or ""
        if isinstance(content, str):
            text_parts.append(content)
        reasoning = (
            _field(delta, "reasoning_content") or _field(delta, "reasoning") or ""
        )
        if isinstance(reasoning, str):
            reasoning_parts.append(reasoning)
        choice_stop_reason = _field(choice, "finish_reason")
        if choice_stop_reason is not None:
            stop_reason = choice_stop_reason

    if not received_choice:
        raise ValueError("Provider stream ended without a completion choice")
    raw_response = {
        "object": "chat.completion.stream",
        "chunks": raw_chunks,
    }
    return (
        raw_response,
        "".join(text_parts),
        stop_reason,
        usage_value,
        "".join(reasoning_parts),
    )


def _consume_bedrock_converse_stream(
    response: dict[str, Any],
) -> tuple[dict[str, Any], str, str | None, Any, str]:
    """Aggregate one Bedrock Converse stream and retain every event."""
    stream = response.get("stream")
    if stream is None:
        raise ValueError("Bedrock streaming response did not include an event stream")
    raw_events = []
    text_parts = []
    reasoning_parts = []
    stop_reason = None
    usage_value = None

    for event in stream:
        raw_events.append(_json_safe(event))
        delta_event = _field(event, "contentBlockDelta")
        if delta_event is not None:
            delta = _field(delta_event, "delta", {}) or {}
            text = _field(delta, "text", "") or ""
            if isinstance(text, str):
                text_parts.append(text)
            reasoning = _field(delta, "reasoningContent")
            reasoning_text = _field(reasoning, "text", "") or ""
            if isinstance(reasoning_text, str):
                reasoning_parts.append(reasoning_text)
        message_stop = _field(event, "messageStop")
        if message_stop is not None:
            stop_reason = _field(message_stop, "stopReason")
        metadata = _field(event, "metadata")
        if metadata is not None and _field(metadata, "usage") is not None:
            usage_value = _field(metadata, "usage")

    raw_response = {
        "object": "bedrock.converse_stream",
        "response_metadata": _json_safe(
            {key: value for key, value in response.items() if key != "stream"}
        ),
        "events": raw_events,
    }
    return (
        raw_response,
        "".join(text_parts),
        stop_reason,
        usage_value,
        "".join(reasoning_parts),
    )


class ConfiguredModelClient:
    """One configured model transport with a lazily-created native client."""

    def __init__(
        self,
        model: dict[str, Any],
        native_client: Any = None,
        *,
        api_mode: str | None = None,
    ):
        self.model = model
        self.native_client = native_client
        if api_mode not in {None, "responses", "chat_completions"}:
            raise ValueError(f"Unsupported API mode: {api_mode!r}")
        if api_mode is not None and model["adapter"] != "aws_bedrock_mantle":
            raise ValueError(
                "An explicit API mode is supported only for aws_bedrock_mantle"
            )
        self.api_mode = api_mode

    def build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            self.model["adapter"] == "aws_bedrock_mantle"
            and self.api_mode == "chat_completions"
        ):
            return build_bedrock_mantle_chat_completions_request(
                self.model,
                messages,
                system_prompt=system_prompt,
                stop_sequences=stop_sequences,
            )
        return build_provider_request(
            self.model,
            messages,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
        )

    def _client(self):
        if self.native_client is not None:
            return self.native_client
        adapter = self.model["adapter"]
        if adapter == "aws_bedrock_converse":
            import boto3
            from botocore.config import Config

            config = Config(
                read_timeout=self.model.get("read_timeout_seconds", 300),
                connect_timeout=self.model.get("connect_timeout_seconds", 60),
                retries={"max_attempts": 0},
            )
            self.native_client = boto3.client(
                "bedrock-runtime",
                region_name=self.model.get(
                    "region", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
                ),
                config=config,
            )
        elif adapter in {
            "openrouter_chat_completions",
            "aws_bedrock_chat_completions",
            "aws_bedrock_mantle",
            "openai_responses",
            "openai_chat_completions",
            "deepseek_chat_completions",
        }:
            from openai import OpenAI

            defaults = {
                "openrouter_chat_completions": (
                    "OPENROUTER_API_KEY",
                    "https://openrouter.ai/api/v1",
                ),
                "aws_bedrock_chat_completions": (
                    "AWS_BEARER_TOKEN_BEDROCK",
                    f"https://bedrock-runtime.{self.model.get('region', 'us-east-1')}.amazonaws.com/openai/v1",
                ),
                "aws_bedrock_mantle": (
                    "AWS_BEARER_TOKEN_BEDROCK",
                    f"https://bedrock-mantle.{self.model.get('region', 'us-east-1')}.api.aws/v1",
                ),
                "openai_chat_completions": ("OPENAI_API_KEY", None),
                "openai_responses": ("OPENAI_API_KEY", None),
                "deepseek_chat_completions": (
                    "DEEPSEEK_API_KEY",
                    "https://api.deepseek.com/v1",
                ),
            }
            env_name, default_url = defaults[adapter]
            env_name = self.model.get("api_key_env") or env_name
            api_key = os.environ.get(env_name)
            if not api_key:
                raise RuntimeError(
                    f"Required credential environment variable is unset: {env_name}"
                )
            kwargs = {"api_key": api_key}
            if adapter in {
                "aws_bedrock_chat_completions",
                "openrouter_chat_completions",
            }:
                # Keep SDK retries disabled for these transports.
                kwargs["max_retries"] = 0
                kwargs["timeout"] = self.model.get("read_timeout_seconds", 600)
            base_url = self.model.get("base_url") or default_url
            if base_url:
                kwargs["base_url"] = base_url
            self.native_client = OpenAI(**kwargs)
        elif adapter == "together_openai_compatible":
            import httpx
            from together import Together

            env_name = self.model.get("api_key_env") or "TOGETHERAI_API_KEY"
            api_key = os.environ.get(env_name)
            if not api_key:
                raise RuntimeError(
                    f"Required credential environment variable is unset: {env_name}"
                )
            self.native_client = Together(
                api_key=api_key,
                base_url=self.model.get("base_url"),
                timeout=httpx.Timeout(
                    timeout=self.model.get("read_timeout_seconds", 900),
                    connect=self.model.get("connect_timeout_seconds", 10),
                ),
                max_retries=self.model.get("sdk_max_retries", 0),
            )
        elif adapter == "google_genai":
            from google import genai

            self.native_client = genai.Client()
        elif adapter == "anthropic_messages":
            import requests

            self.native_client = requests.Session()
        else:
            raise ValueError(f"Unsupported configured adapter: {adapter!r}")
        return self.native_client

    def call(
        self,
        messages: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> ConfiguredResponse:
        adapter = self.model["adapter"]
        request, effective = self.build_request(
            messages,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
        )
        client = self._client()
        if adapter == "aws_bedrock_converse":
            stream_response = self.model.get("stream_responses") is True
            if stream_response:
                effective["stream"] = True
                response = client.converse_stream(**request)
                (
                    response,
                    text,
                    stop_reason,
                    usage_value,
                    reasoning_summary,
                ) = _consume_bedrock_converse_stream(response)
            else:
                response = client.converse(**request)
                blocks = response["output"]["message"]["content"]
                text = "".join(_field(block, "text", "") for block in blocks)
                stop_reason = response.get("stopReason")
                usage_value = response.get("usage")
                reasoning_summary = _reasoning_from_blocks(blocks)
        elif adapter == "openai_responses" or (
            adapter == "aws_bedrock_mantle" and self.api_mode != "chat_completions"
        ):
            response = client.responses.create(**request)
            text = _field(response, "output_text", "") or ""
            stop_reason = _field(response, "status")
            usage_value = _field(response, "usage")
            output = _field(response, "output", []) or []
            reasoning_summary = _reasoning_from_response_output(output)
            text, client_stopped = _apply_client_stop_sequences(
                text,
                effective.get("client_stop_sequences"),
            )
            if client_stopped:
                stop_reason = "client_stop"
        elif adapter in {
            "openrouter_chat_completions",
            "aws_bedrock_chat_completions",
            "aws_bedrock_mantle",
            "openai_chat_completions",
            "deepseek_chat_completions",
            "together_openai_compatible",
        }:
            stream_response = (
                adapter == "together_openai_compatible"
                and self.model.get("stream_responses") is True
            )
            if stream_response:
                request["stream"] = True
                effective["stream"] = True
            response = client.chat.completions.create(**request)
            if stream_response:
                (
                    response,
                    text,
                    stop_reason,
                    usage_value,
                    reasoning_summary,
                ) = _consume_openai_compatible_stream(response)
            else:
                choice = response.choices[0]
                text = choice.message.content or ""
                stop_reason = choice.finish_reason
                usage_value = response.usage
                reasoning_summary = (
                    getattr(choice.message, "reasoning_content", None)
                    or getattr(choice.message, "reasoning", None)
                    or ""
                )
            text, client_stopped = _apply_client_stop_sequences(
                text,
                effective.get("client_stop_sequences"),
            )
            if client_stopped:
                stop_reason = "client_stop"
        elif adapter == "google_genai":
            response = client.models.generate_content(**request)
            text = response.text or ""
            candidates = _field(response, "candidates", []) or []
            stop_reason = str(
                _field(candidates[0], "finish_reason", "") if candidates else ""
            )
            usage_value = _field(response, "usage_metadata")
            reasoning_summary = "\n\n".join(
                _field(part, "text", "")
                for candidate in candidates
                for part in (_field(_field(candidate, "content"), "parts", []) or [])
                if _field(part, "thought", False) and _field(part, "text", "")
            )
        else:
            env_name = self.model.get("api_key_env") or "CLAUDE_API_KEY"
            api_key = os.environ.get(env_name)
            if not api_key:
                raise RuntimeError(
                    f"Required credential environment variable is unset: {env_name}"
                )
            url = self.model.get("base_url") or "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": self.model.get("anthropic_version", "2023-06-01"),
                "content-type": "application/json",
            }
            response = client.post(
                url,
                headers=headers,
                json=request,
                timeout=self.model.get("read_timeout_seconds", 300),
            )
            response.raise_for_status()
            response = response.json()
            blocks = response.get("content", [])
            text = "".join(
                _field(block, "text", "")
                for block in blocks
                if _field(block, "type") == "text"
            )
            stop_reason = response.get("stop_reason")
            usage_value = response.get("usage")
            reasoning_summary = "\n\n".join(
                _field(block, "thinking", "")
                for block in blocks
                if _field(block, "type") == "thinking"
            )
        usage_adapter = (
            "openai_chat_completions"
            if adapter == "aws_bedrock_mantle" and self.api_mode == "chat_completions"
            else adapter
        )
        usage = calculate_usage_cost(
            _normalized_usage(usage_adapter, usage_value),
            self.model,
        )
        raw_response = _as_mapping(response) or {}
        raw_response["request_effective_parameters"] = effective
        return ConfiguredResponse(
            text=text,
            stop_reason=stop_reason,
            usage=usage,
            raw_usage=_as_mapping(usage_value),
            effective_parameters=effective,
            raw_response=raw_response,
            reasoning_summary=reasoning_summary,
        )


def write_raw_response(path, response: ConfiguredResponse) -> None:
    """Persist the JSON-safe native response plus normalized usage."""
    record = {
        "raw_response": response.raw_response,
        "raw_usage": response.raw_usage,
        "normalized_usage": response.usage,
        "stop_reason": response.stop_reason,
        "effective_parameters": response.effective_parameters,
    }
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
