"""Apply FEM-Bench model settings, then start the pinned OpenHands CLI."""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path
from typing import Any


CONFIG_ENV = "FEM_BENCH_OPENHANDS_CONFIG"
METRICS_PATH_ENV = "FEM_BENCH_OPENHANDS_METRICS_PATH"
CONTAINER_STATE_DIR = "/state"
TOKEN_BUDGET_MARKER = "FEM_BENCH_OPENHANDS_TOKEN_BUDGET_EXCEEDED"


def _configure_isolated_state() -> None:
    """Keep libraries that consult ``HOME`` inside the mounted state dir.

    Singularity deliberately refuses to override HOME through
    ``SINGULARITYENV_HOME``.  This entry point therefore sets it before any
    OpenHands modules are imported.  The state bind is temporary and is the
    only writable location besides the attempt workspace.
    """

    os.environ["HOME"] = CONTAINER_STATE_DIR
    os.environ["XDG_CONFIG_HOME"] = f"{CONTAINER_STATE_DIR}/.config"
    os.environ["XDG_CACHE_HOME"] = f"{CONTAINER_STATE_DIR}/.cache"


def _zero_metrics(*, complete: bool = False) -> dict[str, Any]:
    return {
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
        },
        "model_calls": 0,
        "accounting_complete": complete,
    }


def _conversation_metrics(conversation: Any | None) -> dict[str, Any]:
    """Return aggregate usage after a normally terminated conversation."""

    if conversation is None:
        return _zero_metrics()
    stats = conversation.conversation_stats
    combined = stats.get_combined_metrics()
    usage = combined.accumulated_token_usage
    if usage is None:
        return _zero_metrics(complete=True)
    input_tokens = int(usage.prompt_tokens or 0)
    output_tokens = int(usage.completion_tokens or 0)
    cache_read_tokens = int(getattr(usage, "cache_read_tokens", 0) or 0)
    cache_write_tokens = int(getattr(usage, "cache_write_tokens", 0) or 0)
    return {
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": int(usage.reasoning_tokens or 0),
            "cached_tokens": cache_read_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "model_calls": len(combined.token_usages),
        "accounting_complete": True,
    }


def _write_metrics(conversation: Any | None) -> None:
    path = os.environ.get(METRICS_PATH_ENV)
    if not path:
        return
    try:
        payload = _conversation_metrics(conversation)
    except Exception:
        payload = _zero_metrics()
    metrics_path = Path(path)
    try:
        live_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        live_payload = None
    if (
        isinstance(live_payload, dict)
        and live_payload.get("accounting_complete") is True
        and int(live_payload.get("usage", {}).get("total_tokens", 0) or 0)
        >= int(payload.get("usage", {}).get("total_tokens", 0) or 0)
    ):
        payload = live_payload
    metrics_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _usage_counts(usage: Any) -> dict[str, int]:
    """Normalize one OpenHands usage object for cumulative accounting."""

    input_tokens = int(
        getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    )
    output_tokens = int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    input_details = getattr(usage, "prompt_tokens_details", None) or getattr(
        usage, "input_tokens_details", None
    )
    cache_read_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
    if not cache_read_tokens:
        cache_read_tokens = int(getattr(usage, "cached_tokens", 0) or 0)
    output_details = getattr(usage, "completion_tokens_details", None) or getattr(
        usage, "output_tokens_details", None
    )
    reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
    cache_write_tokens = int(getattr(usage, "_cache_creation_input_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cache_read_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _write_live_metrics(usage: dict[str, int], model_calls: int) -> None:
    """Persist usage after each provider response, including interrupted runs."""

    path = os.environ.get(METRICS_PATH_ENV)
    if not path:
        return
    payload = {
        "usage": usage,
        "model_calls": model_calls,
        "accounting_complete": True,
    }
    Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _load_config() -> dict[str, Any]:
    try:
        config = json.loads(os.environ[CONFIG_ENV])
    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid or missing {CONFIG_ENV}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"{CONFIG_ENV} must contain a JSON object")
    return config


def _network_context(config: dict[str, Any]) -> str | None:
    """Return the prompt notice for an explicitly network-enabled run."""

    if not config.get("network_access", False):
        return None
    return (
        "External network access is permitted for this experiment. "
        "Use terminal commands for HTTP requests when needed."
    )


def _append_tool_once(tools: list[Any], name: str, tool_type: Any) -> list[Any]:
    """Append one named tool specification without duplicating it."""

    if any(getattr(tool, "name", None) == name for tool in tools):
        return tools
    return [*tools, tool_type(name=name)]


def _exclude_tools(tools: list[Any], names: set[str]) -> list[Any]:
    """Return tool specifications whose names are not excluded."""

    return [tool for tool in tools if getattr(tool, "name", None) not in names]


def _llm_updates(config: dict[str, Any]) -> dict[str, Any]:
    prompt_caching = config.get("prompt_caching", {"enabled": True})
    updates = {
        "model_canonical_name": config.get("model_canonical_name"),
        "max_output_tokens": config.get("max_output_tokens"),
        "temperature": config.get("temperature"),
        "reasoning_effort": config.get("reasoning_effort"),
        "reasoning_summary": config.get("reasoning_summary"),
        "enable_encrypted_reasoning": bool(
            config.get("enable_encrypted_reasoning", False)
        ),
        "timeout": int(config.get("request_timeout_seconds", 300)),
        "num_retries": int(config.get("model_retry_attempts", 1)),
        "caching_prompt": prompt_caching.get("enabled", True),
    }
    extra_body = {}
    if config.get("provider_routing"):
        extra_body["provider"] = dict(config["provider_routing"])
    parallel = config.get("parallel_tool_calls")
    if parallel is not None:
        extra_body["parallel_tool_calls"] = bool(parallel)
    if (
        prompt_caching.get("enabled") is False
        and config.get("supports_prompt_cache_options") is True
    ):
        extra_body["prompt_cache_options"] = {"mode": "explicit"}
    if extra_body:
        updates["litellm_extra_body"] = extra_body
    return updates


def _enable_tool_parameter_guard(llm_type: Any) -> None:
    """Omit the parallel-tool option from chat requests without tools."""

    original = llm_type._transport_call

    @wraps(original)
    def transport_call(self, *args, **kwargs):
        extra_body = kwargs.get("extra_body") or {}
        if not kwargs.get("tools") and not extra_body.get("tools"):
            kwargs.pop("parallel_tool_calls", None)
            if "parallel_tool_calls" in extra_body:
                # The SDK shares this dictionary with later tool requests.
                kwargs["extra_body"] = dict(extra_body)
                kwargs["extra_body"].pop("parallel_tool_calls")
        return original(self, *args, **kwargs)

    llm_type._transport_call = transport_call


def _enable_forced_prompt_caching(llm_type: Any, enabled: bool) -> None:
    """Force cache markers when the pinned SDK lacks a new model ID."""

    if not enabled:
        return
    original = llm_type.is_caching_prompt_active

    @wraps(original)
    def is_caching_prompt_active(self):
        if self.caching_prompt:
            return True
        return original(self)

    llm_type.is_caching_prompt_active = is_caching_prompt_active


def _setup_headless_conversation(
    setup_conversation: Any,
    conversations: list[Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Create a conversation without the unused headless TUI visualizer."""

    # FEM-Bench always invokes this entry point with ``--headless --json``.
    # OpenHands 1.16 still passes its Textual visualizer to the SDK in that
    # mode. The visualizer treats truncated shell commands as Rich markup,
    # which can raise MissingStyle for valid commands such as ``w[fixed]``.
    # JSON events remain available through the separate event callback.
    kwargs["visualizer"] = None
    conversation = setup_conversation(*args, **kwargs)
    conversations.append(conversation)
    return conversation


def _configure_openhands(config: dict[str, Any]) -> list[Any]:
    from openhands.sdk import AgentContext
    from openhands.sdk.llm import llm as llm_module
    from openhands.sdk.llm.utils.telemetry import Telemetry
    from openhands_cli import setup as setup_module
    from openhands_cli.stores.agent_store import AgentStore
    from openhands_cli.utils import get_os_description
    from openhands_cli.locations import get_work_dir

    _enable_tool_parameter_guard(llm_module.LLM)
    _enable_forced_prompt_caching(
        llm_module.LLM, bool(config.get("force_prompt_caching", False))
    )
    updates = _llm_updates(config)
    original_apply = AgentStore._apply_env_overrides
    original_metadata = AgentStore._with_llm_metadata
    original_resolve_tools = AgentStore._resolve_tools

    @wraps(original_apply)
    def apply_settings(self, agent, overrides):
        agent = original_apply(self, agent, overrides)

        def update_llm(llm):
            updated = llm.model_copy(update=updates)
            cost_limit = config.get("cost_limit")
            if cost_limit is not None:
                updated.metrics.max_budget_per_task = float(cost_limit)
            return updated

        condenser = agent.condenser
        if condenser is not None and hasattr(condenser, "llm"):
            condenser = condenser.model_copy(update={"llm": update_llm(condenser.llm)})
        return agent.model_copy(
            update={"llm": update_llm(agent.llm), "condenser": condenser}
        )

    @wraps(original_metadata)
    def preserve_request_body(self, llm, *, session_id, llm_type):
        configured = dict(llm.litellm_extra_body or {})
        updated = original_metadata(self, llm, session_id=session_id, llm_type=llm_type)
        generated = dict(updated.litellm_extra_body or {})
        if not configured:
            return updated
        return updated.model_copy(
            update={"litellm_extra_body": {**configured, **generated}}
        )

    def benchmark_context(self):
        lines = [
            f"Your current working directory is: {get_work_dir()}",
            f"User operating system: {get_os_description()}",
        ]
        if network_notice := _network_context(config):
            lines.append(network_notice)
        suffix = "\n".join(lines)
        return AgentContext(skills=[], system_message_suffix=suffix)

    AgentStore._apply_env_overrides = apply_settings
    AgentStore._with_llm_metadata = preserve_request_body
    AgentStore._build_agent_context = benchmark_context

    original_record_usage = Telemetry._record_usage
    total_token_budget = int(config["total_token_budget"])
    cumulative_usage = _zero_metrics(complete=True)["usage"]
    model_calls = 0

    @wraps(original_record_usage)
    def record_cache_writes(self, usage, response_id, context_window):
        """Record cache writes and enforce the cumulative inference-token cap."""
        nonlocal model_calls
        details = getattr(usage, "prompt_tokens_details", None) or getattr(
            usage, "input_tokens_details", None
        )
        cache_write_tokens = 0
        if details is not None:
            cache_write_tokens = int(
                getattr(details, "cache_write_tokens", 0)
                or getattr(details, "cache_creation_tokens", 0)
                or 0
            )
        if not cache_write_tokens:
            cache_write_tokens = int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )
        if cache_write_tokens:
            object.__setattr__(
                usage, "_cache_creation_input_tokens", cache_write_tokens
            )
        result = original_record_usage(self, usage, response_id, context_window)
        for name, value in _usage_counts(usage).items():
            cumulative_usage[name] += value
        model_calls += 1
        _write_live_metrics(cumulative_usage, model_calls)
        if cumulative_usage["total_tokens"] >= total_token_budget:
            raise RuntimeError(
                f"{TOKEN_BUDGET_MARKER}: cumulative usage "
                f"{cumulative_usage['total_tokens']} reached the configured "
                f"limit {total_token_budget}"
            )
        return result

    Telemetry._record_usage = record_cache_writes

    enable_browser = bool(config.get("enable_browser", False))
    enable_subagents = bool(config.get("enable_subagents", True))
    if not enable_subagents:
        # Do not register built-in subagent factories in this experimental
        # condition. The tool filter below also removes both current and legacy
        # delegation interfaces from the parent agent.
        setup_module.register_builtins_agents = lambda **kwargs: []

    if enable_browser or not enable_subagents:
        from openhands.sdk.tool import Tool
        from openhands.tools.browser_use import BrowserToolSet
        from openhands.tools.delegate.definition import DelegateTool
        from openhands.tools.task import TaskToolSet

        subagent_tool_names = {DelegateTool.name, TaskToolSet.name}

        @wraps(original_resolve_tools)
        def resolve_tools(self, session_id):
            tools = original_resolve_tools(self, session_id)
            if not enable_subagents:
                tools = _exclude_tools(tools, subagent_tool_names)
            if enable_browser:
                tools = _append_tool_once(tools, BrowserToolSet.name, Tool)
            return tools

        AgentStore._resolve_tools = resolve_tools

    original_responses_options = llm_module.select_responses_options

    @wraps(original_responses_options)
    def require_tool(*args, **kwargs):
        options = original_responses_options(*args, **kwargs)
        options["tool_choice"] = config.get("tool_choice", "required")
        return options

    llm_module.select_responses_options = require_tool

    step_limit = int(config.get("step_limit", 100))
    original_conversation = setup_module.Conversation

    def limited_conversation(*args, **kwargs):
        kwargs["max_iteration_per_run"] = step_limit
        return original_conversation(*args, **kwargs)

    setup_module.Conversation = limited_conversation

    conversations: list[Any] = []
    original_setup_conversation = setup_module.setup_conversation

    @wraps(original_setup_conversation)
    def track_conversation(*args, **kwargs):
        return _setup_headless_conversation(
            original_setup_conversation, conversations, *args, **kwargs
        )

    setup_module.setup_conversation = track_conversation
    return conversations


def main() -> None:
    _configure_isolated_state()
    config = _load_config()
    conversations = _configure_openhands(config)
    from openhands_cli.entrypoint import main as openhands_main

    try:
        openhands_main()
    finally:
        conversation = conversations[-1] if conversations else None
        try:
            _write_metrics(conversation)
        finally:
            # The CLI does not reliably close persisted conversations when its
            # headless Textual app exits.  In particular, an initialized browser
            # tool can otherwise leave its AnyIO executor and Chromium process
            # alive after the agent has finished, preventing Singularity from
            # exiting.  Close explicitly; a cleanup failure must still make the
            # subprocess fail rather than allowing the outer harness to recover
            # a submission from an incompletely terminated run.
            if conversation is not None:
                conversation.close()


if __name__ == "__main__":
    main()
