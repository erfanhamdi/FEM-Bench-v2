"""Minimal execution-feedback agent for FEM-Bench v2.

The model acts through fenced ``python`` and ``submit`` blocks. Python actions
run in fresh, network-disabled Bubblewrap sandboxes and receive raw execution
output but no evaluator or reference-test feedback. Harness prompts and model
settings are supplied by the immutable experiment snapshots; this module does
not contain a second prompt source or perform grading.
"""

import ast
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from llm_api.configured_client import ConfiguredModelClient


SCAFFOLD_VERSION = 9
PROMPT_KEYS = frozenset(
    {
        "SYSTEM_PROMPT",
        "INSTANCE_PROMPT",
        "NO_ACTION_MSG",
        "MULTI_SUBMIT_MSG",
        "SUBMIT_IGNORED_NOTE",
        "TRUNCATION_MSG",
        "EMPTY_OUTPUT_PLACEHOLDER",
        "NO_TESTS_NOTE",
        "MANDATE_MSG",
        "RC_FAIL_SUBMIT_MSG",
        "OBSERVATION_MSG",
    }
)


class SandboxInfrastructureError(RuntimeError):
    """The required isolated execution environment could not be started."""


def _validate_prompts(prompts: dict[str, str]) -> dict[str, str]:
    if set(prompts) != PROMPT_KEYS or any(
        not isinstance(value, str) for value in prompts.values()
    ):
        raise ValueError(
            f"SimpleAgent prompt set mismatch; expected {sorted(PROMPT_KEYS)}, "
            f"got {sorted(prompts)}"
        )
    return dict(prompts)


def _model_metadata(model_config: dict) -> dict:
    """Derive native trajectory metadata from one resolved model snapshot."""
    request = model_config["request"]
    reasoning = request["reasoning"]
    generation = request.get("generation", {})
    max_tokens = generation.get("max_output_tokens", generation.get("max_tokens"))
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError(
            "SimpleAgent models require a positive per-call max_output_tokens/max_tokens"
        )
    parameters = dict(reasoning.get("parameters", {}))
    thinking = parameters.get("thinking", {})
    thinking_enabled = isinstance(thinking, dict) and thinking.get("type") == "enabled"
    thinking_budget_tokens = (
        thinking.get("budget_tokens") if isinstance(thinking, dict) else None
    )
    if thinking_budget_tokens is None:
        thinking_budget_tokens = parameters.get("thinking_budget")
    output_config = parameters.get("output_config", {})
    reasoning_effort = parameters.get("reasoning_effort")
    responses_reasoning = parameters.get("reasoning", {})
    if reasoning_effort is None and isinstance(responses_reasoning, dict):
        reasoning_effort = responses_reasoning.get("effort")
    if reasoning_effort is None and isinstance(output_config, dict):
        reasoning_effort = output_config.get("effort")
    temperature = generation.get("temperature")
    if thinking_budget_tokens is not None:
        decoding_label = f"thinking-{thinking_budget_tokens}"
    elif reasoning_effort is not None:
        decoding_label = f"reasoning-{reasoning_effort}"
    elif temperature is not None:
        decoding_label = f"temp-{temperature}"
    else:
        decoding_label = "provider-default"
    return {
        "model_config_name": model_config["model_key"],
        "provider": model_config["provider"],
        "model_id": model_config["model_id"],
        "model_name": model_config["model_key"],
        "adapter": model_config["adapter"],
        "decoding_label": decoding_label,
        "temperature": temperature,
        "thinking_enabled": thinking_enabled,
        "thinking_control": ("thinking.budget_tokens" if thinking_enabled else None),
        "thinking_budget_tokens": (
            thinking_budget_tokens if thinking_enabled else None
        ),
        "reasoning_effort": reasoning_effort,
        "max_tokens_per_reply": max_tokens,
    }


def _action_contains_tests(code: str) -> bool:
    """Whether an action appears to execute tests, not merely define them.

    Plain ``exec`` does not run ``def test_*`` bodies automatically. Follow
    calls into local helper functions so direct drivers and ``run_tests()``
    wrappers count, while an uncalled test definition does not.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    active_functions = set()
    framework_drivers = {
        "pytest.main",
        "pytest.raises",
        "unittest.main",
        "nose.run",
        "nose2.main",
    }

    def call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def runs_test(node):
        if isinstance(node, list):
            return any(runs_test(item) for item in node)
        if not isinstance(node, ast.AST):
            return False
        if isinstance(node, ast.Assert):
            return True
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            return False
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            leaf = name.rsplit(".", 1)[-1]
            if (
                leaf.startswith("test_")
                or leaf.startswith("assert")
                or name in framework_drivers
            ):
                return True
            if (
                isinstance(node.func, ast.Name)
                and name in functions
                and name not in active_functions
            ):
                active_functions.add(name)
                found = runs_test(functions[name].body)
                active_functions.remove(name)
                if found:
                    return True
        return any(runs_test(child) for child in ast.iter_child_nodes(node))

    return runs_test(tree.body)


def _format_execution_observation(
    code: str, result: dict, prompts: dict[str, str]
) -> str:
    """Format execution output and explicitly report a missing test driver."""
    output = result["output"]
    if not output.strip():
        output = prompts["EMPTY_OUTPUT_PLACEHOLDER"]
    observation = prompts["OBSERVATION_MSG"].format(
        returncode=result["returncode"], output=output
    )
    if not _action_contains_tests(code):
        observation += "\n" + prompts["NO_TESTS_NOTE"]
    return observation


def _restore_fence(text: str, stop_reason: str) -> str:
    """Re-append the closing fence that fence-interrupted decoding ate.

    Stop sequences are excluded from the normalized provider response,
    so an interrupted reply ends mid-block with an ODD number of ``` fences.
    Appending "```\n" makes it well-formed again.
    """
    if stop_reason == "max_tokens":
        return text
    if text.count("```") % 2 == 1:
        return text + "```\n"
    return text


def call_model(
    client,
    messages: list[dict],
    token_counter: dict,
    *,
    system_prompt: str,
    adapter: str,
):
    """Return (reply_text, stop_reason, reasoning_summary) from the model.

    The configured client translates the logged canonical messages to the
    selected provider transport. A length stop is normalized to ``max_tokens``
    so incomplete code is never executed.
    """
    response = client.call(
        messages,
        system_prompt=system_prompt,
        stop_sequences=["```\n"],
    )
    usage = response.usage
    total = usage.get("total_tokens")
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        raise RuntimeError(
            f"{adapter} did not expose total token usage; "
            "the cumulative SimpleAgent budget cannot be enforced"
        )
    token_counter["total"] += total
    token_counter.setdefault("log", []).append(
        {
            "input": usage.get("input_tokens"),
            "output": usage.get("output_tokens"),
            "reasoning": usage.get("reasoning_tokens"),
            "cached": usage.get("cached_tokens"),
            "cache_read": usage.get("cache_read_tokens"),
            "cache_write": usage.get("cache_write_tokens"),
            "total": total,
            "cost_usd": usage.get("cost_usd"),
            "raw_usage": response.raw_usage,
        }
    )
    token_counter.setdefault("provider_log", []).append(response.raw_response)
    token_counter.setdefault("reasoning_records", []).append(
        {
            "call_index": len(token_counter.get("log", [])) - 1,
            "summary_text": response.reasoning_summary,
            "summary_available": bool(response.reasoning_summary),
            "native_response_saved": True,
        }
    )
    token_counter["effective_parameters"] = response.effective_parameters
    stop_reason = response.stop_reason
    if stop_reason in {"length", "max_tokens"}:
        stop_reason = "max_tokens"
    return (
        response.text,
        stop_reason or "end_turn",
        response.reasoning_summary,
    )


def parse_action(reply_text: str, prompts: dict[str, str]):
    """Extract one executable or submission action from a model reply.

    Returns a 3-tuple:

        ("run",    joined_code, submit_was_ignored)  -> execute joined_code
        ("submit", code_str,    False)               -> final answer, stop
        ("error",  message,     False)               -> feedback, retry

    Protocol rules:
    * All ```python blocks (and unlabeled ``` blocks — models emit these
      constantly, and punishing them measures formatting compliance rather
      than capability) are concatenated in reply order into one script.
    * A ```submit block is honored ONLY when the reply contains no python
      blocks. This is the anti-hallucination rule: it forces a REAL
      observation between the last executed code and any accepted
      submission — a model can no longer run tests and submit on imagined
      results inside one breath. If code and submit co-occur, the code
      runs and the caller appends the configured submit-ignored message.
    * Other labels (```text, ```output, ...) are treated as prose and
      skipped — a fabricated ```output block is transcript pollution, not
      an action, and the real observation that follows supersedes it.
    * Zero actionable blocks, or >1 submit blocks in a code-free reply,
      are format errors fed back verbatim.
    """
    fenced = re.findall(r"```(\w*)\n(.*?)```", reply_text, re.DOTALL)
    python_blocks = [c for lab, c in fenced if lab.lower() in ("", "python")]
    submit_blocks = [c for lab, c in fenced if lab.lower() == "submit"]

    if python_blocks:
        joined = "\n\n".join(c.strip() for c in python_blocks)
        return ("run", joined, bool(submit_blocks))
    if len(submit_blocks) == 1:
        return ("submit", submit_blocks[0].strip(), False)
    if len(submit_blocks) > 1:
        return (
            "error",
            prompts["MULTI_SUBMIT_MSG"].format(n=len(submit_blocks)),
            False,
        )
    return ("error", prompts["NO_ACTION_MSG"], False)


def execute_python(
    code: str,
    preamble: str = "",
    *,
    timeout_seconds: int = 60,
    max_output_chars: int = 10_000,
    network_access: bool = False,
    files: dict[str, bytes] | None = None,
) -> dict:
    """Run one benchmark action in a fresh, disposable subprocess environment.

    Each call creates a new temporary directory and executes only the task
    preamble followed by `code`. Python definitions, imports performed by the
    action, and files written by earlier actions are therefore unavailable.
    The conversation contains prior observations, but the execution
    environment itself has no cross-action state.

    `preamble` is the task's required imports + helper-function sources. The
    shared experiment execution layer assembles it in the SAME order the
    grading harness assembles its namespace (imports, dependencies, then the
    model's code). Executing it first keeps sandbox and grading namespaces
    identical without making this loop load benchmark tasks itself.

    The model's code is written VERBATIM to attempt.py and exec'd from there
    by a tiny runner, so tracebacks still report real line numbers into the
    model's own code (there is one extra `_runner.py` frame at the top of a
    traceback; the frames below it are unchanged).

    Returns {"output": str, "returncode": int}. A timeout is reported as
    returncode -1 with an explanatory line appended — the model can react
    to it like any other error.

    When ``network_access`` is false, Bubblewrap exposes only the Python runtime
    and disposable work directory. The benchmark repository and parent
    environment credentials are not mounted or passed into the action.

    ``files`` maps relative paths to bytes copied into the work directory
    before the action runs (used by the MultiAgent tester workspace).
    """
    with tempfile.TemporaryDirectory(prefix="femagent_") as workdir:
        for relative, content in (files or {}).items():
            target = Path(workdir) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        # Write to a file and run it (rather than `python -c`) so tracebacks
        # show real line numbers the model can use.
        script = Path(workdir) / "attempt.py"
        script.write_text(code, encoding="utf-8")
        runner = Path(workdir) / "_runner.py"
        runner.write_text(
            preamble
            + '\n\nwith open("attempt.py", encoding="utf-8") as _f:\n'
            + "    _code = _f.read()\n"
            + 'exec(compile(_code, "attempt.py", "exec"))\n',
            encoding="utf-8",
        )
        command = [sys.executable, str(runner)]
        command_cwd = workdir
        sandbox_env = {
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PATH": f"{Path(sys.executable).resolve().parent}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        if not network_access:
            bubblewrap = shutil.which("bwrap")
            if bubblewrap is None:
                raise SandboxInfrastructureError(
                    "network-disabled SimpleAgent execution requires bubblewrap (bwrap)"
                )
            runtime_candidates = [
                Path(sys.base_prefix).resolve(),
                Path(sys.prefix).resolve(),
                Path(site.getusersitepackages()).resolve(),
                *(Path(path).resolve() for path in site.getsitepackages()),
                *(Path(path) for path in ("/lib", "/lib64", "/usr/lib", "/usr/lib64")),
                Path("/etc/ld.so.cache"),
            ]
            runtime_roots = []
            for path in runtime_candidates:
                if not path.exists() or path in runtime_roots:
                    continue
                if any(
                    root.is_dir() and path.is_relative_to(root)
                    for root in runtime_roots
                ):
                    continue
                runtime_roots.append(path)
            command = [
                bubblewrap,
                "--die-with-parent",
                "--new-session",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
            ]
            for runtime_root in runtime_roots:
                command.extend(["--ro-bind", str(runtime_root), str(runtime_root)])
            command.extend(
                [
                    "--tmpfs",
                    "/tmp",
                    "--dev",
                    "/dev",
                    "--proc",
                    "/proc",
                    "--unshare-net",
                    "--bind",
                    workdir,
                    workdir,
                    "--chdir",
                    workdir,
                    "--",
                    sys.executable,
                    str(runner),
                ]
            )
        try:
            proc = subprocess.run(
                command,
                cwd=command_cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=sandbox_env,
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
            if not network_access and stderr.lstrip().startswith("bwrap:"):
                raise SandboxInfrastructureError(stderr.strip())
        except subprocess.TimeoutExpired as e:
            stdout, stderr = e.stdout or "", e.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            stderr += f"\n[killed: exceeded {timeout_seconds}s timeout]"
            returncode = -1

    output = stdout + stderr
    if len(output) > max_output_chars:
        half = max_output_chars // 2
        output = output[:half] + "\n...[output truncated]...\n" + output[-half:]
    return {
        "output": output,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "network_access": network_access,
    }


def run_agent(
    task: str,
    task_id: str,
    preamble: str = "",
    *,
    model_config: dict,
    harness_config: dict,
    output_dir: Path | str,
    prompts: dict[str, str],
    model_client: ConfiguredModelClient | None = None,
) -> dict:
    """Run SimpleAgent on one task and return its native execution record.

    ``preamble`` is prepended to every sandbox action so execution uses the
    evaluator's imports and helper namespace. It is never shown to the model.
    The trajectory is flushed after each turn; evaluation remains the shared
    execution layer's responsibility.
    """
    prompts = _validate_prompts(prompts)
    model_metadata = _model_metadata(model_config)
    settings = {
        "max_total_tokens": harness_config["total_token_budget"],
        "max_wall_time_seconds": harness_config["max_wall_time_seconds"],
        "max_iterations": harness_config.get("max_iterations", 100),
        "execution_timeout_seconds": harness_config.get(
            "execution_timeout_seconds", 60
        ),
        "max_output_chars": harness_config.get("max_output_chars", 10_000),
        "network_access": harness_config.get("network_access"),
    }
    for name in (
        "max_total_tokens",
        "max_wall_time_seconds",
        "max_iterations",
        "execution_timeout_seconds",
        "max_output_chars",
    ):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"SimpleAgent {name} must be a positive integer")
    if settings["network_access"] is not False:
        raise ValueError("SimpleAgent requires network_access: false")

    experiment_dir = Path(output_dir)
    # Mantle/Gemma uses Responses here as it does elsewhere in FEM-Bench.
    # Responses does not receive the closing-fence stop sequence: the configured
    # client removes it from the request and truncates the returned text locally.
    api_mode = (
        "responses" if model_config["adapter"] == "aws_bedrock_mantle" else None
    )
    model_metadata["api_mode_override"] = api_mode
    client = model_client or ConfiguredModelClient(
        model_config,
        api_mode=api_mode,
    )
    started_monotonic = time.monotonic()

    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_metadata["model_name"]
    trajectory_path = experiment_dir / f"{task_id}_trajectory_{model_name}.json"
    meta_path = experiment_dir / f"generation_meta_{model_name}.json"
    meta = {
        "harness": "simpleagent",
        "scaffold_version": SCAFFOLD_VERSION,
        "execution_environment": "stateless",
        **model_metadata,
        "max_total_tokens": settings["max_total_tokens"],
        "max_iterations": settings["max_iterations"],
        "tasks": {},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    tokens = {
        "total": 0,
        "log": [],
        "provider_log": [],
        "reasoning_records": [],
    }
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": prompts["INSTANCE_PROMPT"].format(
                        task=task, max_chars=settings["max_output_chars"]
                    )
                }
            ],
        }
    ]
    n_calls = 0
    n_executions = 0
    n_rejected_submits = 0
    reasoning_summaries = []
    execution_records = []
    last_returncode = None
    exit_status, submission = None, ""

    def snapshot():
        trajectory_path.write_text(
            json.dumps(
                {
                    "harness": "simpleagent",
                    "scaffold_version": SCAFFOLD_VERSION,
                    "execution_environment": "stateless",
                    "task_id": task_id,
                    **model_metadata,
                    "max_total_tokens": settings["max_total_tokens"],
                    "max_iterations": settings["max_iterations"],
                    "max_wall_time_seconds": settings["max_wall_time_seconds"],
                    "network_access": settings["network_access"],
                    "exit_status": exit_status,
                    "submission": submission,
                    "n_model_calls": n_calls,
                    "n_executions": n_executions,
                    "n_rejected_submits": n_rejected_submits,
                    "total_tokens": tokens["total"],
                    "usage_log": tokens["log"],
                    "provider_log": tokens["provider_log"],
                    "effective_parameters": tokens.get("effective_parameters"),
                    "reasoning_summaries": reasoning_summaries,
                    "reasoning_records": tokens["reasoning_records"],
                    "execution_records": execution_records,
                    "wall_time_seconds": time.monotonic() - started_monotonic,
                    "timestamp": time.time(),
                    "messages": messages,
                },
                indent=2,
            )
        )

    while True:
        if time.monotonic() - started_monotonic >= settings["max_wall_time_seconds"]:
            exit_status = "wall_time_exceeded"
            break
        if n_calls >= settings["max_iterations"]:
            exit_status = "max_iterations_reached"
            break
        if tokens["total"] >= settings["max_total_tokens"]:
            exit_status = "token_budget_exceeded"
            break

        reply, stop_reason, reasoning_summary = call_model(
            client,
            messages,
            tokens,
            system_prompt=prompts["SYSTEM_PROMPT"],
            adapter=model_metadata["adapter"],
        )
        reasoning_summaries.append(reasoning_summary)
        reply = _restore_fence(reply, stop_reason)
        n_calls += 1
        messages.append({"role": "assistant", "content": [{"text": reply}]})

        if tokens["total"] > settings["max_total_tokens"]:
            exit_status = "token_budget_exceeded"
            snapshot()
            break
        if time.monotonic() - started_monotonic >= settings["max_wall_time_seconds"]:
            exit_status = "wall_time_exceeded"
            snapshot()
            break

        if stop_reason == "max_tokens":
            messages.append(
                {
                    "role": "user",
                    "content": [{"text": prompts["TRUNCATION_MSG"]}],
                }
            )
            snapshot()
            continue

        kind, payload, submit_ignored = parse_action(reply, prompts)

        if kind == "submit":
            if n_executions == 0:
                n_rejected_submits += 1
                messages.append(
                    {
                        "role": "user",
                        "content": [{"text": prompts["MANDATE_MSG"]}],
                    }
                )
                snapshot()
                continue
            if last_returncode != 0:
                n_rejected_submits += 1
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompts["RC_FAIL_SUBMIT_MSG"].format(
                                    rc=last_returncode
                                )
                            }
                        ],
                    }
                )
                snapshot()
                continue
            exit_status, submission = "submitted", payload
            break

        if kind == "error":
            messages.append({"role": "user", "content": [{"text": payload}]})
            snapshot()
            continue

        n_executions += 1
        result = execute_python(
            payload,
            preamble,
            timeout_seconds=settings["execution_timeout_seconds"],
            max_output_chars=settings["max_output_chars"],
            network_access=settings["network_access"],
        )
        execution_records.append({"code": payload, **result})
        last_returncode = result["returncode"]
        obs = _format_execution_observation(payload, result, prompts)
        if submit_ignored:
            obs += prompts["SUBMIT_IGNORED_NOTE"]
        messages.append({"role": "user", "content": [{"text": obs}]})
        snapshot()

    snapshot()

    def cumulative_usage(field):
        values = [entry[field] for entry in tokens["log"] if entry[field] is not None]
        return sum(values) if values else None

    meta["tasks"][task_id] = {
        "exit_status": exit_status,
        "n_model_calls": n_calls,
        "total_tokens": tokens["total"],
        "input_tokens": cumulative_usage("input"),
        "output_tokens": cumulative_usage("output"),
        "thinking_budget_tokens": model_metadata["thinking_budget_tokens"],
        "reasoning_effort": model_metadata["reasoning_effort"],
        "max_iterations": settings["max_iterations"],
    }
    meta["total_tokens_all_tasks"] = sum(
        t["total_tokens"] for t in meta["tasks"].values()
    )
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "exit_status": exit_status,
        "submission": submission,
        "n_model_calls": n_calls,
        "total_tokens": tokens["total"],
        "n_executions": n_executions,
        "n_rejected_submits": n_rejected_submits,
        "reasoning_summaries": reasoning_summaries,
        "reasoning_records": tokens["reasoning_records"],
        "execution_records": execution_records,
        "usage": {
            "input_tokens": cumulative_usage("input"),
            "output_tokens": cumulative_usage("output"),
            "reasoning_tokens": cumulative_usage("reasoning"),
            "cached_tokens": cumulative_usage("cached"),
            "cache_read_tokens": cumulative_usage("cache_read"),
            "cache_write_tokens": cumulative_usage("cache_write"),
            "total_tokens": tokens["total"],
            "cost_usd": cumulative_usage("cost_usd"),
        },
        "effective_parameters": tokens.get("effective_parameters", {}),
        "provider_log": tokens["provider_log"],
        "wall_time_seconds": time.monotonic() - started_monotonic,
        "messages": messages,
    }


if __name__ == "__main__":
    print("Run SimpleAgent with python run_simpleagent.py.", file=sys.stderr)
    raise SystemExit(2)
