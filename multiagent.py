"""Coder/tester execution-feedback agent for FEM-Bench v2.

Two agents built on the SimpleAgent sandbox share one task. The coder works in
stateless scratch executions and publishes ```candidate implementations. The
tester never sees that source: it imports a sourceless ``solution.pyc`` and
maintains a pytest suite (add, rewrite, or retire files) that reruns on every
candidate. The coder
receives only the tester's written feedback and the names of failing tests.
Only the tester can submit, and only when the full suite passes on the
current candidate. An optional ``max_coder_runs_per_round`` caps the coder's
scratch executions between candidates, pushing it to publish more often.
"""

import ast
import json
import py_compile
import re
import sys
import tempfile
import time
from pathlib import Path

import openai
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from botocore.exceptions import HTTPClientError

from llm_api.configured_client import ConfiguredModelClient
from simpleagent import _model_metadata, _restore_fence, call_model, execute_python


SCAFFOLD_VERSION = 1
PROMPT_KEYS = frozenset(
    {
        "CODER_SYSTEM_PROMPT",
        "CODER_INSTANCE_PROMPT",
        "CODER_NO_ACTION_MSG",
        "CODER_FEEDBACK_MSG",
        "CANDIDATE_ERROR_MSG",
        "TESTER_SYSTEM_PROMPT",
        "TESTER_INSTANCE_PROMPT",
        "TESTER_NO_ACTION_MSG",
        "NEW_CANDIDATE_MSG",
        "TESTS_ADDED_MSG",
        "TESTS_REJECTED_MSG",
        "TESTS_RETIRED_MSG",
        "SUBMIT_REJECTED_MSG",
        "SUITE_RESULT_MSG",
        "OBSERVATION_MSG",
        "EMPTY_OUTPUT_PLACEHOLDER",
        "TRUNCATION_MSG",
    }
)
ACTIONS = {
    "coder": {"python", "candidate"},
    "tester": {"python", "tests", "retire", "feedback", "submit"},
}

# A call that times out or loses its connection returns nothing and costs no
# counted tokens, so it is retried instead of ending the whole run.
TRANSIENT_ERRORS = (
    HTTPClientError,
    BotocoreConnectionError,
    openai.APIConnectionError,
)
MAX_CALL_ATTEMPTS = 3

SUITE_MARKER = "__FEM_MULTIAGENT_SUITE__"
SUITE_RUNNER = f"""
import json
import pytest


class _Reports:
    def __init__(self):
        self.failed = []
        self.collect_errors = []

    def pytest_runtest_logreport(self, report):
        if report.failed and report.nodeid not in self.failed:
            self.failed.append(report.nodeid)

    def pytest_collectreport(self, report):
        if report.failed:
            self.collect_errors.append(report.nodeid)


_reports = _Reports()
_code = pytest.main(
    ["-q", "-p", "no:cacheprovider", "--tb=short", "tests"], plugins=[_reports]
)
print("{SUITE_MARKER}" + json.dumps(vars(_reports)))
raise SystemExit(int(_code))
"""


def compile_solution(source: str, preamble: str) -> bytes:
    """Compile preamble + candidate into a sourceless ``solution.pyc``.

    The preamble gives the module the grading namespace (imports and helper
    functions). Like the grader, only the candidate's first top-level function
    is kept, so the coder's own imports and helpers are dropped and
    ``from solution import fn`` behaves like the graded code.
    """
    functions = [
        ast.unparse(node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    ]
    with tempfile.TemporaryDirectory(prefix="femsolution_") as tmp:
        source_path = Path(tmp) / "solution.py"
        source_path.write_text(
            preamble + "\n\n" + "\n".join(functions[:1]), encoding="utf-8"
        )
        compiled = Path(tmp) / "solution.pyc"
        py_compile.compile(
            str(source_path),
            cfile=str(compiled),
            dfile="solution.py",
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        return compiled.read_bytes()


def first_block(reply_text: str) -> str:
    """Cut a reply after its first fenced block, as the stop sequence would.

    Some endpoints only apply the closing-fence stop sequence client side and
    miss fences not followed by a newline, so a reply can carry several
    blocks. Only the first is ever acted on; the rest must not stay in history.
    """
    match = re.search(r"```\w*\n.*?```", reply_text, re.DOTALL)
    return reply_text[: match.end()] + "\n" if match else reply_text


def parse_action(reply_text: str, role: str) -> tuple[str | None, str]:
    """Return the first fenced block whose label is an action for ``role``.

    Unlabeled blocks count as ``python``. Generation stops at the first
    closing fence, so a reply normally holds a single block.
    """
    for label, body in re.findall(r"```(\w*)\n(.*?)```", reply_text, re.DOTALL):
        label = label.lower() or "python"
        if label in ACTIONS[role]:
            return label, body.strip()
    return None, ""


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
    tester_model_config: dict | None = None,
    tester_client: ConfiguredModelClient | None = None,
) -> dict:
    """Run the coder/tester pair on one task and return its execution record.

    ``model_config`` is the coder; the tester defaults to the same model.
    """
    max_coder_runs = harness_config.get("max_coder_runs_per_round")
    if set(prompts) != PROMPT_KEYS | (
        {"SCRATCH_LIMIT_MSG"} if max_coder_runs is not None else set()
    ):
        raise ValueError(f"MultiAgent prompt set mismatch: {sorted(prompts)}")
    tester_model_config = tester_model_config or model_config
    model_metadata = _model_metadata(model_config)
    tester_metadata = _model_metadata(tester_model_config)
    budget = harness_config["total_token_budget"]
    max_iterations = harness_config["max_iterations"]
    max_rounds = harness_config["max_rounds"]
    max_wall_time = harness_config["max_wall_time_seconds"]
    max_chars = harness_config["max_output_chars"]
    clients = {
        "coder": model_client or ConfiguredModelClient(model_config),
        "tester": tester_client or ConfiguredModelClient(tester_model_config),
    }
    adapters = {
        "coder": model_config["adapter"],
        "tester": tester_model_config["adapter"],
    }
    started = time.monotonic()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = (
        output_dir / f"{task_id}_trajectory_{model_metadata['model_name']}.json"
    )
    agents = {
        role: {
            "messages": [],
            "n_calls": 0,
            "tokens": {
                "total": 0,
                "log": [],
                "provider_log": [],
                "reasoning_records": [],
            },
        }
        for role in ACTIONS
    }
    candidates = []
    test_files = {}
    test_history = []
    n_created_tests = 0
    coder_runs = 0  # scratch executions since the coder's last candidate
    execution_records = []
    suite_runs = []
    timeline = []
    solution_pyc = b""
    call_failures = []
    exit_status, submission = None, ""

    def used_tokens():
        return sum(agent["tokens"]["total"] for agent in agents.values())

    def say(role, text):
        agents[role]["messages"].append({"role": "user", "content": [{"text": text}]})

    def run(code, role, files=None, timeout=None):
        result = execute_python(
            code,
            preamble,
            timeout_seconds=timeout or harness_config["execution_timeout_seconds"],
            max_output_chars=max_chars,
            network_access=harness_config["network_access"],
            files=files,
        )
        execution_records.append({"agent": role, "code": code, **result})
        return result

    def tester_files():
        return {
            "solution.pyc": solution_pyc,
            **{f"tests/{name}": code.encode() for name, code in test_files.items()},
        }

    def observation(result):
        return prompts["OBSERVATION_MSG"].format(
            returncode=result["returncode"],
            output=result["output"] or prompts["EMPTY_OUTPUT_PLACEHOLDER"],
        )

    def run_suite():
        result = run(
            SUITE_RUNNER,
            "harness",
            tester_files(),
            timeout=harness_config["suite_timeout_seconds"],
        )
        reports = {"failed": ["<test suite did not complete>"], "collect_errors": []}
        for line in result["stdout"].splitlines():
            if line.startswith(SUITE_MARKER):
                reports = json.loads(line[len(SUITE_MARKER) :])
        output = "\n".join(
            line
            for line in result["output"].splitlines()
            if not line.startswith(SUITE_MARKER)
        )
        suite = {
            "candidate": len(candidates),
            "test_files": sorted(test_files),
            "returncode": result["returncode"],
            **reports,
        }
        suite_runs.append(suite)
        suite["message"] = prompts["SUITE_RESULT_MSG"].format(
            n_files=len(test_files),
            round=len(candidates),
            returncode=result["returncode"],
            output=output or prompts["EMPTY_OUTPUT_PLACEHOLDER"],
        )
        return suite

    def snapshot():
        trajectory_path.write_text(
            json.dumps(
                {
                    "harness": "multiagent",
                    "scaffold_version": SCAFFOLD_VERSION,
                    "execution_environment": "stateless+tester_workspace",
                    "task_id": task_id,
                    **model_metadata,
                    "tester_model": tester_metadata,
                    "harness_config": harness_config,
                    "exit_status": exit_status,
                    "submission": submission,
                    "candidates": candidates,
                    "test_files": test_files,
                    "test_history": test_history,
                    "suite_runs": suite_runs,
                    "timeline": timeline,
                    "call_failures": call_failures,
                    "execution_records": execution_records,
                    "agents": agents,
                    "wall_time_seconds": time.monotonic() - started,
                    "timestamp": time.time(),
                },
                indent=2,
            )
        )

    say(
        "coder",
        prompts["CODER_INSTANCE_PROMPT"].format(
            task=task, max_chars=max_chars, max_runs=max_coder_runs
        ),
    )
    role = "coder"
    while True:
        agent = agents[role]
        if time.monotonic() - started >= max_wall_time:
            exit_status = "wall_time_exceeded"
            break
        if agent["n_calls"] >= max_iterations:
            exit_status = "max_iterations_reached"
            break
        if used_tokens() >= budget:
            exit_status = "token_budget_exceeded"
            break

        for attempt in range(MAX_CALL_ATTEMPTS):
            try:
                reply, stop_reason, _ = call_model(
                    clients[role],
                    agent["messages"],
                    agent["tokens"],
                    system_prompt=prompts[f"{role.upper()}_SYSTEM_PROMPT"],
                    adapter=adapters[role],
                )
                break
            except TRANSIENT_ERRORS as error:
                call_failures.append(
                    {
                        "agent": role,
                        "error": repr(error),
                        "t": time.monotonic() - started,
                    }
                )
                if attempt == MAX_CALL_ATTEMPTS - 1:
                    raise
        full_reply = _restore_fence(reply, stop_reason)
        # Converse rejects empty text blocks in history.
        reply = first_block(full_reply) if full_reply.strip() else "(empty reply)"
        agent["n_calls"] += 1
        agent["messages"].append({"role": "assistant", "content": [{"text": reply}]})
        if used_tokens() > budget:
            exit_status = "token_budget_exceeded"
            snapshot()
            break
        kind, payload = (
            ("truncated", "")
            if stop_reason == "max_tokens"
            else parse_action(reply, role)
        )
        timeline.append(
            {
                "agent": role,
                "action": kind,
                "t": time.monotonic() - started,
                "discarded": full_reply[len(reply.rstrip("\n")) :].strip(),
            }
        )

        if kind == "truncated":
            say(role, prompts["TRUNCATION_MSG"])
        elif kind is None:
            say(role, prompts[f"{role.upper()}_NO_ACTION_MSG"])
        elif kind == "python" and role == "coder" and coder_runs == max_coder_runs:
            say("coder", prompts["SCRATCH_LIMIT_MSG"].format(max_runs=max_coder_runs))
        elif kind == "python":
            files = tester_files() if role == "tester" else None
            coder_runs += role == "coder"
            say(role, observation(run(payload, role, files)))
        elif kind == "candidate":
            if len(candidates) == max_rounds:
                exit_status = "max_rounds_reached"
                break
            try:
                compile(payload, "candidate", "exec")
            except SyntaxError as error:
                say(
                    "coder",
                    prompts["CANDIDATE_ERROR_MSG"].format(
                        error=f"line {error.lineno}: {error.msg}"
                    ),
                )
                snapshot()
                continue
            solution_pyc = compile_solution(payload, preamble)
            candidates.append(payload)
            coder_runs = 0
            intro = prompts["TESTER_INSTANCE_PROMPT"].format(
                task=task, max_chars=max_chars
            )
            notice = prompts["NEW_CANDIDATE_MSG"].format(round=len(candidates))
            say(
                "tester",
                (intro + "\n\n" if len(candidates) == 1 else "")
                + notice
                + "\n\n"
                + run_suite()["message"],
            )
            role = "tester"
        elif kind == "tests":
            header = re.match(r"#\s*file:\s*(\S+)", payload)
            if header:
                name = header.group(1)
            else:
                n_created_tests += 1
                name = f"test_{n_created_tests:03d}.py"
            previous = test_files.get(name)
            test_files[name] = payload
            suite = run_suite()
            if f"tests/{name}" in suite["collect_errors"]:
                if previous is None:
                    del test_files[name]
                else:
                    test_files[name] = previous
                say("tester", prompts["TESTS_REJECTED_MSG"] + "\n" + suite["message"])
            else:
                test_history.append(
                    {
                        "candidate": len(candidates),
                        "action": "add" if previous is None else "rewrite",
                        "file": name,
                        "code": payload,
                    }
                )
                say(
                    "tester",
                    prompts["TESTS_ADDED_MSG"].format(name=name)
                    + "\n"
                    + suite["message"],
                )
        elif kind == "retire":
            retired = [name for name in payload.split() if name in test_files]
            for name in retired:
                test_history.append(
                    {
                        "candidate": len(candidates),
                        "action": "retire",
                        "file": name,
                        "code": test_files.pop(name),
                    }
                )
            say(
                "tester",
                prompts["TESTS_RETIRED_MSG"].format(names=", ".join(retired) or "none")
                + "\n"
                + run_suite()["message"],
            )
        elif kind == "feedback":
            suite = run_suite()
            failing = suite["failed"] + suite["collect_errors"]
            say(
                "coder",
                prompts["CODER_FEEDBACK_MSG"].format(
                    round=len(candidates),
                    max_runs=max_coder_runs,
                    feedback=payload,
                    n_failing=len(failing),
                    failing="\n".join(f"- {nodeid}" for nodeid in failing)
                    or "(none)",
                ),
            )
            role = "coder"
        else:  # submit
            suite = run_suite()
            if suite["returncode"] == 0:
                exit_status, submission = "submitted", candidates[-1]
                break
            say("tester", prompts["SUBMIT_REJECTED_MSG"] + "\n" + suite["message"])
        snapshot()

    snapshot()

    logs = [entry for agent in agents.values() for entry in agent["tokens"]["log"]]

    def cumulative_usage(field):
        values = [entry[field] for entry in logs if entry[field] is not None]
        return sum(values) if values else None

    return {
        "exit_status": exit_status,
        "submission": submission,
        "n_model_calls": sum(agent["n_calls"] for agent in agents.values()),
        "n_executions": len(execution_records),
        "n_candidates": len(candidates),
        "execution_records": execution_records,
        "usage": {
            "input_tokens": cumulative_usage("input"),
            "output_tokens": cumulative_usage("output"),
            "reasoning_tokens": cumulative_usage("reasoning"),
            "cached_tokens": cumulative_usage("cached"),
            "cache_read_tokens": cumulative_usage("cache_read"),
            "cache_write_tokens": cumulative_usage("cache_write"),
            "total_tokens": cumulative_usage("total"),
            "cost_usd": cumulative_usage("cost_usd"),
        },
        "effective_parameters": agents["coder"]["tokens"].get(
            "effective_parameters", {}
        ),
        "wall_time_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    print("Run MultiAgent with python run_multiagent.py.", file=sys.stderr)
    raise SystemExit(2)
