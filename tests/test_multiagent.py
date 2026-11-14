import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import multiagent

requires_sandbox = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bwrap") is None,
    reason="Sandbox execution requires Linux and Bubblewrap",
)

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = json.loads((ROOT / "prompt_sets/multiagent_v2/prompts.json").read_text())
MODEL = {
    "model_key": "fake",
    "provider": "aws",
    "model_id": "offline.fake",
    "adapter": "aws_bedrock_converse",
    "request": {
        "reasoning": {"enabled": False, "parameters": {}},
        "generation": {"max_output_tokens": 512},
    },
}
HARNESS = {
    "total_token_budget": 10_000,
    "max_wall_time_seconds": 600,
    "network_access": False,
    "max_iterations": 20,
    "max_rounds": 5,
    "max_coder_runs_per_round": 3,
    "execution_timeout_seconds": 60,
    "suite_timeout_seconds": 120,
    "max_output_chars": 10_000,
}
TESTS = """from solution import add

def test_add():
    assert add(1.0, 2.0) == 3.0
"""


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def call(self, messages, *, system_prompt, stop_sequences):
        self.calls.append((system_prompt, [m["content"][0]["text"] for m in messages]))
        return SimpleNamespace(
            text=self.replies.pop(0),
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            raw_usage={},
            raw_response={},
            reasoning_summary=None,
            effective_parameters={},
        )


@requires_sandbox
def test_tester_gates_submission_and_coder_never_sees_tests(tmp_path):
    client = ScriptedClient(
        [
            "```candidate\ndef add(x, y):\n    return np.subtract(x, y)\n```\n",
            "```python\nimport inspect, solution\n"
            "print(inspect.getsource(solution.add))\n```\n",
            f"```tests\n{TESTS}```\n",
            "```submit\n```\n",
            "```feedback\nadd subtracts instead of adding.\n```\n",
            "```candidate\ndef add(x, y):\n    return np.add(x, y)\n```\n",
            "```submit\n```\n",
        ]
    )
    result = multiagent.run_agent(
        "Implement add(x, y).",
        "toy",
        "import numpy as np",
        model_config=MODEL,
        harness_config=HARNESS,
        output_dir=tmp_path,
        prompts=PROMPTS,
        model_client=client,
        tester_client=client,
    )

    assert result["exit_status"] == "submitted"
    assert "np.add" in result["submission"]
    assert result["n_candidates"] == 2
    trajectory = json.loads(next(tmp_path.glob("*_trajectory_*.json")).read_text())
    tester_view = "\n".join(
        m["content"][0]["text"] for m in trajectory["agents"]["tester"]["messages"]
    )
    assert "np.subtract" not in tester_view and "could not get source" in tester_view
    assert "Submission rejected" in tester_view
    coder_view = "\n".join(
        m["content"][0]["text"] for m in trajectory["agents"]["coder"]["messages"]
    )
    assert "tests/test_001.py::test_add" in coder_view
    assert "assert add" not in coder_view and "3.0" not in coder_view
    assert [run["returncode"] for run in trajectory["suite_runs"]][-1] == 0


@requires_sandbox
def test_token_budget_is_shared_by_coder_and_tester(tmp_path):
    probe = "```python\nprint(1)\n```\n"
    coder = ScriptedClient(["```candidate\ndef add(x, y):\n    return x + y\n```\n"])
    tester = ScriptedClient([probe] * 3)
    result = multiagent.run_agent(
        "Implement add(x, y).",
        "toy",
        "import numpy as np",
        model_config=MODEL,
        harness_config={**HARNESS, "total_token_budget": 7},
        output_dir=tmp_path,
        prompts=PROMPTS,
        model_client=coder,
        tester_client=tester,
    )

    # Each call costs 2 tokens: the tester alone stays at 6, the pair reaches 8.
    assert result["exit_status"] == "token_budget_exceeded"
    assert len(coder.calls) == 1 and len(tester.calls) == 3


@requires_sandbox
def test_tester_can_rewrite_and_retire_its_own_wrong_tests(tmp_path):
    wrong = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 4\n"
    client = ScriptedClient(
        [
            "```candidate\ndef add(x, y):\n    return x + y\n```\n",
            f"```tests\n{wrong}```\n",
            f"```tests\n{wrong}```\n",
            "```submit\n```\n",
            f"```tests\n# file: test_001.py\n{TESTS}```\n",
            "```retire\ntest_002.py\n```\n",
            "```submit\n```\n",
        ]
    )
    result = multiagent.run_agent(
        "Implement add(x, y).",
        "toy",
        model_config=MODEL,
        harness_config=HARNESS,
        output_dir=tmp_path,
        prompts=PROMPTS,
        model_client=client,
        tester_client=client,
    )

    assert result["exit_status"] == "submitted"
    trajectory = json.loads(next(tmp_path.glob("*_trajectory_*.json")).read_text())
    assert list(trajectory["test_files"]) == ["test_001.py"]
    assert [event["action"] for event in trajectory["test_history"]] == [
        "add",
        "add",
        "rewrite",
        "retire",
    ]


def test_reply_is_cut_after_its_first_block():
    reply = "Probe.```python\nprint(1)\n```Then tests.```tests\ndef test_x():\n    pass\n```\n"
    assert multiagent.first_block(reply) == "Probe.```python\nprint(1)\n```\n"
    assert multiagent.parse_action(multiagent.first_block(reply), "tester") == (
        "python",
        "print(1)",
    )


@requires_sandbox
def test_empty_reply_is_kept_non_empty_in_history(tmp_path):
    client = ScriptedClient(
        [
            "",
            "```candidate\ndef add(x, y):\n    return x + y\n```\n",
            f"```tests\n{TESTS}```\n",
            "```submit\n```\n",
        ]
    )
    result = multiagent.run_agent(
        "Implement add(x, y).",
        "toy",
        model_config=MODEL,
        harness_config=HARNESS,
        output_dir=tmp_path,
        prompts=PROMPTS,
        model_client=client,
        tester_client=client,
    )

    assert result["exit_status"] == "submitted"
    assert all(text.strip() for _, history in client.calls for text in history)


@requires_sandbox
def test_timed_out_call_is_retried(tmp_path):
    from botocore.exceptions import ReadTimeoutError

    client = ScriptedClient(
        [
            "```candidate\ndef add(x, y):\n    return x + y\n```\n",
            f"```tests\n{TESTS}```\n",
            "```submit\n```\n",
        ]
    )
    call = client.call
    failures = [ReadTimeoutError(endpoint_url="https://bedrock")]

    def flaky(*args, **kwargs):
        if failures:
            raise failures.pop()
        return call(*args, **kwargs)

    client.call = flaky
    result = multiagent.run_agent(
        "Implement add(x, y).",
        "toy",
        model_config=MODEL,
        harness_config=HARNESS,
        output_dir=tmp_path,
        prompts=PROMPTS,
        model_client=client,
        tester_client=client,
    )

    assert result["exit_status"] == "submitted"
    trajectory = json.loads(next(tmp_path.glob("*_trajectory_*.json")).read_text())
    assert len(trajectory["call_failures"]) == 1


@requires_sandbox
def test_tester_sees_only_the_graded_function(tmp_path):
    client = ScriptedClient(
        [
            "```candidate\nfrom math import fsum\n\ndef add(x, y):\n"
            "    return fsum([x, y])\n```\n",
            "```python\nfrom solution import add\nadd(1.0, 2.0)\n```\n",
            "```feedback\nNameError: fsum is not defined.\n```\n",
            "```candidate\ndef add(x, y):\n    return x + y\n```\n",
            f"```tests\n{TESTS}```\n",
            "```submit\n```\n",
        ]
    )
    result = multiagent.run_agent(
        "Implement add(x, y).",
        "toy",
        "import numpy as np",
        model_config=MODEL,
        harness_config=HARNESS,
        output_dir=tmp_path,
        prompts=PROMPTS,
        model_client=client,
        tester_client=client,
    )

    assert result["exit_status"] == "submitted"
    trajectory = json.loads(next(tmp_path.glob("*_trajectory_*.json")).read_text())
    tester_view = "\n".join(
        m["content"][0]["text"] for m in trajectory["agents"]["tester"]["messages"]
    )
    assert "NameError: name 'fsum' is not defined" in tester_view


@requires_sandbox
def test_coder_scratch_runs_are_capped_per_round(tmp_path):
    prompts = json.loads(
        (ROOT / "prompt_sets/multiagent_v2/prompts.json").read_text()
    )
    run = "```python\nprint('hi')\n```\n"
    client = ScriptedClient(
        [
            run,
            run,
            run,
            "```candidate\ndef add(x, y):\n    return x - y\n```\n",
            "```feedback\nadd subtracts.\n```\n",
            run,
            run,
            "```candidate\ndef add(x, y):\n    return x + y\n```\n",
            f"```tests\n{TESTS}```\n",
            "```submit\n```\n",
        ]
    )
    result = multiagent.run_agent(
        "Implement add(x, y).",
        "toy",
        "import numpy as np",
        model_config=MODEL,
        harness_config={**HARNESS, "max_coder_runs_per_round": 2},
        output_dir=tmp_path,
        prompts=prompts,
        model_client=client,
        tester_client=client,
    )

    assert result["exit_status"] == "submitted"
    trajectory = json.loads(next(tmp_path.glob("*_trajectory_*.json")).read_text())
    coder_runs = [r for r in trajectory["execution_records"] if r["agent"] == "coder"]
    assert len(coder_runs) == 4  # 2 in round 1 (third refused), 2 in round 2
    coder_view = [
        m["content"][0]["text"]
        for m in trajectory["agents"]["coder"]["messages"]
        if m["role"] == "user"
    ]
    assert "Scratch limit reached" in coder_view[3]
    assert sum("Scratch limit reached" in text for text in coder_view) == 1
