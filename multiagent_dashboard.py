"""Web dashboard for MultiAgent coder/tester trajectories.

Serves every MultiAgent trial under ``experiment_runs/`` (finished or still
running) and reconstructs the coder/tester back-and-forth in chronological
order. Usage:

    python multiagent_dashboard.py [--port 8765] [--runs experiment_runs]
    python multiagent_dashboard.py --export dashboard.html   # static snapshot
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parent
PAGE = REPO_ROOT / "multiagent_dashboard.html"
CANDIDATE_ERROR_PREFIX = "Candidate rejected"


def find_trials(runs_root: Path) -> list[Path]:
    """Trial directories whose trajectory comes from the MultiAgent harness."""
    trials = []
    for trial in sorted(runs_root.glob("*/trials/*/*/trial_*")):
        path = trajectory_path(trial)
        if path and '"harness": "multiagent"' in path.open().read(200):
            trials.append(trial)
    return trials


def trajectory_path(trial: Path) -> Path | None:
    """Finished trials hold trajectory.json; running ones the native name."""
    if (trial / "trajectory.json").exists():
        return trial / "trajectory.json"
    return next(trial.glob("*_trajectory_*.json"), None)


def text(message: dict) -> str:
    return message["content"][0]["text"]


def conversation(trajectory: dict) -> list[dict]:
    """Merge both agent histories into one chronological event list.

    Each history alternates user/assistant. The timeline orders assistant
    turns; each turn's harness response lands in the acting agent's history,
    except a compiled candidate (goes to the tester) and feedback (goes to
    the coder).
    """
    users, replies = {}, {}
    for role, agent in trajectory["agents"].items():
        users[role] = [text(m) for m in agent["messages"] if m["role"] == "user"]
        replies[role] = [
            text(m) for m in agent["messages"] if m["role"] == "assistant"
        ]
    events = [
        {"from": "harness", "to": "coder", "kind": "task", "text": users["coder"][0]}
    ]
    next_user = {"coder": 1, "tester": 0}
    next_reply = {"coder": 0, "tester": 0}
    for step in trajectory["timeline"]:
        role = step["agent"]
        events.append(
            {
                "from": role,
                "kind": step["action"] or "format_error",
                "text": replies[role][next_reply[role]],
                "discarded": step.get("discarded", ""),
                "t": step["t"],
            }
        )
        next_reply[role] += 1
        target = role
        if step["action"] == "feedback":
            target = "coder"
        elif step["action"] == "candidate":
            pending = users["coder"][next_user["coder"] : next_user["coder"] + 1]
            if not (pending and pending[0].startswith(CANDIDATE_ERROR_PREFIX)):
                target = "tester"
        if next_user[target] < len(users[target]):
            events.append(
                {
                    "from": "harness",
                    "to": target,
                    "kind": f"{step['action'] or 'format_error'}_response",
                    "text": users[target][next_user[target]],
                    "t": step["t"],
                }
            )
            next_user[target] += 1
    return events


def load_trial(trial: Path, runs_root: Path, detail: bool) -> dict:
    trajectory = json.loads(trajectory_path(trial).read_text())
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    agents = trajectory["agents"]
    summary = {
        "id": trial.relative_to(runs_root).as_posix(),
        "run_id": trial.relative_to(runs_root).parts[0],
        "task_id": trajectory["task_id"],
        "trial": trial.name,
        "coder_model": trajectory["model_name"],
        "tester_model": trajectory["tester_model"]["model_name"],
        "status": trajectory["exit_status"] or "running",
        "correct": result["evaluation"].get("function_correctness") if result else None,
        "n_candidates": len(trajectory["candidates"]),
        "n_test_files": len(trajectory["test_files"]),
        "n_suite_runs": len(trajectory["suite_runs"]),
        "tokens": {role: agents[role]["tokens"]["total"] for role in agents},
        "calls": {role: agents[role]["n_calls"] for role in agents},
        "wall_time_seconds": trajectory["wall_time_seconds"],
        "updated": trajectory["timestamp"],
    }
    if not detail:
        return summary
    return {
        **summary,
        "harness_config": trajectory["harness_config"],
        "events": conversation(trajectory),
        "candidates": trajectory["candidates"],
        "test_files": trajectory["test_files"],
        "test_history": trajectory["test_history"],
        "suite_runs": trajectory["suite_runs"],
        "submission": trajectory["submission"],
        "evaluation": result["evaluation"] if result else None,
    }


def make_handler(runs_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/trials":
                trials = [load_trial(t, runs_root, False) for t in find_trials(runs_root)]
                self.reply(json.dumps(trials).encode(), "application/json")
            elif url.path == "/api/trial":
                trial = (runs_root / parse_qs(url.query)["id"][0]).resolve()
                if not trial.is_relative_to(runs_root):
                    self.send_error(404)
                    return
                data = load_trial(trial, runs_root, True)
                self.reply(json.dumps(data).encode(), "application/json")
            else:
                self.send_error(404)

        def reply(self, body: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def export(runs_root: Path, output: Path) -> None:
    """Write one self-contained HTML file with every trial embedded."""
    trials = [load_trial(t, runs_root, True) for t in find_trials(runs_root)]
    data = json.dumps(trials).replace("</", "<\\/")
    page = PAGE.read_text().replace(
        "<!--EMBEDDED-->", f"<script>window.__TRIALS__ = {data};</script>"
    )
    output.write_text(page, encoding="utf-8")
    print(f"Wrote {output} ({len(trials)} trials)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "experiment_runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export", type=Path, help="write a static HTML snapshot")
    args = parser.parse_args()
    runs_root = args.runs.resolve()
    if args.export:
        export(runs_root, args.export)
    else:
        print(f"Serving {runs_root} at http://{args.host}:{args.port}")
        ThreadingHTTPServer((args.host, args.port), make_handler(runs_root)).serve_forever()
