"""Run configured LLM judges on prepared agent input packages."""

import argparse
import ast
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fem_bench.experiment import load_models_catalog, resolve_generation_by_model, resolve_models
from llm_api.configured_client import ConfiguredModelClient


DEFAULT_CODEBOOK = "llm-as-judge/Codebook.csv"
DEFAULT_SYSTEM_PROMPT = "llm-as-judge/judge_system_prompt.tx"


def codebook_csv_to_json(csv_path):
    csv_path = Path(csv_path)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(load_codebook(csv_path) + "\n", encoding="utf-8")
    return json_path


def load_codebook(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        if path.suffix == ".csv":
            return json.dumps(list(csv.DictReader(file)), indent=2, ensure_ascii=False)
        return file.read()


def build_system_prompt(template_path, codebook):
    return Path(template_path).read_text(encoding="utf-8").replace("{CODEBOOK}", codebook)


def content_text(content):
    return "\n".join(block["text"] for block in content if block["type"] == "text")


def trajectory_entry(thought="", action="", observation=""):
    """Use the same three fields for both harnesses."""
    return {"thought": thought, "action": action, "observation": observation}


def clean_simpleagent_trajectory(trajectory):
    messages = trajectory["messages"]
    initial_prompt = "\n".join(block["text"] for block in messages[0]["content"])
    entries = []
    call_index = 0
    for message in messages[1:]:
        text = "\n".join(block["text"] for block in message["content"])
        if message["role"] == "assistant":
            thought, fence, code = text.partition("```")
            reasoning = trajectory["reasoning_summaries"][call_index]
            thought = thought.removeprefix("THOUGHT:").strip()
            entries.append(trajectory_entry(
                thought="\n\n".join(part for part in (reasoning, thought) if part),
                action=fence + code,
            ))
            call_index += 1
        else:
            entries.append(trajectory_entry(observation=text))
    return initial_prompt, entries


def clean_openhands_trajectory(trajectory):
    summaries = {
        record["event_index"]: record["summary_text"]
        for record in trajectory.get("reasoning_records", [])
    }
    initial_prompt = ""
    entries = []
    for index, event in enumerate(trajectory["events"]):
        if event["kind"] == "MessageEvent":
            message = event["llm_message"]
            if index == 0:
                initial_prompt = content_text(message["content"])
            elif message["role"] == "assistant":
                entries.append(trajectory_entry(thought=content_text(message["content"])))
            elif message["role"] == "user":
                entries.append(trajectory_entry(observation=content_text(message["content"])))
        elif event["kind"] == "ActionEvent":
            reasoning = event["reasoning_content"] or "\n".join(
                block["thinking"]
                for block in event["thinking_blocks"]
                if block["type"] == "thinking"
            ) or summaries.get(index, "")
            thought = content_text(event["thought"])
            tool_call = event["tool_call"]
            arguments = {
                key: value
                for key, value in json.loads(tool_call["arguments"]).items()
                if key != "security_risk"
            }
            entries.append(trajectory_entry(
                thought="\n\n".join(part for part in (reasoning, thought) if part),
                action=tool_call["name"] + "\n" + json.dumps(arguments, ensure_ascii=False),
            ))
        elif event["kind"] == "ObservationEvent":
            entries.append(trajectory_entry(
                observation=content_text(event["observation"]["content"]),
            ))
        elif event["kind"] == "AgentErrorEvent":
            entries.append(trajectory_entry(
                observation="\n".join(
                    part for part in (event.get("tool_name"), event["error"]) if part
                ),
            ))
    return initial_prompt, entries


def clean_evaluation(value):
    """Keep evaluator checks without bulky inputs and outputs or model metadata."""
    if isinstance(value, dict):
        return {
            key: clean_evaluation(item)
            for key, item in value.items()
            if key not in {
                "llm_name", "test_generation_summary",
                "generated_output", "inputs", "reference_output",
            }
            and not key.startswith("joint_test")
        }
    if isinstance(value, list):
        return [clean_evaluation(item) for item in value]
    return value


def prepare_input_package(initial_prompt, trajectory, task_dir, evaluation, reference_root, submitted_code):
    task_name = task_dir.name
    source = (reference_root / f"{task_name}.py").read_text()
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == task_name
    )
    cleaned_evaluation = clean_evaluation(evaluation)
    del cleaned_evaluation["original_evaluator_result"]["test_results"]
    cleaned_evaluation["test_results"] = [
        {
            "test_case": case["test_case"],
            "verification_inputs": case["inputs"],
            "test_results": {
                key: value for key, value in clean_evaluation(case).items()
                if key != "test_case"
            },
        }
        for case in evaluation["original_evaluator_result"]["test_results"]
    ]
    return {
        "initial_prompt": initial_prompt,
        "trajectory": trajectory,
        "submitted_code": submitted_code,
        "reference_code": ast.get_source_segment(source, function),
        "evaluation": cleaned_evaluation,
    }


def extract_judgment(response_text):
    if not isinstance(response_text, str) or "{" not in response_text:
        raise ValueError("Judge response contains no JSON object")
    text = response_text[response_text.index("{"):]
    decoder = json.JSONDecoder()
    while True:
        try:
            judgment, _ = decoder.raw_decode(text)
            return judgment
        except json.JSONDecodeError as error:
            if error.msg != "Invalid \\escape":
                raise
            # Preserve literal backslashes in model prose, e.g. LaTeX \sigma.
            text = text[:error.pos] + "\\" + text[error.pos:]


def validate_judgment(judgment, valid_codes):
    if not isinstance(judgment, dict) or not isinstance(judgment.get("codes"), list):
        raise ValueError("Judgment must be an object with a codes list")
    if not isinstance(judgment.get("flag_for_review"), bool):
        raise ValueError("Judgment flag_for_review must be boolean")
    seen = set()
    for code in judgment["codes"]:
        if not isinstance(code, dict) or not isinstance(code.get("code_id"), str):
            raise ValueError("Each code must be an object with a string code_id")
        code_id = code["code_id"]
        if code_id not in valid_codes or code_id in seen:
            raise ValueError(f"Unknown or duplicate code_id: {code_id}")
        if not isinstance(code.get("rationale"), str) or not code["rationale"].strip():
            raise ValueError(f"Missing rationale for {code_id}")
        seen.add(code_id)


def run_judge(config, input_packages, codebook):
    repo_root = Path(__file__).resolve().parent
    run_dir = repo_root / "judge_runs" / config["experiment"]["id"] / config["codebook_version"]
    system_prompt = build_system_prompt(repo_root / config.get("system_prompt", DEFAULT_SYSTEM_PROMPT), codebook)
    models = resolve_models(load_models_catalog(repo_root / "configs/models.json"), config["models"])
    generation = resolve_generation_by_model(config["generation"], models)
    valid_codes = {row["Code ID"] for row in json.loads(codebook)}
    completed = skipped = failed = 0
    for model_key, model in models["models"].items():
        model["request"]["generation"].update(generation[model_key])
        client = ConfiguredModelClient(model)
        for package_path in input_packages:
            relative_path = package_path.relative_to(repo_root / config["input_dir"])
            output_path = run_dir / model_key / relative_path.with_name("judge_eval.json")
            if output_path.exists():
                try:
                    validate_judgment(json.loads(output_path.read_text(encoding="utf-8")), valid_codes)
                except ValueError as error:
                    print(f"Rejudging invalid saved judgment {output_path}: {error}", flush=True)
                else:
                    print(f"Skipping existing judgment: {output_path}", flush=True)
                    skipped += 1
                    continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            messages = [{"role": "user", "content": package_path.read_text(encoding="utf-8")}]
            for attempt in range(1, 4):
                response = None
                try:
                    response = client.call(messages, system_prompt=system_prompt)
                    judgment = extract_judgment(response.text)
                    validate_judgment(judgment, valid_codes)
                except Exception as error:
                    # Provider SDKs use different exception types; isolate each request.
                    record = {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "attempt": attempt,
                        "error": f"{type(error).__name__}: {error}",
                        "response_text": getattr(response, "text", None),
                        "stop_reason": getattr(response, "stop_reason", None),
                        "usage": getattr(response, "usage", None),
                        "raw_response": getattr(response, "raw_response", None),
                    }
                    with output_path.with_name("judge_errors.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"{model_key}: {package_path} attempt {attempt}/3 failed: {record['error']}", flush=True)
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                else:
                    break
            else:
                failed += 1
                print(f"Leaving judgment pending: {package_path}", flush=True)
                continue
            temporary_path = output_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(judgment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary_path.replace(output_path)
            completed += 1
            print(f"{model_key}: {package_path}")
            print(json.dumps(judgment, indent=2, ensure_ascii=False))
    print(f"Judgments: {completed} completed, {skipped} skipped, {failed} failed", flush=True)
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/llm_as_judge.yaml")
    parser.add_argument("--codebook")
    parser.add_argument("--system-prompt", help="Prompt template path (relative to the repository or absolute)")
    parser.add_argument("--models", nargs="+", help="Judge model aliases from configs/models.json")
    parser.add_argument("--run-id", help="Override experiment.id; use a new ID for a fresh evaluation")
    parser.add_argument("--codebook-version", help="Output directory label for this codebook/prompt version")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", nargs="+", help="Task IDs; select all prepared cases for these tasks, overriding trial_dirs")
    selection.add_argument("--trial-dirs", nargs="+", help="Exact prepared case directories, overriding trial_dirs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list the selection without calling judges")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    config = yaml.safe_load((repo_root / args.config).read_text())
    for field in ("models", "system_prompt", "codebook_version", "trial_dirs"):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.run_id is not None:
        config.setdefault("experiment", {})["id"] = args.run_id
    for label in (args.run_id, args.codebook_version):
        if label is not None and (not label or Path(label).name != label or label in {".", ".."}):
            parser.error("--run-id and --codebook-version must be single directory names")
    codebook_csv = args.codebook or config.get("codebook") or DEFAULT_CODEBOOK
    codebook = load_codebook(repo_root / codebook_csv)
    input_root = (repo_root / config["input_dir"]).resolve()
    if config.get("trial_dirs") and args.tasks is None:
        input_packages = [
            (repo_root / path / "input_package.json").resolve()
            for path in config["trial_dirs"]
        ]
    else:
        input_packages = sorted(
            package
            for harness in ("simpleagent", "openhands")
            for package in (input_root / harness).rglob("input_package.json")
        )
    if args.tasks is not None:
        missing = set(args.tasks) - {path.parent.parent.name for path in input_packages}
        if missing:
            parser.error(f"No prepared input packages for task IDs: {', '.join(sorted(missing))}")
        input_packages = [path for path in input_packages if path.parent.parent.name in args.tasks]
    input_packages = sorted(set(input_packages))
    if not input_packages:
        parser.error("No prepared input packages selected")
    for path in input_packages:
        if not path.is_relative_to(input_root) or not path.is_file():
            parser.error(f"Expected a prepared input package under {input_root}: {path}")
    print(f"Selected {len(input_packages)} prepared input packages")
    if args.dry_run:
        models = resolve_models(load_models_catalog(repo_root / "configs/models.json"), config["models"])
        resolve_generation_by_model(config["generation"], models)
        prompt_path = repo_root / config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        if "{CODEBOOK}" not in prompt_path.read_text(encoding="utf-8"):
            parser.error(f"Prompt must contain {{CODEBOOK}}: {prompt_path}")
        codes = [row["Code ID"] for row in json.loads(codebook)]
        if not codes or len(codes) != len(set(codes)):
            parser.error("Codebook must contain unique Code ID values")
        print(f"Judges: {', '.join(config['models'])}")
        print(f"Codebook: {repo_root / codebook_csv}\nPrompt: {prompt_path}")
        print(f"Results: {repo_root / 'judge_runs' / config['experiment']['id'] / config['codebook_version']}")
        for path in input_packages:
            print(path.parent.relative_to(input_root))
        print(f"Dry run: {len(input_packages) * len(config['models'])} judgments selected; existing valid results will be skipped.")
        return
    if run_judge(config, input_packages, codebook):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
