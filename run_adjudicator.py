"""Prepare disagreement packages and run GPT-6 Astra as the final judge."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from llm_api.configured_client import ConfiguredModelClient
from run_judge import extract_judgment, load_codebook, load_models_catalog, resolve_models, validate_judgment


MODEL_MAP = {
    "claude_opus_5_aws": "model_1",
    "deepseek_v4_flash_together": "model_2",
    "glm_5_2_together": "model_3",
    "gpt_5_6_luna_aws": "model_4",
    "gpt_5_6_sol_aws": "model_5",
    "kimi_k3_together": "model_6",
}
JUDGES = {
    "gpt_6_1_sol_aws": "judge_1",
    "claude_opus_5_5_aws": "judge_2",
    "grok_4_7_aws": "judge_3",
}
REPO_ROOT = Path(__file__).resolve().parent
RUN_DIR = Path("judge_runs/llm-as-judge/new-v5")
CODEBOOK = Path("llm-as-judge/Codebook.csv")
PROMPT = Path("llm-as-judge/adjudicator_system_prompt.txt")


def load_system_prompt():
    prompt = (REPO_ROOT / PROMPT).read_text(encoding="utf-8")
    codebook = load_codebook(REPO_ROOT / CODEBOOK)
    return prompt.replace("{CODEBOOK}", codebook)


def load_results(source_run=None, input_dir="llm-as-judge"):
    run_dir = REPO_ROOT / (source_run or RUN_DIR)
    input_root = REPO_ROOT / input_dir
    valid_codes = {row["Code ID"] for row in json.loads(load_codebook(REPO_ROOT / CODEBOOK))}
    cases = {path.parent.relative_to(input_root).as_posix()
             for harness in ("openhands", "simpleagent")
             for path in (input_root / harness).rglob("input_package.json")}
    if not cases:
        raise ValueError("No prepared input packages found")
    results = {case: {} for case in sorted(cases)}
    for judge in JUDGES:
        root = run_dir / judge
        paths = {path.parent.relative_to(root).as_posix(): path for path in root.rglob("judge_eval.json")}
        if set(paths) != cases:
            raise ValueError(f"{judge}: missing {len(cases - set(paths))}, unexpected {len(set(paths) - cases)} judgments")
        for case in results:
            judgment = json.loads(paths[case].read_text())
            validate_judgment(judgment, valid_codes)
            results[case][judge] = judgment
    return results


def check_agreement(results):
    agreements = {}
    disagreements = {}
    for case, judgments in results.items():
        code_sets = [{entry["code_id"] for entry in judgment["codes"]}
                     for judgment in judgments.values()]
        if all(codes == code_sets[0] for codes in code_sets[1:]):
            agreements[case] = judgments
        else:
            disagreements[case] = judgments
    return agreements, disagreements


def input_package_builder(case, judgments, output_root=None, *, input_dir="llm-as-judge", model_map=None):
    run_dir = REPO_ROOT / RUN_DIR
    model_map = MODEL_MAP if model_map is None else model_map
    if output_root is None:
        output_root = run_dir / "final_adjudicator"
    harness, model, task_id, trial = Path(case).parts
    case_id = f"{harness}/{model_map[model]}/{task_id}/{trial}"
    package = {
        "case_id": case_id,
        "task_id": task_id,
        "input_package": json.loads(
            (REPO_ROOT / input_dir / case / "input_package.json").read_text()
        ),
        "judgments": {
            JUDGES[judge]: judgment for judge, judgment in judgments.items()
        },
    }
    output_path = (
        output_root / task_id / harness / model_map[model] / trial
        / "input_package.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if json.loads(output_path.read_text()) != package:
            raise ValueError(f"Adjudication input changed: {output_path}; use a new output directory")
    else:
        output_path.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def run_final_judge(input_packages):
    models = resolve_models(
        load_models_catalog(REPO_ROOT / "configs/models.json"), ["gpt_6_astra_aws"]
    )
    client = ConfiguredModelClient(models["models"]["gpt_6_astra_aws"])
    system_prompt = load_system_prompt()
    valid_codes = {row["Code ID"] for row in json.loads(load_codebook(REPO_ROOT / CODEBOOK))}
    completed = skipped = failed = 0
    for package_path in input_packages:
        output_path = package_path.with_name("final_judgment.json")
        if output_path.exists():
            try:
                validate_judgment(json.loads(output_path.read_text()), valid_codes)
            except ValueError:
                pass
            else:
                skipped += 1
                continue
        for attempt in range(1, 4):
            response = None
            try:
                response = client.call(
                    [{"role": "user", "content": package_path.read_text(encoding="utf-8")}],
                    system_prompt=system_prompt,
                )
                judgment = extract_judgment(response.text)
                validate_judgment(judgment, valid_codes)
            except Exception as error:
                record = {"time": datetime.now(timezone.utc).isoformat(), "attempt": attempt,
                          "error": f"{type(error).__name__}: {error}",
                          "response_text": getattr(response, "text", None),
                          "usage": getattr(response, "usage", None)}
                with package_path.with_name("adjudication_errors.jsonl").open("a") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{package_path}: attempt {attempt}/3 failed: {record['error']}", flush=True)
                if attempt < 3:
                    time.sleep(2 ** attempt)
            else:
                break
        else:
            failed += 1
            continue
        package_path.with_name("usage.json").write_text(json.dumps(response.usage, indent=2) + "\n")
        temporary = output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(judgment, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
        completed += 1
        print(output_path, flush=True)
    print(f"Adjudications: {completed} completed, {skipped} skipped, {failed} failed", flush=True)
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("disagreements", "review", "all"), default="disagreements",
                        help="Review includes disagreements and any judge's review flags")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--source-run", type=Path, default=RUN_DIR,
                        help="Directory containing the three initial judge result trees")
    parser.add_argument("--input-dir", type=Path, default=Path("llm-as-judge"),
                        help="Root of the prepared SimpleAgent/OpenHands packages")
    parser.add_argument("--agent-models", nargs="+",
                        help="Agent model aliases in anonymization order; defaults to the paper mapping")
    parser.add_argument("--cases", nargs="+", help="Exact harness/model/task/trial case IDs")
    parser.add_argument("--output-dir", help="Separate destination for a fresh adjudication; relative to repository or absolute")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Maximum selected cases for a validation run")
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    model_map = MODEL_MAP
    if args.agent_models is not None:
        if len(args.agent_models) != len(set(args.agent_models)):
            parser.error("--agent-models must contain unique aliases")
        model_map = {model: f"model_{index}" for index, model in enumerate(args.agent_models, 1)}
    results = load_results(args.source_run, args.input_dir)
    unknown = {Path(case).parts[1] for case in results} - set(model_map)
    if unknown:
        parser.error(f"Agent models missing from anonymity mapping: {sorted(unknown)}; use --agent-models")
    agreements, disagreements = check_agreement(results)
    selected = {case: judgments for case, judgments in results.items()
                if args.scope == "all" or case in disagreements
                or (args.scope == "review" and any(j["flag_for_review"] for j in judgments.values()))}
    if args.cases:
        missing = set(args.cases) - set(selected)
        if missing:
            parser.error(f"Cases absent from selected scope: {sorted(missing)}")
        selected = {case: judgments for case, judgments in selected.items() if case in args.cases}
    print(f"Cases: {len(results)}; unanimous: {len(agreements)}; disagreements: {len(disagreements)}; "
          f"selected ({args.scope}): {len(selected)}", flush=True)
    # Preserve the prompt and identity mapping for audit and safe resumption.
    output_root = ((REPO_ROOT / args.output_dir).resolve() if args.output_dir
                   else REPO_ROOT / args.source_run / "final_adjudicator")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"source_run": str(args.source_run), "judges": JUDGES, "agent_models": model_map,
                "adjudicator": "gpt_6_astra_aws", "scope": args.scope,
                "selected_cases": list(selected), "unanimous_cases": list(agreements),
                "codebook": str(CODEBOOK), "prompt": str(PROMPT)}
    if args.input_dir != Path("llm-as-judge"):
        manifest["input_dir"] = str(args.input_dir)
    for name, text in [("manifest.json", json.dumps(manifest, indent=2) + "\n"),
                       ("system-prompt.txt", load_system_prompt())]:
        path = output_root / name
        if path.exists():
            if path.read_text() != text:
                raise ValueError(f"Adjudication configuration changed: {path}")
        else:
            path.write_text(text)
    cases = list(selected)[args.shard_index::args.shard_count]
    if args.limit is not None:
        cases = cases[:args.limit]
    packages = [input_package_builder(case, selected[case], output_root,
                                      input_dir=args.input_dir, model_map=model_map) for case in cases]
    print(f"Worker {args.shard_index + 1}/{args.shard_count}: {len(packages)} cases", flush=True)
    if not args.prepare_only and run_final_judge(packages):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
