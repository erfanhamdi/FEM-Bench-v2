"""Judge selected tasks or exact cases assigned a code by a previous judge."""

# Edit these values, then run: .venv/bin/python submit_judge_subset.py
SETTINGS = {
    "trial_dirs": [],  # Exact case folders (or input_package.json paths); leave tasks=[] when using this.
    "tasks": [],  # Task IDs to judge; [] selects cases by the previous judge/code below.
    "source_run": "judge_runs/llm-as-judge/new-v5",
    "source_judge": "claude_opus_5_5_aws",
    "code": "VF-04",
    "codebook": "llm-as-judge/Codebook.csv",
    "system_prompt": "llm-as-judge/judge_system_prompt.tx",
    "models": ["claude_opus_5_5_aws"],  # Same judge; or ["claude_sonnet_5_5_aws", "gpt_6_1_sol_aws"]
    "run_id": "VF-04-recheck-v5",  # Choose a new name for each fresh run.
    "codebook_version": "rejudge",
    "input_dir": "llm-as-judge",
    "mode": "terminal",  # "qsub" queues a job; "terminal" runs here and shows live output.
    "dry_run": False,  # False submits/runs immediately; True only previews.
}

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

from run_judge import load_codebook
from fem_bench.experiment import load_models_catalog, resolve_generation_by_model, resolve_models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dirs", nargs="+", help="Prepared case paths (input-root-relative, repository-relative, or absolute); a sole trial can be omitted")
    parser.add_argument("--tasks", nargs="+", help="Task IDs; select all prepared cases for these tasks instead of previous code labels")
    parser.add_argument("--source-run", help="Previous run/version directory, e.g. judge_runs/llm-as-judge/new-v5")
    parser.add_argument("--source-judge", help="Judge alias whose previous labels select the cases")
    parser.add_argument("--code", type=str.upper, help="Select judgments containing this code, e.g. VF-01")
    parser.add_argument("--codebook", help="CSV or JSON codebook for the new judgments")
    parser.add_argument("--system-prompt", help="New prompt template containing {CODEBOOK}")
    parser.add_argument("--models", nargs="+", help="New judge aliases; defaults to the source judge")
    parser.add_argument("--run-id", help="New experiment ID; an existing destination is rejected")
    parser.add_argument("--codebook-version", help="Output subdirectory label")
    parser.add_argument("--input-dir", help="Prepared package root")
    parser.add_argument("--mode", choices=("qsub", "terminal"), help="Queue a job or run directly in this terminal")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, help="Validate and preview without writing files or submitting")
    parser.set_defaults(**SETTINGS)
    args = parser.parse_args(argv)
    if args.mode not in {"qsub", "terminal"}:
        parser.error("mode must be 'qsub' or 'terminal'")
    if not isinstance(args.tasks, list) or any(not isinstance(task, str) or not task.strip() for task in args.tasks):
        parser.error("tasks must be a list of nonempty task IDs; use [] to select by previous code labels")
    if not isinstance(args.trial_dirs, list) or any(not isinstance(path, str) or not path.strip() for path in args.trial_dirs):
        parser.error("trial_dirs must be a list of nonempty case paths")
    if args.trial_dirs and args.tasks:
        parser.error("Use either trial_dirs for exact cases or tasks for all cases of a task, not both")
    repo_root = Path(__file__).resolve().parent
    labels = [args.run_id, args.codebook_version]
    if not args.tasks and not args.trial_dirs:
        labels.append(args.source_judge)
    for label in labels:
        if not label or Path(label).name != label or label in {".", ".."}:
            parser.error("Judge alias, run ID, and codebook version must be single directory names")
    input_root = (repo_root / args.input_dir).resolve()
    output_root = repo_root / "judge_runs" / args.run_id / args.codebook_version
    if output_root.exists():
        parser.error(f"Destination already exists: {output_root}. Choose a new --run-id, or resume using its saved config.")
    if args.trial_dirs:
        selected = []
        for address in args.trial_dirs:
            path = Path(address)
            base = input_root if path.parts and path.parts[0] in {"simpleagent", "openhands"} else repo_root
            folder = (base / path).resolve()
            if folder.name == "input_package.json":
                folder = folder.parent
            if not folder.is_relative_to(input_root):
                parser.error(f"Case path must be under {input_root}: {folder}")
            if len(folder.relative_to(input_root).parts) == 3:
                packages = sorted(folder.glob("trial_*/input_package.json"))
                if len(packages) != 1:
                    parser.error(f"Found {len(packages)} prepared trials under {folder}; "
                                 "provide an explicit trial folder or input_package.json path")
                folder = packages[0].parent
            selected.append(folder.relative_to(input_root))
        selected = sorted(set(selected))
        selection = {"trial_dirs": [str(input_root / case) for case in selected]}
        summary = f"Selected {len(selected)} exact prepared cases."
    elif args.tasks:
        packages = sorted(path for harness in ("simpleagent", "openhands")
                          for path in (input_root / harness).rglob("input_package.json"))
        missing = set(args.tasks) - {path.parent.parent.name for path in packages}
        if missing:
            parser.error(f"No prepared input packages for task IDs: {', '.join(sorted(missing))}")
        selected = [path.parent.relative_to(input_root) for path in packages
                    if path.parent.parent.name in args.tasks]
        selection = {"tasks": sorted(set(args.tasks))}
        summary = f"Selected {len(selected)} prepared cases for {len(selection['tasks'])} tasks."
    else:
        source_root = (repo_root / args.source_run / args.source_judge).resolve()
        judgments = sorted(source_root.rglob("judge_eval.json"))
        if not judgments:
            parser.error(f"No judgments found under {source_root}")
        selected = []
        for path in judgments:
            try:
                judgment = json.loads(path.read_text(encoding="utf-8"))
                codes = [entry["code_id"] for entry in judgment["codes"]]
            except (OSError, ValueError, KeyError, TypeError) as error:
                parser.error(f"Cannot read judgment {path}: {error}")
            if args.code in codes:
                selected.append(path.parent.relative_to(source_root))
        if not selected:
            parser.error(f"No judgments containing {args.code} under {source_root}; no job submitted")
        selection = {"source_judge_dir": str(source_root), "code": args.code}
        summary = f"Selected {len(selected)} of {len(judgments)} judgments containing {args.code} from {args.source_judge}."

    for case in selected:
        if len(case.parts) != 4 or case.parts[0] not in {"simpleagent", "openhands"}:
            parser.error(f"Expected harness/model/task/trial for selected case: {case}")
        package = (input_root / case / "input_package.json").resolve()
        if not package.is_relative_to(input_root) or not package.is_file():
            parser.error(f"Missing prepared package under {input_root}: {package}")

    models = args.models or [args.source_judge]
    generation = {"max_output_tokens": "provider_max"}
    try:
        snapshot = resolve_models(load_models_catalog(repo_root / "configs/models.json"), models)
        resolve_generation_by_model(generation, snapshot)
        codebook = load_codebook(repo_root / args.codebook)
        codes = [row["Code ID"] for row in json.loads(codebook)]
        if not codes or any(not isinstance(code, str) or not code.strip() for code in codes) or len(codes) != len(set(codes)):
            raise ValueError("Codebook must contain unique, nonempty Code ID strings")
        prompt = (repo_root / args.system_prompt).read_text(encoding="utf-8")
        if "{CODEBOOK}" not in prompt:
            raise ValueError("Prompt must contain {CODEBOOK}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    config = {
        "experiment": {"name": "llm_as_judge", "id": args.run_id},
        "codebook_version": args.codebook_version,
        "codebook": str(output_root / "codebook.json"),
        "system_prompt": str(output_root / "system-prompt.txt"),
        "models": models, "generation": generation, "input_dir": str(input_root),
        "trial_dirs": [str(input_root / case) for case in selected],
        "selection": selection,
    }
    if args.mode == "qsub":
        command = ["qsub", "-wd", str(repo_root), "-o", str(output_root / "logs"),
                   str(repo_root / "submit_judge.sh"), "--config", str(output_root / "config.yaml")]
    else:
        command = [sys.executable, "-u", str(repo_root / "run_judge.py"),
                   "--config", str(output_root / "config.yaml")]
    print(summary)
    print(f"New judges: {', '.join(models)}; {len(selected) * len(models)} judgments")
    print(f"Codebook: {repo_root / args.codebook}\nPrompt: {repo_root / args.system_prompt}")
    for case in selected:
        print(case.as_posix())
    print(f"Results: {output_root}")
    print(f"Command ({args.mode}): " + shlex.join(command), flush=True)
    if args.dry_run:
        print("Dry run: no files written, jobs submitted, or API calls made.")
        return
    if args.mode == "qsub" and shutil.which("qsub") is None:
        parser.error("qsub is not available; use mode='terminal', a scheduler host, or --dry-run to preview")
    output_root.mkdir(parents=True, exist_ok=False)
    if args.mode == "qsub":
        (output_root / "logs").mkdir()
    (output_root / "codebook.json").write_text(codebook + "\n", encoding="utf-8")
    (output_root / "system-prompt.txt").write_text(prompt, encoding="utf-8")
    (output_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # The saved config also resumes this exact selection if submission or a job fails.
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
