"""Split unfinished judgments into disjoint jobs using the existing judge runner."""

# Edit these values, then run: .venv/bin/python submit_judge_parallel.py
SETTINGS = {
    "config": "configs/experiments/judge_grok.yaml",
    "jobs": 4,
    "dry_run": False,
}

import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

import yaml

import run_judge


def split_pending(repo_root, config, jobs):
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    input_root = (repo_root / config["input_dir"]).resolve()
    if config.get("trial_dirs"):
        packages = [(repo_root / path / "input_package.json").resolve()
                    for path in config["trial_dirs"]]
    else:
        packages = [path.resolve() for harness in ("openhands", "simpleagent")
                    for path in (input_root / harness).rglob("input_package.json")]
    codebook = config.get("codebook") or run_judge.DEFAULT_CODEBOOK
    valid_codes = {row["Code ID"] for row in json.loads(run_judge.load_codebook(repo_root / codebook))}
    output_root = repo_root / "judge_runs" / config["experiment"]["id"] / config["codebook_version"]
    pending = []
    for package in sorted(set(packages)):
        if not package.is_relative_to(input_root) or not package.is_file():
            raise ValueError(f"Invalid input package: {package}")
        for model in config["models"]:
            result = output_root / model / package.relative_to(input_root).with_name("judge_eval.json")
            try:
                run_judge.validate_judgment(json.loads(result.read_text()), valid_codes)
            except (FileNotFoundError, ValueError):
                pending.append(package.parent)
                break
    return [pending[index::jobs] for index in range(min(jobs, len(pending)))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Existing judge YAML; preserves its models, prompt and output location")
    parser.add_argument("--jobs", type=int, help="Maximum number of concurrent scheduler jobs")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction)
    parser.set_defaults(**SETTINGS)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("jobs must be at least 1")
    repo_root = Path(__file__).resolve().parent
    config_path = (repo_root / args.config).resolve()
    # Reuse the runner's full configuration validation before submitting anything.
    with contextlib.redirect_stdout(io.StringIO()):
        run_judge.main(["--config", str(config_path), "--dry-run"])
    config = yaml.safe_load(config_path.read_text())
    shards = split_pending(repo_root, config, args.jobs)
    print(f"Pending cases: {sum(map(len, shards))}; jobs: {len(shards)}; "
          f"cases per job: {[len(shard) for shard in shards]}", flush=True)
    if args.dry_run or not shards:
        return

    output_root = repo_root / "judge_runs" / config["experiment"]["id"] / config["codebook_version"]
    submissions = output_root / "parallel_submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    batch = Path(tempfile.mkdtemp(prefix="batch-", dir=submissions))
    (batch / "logs").mkdir()
    codebook = config.get("codebook") or run_judge.DEFAULT_CODEBOOK
    (batch / "codebook.json").write_text(run_judge.load_codebook(repo_root / codebook) + "\n")
    prompt = repo_root / config.get("system_prompt", run_judge.DEFAULT_SYSTEM_PROMPT)
    (batch / "system-prompt.txt").write_text(prompt.read_text())
    paths = []
    for index, shard in enumerate(shards, 1):
        worker = {**config, "trial_dirs": [str(path) for path in shard],
                  "codebook": str(batch / "codebook.json"),
                  "system_prompt": str(batch / "system-prompt.txt")}
        path = batch / f"worker-{index}.yaml"
        path.write_text(yaml.safe_dump(worker, sort_keys=False))
        paths.append(path)
    print(f"Saved worker configs and logs: {batch}", flush=True)
    for index, path in enumerate(paths, 1):
        command = ["qsub", "-terse", "-N", f"judge_parallel_{index}",
                   "-wd", str(repo_root), "-o", str(batch / "logs"),
                   str(repo_root / "submit_judge.sh"), "--config", str(path)]
        result = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip()
        # Record each accepted job immediately, including partial submissions.
        with (batch / "jobs.jsonl").open("a") as file:
            file.write(json.dumps({"job_id": job_id, "config": str(path)}) + "\n")
        print(f"Submitted job {job_id}: {len(shards[index - 1])} cases", flush=True)


if __name__ == "__main__":
    main()
