"""Prepare judge evidence for completed, incorrect agent implementation trials."""

import argparse
import json
from pathlib import Path

from run_judge import (
    clean_openhands_trajectory,
    clean_simpleagent_trajectory,
    prepare_input_package,
)
from fem_bench.experiment import load_experiment_run
from fem_bench.run_audit import audit_experiment_run


def prepare_run(run_dir, output_root):
    run = load_experiment_run(run_dir)
    harness = run.resolved["harness"]
    # Use the recorded agent so archived harness labels remain readable.
    if run.resolved.get("harness_config", {}).get("agent") == "openhands":
        harness = "openhands"
    if run.resolved["track"] != "implementation" or harness not in {"simpleagent", "openhands"}:
        raise ValueError("Judge preparation requires a SimpleAgent or OpenHands implementation run")
    report = audit_experiment_run(run)
    invalid = [issue for issue in report["issues"]
               if issue["kind"] not in {"missing_trial", "execution_failure", "unevaluated_trial"}]
    if invalid:
        raise ValueError(f"Invalid run records: {invalid}")
    harness_name = "simpleagent" if harness == "simpleagent" else "openhands"
    cleaner = clean_simpleagent_trajectory if harness == "simpleagent" else clean_openhands_trajectory
    tasks = {task["task_id"]: task for task in run.manifest["tasks"]}
    paths = []
    for result_path in sorted((run.run_dir / "trials").glob("*/*/*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evaluation = result["evaluation"]
        if (result["execution"]["status"] != "completed"
                or evaluation["evaluated"] is not True
                or evaluation["function_correctness"] != 0):
            continue
        trial_dir = result_path.parent
        artifacts = {}
        for key in ("trajectory", "submission"):
            artifact = (trial_dir / result["artifacts"][key]).resolve()
            if not artifact.is_relative_to(trial_dir.resolve()):
                raise ValueError(f"Artifact outside trial directory: {artifact}")
            artifacts[key] = artifact.read_text(encoding="utf-8")
        initial_prompt, trajectory = cleaner(json.loads(artifacts["trajectory"]))
        task = tasks[result["task"]["task_id"]]
        package = prepare_input_package(
            initial_prompt, trajectory, trial_dir.parent, evaluation,
            run.run_dir / "benchmark_tasks" / task["split"], artifacts["submission"],
        )
        path = (Path(output_root) / harness_name
                / trial_dir.relative_to(run.run_dir / "trials") / "input_package.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != package:
                raise ValueError(f"Different evidence already exists: {path}; use a new output directory")
        else:
            with path.open("x", encoding="utf-8") as file:
                file.write(json.dumps(package, indent=2, ensure_ascii=False) + "\n")
        paths.append(path)
    print(f"{run.run_dir}: {len(paths)} completed, incorrect trials prepared; "
          f"{len(report['issues'])} incomplete/unsuccessful run audit issues")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing resolved_experiment.json")
    parser.add_argument("--output-dir", type=Path, default=Path("llm-as-judge"),
                        help="Package root, containing simpleagent/ and openhands/")
    args = parser.parse_args(argv)
    for run_dir in args.runs:
        prepare_run(run_dir, args.output_dir)


if __name__ == "__main__":
    main()
