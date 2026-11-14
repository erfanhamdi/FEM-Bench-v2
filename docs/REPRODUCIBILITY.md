# Reproducing the evaluation workflow

This release contains the evaluation code, benchmark-authored prompts, model
configuration catalog, OpenHands container recipe, and public legacy tasks.
It deliberately excludes the new tasks and all experiment outputs. You can run
and test the complete workflow on the public tasks. Reproducing the paper's
numerical results additionally requires the withheld task sources and retained
run-specific inputs; this repository alone does not reproduce those results.

## Public examples and paper settings

The default single-pass, SimpleAgent, and OpenHands files in
`configs/experiments/` are small examples: one public task, one model, and one
trial. Their generation and agent budgets are smaller than the paper settings.
They are suitable for checking an installation, not for recreating the paper's
experimental matrix. The coder–tester example uses the final prompt set with its own
budgets; use the settings below for the paper configuration.

The paper reports five trials per task, separate single-pass implementation and
test-generation tracks, and implementation-only agentic evaluations. The main
agent settings are:

| Setting | SimpleAgent | OpenHands |
| --- | --- | --- |
| Prompt set | `simpleagent_v1` / `simpleagent_v1_avoid_guard_rail` | `openhands_v2` |
| `generation.max_output_tokens` | 30,000 | 30,000 |
| Cumulative input/output token budget | 2,000,000 | 2,000,000 |
| Wall-time limit | 7,200 seconds | 7,200 seconds |
| Iteration/step limit | 200 | 200 |
| Prompt caching | Enabled | Enabled |
| Python execution timeout | 120 seconds | Framework-controlled |
| Execution-output limit | 10,000 characters | Framework-controlled |
| Model-request timeout | Provider default | 600 seconds |
| API-cost ceiling | None configured | USD 75 |
| Network access | Disabled | Enabled |

For SimpleAgent, these settings live directly under `harness_config`, using
`total_token_budget`, `max_wall_time_seconds`, `max_iterations`,
`execution_timeout_seconds`, `max_output_chars`, and `network_access`.
For OpenHands they live under `harness_config.agent_config`, using
`total_token_budget`, `timeout_seconds`, `step_limit`,
`request_timeout_seconds`, `cost_limit`, and `network_access`.
The adapter passes `model_retry_attempts: 1` to OpenHands.

The release retains `onepass_implementation_v1`, `onepass_test_generation_v1`,
`openhands_v2`, and `multiagent_v2`, as verified against the final-run snapshots.
The shared task templates are `code_prompt.j2` and `test_prompt.j2`. The
OpenHands harness and prompt-set identifiers are named `openhands` and
`openhands_v2` in this release; the archived instruction text is unchanged.
Saved runs retain their original identities and can still be loaded for audit
and judge-input preparation.

The final SimpleAgent runs used two prompt texts under the same historical ID,
`simpleagent_v1`. The release uses `simpleagent_v1` for the primary text and
`simpleagent_v1_avoid_guard_rail` for the Opus retry text. The example config
selects the primary text. Only their `SYSTEM_PROMPT` strings differ.
Both bundles retain the archived prompt text exactly. Some historical snapshots
are unavailable, so use the original run's snapshot when reproducing a specific
trial; a matching version name alone does not establish identical prompt text.

Single-pass output caps, reasoning settings, and providers differ by model and
some retained trials used different providers or limits. `provider_max` selects
the ceiling recorded in the supplied model catalog; it does not query a provider
and does not reconstruct a historical run's configuration. Use the saved
`models_snapshot.json`, `resolved_experiment.json`, and per-trial effective
parameters when reproducing a specific condition.

The exploratory coder–tester implementation is also retained. Its paper settings
use `multiagent_v2`, a shared 2,000,000-token and 7,200-second budget, 32,000
output tokens per call, 200 iterations per agent, 50 candidate versions, three
coder scratch executions per candidate, 60-second scratch timeouts, 300-second
full-suite timeouts, and 10,000-character execution output. Match the coder and
tester assignments to the intended run.

## Runtime requirements

Run commands from the repository root after an editable installation with
Python 3.12. The repository contains root-level runners and support modules;
it is intended to be used as a source checkout.

Single-pass evaluation executes submitted Python against task references.
SimpleAgent and the coder–tester harness additionally require Linux and
Bubblewrap (`bwrap`) with working user namespaces. Each execution uses a fresh
sandbox. OpenHands requires Linux, Singularity, and a built container image;
its workspace files persist across actions.

Build the supplied OpenHands recipe from the repository root:

```bash
singularity build --fakeroot containers/openhands-tool-parameters-20260919.sif containers/openhands.def
```

Set `harness_config.agent_config.image` to the resulting image. Even an
OpenHands `--dry-run` validates that the image file exists. A dry run saves the
resolved setup without launching the container or contacting a model.

The recipe pins OpenHands CLI 1.16.0, SDK 1.21.0, agent server 1.44.0, and
numerical packages, with additional constraints in
`containers/openhands-agent-controls.constraints.txt`. It downloads its base
image and operating-system packages during the build. Rebuilding it therefore
does not establish a byte-identical copy of a historical image. Preserve the
actual image and its SHA-256 hash for exact environment provenance. The paper's
retained runs used more than one image and include run-specific prompt changes;
match those assets to the trial being reproduced.

## Supplying privately held tasks

Private task modules can be restored locally under the ignored `new_tasks/`
directory. Do not commit that directory or generated run folders. Regenerate a
separate, ignored local manifest after restoring them:

```bash
python -m fem_bench.benchmark_manifest --output benchmark_manifest.local.json
```

Make a local experiment YAML selecting `benchmark.split: heldout`, the intended
task IDs and models, and the appropriate trial numbers. Then pass both paths:

```bash
python run_simpleagent.py --experiment configs/experiments/local/simpleagent.yaml \
  --manifest benchmark_manifest.local.json --output-root experiment_runs \
  --run-id my-private-reproduction --dry-run
```

Remove `--dry-run` to run the experiment. This makes paid provider requests.
Use a new run ID for every new benchmark invocation; benchmark runs are never
overwritten or automatically resumed. Judge and adjudicator runs have separate
resume behavior described in the main README.

Task names alone do not establish identical task contracts: match the helper
functions, target docstring, reference implementation, test inputs, and faulty
implementations to the run-specific source. Restoring only the newest task
version does not recreate all historical conditions.

## Final judge assets

The final `pilot-v6/new-v5` runs used
[`Codebook.csv`](../llm-as-judge/Codebook.csv),
[`judge_system_prompt.tx`](../llm-as-judge/judge_system_prompt.tx) for the three initial judges,
and [`adjudicator_system_prompt.txt`](../llm-as-judge/adjudicator_system_prompt.txt) for adjudication.
These are the only judge assets included; their contents match the archived
run snapshots. Other codebooks or prompts can be supplied explicitly by path.

## Saved evidence and validation

Each benchmark run saves its experiment YAML, resolved settings, model catalog
subset, task sources, and prompt assets. These files define the condition that
was actually run. Trial artifacts retain the submission, trajectory or response,
evaluation output, usage, and effective generation parameters. Preserve the
repository source revision and Python dependency inventory alongside these
artifacts when creating new experiments.

Run the offline regression suite with:

```bash
python -m pytest -q
```

The suite checks scoring, run identity and snapshots, and uses mocked model or
container responses for agent integration. Seven coder–tester tests execute
Bubblewrap when Linux and the binary are available; otherwise they are skipped.
Passing the suite does not verify a live provider deployment or a rebuilt
OpenHands image.
Historical model aliases, prices, and endpoint settings in `configs/models.json`
are retained configuration values; provider availability can change.

Model generation is stochastic and hosted deployments can change. The retained
code preserves the evaluation procedure; it cannot guarantee identical model
responses or exact numerical reproduction without the withheld task/run assets.
