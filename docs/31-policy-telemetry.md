# Frozen-policy drone telemetry

This command replays **one existing checkpoint** in a fresh simulator process and saves one row per drone per evaluation step. It makes a policy command, the controller's requested velocity, the measured next velocity/pose, rotor control, and reward terms visible in the same row. It performs **zero optimizer updates**. The source training run is mounted read-only; the replay artifacts live in a new `runs/policy-telemetry/` directory.

## Run on the lab

From `align/`, first prepare a new pinned image context after any code changes:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
```

Use the printed `runs/runtime-build/<ID>` in the following commands. The context is single-use; never reuse an already built `--prepared-run`.

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<ID>
uv run --locked python scripts/run_policy_telemetry.py \
  --source-run runs/plane-baseline-training/20260918T105406.233167IST-d74f8b47 \
  --build-report runs/runtime-build/<ID>/report.json \
  --policy-seed 41 --evaluation-update 12 --accept-eula --gpu 0
```

The selected GPU must be allocated and idle. This replay uses the seed-41, update-12 plane specialist, four parallel worlds, and the source run's 800-step evaluation configuration. The launcher checks source checkpoint integrity, validates the new image, checks GPU availability, and records the exact command. Source configuration and checkpoint hashes are retained in `report.json`. No training rerun is needed. To inspect the matched generalist, pass the `runs/multi-template-training/20260918T111606.650830IST-7fb24d49` source with a separate new replay run; use the same build report.

## Saved data and audit

- `evaluation/evaluation.csv`: existing environment-step metrics; schema unchanged.
- `evaluation/policy-telemetry.csv`: step, environment, phase, template, drone identity, pre/post position and velocity in metres and metres/second, assigned target, WXYZ quaternion, four bounded actor actions, commanded world-axis velocity, raw/applied rotor actions, raw and weighted reward terms, per-drone and team rewards, and contact force.
- `evaluation/metrics.json`, `evaluation/probe-result.json`, `evaluation/console.log`, `evaluation/kit-logs/`: simulator result and native logs.
- `report.json`: source and new image identities, checkpoint hash, run command, GPU inventory, and CPU audit.

The CPU audit in `align.tasks.policy_telemetry` rejects missing/duplicate drone rows, nonfinite values, invalid action-to-velocity mapping, rotor clamp differences, reward-weight errors, team reward mismatch, and assigned/pairwise RMSE disagreement with the environment-level CSV. It also reports descriptive fractions of formation-phase commands pointing toward the assigned target and velocity changes pointing toward the commanded velocity. These fractions are **diagnostics**, not a pass criterion or evidence of stable control. The target may be unreachable within one step, and velocity response depends on inertia and the inner controller.

The raw policy-telemetry CSV is ignored by Git with other `runs/` data. Back it up separately with its source run and build context before moving machines. A passing telemetry replay establishes that these observations were recorded and internally reconciled; it does not establish formation convergence, reward causality, or reproduction of paper results. Video remains unvalidated for this command.

## Accepted lab replay and CPU analysis

The first replay passed on 2026-09-18 IST:

- Training source: `runs/plane-baseline-training/20260918T105406.233167IST-d74f8b47`, seed 41, completed update 12.
- Replay: `runs/policy-telemetry/20260918T130125.401428IST-bee09d21`; image `sha256:aeab46b318a5d1581a1068529d032a9718eeaf83f6ba03690ba7ebcfa69470ff`.
- Checkpoint: `000000000012-020beda227f34911adaba9926198cfb4`, payload SHA-256 `fb98d68557e4a2f5a5ac7c97b8880f620a389073388cd33d744dfd213c199851`.
- 3,200 environment rows and 12,800 drone rows passed the cross-file audit. All 4 episodes timed out; optimizer updates were 0. These four cloned worlds are not independent policy seeds. The source and instrumented replay have exactly the same aggregate evaluation measurements, including mean assigned RMSE `1.355322470263029 m`, so this replay did not show a metric change from logging.

Reproduce the phase summary without starting Isaac Sim:

```sh
uv run --locked python scripts/analyze_policy_telemetry.py \
  --source-run runs/policy-telemetry/20260918T130125.401428IST-bee09d21
```

The accepted analysis is `runs/policy-telemetry-analysis/20260918T130838.847388IST-ea74ac9d/report.json`. Its hash-linked report finds, during formation, mean command speed `0.0953 m/s`, mean realized speed `0.0861 m/s`, mean assigned-target distance `1.5766 m`, and target-aligned commands on `45.775%` of drone steps. The mean weighted tracking term was `−0.02502` per drone step versus `−0.004925` for the formation term. This describes the frozen flight; it does not isolate why the actor chose those commands.

At the first formation step (`episode_step=551`), the drones were about `1.21–1.31 m` below the `1.5 m` target altitude. The 200-step success dwell must begin by step 601 in an 800-step episode, leaving 50 steps or `0.5 s` from that observed state. The configured velocity command is capped at `0.5 m/s`, allowing `0.25 m` commanded travel before that deadline. Even granting all that travel toward the target, the altitude difference alone gives a calculated assigned-RMSE floor of `1.013–1.014 m` across the four worlds, above the `0.10 m` tolerance. This is a **command-budget bound under the declared interface**; simulator dynamics could briefly exceed a command. It does show that the observed takeoff-to-formation state and current dwell schedule are poorly matched. An unchanged-policy rerun with more time would test the schedule effect before changing reward weights or controller gains.
