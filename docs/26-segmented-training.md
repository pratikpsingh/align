# Segmented training across fresh simulator processes

## Purpose

A longer training command should not keep one Isaac Sim process alive for the
entire experiment. ALiGn can now divide one declared learning budget into short
segments. Every segment ends at a committed PPO update boundary. The next
segment starts a fresh simulator process, loads the latest validated checkpoint,
resets the physical environments and recurrent memory, and continues the same
learner lineage.

This is reset-mode learner recovery. It restores the actor, critic, both Adam
optimizers, frozen critic normalization, random-number generator states, and
counters. It does not restore an unfinished PhysX trajectory.

## Configuration

[learning-curve-segmented.json](../configs/learning-curve-segmented.json) is a
schema-2 learning-curve configuration:

| Setting | Value |
|---|---:|
| Policy seeds | 41, 73 |
| Updates per seed | 12 |
| Updates per simulator process | 4 |
| Segment ranges | 0-4, 4-8, 8-12 |
| Declared process restart | after segment 1 |
| Evaluation milestones | 0, 6, 12 |
| Evaluation steps per environment | 800 |

Ranges use completed-update counters. Segment 4-8 restores checkpoint 4 and
commits checkpoints 5 through 8. The total budget stays at 12 in every segment's
resolved configuration and checkpoint. Segment bounds are launcher arguments,
so they do not change the experiment identity or checkpoint configuration hash.

The acceptance command explicitly selects
[critic-normalization-active-groups-floor.json](../configs/critic-normalization-active-groups-floor.json).
Only the first segment may collect normalization warmup samples. Later segments
must recover the exact frozen normalization state and report zero warmup
transitions.

## Acceptance checks

Each segment retains the existing live stability checks. The host then joins
the segment records and requires:

- the configured ranges appear exactly once and in order;
- every segment passes and produces the expected update counters;
- the next segment's starting checkpoint ID and payload hash equal the previous
  segment's final checkpoint;
- completed updates are exactly 1 through 12, with no gap or duplicate;
- the final counter is 12;
- critic normalization is identical across segments;
- no resumed segment repeats normalization warmup;
- the declared boundary uses two distinct container processes.

Checkpoint 0 and every completed update remain available for exact historical
selection. Evaluations still run in separate simulator processes and load
updates 0, 6, and 12 explicitly.

## Run it

Prepare and build a fresh immutable runtime image, then run:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&

uv run --locked python scripts/run_segmented_learning.py \
  --build-report <build-run>/report.json \
  --critic-normalization-config configs/critic-normalization-active-groups-floor.json \
  --accept-eula \
  --gpu 0
~~~

The launcher first verifies that the requested GPU is idle. It starts
containers sequentially and keeps Docker networking disabled. A per-container
timeout protects each segment independently.

## Artifacts

A run is written under `runs/segmented-learning/<run-id>/`. Each seed contains:

- `checkpoints/`: initialization and per-update immutable checkpoints;
- `train-segment-0000-0004/`, `train-segment-0004-0008/`, and
  `train-segment-0008-0012/`: separate simulator logs, events, raw rollout
  metrics, critic-distribution tables, and segment results;
- `evaluation-update-0000/`, `evaluation-update-0006/`, and
  `evaluation-update-0012/`: deterministic physical evaluation records.

The top-level report keeps every container command, exit status, duration,
checkpoint boundary, and joined checks. The learning-curve JSON and CSV tables
retain per-update training diagnostics and cross-seed evaluation metrics.
`runs/` is ignored by Git and requires separate backup.

## Accepted lab evidence

Run `20260917T144058.713679IST-643868db` passed on GPU 0 using derived image
`sha256:655e092493103af4b36cbf0f67aa5b388286b1c5c753b23e63e1164db097c520`.
It ran from 2026-09-17 14:40:58 IST to 15:00:09 IST and took 1,150.737
seconds. GPU 0 returned to 16 MiB at zero utilization.

Both seeds completed all three four-update segments and all three exact
checkpoint evaluations. The run performed 24 PPO updates, six deterministic
evaluations, and four fresh-process training resumes. At every process boundary,
the restored checkpoint ID and payload hash matched the preceding committed
checkpoint. Update counters were exactly 1 through 12. Each seed collected one
3,072-environment-transition normalization warmup; resumed segments collected
none. All joined segment checks passed.

Training value clipping stayed at zero. Critic explained variance was finite
and positive across all 24 updates, ranging from 0.0278 to 0.7861. Velocity
clipping remained zero through update 11; at update 12 one seed clipped
0.0407 percent of velocity scalars. This small observed tail exceeds the
historically calibrated floor without invalidating the segment/recovery
contract and should remain visible in later scaling decisions.

Cross-seed deterministic evaluation means were:

| Update | Reward | Assigned RMSE (m) | Pairwise RMSE (m) | Minimum separation (m) |
|---:|---:|---:|---:|---:|
| 0 | -0.02291 | 1.42304 | 0.27672 | 0.97477 |
| 6 | -0.02091 | 1.36412 | 0.32204 | 0.74826 |
| 12 | -0.02148 | 1.37596 | 0.36523 | 0.60131 |

Assigned error and reward improved at update 6, then regressed partly at update
12. Pairwise shape error worsened and minimum separation fell. All 24 evaluated
episodes reached the time limit; none satisfied formation success. The run
validates segmented process recovery and extends the measured budget. It does
not establish reliable formation learning or convergence.

The schema, range construction, Docker arguments, joined counter order,
checkpoint continuity, restart identity, and warmup rules also pass all 192
host tests under Python 3.12 and the isolated Python 3.10 environment. Ruff,
formatting, compile, and Git diff checks pass.
