# Interrupted segmented-training recovery

## Purpose

A process can stop after collecting part of a rollout but before PPO changes the
policy. ALiGn now records that unfinished work explicitly and resumes from the
last committed update in a new run directory. The source run remains immutable.

A checkpoint is the transaction boundary. A completed checkpoint contains the
actor, critic, optimizers, frozen critic normalization, random-number generator
states, counters, and lineage. An unfinished rollout contains simulator state
and recurrent memory that are not checkpointed. Recovery therefore discards the
partial rollout, resets the environments and recurrent memory, and recollects
that update from the committed checkpoint.

## Deliberate interruption acceptance

The interruption flags are acceptance-test controls. They are accepted only as
a complete set, only by segmented training, and only for the first update after
a noninitial segment boundary. This prevents an injected fault from being
mistaken for an ordinary training option.

For the current segmented configuration, seed 41 finishes updates 1 through 4.
The next fresh process starts update 5 and is stopped after rollout step 64,
before PPO or checkpoint publication. With four environments and four agents,
that records and then discards 256 environment transitions and 1,024 agent
transitions.

Run the prepared derived-image build, then the source interruption command:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<build-run-id> && \
uv run --locked python scripts/run_segmented_learning.py \
  --build-report runs/runtime-build/<build-run-id>/report.json \
  --critic-normalization-config configs/critic-normalization-active-groups-floor.json \
  --inject-interruption-seed 41 \
  --inject-interruption-update 5 \
  --inject-interruption-rollout-step 64 \
  --accept-eula \
  --gpu 0
```

The expected outcome is `interrupted` and process exit code 2. That exit code is
part of the acceptance contract; it does not mean the injected fault was
missed. The source report is under `runs/segmented-learning/<source-run-id>/`.
Do not rerun into that directory.

## Recovery command

Use the exact interrupted source directory printed by the first command:

```sh
sudo -v
uv run --locked python scripts/recover_segmented_learning.py \
  --source-run runs/segmented-learning/<source-run-id> \
  --accept-eula \
  --gpu 0
```

Recovery creates `runs/segmented-learning-recovery/<recovery-run-id>/`. It does
not write into the source. It copies and reverifies the committed checkpoint
prefix, retains the original logical learner identity, completes missing
training segments, and evaluates checkpoints 0, 6, and 12.

## Acceptance contract

The host and recovery runner require all of the following:

- the source status and interruption record agree with the requested seed,
  update, and rollout step;
- the flushed `rollout.csv` has every expected step/environment row;
- no PPO result or checkpoint exists for the interrupted update;
- completed source segments form one valid prefix;
- copied checkpoint payload hashes match the source through update 4;
- joined updates for each seed are exactly 1 through 12, without gaps or
  duplicates;
- frozen critic normalization and checkpoint lineage continue across the new
  processes;
- normalization warmup appears exactly once per seed;
- the partial rollout is labeled discarded and is never included in an update;
- hashes of every source-run file match before and after recovery.

The recovery run stores complete source inventories, copied-checkpoint records,
source-versus-recovery artifact origins, container commands, raw update tables,
and deterministic evaluation results. `runs/` remains ignored and needs a
separate backup.

## Accepted lab evidence

Interrupted source run `20260917T195211.368696IST-61fd6fc0` used derived image
`sha256:d6cbb8a18616bcdf9c8e07616b1f0fd0b75969ccadf65f5b52209f5fc71f17b2`.
It ran from 2026-09-17 19:52:11 IST to 19:55:24 IST and took 192.752 seconds.
Seed 41 committed updates 1 through 4, restored checkpoint 4 in a fresh
process, then stopped update 5 after 64 rollout steps. Its flushed file contains
256 ordered environment rows, representing 1,024 agent transitions. The source
contains neither PPO metrics nor a checkpoint for update 5; checkpoint 4 and
payload hash
`b7100da308cadba02bb6601ecf171d20cdfcc93920ad8e2360a5fea92b8e4775`
remain latest.

Recovery run `20260917T200314.721203IST-6c0eec2f` passed using the same image.
It ran from 2026-09-17 20:03:14 IST to 20:20:29 IST and took 1,034.817 seconds.
It copied and reverified checkpoints 0 through 4 for seed 41, then performed 20
remaining PPO updates: updates 5 through 12 for seed 41 and 1 through 12 for
seed 73. Joined counters for each seed are exactly 1 through 12 with no
duplicates. Six fresh-process evaluations loaded exact checkpoints 0, 6, and
12. Seed 41 retained its 3,072-transition warmup from the source; its resumed
segments performed zero warmup. Seed 73 warmed up once and also performed zero
warmup in later segments.

The complete source inventory matched before and after recovery: 60 files,
997,697,590 bytes, and tree SHA-256
`47107e7e96fbf7849b4c06f947bac07f2696581fcab1a87bc5ad6a45fcf12826`.
GPU 0 returned to 16 MiB at zero utilization. The interrupted partial rollout
is preserved and explicitly counted as discarded.

Deterministic cross-seed evaluation means were:

| Update | Reward | Assigned RMSE (m) | Pairwise RMSE (m) | Minimum separation (m) |
|---:|---:|---:|---:|---:|
| 0 | -0.02291 | 1.42304 | 0.27672 | 0.97477 |
| 6 | -0.02091 | 1.36412 | 0.32204 | 0.74826 |
| 12 | -0.02148 | 1.37596 | 0.36523 | 0.60131 |

All 24 evaluated episodes reached their time limit and none achieved formation
success. These measurements validate interruption and recovery mechanics; they
do not establish reliable formation learning or convergence.

The implementation passes 197 CPU tests under locked Python 3.12 and an
isolated Python 3.10 installation. Ruff, formatting, compilation, and patch
checks pass.
