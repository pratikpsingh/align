# Frozen active-group critic normalization

## Purpose

The accepted semantic diagnostic found that active velocity features varied
roughly ten to fourteen times less than position and target features. This
experiment standardizes the centralized critic's active position, velocity,
and target groups separately. It does not change the shared actor's 55-value
local observation or the physical observation scales used to assemble state.

This is a controlled representation experiment. A passing run establishes that
the transform, recovery, and evaluation contracts work. It does not establish
that the policy learns a better formation.

## Configuration

`configs/critic-normalization-active-groups.json` declares:

- 768 warmup simulator steps, equal to one configured rollout horizon;
- epsilon `1e-6` for a nonzero denominator;
- clipping at `[-5, 5]` after standardization;
- one frozen population mean and variance for each of position, velocity, and
  target.

`configs/critic-normalization-disabled.json` preserves the prior declared-scale
baseline. The older `training.normalization_enabled` field remains false and is
rejected if enabled; it is not the critic experiment switch.

## Data contract

The 80-value critic state contains eight agent slots with nine physical values
per slot, followed by eight masks. For each active slot, the warmup accumulates
all xyz values in these groups:

~~~text
position = slot[0:3]
velocity = slot[3:6]
target   = slot[6:9]
~~~

Inactive physical slots must be exactly zero and do not contribute statistics.
Masks must be binary and remain unchanged. The transform is:

~~~text
normalized = clip((value - group_mean) / sqrt(max(group_variance, epsilon^2)), -5, 5)
~~~

Inactive physical slots are written back as zero after the transform. This
prevents centering from turning padding into a signal.

## Warmup, training, and recovery

A new logical run performs the warmup before publishing checkpoint zero:

1. reset the vector task;
2. use the initialized shared actor without gradients or optimizer updates;
3. collect active critic-group reductions for 768 steps;
4. freeze the three group moments;
5. reset every environment and recurrent state;
6. publish checkpoint zero with the frozen moments and RNG state;
7. collect the first PPO rollout using normalized critic states.

Warmup transitions are reported separately and do not increment training
transition counters. They still consume simulator time and actor RNG, so the
checkpoint captures RNG after warmup.

A resumed attempt restores the frozen state and skips warmup. Fresh-process
evaluation restores and validates the same state but performs no normalization
updates. The actor does not consume the centralized critic state during
evaluation.

Learner-state schema version 2 makes the normalization payload mandatory and
validates its exact fields before atomic checkpoint publication and after load.
A source/configuration change therefore starts a new experiment lineage rather
than silently loading an older learner state.

## Acceptance checks

The simulator and host jointly require:

- exactly one warmup per seed and no repeated warmup on resumed updates;
- exactly 768 warmup rows per seed;
- exactly `4 environments x 4 active agents x 3 axes = 48` scalar samples per
  group in every warmup row;
- finite reductions and actions in `[-1, 1]`;
- binary masks, exactly four active slots, and zero inactive padding;
- a frozen state matching the resolved configuration in every checkpoint;
- identical normalization state across all three PPO updates;
- finite normalized critic states;
- fresh-process evaluation restoration with zero statistic updates;
- ordinary PPO counters that exclude warmup transitions.

The host independently parses `normalization-warmup.csv`; a self-reported
container flag alone is insufficient.

## Run the bounded comparison

Prepare and build a fresh immutable runtime context, then run the existing
three-update, two-seed stability protocol with the normalization configuration:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&

uv run --locked python scripts/run_training_stability.py \
  --build-report <build-run>/report.json \
  --critic-normalization-config configs/critic-normalization-active-groups.json \
  --accept-eula \
  --gpu 0
~~~

Use an allocated idle GPU. The launcher runs training and fresh-process
evaluation sequentially for seeds 41 and 73 with networking disabled.

## Artifacts

Artifacts are written under `runs/training-stability/<run-id>/`:

- `config.json` records the selected normalization configuration;
- each seed's `train/update-0001/normalization-warmup.csv` contains raw per-step
  reductions;
- each update's `metrics.json` records whether warmup ran or reused a
  checkpoint and reports warmup cost separately;
- checkpoint zero and later checkpoints contain the frozen normalization state;
- each evaluation `metrics.json` records restoration, immutability, and zero
  normalization updates;
- `stability-summary.json` aggregates training/evaluation ranges and warmup
  transitions;
- `report.json` contains the host audit result, commands, image, source,
  hardware, exits, and final status.

The run directory is ignored by Git and requires separate backup.

## Validation status

The configuration, schemas, launcher integration, raw-table audit, and existing
host contracts passed all 181 tests under locked Python 3.12 and Python 3.10.
The lab GPU contract is accepted below. Reliable formation learning remains
unestablished.

## Accepted lab evidence

Run `20260917T125147.124653IST-920dcb9c` passed on GPU 0 using derived image
`sha256:a5102a334f90d6043e7ac28f1cbc06a798ea8132bde30ec70d17d15e8fb74142`.
It started at 2026-09-17 12:51:47 IST, finished at 12:58:01 IST, and
took 374.752 seconds. All four sequential containers exited successfully, GPU
0 returned to 16 MiB, and no compute process remained.

For each seed, the host parsed exactly 768 warmup rows. The warmup contributed
3,072 environment transitions, 12,288 active-agent samples, and 36,864 scalar
samples to each xyz group. It took 7.715 seconds for seed 41 and 7.668 seconds
for seed 73. Those transitions remained separate from the 9,216 PPO environment
transitions per seed.

| Group | Seed 41 mean / standard deviation | Seed 73 mean / standard deviation |
|---|---:|---:|
| Position | 0.003741 / 0.129468 | 0.004927 / 0.130112 |
| Velocity | -0.000638 / 0.013491 | 0.001299 / 0.012572 |
| Target | 0.093750 / 0.181904 | 0.093750 / 0.181904 |

Every seed warmed up exactly once, reused an identical frozen state for updates
2 and 3, kept normal PPO counters exact, and published four contiguous
checkpoints. Both fresh evaluation processes loaded update 3, validated finite
normalized state, left normalization and actor parameters unchanged, and
performed zero normalization updates. All six PPO updates had zero value clip
fraction and stayed within the configured diagnostic guidance.

Explained variance changed substantially relative to the accepted declared-scale
three-update run:

| Seed | Declared-scale updates 1/2/3 | Normalized updates 1/2/3 |
|---:|---:|---:|
| 41 | -1.956 / -1.776 / -1.885 | 0.180 / 0.466 / 0.623 |
| 73 | -2.692 / -2.540 / -2.513 | 0.407 / 0.230 / -0.002 |

This supports testing normalization over a longer budget, but it is not a
formation-performance result. Deterministic evaluation still produced zero
successes and four time limits per seed. Compared with the earlier short
baseline, mean assigned RMSE was 0.0054 m worse, pairwise RMSE was 0.0390 m
lower, minimum separation was 0.0172 m lower, and reward differed by only
0.000042.

The two short runs are descriptive rather than a matched causal ablation. The
normalization warmup consumes actor RNG before PPO collection, so the stochastic
training rollouts are not identical to the earlier baseline rollouts. Two seeds
and three updates are also too small a budget for a performance conclusion.
The accepted result establishes the complete normalization, checkpoint,
recovery, evaluation, and accounting contracts and provides a justified input
to a longer bounded curve.
