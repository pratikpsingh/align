# Training stability calibration and deterministic evaluation

## Purpose and limits

This command extends the accepted two-update integration check in three ways:

1. it collects three full recurrent PPO batches for each of two declared policy seeds;
2. it measures PPO diagnostics again after every parameter update;
3. it starts a separate simulator process, loads the final checkpoint, and evaluates deterministic actor actions without optimizer steps.

The command validates the experiment path and reports whether diagnostic values cross declared guidance thresholds. It does not claim convergence, successful formation learning, or statistical performance from two seeds and three updates.

## Configuration

The defaults are explicit in:

- [training-stability.json](../configs/training-stability.json): seeds 41 and 73, three updates per seed, 800 evaluation steps, and diagnostic guidance;
- [recurrent-stability-rollout.json](../configs/recurrent-stability-rollout.json): 768 simulator steps per update, four environments, four agents, and 16-timestep recurrent chunks;
- [recurrent-stability-task.json](../configs/recurrent-stability-task.json): an 800-step task time limit;
- [recurrent-ppo-calibration.json](../configs/recurrent-ppo-calibration.json): one full-batch PPO epoch, actor learning rate 3e-4, and critic learning rate 1e-4.

The earlier integration profile used two PPO epochs and a critic learning rate of 1e-3. Its second epoch produced full value clipping and a large critic gradient in one batch. The calibration profile is a conservative candidate chosen to measure that problem. It is not yet an accepted final hyperparameter set.

Construction enters its formation phase after 50 ground steps and 500 takeoff steps. A 768-step batch therefore has enough configured time to include the formation phase if the policy remains safe. The reports separately record whether evaluation trajectories physically reached that phase; early termination is retained as an outcome rather than hidden.

## Acceptance and guidance

The run passes its engineering contract only when:

- both seeds complete exactly three updates;
- counters equal the configured environment and agent transition totals;
- initialization plus all update checkpoints form complete per-seed histories;
- rollout tensors, losses, gradients, and post-update diagnostics are finite;
- each final checkpoint loads in a new simulator process;
- deterministic evaluation writes exactly 800 rows per environment;
- actions remain bounded, simulator action clipping remains zero, and evaluated actor parameters do not change.

The configured KL, policy clip, value clip, and pre-clipping gradient limits are **guidance flags**. Crossing them does not erase a mechanically valid run. Instead, the guidance field in the training metrics shows which update needs investigation. This separation prevents an arbitrary early threshold from discarding useful calibration evidence.

Post-update measurements evaluate the stored rollout after the optimizer step. This is needed for the one-epoch profile: the ordinary epoch row is evaluated before that epoch's parameter update and cannot by itself show the resulting policy/value displacement.

## Run it

Prepare and build a fresh immutable runtime context so the new source and configuration are included:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&

uv run --locked python scripts/run_training_stability.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The launcher checks that the selected GPU is idle, then runs four sequential containers: training and evaluation for seed 41, followed by training and evaluation for seed 73. It uses one allocated GPU and disables container networking. Each phase has a timeout of 1,800 seconds by default.

## Artifacts

Artifacts are written under runs/training-stability/<run-id>/:

- report.json: commands, exits, machine/runtime identity, per-seed results, and final status;
- config.json: the shared resolved configuration;
- seed-0000000041/ and seed-0000000073/: seed-resolved configuration, checkpoints, training artifacts, and evaluation artifacts;
- train/update-0001/ through update-0003/: raw rollout CSV, PPO epoch CSV, and metrics including post-update diagnostics;
- evaluation/evaluation.csv: one row per environment and evaluation step, including reward, formation errors, separation, action range, phase, and outcome;
- stability-summary.json: ranges and means across seeds for selected update and evaluation measurements.

Native Isaac Sim logs remain under each phase's kit-logs directory. runs/ is ignored by Git and must be backed up separately.

## Reading a result

Start with report.json. A passed status says the calibration and independent evaluation paths completed their contracts. Then inspect:

1. each update's stability_guidance flags;
2. post-update approximate KL and clip fractions;
3. critic gradient norms and explained variance;
4. evaluation phase_rows, formation_phase_reached, outcome counts, formation errors, and minimum separation;
5. raw CSV rows whenever a summary looks surprising.

Reward means from stochastic training and deterministic evaluation are not directly comparable. The action mode, policy state, and trajectories differ. Use deterministic evaluation across matched checkpoints and declared episodes for performance comparisons.

## Accepted lab evidence

Run 20260916T230826.067057IST-2534c4be passed on GPU 0, an RTX A4000,
using derived image
sha256:6c77caa126320cb01ce6fa25b4d82d2498f7eebe095efba376810ead748274c9.
It started at 2026-09-16 23:08:26 IST, finished at 23:14:30 IST, and
took 364.050 seconds. All four containers exited successfully: training and
fresh-process evaluation for seeds 41 and 73.

The exact totals were:

| Measurement | Per seed | Combined |
|---|---:|---:|
| PPO updates | 3 | 6 |
| Environment transitions | 9,216 | 18,432 |
| Agent transitions | 36,864 | 73,728 |
| Checkpoints including initialization | 4 | 8 |
| Deterministic evaluation rows | 3,200 | 6,400 |

All hard training and evaluation checks passed. Both evaluators loaded their
update-3 checkpoint, wrote 800 steps for each of four environments, kept actor
parameters unchanged, produced finite metrics and bounded actions, and required
no action clipping. Each evaluation reached the formation phase. The phase
counts per seed were 200 ground rows, 2,000 takeoff rows, and 1,000 formation
rows. Each environment ended once by the 800-step time limit; there were no
successes or safety terminations.

Deterministic evaluation measurements were:

| Seed | Mean team reward | Mean assigned RMSE | Mean pairwise RMSE | Minimum separation |
|---:|---:|---:|---:|---:|
| 41 | -0.0221353 | 1.40203 m | 0.289348 m | 0.890459 m |
| 73 | -0.0214139 | 1.36446 m | 0.375418 m | 0.997040 m |

These are measurements of short, minimally trained policies. No episode
satisfied the formation success contract, so the result is not evidence of a
learned formation controller.

## Optimizer finding

Post-update approximate KL ranged from 0.0036704 to 0.0127210, and policy
clip fraction ranged from 0.0327962 to 0.2027995; both stayed within the
declared guidance. Pre-clipping critic gradient norms ranged from 3.5815 to
11.1496, below the guidance limit of 50.

The post-update value clip fraction was 1.0 for all six updates, exceeding
the 0.5 guidance threshold every time. Post-update explained variance was also
negative for all updates, ranging from -7.4858 to -3.7917. Reducing the
critic learning rate from 1e-3 to 1e-4 and using one PPO epoch therefore
did not resolve critic displacement. This calibration profile is operationally
validated but is not accepted for sustained learning.

The next optimizer investigation should record return and predicted-value
distributions, test value/return scaling or normalization, and compare a
smaller critic step using matched seeds and rollouts. Do this before a long
baseline training run.

The host-safe configuration, result gates, and summary aggregation pass with
the full 160-test suite under Python 3.10 and Python 3.12. Rendering and video
remain unvalidated. The run directory is ignored by Git and requires separate
backup.
