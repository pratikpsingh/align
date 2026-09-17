# Critic distribution drift and normalized learning curves

## Purpose

The accepted frozen-normalization run used one 768-step warmup to estimate
position, velocity, and target statistics. Those moments remained fixed for
three PPO updates. The next bounded experiment must establish whether later
rollouts remain represented by those initial statistics before increasing the
training budget to ten updates.

This implementation measures the input distribution seen by the centralized
critic on every update. It does not change actor observations, rewards, actions,
PPO losses, or frozen normalization statistics.

## Measurements

For every active position, velocity, and target group, each update records:

- raw count, mean, population standard deviation, minimum, and maximum;
- the checkpointed warmup mean and standard deviation;
- raw mean shift measured in warmup standard deviations;
- raw standard deviation divided by warmup standard deviation;
- normalized mean, standard deviation, minimum, and maximum;
- count and fraction that exceeded the configured `[-5, 5]` range before
  clipping.

The current task contributes exactly:

~~~text
768 rollout steps x 4 environments x 4 active agents x 3 axes
= 36,864 scalar samples per group per update
~~~

Padding is excluded using the critic masks. Masks remain unchanged, inactive
physical slots remain zero, and raw declared-scale features must remain within
`[-1, 1]`.

## Relationship to PPO

At each rollout step the collector retains the raw critic state before applying
the frozen transform. PPO continues to store and train on the normalized state.
After collection, the reporter reapplies the checkpointed transform to every raw
state and requires bit-for-bit equality with the normalized rollout tensor.
This ties the measurements to the exact data used by the critic.

No running statistics are updated. Distribution shift is evidence about the
fixed transform, not a mechanism that silently changes it.

## Raw and aggregate artifacts

Each update writes:

~~~text
train/update-XXXX/critic-distribution.csv
~~~

The table has exactly three rows: position, velocity, and target. The host
launcher independently validates its schema, update number, active scalar
count, finite values, declared raw bounds, configured normalized bounds, and
clipped fraction arithmetic.

For a learning curve, `learning-curve-summary.json` contains one cross-seed
entry per update and group. It reports minimum, maximum, and mean for:

- clipping fraction;
- mean shift in warmup standard deviations;
- raw standard deviation ratio.

`critic-distribution-curve.csv` stores the same trends in long form for plotting.
Historical curves without these measurements remain readable and receive an
empty drift section.

## Interpretation

A clipping fraction near zero means later active values remain inside five
warmup standard deviations. It does not by itself prove good conditioning.

A mean shift near zero means the group center resembles the warmup center. A
standard-deviation ratio near one means its spread resembles the warmup spread.
Large values indicate that the frozen transform is becoming stale even if
clipping has not begun.

These measurements must be read alongside explained variance, value loss,
formation error, separation, outcomes, and reward. Better critic fit without
better deterministic flight remains a calibration result.

## Run the normalized ten-update curve

First prepare and build a fresh immutable runtime context:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&

uv run --locked python scripts/run_learning_curve.py \
  --build-report <build-run>/report.json \
  --critic-normalization-config configs/critic-normalization-active-groups.json \
  --accept-eula \
  --gpu 0
~~~

The launcher uses one allocated GPU. It trains ten updates for seeds 41 and 73
and evaluates exact checkpoints 0, 5, and 10 in fresh processes. It also audits
one warmup table and ten critic-distribution tables for each seed. Networking is
disabled.

The earlier learning-curve run needed recovery after one evaluation container
stalled. The existing immutable recovery command remains applicable if this run
finishes training but later evaluation startup stalls. Do not overwrite or
modify the source run.

## Acceptance

A passing run requires:

- one valid frozen-statistics warmup per seed;
- ten contiguous checkpointed PPO updates per seed;
- thirty valid distribution rows per seed;
- exact 36,864 active scalar samples in every group and update;
- finite drift and clipping measurements;
- exact fresh-process evaluation of updates 0, 5, and 10;
- restored normalization with no evaluation updates;
- retained raw tables, checkpoints, configuration, source/runtime identity,
  logs, and failures.

The experiment may show that normalization stays suitable, becomes stale, or
fails to improve flight. All three outcomes are useful. The accepted lab evidence and its limits are recorded below.

## Accepted lab evidence

Run `20260917T131555.325446IST-648f12e3` passed on GPU 0 using derived image
`sha256:50b4ce61d1097c238305df90110c2da0f30fd5acbe430187e0cedaa8cfb1ea8b`.
It started at 2026-09-17 13:15:55 IST, finished at 13:29:42 IST, and
took 827.057 seconds. It completed twenty PPO updates and six fresh-process
evaluations. GPU 0 returned to 16 MiB at zero utilization.

Both seed reports passed their independent warmup and critic-distribution CSV
audits. The retained artifacts contain exactly two warmup tables and twenty
per-update distribution tables. Every group row contains 36,864 active scalar
samples. All twenty value clip fractions were zero and all configured training
guidance checks passed.

The final host validation passed all 184 tests under locked Python 3.12. Ruff
lint, Ruff formatting, and Git diff-integrity checks also passed. The same 184
tests had passed under the simulator-compatible Python 3.10 environment before
the image was built.

### Distribution drift

Position stayed close to its warmup distribution over ten updates: clipping was
zero, absolute mean shift stayed below 0.161 warmup standard deviations, and
the standard-deviation ratio stayed between 0.984 and 1.029. Target was fixed by
this task, with zero shift, unit spread ratio, and zero clipping.

Velocity did not remain close to warmup:

| Seed | Update 1 clip / shift / spread ratio | Update 10 clip / shift / spread ratio |
|---:|---:|---:|
| 41 | 0.0000 / -0.024 / 0.959 | 0.1811 / +2.329 / 2.539 |
| 73 | 0.0002 / -0.084 / 1.067 | 0.0296 / -0.405 / 1.993 |

Seed 41 velocity clipping began at update 4 and rose to 18.1 percent by update
10. Seed 73 had intermittent early clipping and reached 3.0 percent. The
warmup-derived velocity denominator is therefore too narrow for the later
policy-state distributions, even though the transform remains finite and
mechanically correct.

### Critic and flight outcomes

Seed 41 post-update explained variance remained positive and reached 0.755 at
update 10. Seed 73 varied around zero but ended at 0.237. The declared-scale
curve ended between -2.711 and -2.628, so normalized critic fit remained much
better across this bounded run.

Deterministic evaluation still produced no successful episodes. All eight
environments at each milestone reached the time limit. Cross-seed means were:

| Update | Reward | Assigned RMSE (m) | Pairwise RMSE (m) | Minimum separation (m) |
|---:|---:|---:|---:|---:|
| 0 | -0.02291 | 1.42304 | 0.27672 | 0.97477 |
| 5 | -0.02134 | 1.37776 | 0.31470 | 0.76309 |
| 10 | -0.02108 | 1.36943 | 0.33990 | 0.61232 |

Assigned error and reward improved modestly from initialized evaluation, while
pairwise error worsened and minimum separation declined substantially. At
update 10, the normalized curve had slightly lower assigned error than the
prior declared-scale curve, but worse pairwise error and much lower separation.
No superiority or reliable formation learning is established.

### Decision

The fixed warmup normalizer is accepted as a correct implemented mechanism, but
its velocity statistics are not accepted as stable for a longer training
budget. Further training should not simply extend this configuration. The next
controlled normalization design must reduce velocity amplification or update
statistics at explicit rollout boundaries while preserving recurrent PPO
reconstruction, checkpoints, resume behavior, and frozen evaluation.
