# Critic normalization denominator floor

## Purpose

The accepted normalized learning curve showed that the initialized-policy
velocity warmup standard deviation, about 0.013 for both seeds, became too small
for later policy-state distributions. By update 10, the original transform
clipped 18.1 percent of seed 41 velocity scalars and 3.0 percent of seed 73
velocity scalars.

This experiment adds a fixed minimum standard deviation to the frozen critic
normalizer. It changes only the centralized critic input transform. Actor
observations, actions, rewards, PPO losses, warmup means, raw warmup variances,
and environment dynamics remain unchanged.

## Transform

For group `g`, the schema-2 transform uses:

~~~text
raw_warmup_std[g] = sqrt(max(warmup_variance[g], epsilon^2))
effective_std[g] = max(raw_warmup_std[g], minimum_standard_deviation)
normalized[g] = clip((raw[g] - warmup_mean[g]) / effective_std[g], -5, 5)
~~~

The configured minimum is `0.025`. The accepted warmup position and target
standard deviations were about 0.13 and 0.18, so their denominators are
unchanged. Only velocity uses the floor in those observed runs.

Schema-1 configs and checkpoints remain readable and imply a zero floor. A new
schema-2 checkpoint records `minimum_standard_deviation` and uses the explicit
`frozen_active_critic_group_standardization_with_floor` contract. Resume and
evaluation must match that value exactly.

## Historical calibration

The host command reads an accepted drift run without changing it:

~~~sh
uv run --locked python scripts/calibrate_critic_normalization_floor.py \
  --source-run runs/learning-curve/20260917T131555.325446IST-648f12e3
~~~

It validates ten distribution tables for seeds 41 and 73, hashes the source
report, configuration, metrics, and input tables, and compares the candidates
`0.015`, `0.020`, `0.025`, and `0.030`. For each candidate it recomputes, from
the same observed raw extrema and moments:

- the largest absolute velocity value before the `[-5, 5]` clamp;
- the largest absolute velocity mean shift using the effective denominator;
- the largest velocity spread ratio using the effective denominator;
- whether position and target denominators would change.

A candidate passes when the observed extrema prove zero clipping, effective mean
shift and spread ratio are each at most 1.5, and position and target stay
unchanged. The smallest passing candidate is selected.

Host calibration run `20260917T135710.911636IST-36a377ac` passed and retained 42
source hashes. Its results were:

| Floor | Maximum pre-clamp | Effective mean shift | Effective spread ratio | Pass |
|---:|---:|---:|---:|:---:|
| 0.015 | 7.857 | 2.094 | 2.283 | no |
| 0.020 | 5.893 | 1.571 | 1.712 | no |
| 0.025 | 4.714 | 1.257 | 1.370 | yes |
| 0.030 | 3.928 | 1.047 | 1.142 | yes |

The selection is a conservative calibration against already observed states. It
does not establish that a new policy will visit the same states or learn a
better formation.

## Distribution artifacts

New per-update tables distinguish:

- `warmup_standard_deviation`: the measured raw warmup spread;
- `normalization_standard_deviation`: the denominator actually used;
- mean shift and spread ratio relative to the raw warmup spread;
- mean shift and spread ratio relative to the effective denominator.

Historical tables without the effective fields remain readable. New learning
curve summaries retain both raw-drift and effective-transform trends.

## Run the GPU acceptance

Prepare and build a fresh derived image, then run the same two-seed, ten-update
curve with the schema-2 configuration:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&

uv run --locked python scripts/run_learning_curve.py \
  --build-report <build-run>/report.json \
  --critic-normalization-config configs/critic-normalization-active-groups-floor.json \
  --accept-eula \
  --gpu 0
~~~

Acceptance requires twenty valid updates, six exact fresh-process evaluations,
zero or materially reduced velocity clipping, finite effective-scale metrics,
checkpoint restoration of the schema-2 floor, and all existing learner gates.
Formation error, separation, reward, and outcome counts must still be reported;
critic conditioning alone is not flight success.

## Accepted lab evidence

Run `20260917T140141.720844IST-685b2854` passed on GPU 0 using derived image
`sha256:844a0f348f76323ebb9c74cc59507f56689fbdb3779c136fbcf5c36488eed81d`.
It started at 2026-09-17 14:01:41 IST, finished at 14:15:38 IST, and took
836.446 seconds. It completed twenty PPO updates and six exact fresh-process
evaluations. GPU 0 returned to 16 MiB at zero utilization.

Both seeds passed independent warmup and distribution-table audits. All twenty
value clip fractions were zero. The schema-2 floor was checkpointed, restored
across updates, and restored without modification for evaluation.

Velocity clipping was zero in every update, compared with a maximum of 18.1
percent in the unfloored run. Raw velocity drift still reached 2.424 warmup
standard deviations and 2.488 times the warmup spread. The effective values
seen by the critic were bounded to a 1.308 mean shift and 1.343 spread ratio.
This confirms that the report preserves raw drift while the floor controls the
numerical input scale.

At update 10, critic explained variance was 0.775 for seed 41 and 0.208 for seed
73. Both remained positive. Deterministic evaluation still produced no success;
all eight environments reached the time limit. Cross-seed means were:

| Update | Reward | Assigned RMSE (m) | Pairwise RMSE (m) | Minimum separation (m) |
|---:|---:|---:|---:|---:|
| 0 | -0.02291 | 1.42304 | 0.27672 | 0.97477 |
| 5 | -0.02134 | 1.37845 | 0.31075 | 0.78806 |
| 10 | -0.02093 | 1.36352 | 0.34842 | 0.62057 |

Relative to the matched unfloored update-10 evaluation, assigned error was
0.00591 m lower, reward was 0.000147 higher, and minimum separation was 0.00825
m higher, while pairwise error was 0.00853 m worse. These are small mixed
differences from two seeds and do not establish formation learning or
superiority.

The final validation passed all 188 tests under locked Python 3.12 and the
isolated simulator-compatible Python 3.10 environment. Ruff lint, formatting,
compile, and Git diff checks passed. The fixed `0.025` denominator floor is
accepted as the normalization configuration for the next bounded training
work; dependable formation completion remains unresolved.
