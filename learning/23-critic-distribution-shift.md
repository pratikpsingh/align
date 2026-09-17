# Critic distribution shift

## Why frozen statistics can become stale

The normalizer learns its mean and variance from an initialized policy. Training
then changes the policy, which changes where the drones fly and therefore the
states seen by the centralized critic.

Imagine that warmup positions have mean `0.00` and standard deviation `0.13`.
If a later rollout has mean `0.26`, its center has shifted by:

~~~text
(0.26 - 0.00) / 0.13 = 2 warmup standard deviations
~~~

The values may still fit inside the five-standard-deviation clip range, but the
critic now sees a noticeably displaced distribution.

## Three complementary signals

### Clipping fraction

This is the fraction of active values whose standardized magnitude exceeds
five before clipping. It detects severe outliers or drift.

### Mean shift

This reports how far the new center moved, expressed in warmup standard
deviations. Values near zero mean the center stayed similar.

### Standard-deviation ratio

This divides the new raw standard deviation by the warmup standard deviation.
A ratio of one means similar spread; two means the later rollout is twice as
spread out.

Clipping can remain zero while the other two quantities reveal meaningful
drift. That is why all three are retained.

## Worked example

Suppose the velocity warmup has mean `0.001` and standard deviation `0.013`.
A later rollout has mean `0.014` and standard deviation `0.0195`:

~~~text
mean shift = (0.014 - 0.001) / 0.013 = 1.0
spread ratio = 0.0195 / 0.013 = 1.5
~~~

The group moved one warmup standard deviation and became 50 percent more
variable. If all standardized values remain inside `[-5, 5]`, clipping is still
zero. The transform is usable, but no longer makes the group exactly centered
with unit variance.

## Data path

~~~text
raw centralized state
  -> save active values for audit
  -> frozen group normalization
  -> store normalized state in recurrent rollout
  -> PPO critic update

raw audit tensor + frozen checkpoint statistics
  -> reproduce normalized rollout exactly
  -> compute three group rows
  -> host validates raw CSV
  -> aggregate both seeds by update
~~~

The actor's local observation never enters this path. Padded critic slots are
excluded, and masks are neither centered nor scaled.

## What correctness looks like

For the present task, every group row must represent exactly 36,864 active
scalars. Each update must have position, velocity, and target rows. The raw
values must remain within their declared `[-1, 1]` scales, normalized values
must remain within `[-5, 5]`, and clipped counts must agree with clipped
fractions.

The strongest internal check reapplies the frozen normalizer to the raw states
and compares the result with the exact recurrent rollout tensor used by PPO.
If they differ, the run fails before its learning statistics are interpreted.

## What the longer curve was designed to answer

The ten-update run tested whether the positive short-run critic explained
variance persisted, whether input groups drifted away from warmup, and whether
critic changes coincided with deterministic formation changes at updates 0, 5,
and 10.

Two seeds and ten updates are still a bounded calibration. A positive trend
would justify a larger controlled training run. It would not establish the
paper's reported results, deployment feasibility, or superiority over another
method.

## What the ten-update run found

Accepted run `20260917T131555.325446IST-648f12e3` showed why drift measurements
were necessary. Position stayed close to its warmup distribution and target did
not change. Velocity spread grew as the policy changed.

For seed 41, the velocity mean moved from nearly centered to 2.33 warmup
standard deviations above the original mean. Its spread grew to 2.54 times the
warmup spread, and 18.1 percent of its scalar values exceeded the five-standard-
deviation range. Seed 73 reached 1.99 times the warmup spread and 3.0 percent
clipping.

The critic still fit returns much better than the earlier declared-scale
critic. That did not yield successful formations. Assigned error improved
slightly, while pairwise error increased and drones came closer together during
deterministic evaluation.

This separates three ideas:

1. the normalization implementation is correct and recoverable;
2. normalization can improve short-run value prediction;
3. one initialized-policy warmup is not a stable velocity scale as the learned
   policy changes.

The next design should address velocity drift explicitly. Updating moments
between rollouts is possible only if each rollout uses one frozen snapshot and
the checkpoint records the running accumulator and active snapshot. A simpler
fixed denominator floor is another controlled option. Either choice needs a
new lineage and the same reconstruction and evaluation checks.
