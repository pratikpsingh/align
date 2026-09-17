# Frozen critic input normalization

## The problem

The centralized critic predicts one team value from the full training state.
The actor makes each drone's action from a different, local observation. The
semantic diagnostic showed that the critic's active position and target values
changed much more than its velocity values. A neural network can learn around
that difference, but in a short data budget the larger-varying groups may
dominate the first layers.

This experiment asks one narrow question: does giving the critic similarly
scaled active feature groups improve its value learning? It leaves the actor,
reward, actions, PPO update, and task unchanged.

## Standardization in a small example

Suppose the warmup velocity group has mean `0.02` and variance `0.0004`. Its
standard deviation is `sqrt(0.0004) = 0.02`. A later velocity value of `0.03`
becomes:

~~~text
(0.03 - 0.02) / 0.02 = 0.5
~~~

Suppose position has mean `0.10`, variance `0.04`, and a later value is `0.20`:

~~~text
(0.20 - 0.10) / 0.20 = 0.5
~~~

The raw changes had different scales, but both normalized deviations now mean
"half a standard deviation above the warmup mean." Values beyond five standard
deviations are clipped so one unusual state cannot create an unbounded critic
input.

## Why only active slots count

The critic has capacity for eight drones, while the current task uses four. The
remaining four slots are padding. Including padded zeros in the statistics
would mix "drone absent" with a physical value of zero and pull every group
mean toward zero.

The warmup therefore uses masks to select only active slots. After
standardizing active values, it explicitly writes inactive physical features
back to zero. Mask bits themselves remain zero or one and are never
standardized.

## Why statistics freeze before PPO

PPO compares values recorded during collection with values recomputed during
training. If the mean or variance changed while a rollout was being collected,
the same physical state could have two different numerical representations.
The value-clipping comparison would then mix a policy update with a preprocessing
change.

ALiGn collects one separate warmup rollout, freezes the moments, resets all
environments and recurrent memories, and only then starts PPO collection. Every
state in that experiment lineage uses the same transform.

## Data flow

~~~text
new run
  -> reset task
  -> actor actions, no gradient
  -> raw critic states
  -> active group sums and squared sums
  -> frozen mean/variance
  -> reset task and LSTM memory
  -> normalized critic rollout
  -> recurrent PPO
  -> checkpoint model + optimizer + RNG + frozen statistics

resume
  -> restore checkpoint
  -> skip warmup
  -> reset task and LSTM memory
  -> reuse the same transform

evaluation
  -> restore checkpoint
  -> validate frozen statistics
  -> run deterministic local actor
  -> update no statistics
~~~

The actor path never passes through the critic normalizer. Centralized
information remains training-only.

## Counts for the configured experiment

One warmup uses 768 steps, four vector environments, and four active drones:

~~~text
768 x 4 = 3,072 environment transitions
3,072 x 4 = 12,288 active-agent samples
12,288 x 3 = 36,864 scalar samples per xyz group
~~~

Each CSV row covers one vector step, so each group reduction contains:

~~~text
4 environments x 4 agents x 3 axes = 48 scalars
~~~

These warmup transitions are a real compute cost, but they are not PPO training
samples and do not advance training counters.

## Where it is implemented

- `critic_normalization_config.py` defines the host-safe experiment settings.
- `torch_normalization.py` accumulates active moments, validates checkpoint
  state, and transforms centralized critic inputs.
- `task_training.py` performs warmup once, resets the task, applies the frozen
  transform, and records separate costs.
- `torch_recovery.py` requires the normalization state in learner-state schema
  version 2.
- `policy_evaluation.py` verifies that fresh-process evaluation restores but
  never changes the state.
- `stability_runtime.py` independently audits raw warmup rows and aggregates the
  experiment.

## What establishes correctness

A valid run must show more than finite loss. It must demonstrate that:

1. masks and inactive padding preserve their meanings;
2. active counts and raw reductions match the configured task exactly;
3. normalized states stay finite;
4. resumed updates do not collect another warmup;
5. every update uses identical frozen moments;
6. fresh evaluation restores the moments without updating them;
7. training counters exclude the warmup while reports include its cost.

Only after those checks pass should we compare explained variance, value loss,
formation error, separation, outcomes, and reward with the prior
un-normalized runs.

## Limitations

Group normalization does not guarantee useful value ordering. One scalar mean
and variance per semantic group also assumes its x, y, and z components should
share a scale. The warmup comes from an untrained initialized policy and may not
cover states reached later in training. Clipping can hide distribution shift,
so future longer runs should measure normalized saturation.

A positive short-run result would justify a longer controlled comparison. A
negative result would be useful evidence to investigate target/value
normalization, critic representation, reward horizon, or training budget rather
than repeatedly changing the optimizer step.

## What the lab run showed

Accepted run `20260917T125147.124653IST-920dcb9c` exercised the complete path on
two seeds. Each seed produced the expected 768 warmup rows, 36,864 values per
physical group, one frozen checkpoint state, three PPO updates, and a fresh
evaluation process that made no statistic updates.

The measured standard deviations reproduced the diagnostic imbalance:
position was about `0.13`, target about `0.182`, and velocity only
`0.0126-0.0135`. After group normalization, five of six post-update explained
variance values were positive; the sixth was close to zero. The earlier
three-update declared-scale run had negative explained variance for all six
updates.

That improvement concerns the critic's fit to its sampled returns. It did not
produce formation success after three updates. Assigned formation error was
slightly worse on average, pairwise error was lower, minimum separation was
lower, and reward was essentially unchanged relative to the earlier short run.

The comparison is not sample-for-sample matched because the warmup consumes
stochastic actor draws before PPO collection. The result therefore motivates a
longer curve; it does not prove that normalization caused better flight or will
remain beneficial across more seeds and states.
