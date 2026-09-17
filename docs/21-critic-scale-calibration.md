# Critic scale and matched-rollout calibration

## Purpose

The accepted stability run showed a post-update value clip fraction of 1.0 for
all six updates. This command isolates the critic step from trajectory
variation. For each policy seed it collects one real 768-step rollout, freezes
that rollout in memory, and applies three candidate critic learning rates to
separate critic copies that start from identical weights.

This is an optimizer calibration. Candidate critic copies are temporary and are
not published as training checkpoints. The ordinary primary learner still
performs one task-connected PPO update and publishes its normal atomic
checkpoint, which confirms that calibration did not replace the training path.

## Controlled comparison

The configured candidates are:

| Candidate | Critic learning rate |
|---|---:|
| Current calibration baseline | 0.0001 |
| Intermediate step | 0.00003 |
| Small step | 0.00001 |

The actor, rollout, returns, recurrent chunks, initial critic parameters, Adam
epsilon, value loss, value clip range, and gradient clip are identical within
each seed. Only the candidate critic learning rate changes.

Each seed contributes exactly:

~~~text
768 steps x 4 environments = 3,072 critic samples
3,072 samples x 3 candidate rates = 9,216 raw comparison rows
~~~

Seeds 41 and 73 use different initialized policies and stochastic rollouts.
Candidate comparisons are matched within a seed; samples are not matched
between seeds.

## Recorded distributions

The run records count, minimum, maximum, mean, population standard deviation,
and the 5th, 25th, 50th, 75th, and 95th percentiles for:

- team rewards;
- old rollout values;
- bootstrap values;
- GAE returns and advantages;
- return minus old value;
- post-update predictions;
- signed and absolute value deltas;
- post-update return residuals.

For every candidate, the raw critic-samples.csv table retains environment ID,
rollout step, reward, old value, return, advantage, pre-update prediction,
post-update prediction, value delta, absolute delta, and value-clip indicator.

The pre-update prediction must reproduce the value stored during collection
within 1e-5. This check confirms that recurrent chunks reconstruct the same
critic calculation before comparing candidate steps.

## Semantic critic-input diagnostics

The critic state has 80 scalars: eight fixed agent slots, nine physical values
per slot, followed by eight presence masks. Each physical slot contains world
position xyz, velocity xyz, and target xyz. The new diagnostic assigns an
explicit name to every index and writes `critic-feature-summary.csv`.

For every active feature it records the same distribution statistics as the
value targets, plus the fraction equal to zero and the fraction at the declared
`[-1, 1]` scaling boundary. It also pools active values into position, velocity,
and target groups. Padding is excluded from these physical distributions. Mask
features retain all samples so unused capacity remains visible.

Hard acceptance now also requires:

- every state has exactly the configured 80 dimensions;
- every state contains exactly four active-agent masks for this task;
- masks are binary;
- all inactive feature slots are zero;
- declared scaled physical features remain in `[-1, 1]`;
- the host independently parses exactly 80 named feature rows.

The calibration also reports Pearson correlation, mean error, mean absolute
error, and RMSE between old values and returns and between immediate team
reward and returns. Correlation measures ordering; error measures scale and
offset. Both are needed because explained variance can be poor even when the
mean prediction is close.

## Acceptance and selection

Hard acceptance requires:

- both seed containers and their physics rollouts to pass;
- exactly three candidates per seed;
- exactly 3,072 valid critic samples per candidate;
- finite input, loss, gradient, prediction, and delta measurements;
- identical rollout-value reconstruction before every candidate update;
- positive parameter changes;
- an independently parsed raw sample CSV with the exact row count;
- an independently parsed 80-row semantic feature summary whose mask, padding,
  bounds, and active-agent checks all pass.

A candidate is within initial guidance when no more than 50% of its samples
move beyond the configured value clip range of 0.2. The host reports the
largest candidate rate satisfying that guidance for both seeds. This is a
calibration recommendation, not proof that the critic will learn accurately
over many updates.

## Why return normalization is not included yet

Return normalization changes the meaning of the critic output. A correct
implementation must store running statistics in checkpoints, transform targets
during training, convert values consistently for GAE and bootstrapping, freeze
statistics during evaluation, and restore them during recovery. Adding it
before measuring the unnormalized value and target scales would mix two
questions.

This command first determines whether a smaller optimizer step alone controls
value displacement. The saved distributions provide evidence for deciding
whether normalization or a value-function redesign is needed next.

## Run it

Prepare and build a fresh immutable runtime context, then run:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&

uv run --locked python scripts/run_critic_calibration.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The launcher uses one allocated GPU and two sequential containers, one per
policy seed. Networking is disabled.

## Artifacts

The run is saved under runs/critic-calibration/<run-id>/:

- report.json contains runtime identity, commands, exits, per-seed results, and
  the cross-seed summary;
- critic-summary.json compares candidate ranges and gives the largest rate
  within guidance on both seeds;
- each seed directory contains its exact resolved configuration and checkpoint;
- probe/rollout.csv contains the physical rollout;
- probe/critic-samples.csv contains every matched candidate sample;
- probe/critic-feature-summary.csv contains one named row for every critic input;
- probe/updates.csv and metrics.json retain the primary PPO update and all
  critic calibration summaries;
- native simulator logs, events, scene, and assets remain with each seed.

The generated run directory is ignored by Git and requires separate backup.

## Accepted lab evidence

Run `20260916T233640.753380IST-74e912b3` passed on GPU 0 using derived
image
`sha256:43190c5528e6f7023dfbea577b63183883fdbe7f83c7084265dc7449796be36d`.
It started at 2026-09-16 23:36:40 IST, finished at 23:39:29 IST, and
took 168.837 seconds. Both seed containers exited successfully, and the host
independently validated both raw CSV files.

Each seed supplied 3,072 critic samples to every candidate. The run retained
9,216 candidate rows per seed and 18,432 rows overall. Recurrent replay matched
the values recorded during collection with a maximum absolute error between
7.15e-7 and 8.05e-7, well below the 1e-5 contract. Candidate branches changed
their own parameters, while the primary critic parameters and learner RNG state
remained unchanged before the ordinary PPO update.

| Critic rate | Post-update clip fraction, both seeds | Absolute value-delta p95 range | Guidance |
|---:|---:|---:|---|
| 1e-4 | 1.000000 | 0.993358-1.010998 | exceeded |
| 3e-5 | 0.938802 | 0.297137-0.304019 | exceeded |
| 1e-5 | 0.000000 | 0.098899-0.101336 | passed |

The cross-seed selection is therefore `1e-5`. This is the largest tested rate
that stayed within the declared value-displacement guidance for both seeds.

The input measurements also explain why this remains a calibration result.
Mean old values were 0.27850 and 0.67934, while mean returns were -0.25585 and
0.07548. Mean advantages were consequently -0.53435 and -0.60386. Post-update
explained variance stayed negative for every candidate. The smaller rate
controls one-step displacement; it does not establish an accurate critic or a
learned flight policy.

Selected-step run `20260916T234710.834029IST-0733818e` completed that
experiment. Its six consecutive updates all had a post-update value clip
fraction of 0.0. Mean explained variance improved from -4.9114 at rate `1e-4`
to -2.2270 at `1e-5`, but stayed negative. Deterministic evaluation still
produced no successful formation episodes after three updates. The rate
`1e-5` is accepted as the current unnormalized optimizer setting; formation
learning and critic accuracy remain unestablished.

The full 166-test suite passed under Python 3.10 and Python 3.12, along with
Ruff lint, format, compilation, and diff checks. Rendering and video remain
unvalidated. The ignored run directory requires separate backup.


## Accepted semantic-diagnostic evidence

Run `20260917T121849.878540IST-4cabd9a6` passed on GPU 0 in 167.734 seconds
using image
`sha256:efcb66564b781556fac6b83196b3bde56dfb3831f4061a52ebfea926ee23d3f2`.
Both simulator probes, all critic-input checks, both raw candidate tables, and
both 80-row feature tables passed. GPU 0 returned to 16 MiB and no compute
process remained.

Both seeds had exactly four active masks per state, binary masks, zero inactive
padding, bounded physical inputs, and no physical values at the `[-1, 1]`
boundary. Group standard deviations were approximately 0.130 for position,
0.182 for target, and 0.0126-0.0135 for velocity. Velocity therefore enters the
critic at roughly one tenth to one fourteenth the variation of the other
physical groups before the critic's whole-vector LayerNorm.

Old-value versus return correlation was -0.1524 for seed 41 and +0.1554 for
seed 73. The selected `1e-5` update changed these to -0.1514 and +0.1494. It
controlled value displacement but did not improve ordering in this single
step.

That controlled change is now implemented and accepted as run
`20260917T125147.124653IST-920dcb9c`. It collected one declared active-slot
warmup per seed, kept padding and masks unchanged, froze and checkpointed group
moments, skipped warmup on resume, and restored them without evaluation updates.
Short-run explained variance improved markedly, while formation outcomes stayed
mixed with no successes. See [frozen critic normalization](23-frozen-critic-normalization.md)
for the contract, raw counts, results, and limits.
