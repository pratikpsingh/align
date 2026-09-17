# Critic values and matched optimizer updates

## What the critic predicts

The centralized critic receives the declared global training state and predicts
one expected future team return for each environment step. It does not choose
drone actions. Its output helps estimate whether the actor's sampled actions
performed better or worse than expected.

For one transition, generalized advantage estimation produces an advantage A.
The target return is

\[
R = V_{old} + A.
\]

The critic is trained to move its current prediction toward R.

## What value clipping measures

Let V_old be the value saved during collection and V_new the current critic
prediction. PPO constructs a clipped alternative:

\[
V_{clip} = V_{old} + \operatorname{clip}
  (V_{new} - V_{old}, -0.2, 0.2).
\]

The loss uses the larger squared error from the unclipped and clipped
predictions. Value clip fraction is the fraction of samples where

\[
|V_{new} - V_{old}| > 0.2.
\]

A fraction of 1.0 means every valid sample crossed that interval. It does not
mean the code numerically clipped the final network output. It means the
diagnostic found that all outputs moved farther than the configured trust
region.

## Why matched rollouts matter

Consider two critic learning rates tested in separate flights. The policies may
sample different actions, produce different rewards, terminate at different
times, and create different returns. A smaller value delta could come from an
easier batch instead of the learning rate.

The matched experiment removes that ambiguity:

~~~text
one initialized policy and critic
        |
one real stochastic physics rollout
        |
one set of recurrent chunks and returns
        |
        +--> critic copy at learning rate 1e-4
        +--> critic copy at learning rate 3e-5
        +--> critic copy at learning rate 1e-5
~~~

Every copy starts with the same parameters and sees the same ordered recurrent
samples. Only its Adam learning rate differs.

## Why recurrent reconstruction is checked

The rollout stores the critic's old value at every physical step. Later, PPO
groups samples into episode-safe recurrent chunks. Each chunk stores the LSTM
state from the start of that sequence.

Before updating a candidate, ALiGn reruns the critic over those chunks. Its
prediction should reproduce the stored old value. A mismatch would mean the
comparison starts from a different recurrent calculation, making later value
deltas hard to interpret.

The calibration requires the maximum mismatch to stay below 1e-5.

## Reading distribution statistics

A mean alone can hide a small group of extreme samples. The calibration stores
percentiles as well:

- p05 and p95 show the central 90% range;
- the median shows a typical sample without sensitivity to extremes;
- standard deviation shows overall spread;
- minimum and maximum retain the extremes;
- absolute value-delta p95 says how far 95% of predictions moved.

The raw CSV remains the source evidence when these summaries are surprising.

Useful comparisons include:

~~~text
return standard deviation
versus
old-value standard deviation

absolute value-delta p95
versus
the 0.2 value clip range
~~~

If returns are much larger than predictions, the critic may need many careful
steps or a well-defined normalization scheme. If a single small Adam step
moves all predictions too far, the learning rate or network sensitivity is the
immediate concern.

## Gradient clipping and Adam step size

Gradient clipping limits the global gradient norm before Adam applies its
adaptive update. Adam divides each parameter's running gradient estimate by an
estimate of its recent scale. On the first step, many parameter changes can be
close to the configured learning rate even when raw gradients differ greatly.

A small maximum parameter change does not guarantee a small output change.
Thousands of coordinated parameter changes pass through the recurrent network
and can shift every predicted value.

This is why the experiment measures outputs directly instead of choosing a
learning rate from gradient norm alone.

## Why candidate critics are temporary

The candidate branches are diagnostic counterfactuals. Publishing all of them
as training checkpoints would create several competing lineages from one
rollout and make it easy to resume the wrong experiment.

ALiGn saves their complete measurements and raw sample rows. The normal primary
PPO path publishes one checkpoint using the declared baseline configuration.
A later training configuration can deliberately adopt a selected rate with
clear experiment lineage.

## What a passing calibration establishes

A pass establishes that:

- the live rollout and critic targets are finite;
- recurrent replay reproduces the collected values;
- candidate comparisons use matched data and weights;
- raw rows and summaries are complete;
- the launcher can repeat the comparison across two policy seeds.

If a smaller rate stays within value-clip guidance on both seeds, it becomes a
reasonable candidate for another multi-update stability run. It does not prove
positive explained variance, reward improvement, convergence, or formation
success.

## When normalization should be added

Value or return normalization becomes appropriate only with an explicit state
contract. Running mean, variance, and count must be checkpointed. Values used
for GAE and bootstrapping must be in a consistent scale. Evaluation must freeze
statistics. Recovery must restore them exactly.

The matched calibration measures the unnormalized scale first. Its evidence
will tell us whether a smaller optimizer step is sufficient or whether the
normalization work is justified.

## Accepted lab result

The accepted two-seed run compared all three learning rates on exactly the same
rollout and initial critic within each seed:

| Critic rate | Fraction beyond 0.2 | Typical p95 absolute movement |
|---:|---:|---:|
| 1e-4 | 100% | about 1.00 |
| 3e-5 | 93.88% | about 0.30 |
| 1e-5 | 0% | about 0.10 |

The nearly proportional output movement is strong evidence that the optimizer
step was the immediate source of full value clipping. The `1e-5` candidate
kept every sampled prediction inside the 0.2 interval on both seeds.

The critic still had negative explained variance. Its initial prediction mean
also differed substantially between seeds, while each seed had a mean advantage
near -0.6. This means the selected step controls how far the critic moves, but
does not show that its predictions rank future returns correctly.

The repeated multi-update check at `1e-5` then kept all six updates at
zero value clipping. The declared settings are retained in
`configs/recurrent-ppo-selected.json`, which the stability launcher uses by
default.

Explained variance became less negative, but did not become positive, and the
three-update policies did not complete the formation task. This separates two
findings: the optimizer now moves predictions by a controlled amount, while the
critic has not yet learned useful return structure. A longer bounded run can
test whether the second finding changes with experience. If it does not,
normalization or the critic target design should be investigated with its own
checkpoint and evaluation contract.


## Why feature-level measurements come before normalization

The centralized state is already divided by declared physical scales and then
passes through LayerNorm. Negative explained variance alone does not prove that
one input feature is too large. It says the critic's prediction errors vary
more than the returns, which can also come from poor ordering, bootstrapped
targets, or insufficient representation.

The critic receives eight slots. Each slot has position xyz, velocity xyz, and
target xyz, followed by one mask per slot. For four drones, the last four slots
must remain zero with masks equal to zero. The semantic diagnostic measures
active physical values separately from this padding. Otherwise the zeros from
unused capacity would make feature variance look artificially small.

For example, suppose a target feature has standard deviation 0.01 after scaling
while velocity has standard deviation 0.4. That imbalance supports testing
empirical feature normalization. If all physical groups occupy useful ranges
but predicted values are negatively correlated with returns, changing input
scale alone has weaker justification. Return normalization or critic-target
design may then be the better controlled experiment.

The new calibration therefore answers four questions before changing the
learner:

1. Are masks, active-agent counts, and padded slots correct?
2. Are physical features finite, bounded, saturated, or nearly constant?
3. How do value and reward signals correlate with GAE returns?
4. Does a candidate update improve value-to-return alignment as well as loss?

A passing diagnostic establishes trustworthy measurements, not that
normalization is beneficial. The selected active-group experiment now has an
explicit warmup, frozen checkpoint state, fresh-evaluation contract, and
accepted two-seed run; [the normalization lesson](22-frozen-critic-input-normalization.md)
explains its behavior and limits.


## What the semantic diagnostic found

Accepted run `20260917T121849.878540IST-4cabd9a6` showed that the state assembly
is correct: every row had four active agents, every unused slot was zero, and
no physical input reached its clipping boundary. Input corruption and clipping
are therefore poor explanations for the negative explained variance.

The physical groups do not vary on similar scales. Position and target values
had standard deviations near 0.130 and 0.182, while velocity was near 0.013.
Whole-vector LayerNorm does not give each semantic group its own variance; the
small velocity changes can remain weak relative to the fixed positions,
targets, and masks.

Value-to-return correlation also changed sign between seeds and stayed almost
unchanged after the conservative critic step. This pointed to representation
and input balance rather than optimizer displacement. The accepted next
experiment normalized only active critic position, velocity, and target groups;
it did not change decentralized actor inputs.

Statistics must be collected in a separate warmup rollout and then frozen. If
they changed inside the same rollout, values stored during collection would no
longer match values reconstructed for PPO clipping. Frozen statistics preserve
that identity. Padding remains zero and masks remain binary, because centering
padded zeros would turn absence into a nonzero feature.
