# Normalization denominator floors

## Why a small denominator matters

Normalization subtracts a mean and divides by a standard deviation. If the
denominator is very small, an ordinary later value becomes a large normalized
number.

The velocity warmup in this task had a standard deviation close to `0.013`.
Suppose a later velocity differs from the warmup mean by `0.10`:

~~~text
0.10 / 0.013 = 7.69
~~~

The critic input is clipped to 5, so it loses the distinction between values
that produce 5.1, 6, or 7.69.

## What a floor does

A standard-deviation floor places a lower bound on the divisor:

~~~text
effective denominator = max(measured standard deviation, configured floor)
~~~

With the selected floor of `0.025`, the same example becomes:

~~~text
0.10 / 0.025 = 4.0
~~~

That value remains inside the clamp. A floor does not change the raw simulator
state. It controls how strongly the critic magnifies that state.

## Why 0.025 was selected

The previous ten-update run retained the raw minimum, maximum, mean, and spread
for every group and update. The largest recorded velocity deviation required a
floor of at least `0.02357` to fit inside five effective standard deviations.

The project compared rounded candidates on exactly those same twenty velocity
distributions. `0.020` was too small. Both `0.025` and `0.030` passed, so the
smaller passing value was selected. This avoids adding more compression than
the evidence requires.

Position and target had warmup spreads near `0.13` and `0.18`. Since both exceed
`0.025`, their transforms are unchanged.

## Raw drift versus effective conditioning

The new reports keep two views:

1. Raw drift compares a later rollout with the measured warmup statistics. This
   tells us whether the policy changed its state distribution.
2. Effective conditioning compares the same rollout with the divisor actually
   used by the critic. This tells us what numerical scale reaches the network.

The floor should improve effective conditioning. It does not make the raw drift
disappear, and the reports must not claim that it does.

## Checkpoints and evaluation

The floor is part of the checkpointed normalization state. Resumed training and
fresh evaluation reject a different floor. Statistics remain frozen within and
between rollouts; evaluation never updates them.

This creates a clean experiment lineage. A checkpoint trained with the old
`0.013` velocity divisor cannot silently resume with `0.025`.

## What the GPU curve found

The new ten-update curve confirmed the expected numerical effect on newly
collected states. Velocity clipping stayed at zero for both seeds. Raw velocity
still moved far from warmup, but the largest effective mean shift was 1.31 and
the largest effective spread ratio was 1.34.

Both critics retained positive explained variance at update 10. Flight metrics
were still mixed: assigned error and minimum separation were slightly better
than the unfloored run, pairwise shape error was slightly worse, and no episode
formed successfully.

This is the practical lesson: a well-conditioned critic input is a requirement
for reliable value learning, but it does not supply the policy with enough
training, exploration, or reward signal to solve the formation task by itself.
The floor is now a stable foundation for the next experiment rather than a
claim that formation learning is complete.
