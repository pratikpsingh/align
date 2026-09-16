# Recurrent PPO updates

## What this component does

The collector records what the old policy did. The PPO updater asks how the
actor and critic should change after seeing that experience.

For each drone action, the actor stores the probability assigned when the
action was collected. During an update, the actor evaluates the same action
again. The ratio between the new and old probabilities says how much the policy
has changed its preference for that action.

## Why PPO clips the ratio

Suppose an action has positive advantage: it worked better than the critic
expected. Increasing its probability is useful, but one batch should not make
the policy completely different. PPO caps the reward for moving the ratio too
far.

With a clip ratio of 0.2, the useful interval is approximately `[0.8, 1.2]`.
The actor can still move outside it, but the clipped objective stops rewarding
that larger move for the affected sample. The clip fraction reports how many
valid samples crossed the interval.

The advantage is a team quantity in this cooperative task. One environment
timestep has one team advantage, and the updater repeats it for the four drone
actions from that timestep. Each actor sequence still uses its own local
observation and LSTM memory.

## Why advantages are normalized

Advantage magnitude depends on reward scale and episode history. Subtracting
the valid-batch mean and dividing by its standard deviation gives a more stable
optimization scale. Padding must be removed before calculating those
statistics. Otherwise the number of padded rows changes the update even though
no additional experience was collected.

## Why the critic is clipped separately

The critic estimates one future team return per environment. It can also move
too far in a single update, so its new prediction is compared with a version
clipped around the stored old value. The loss uses the worse of the clipped and
unclipped squared errors.

Actor clipping controls probability ratios. Critic clipping controls value
changes. They solve related stability problems but operate on different
quantities.

## Recurrent training versus length-one samples

The updater passes an entire ordered chunk through the LSTM:

~~~text
initial memory -> step 0 -> step 1 -> step 2 -> ...
~~~

It does not turn each timestep into an independent one-step sequence. The
initial hidden and cell states come from the rollout at the chunk's exact start.
This lets gradients teach the network how earlier observations should affect a
later action.

## The padding experiment

One fixture sequence contains a padded final row. The probe makes a second copy
and replaces every padded field with obviously different values, including an
action outside the controller range and large advantages and returns.

Both copies begin with identical networks, Adam states, and random seeds. After
two updates, actor and critic parameters remain exactly identical. This shows
that padded loss fields cannot affect learning and that the LSTM's valid mask
prevents a padded observation from advancing memory.

## How to read the diagnostics

- `approximate_kl` summarizes policy-distribution movement;
- `policy_clip_fraction` reports how many actor samples crossed the PPO ratio
  interval;
- `value_clip_fraction` reports how many critic changes crossed its limit;
- gradient norm measures update pressure before clipping;
- explained variance compares critic errors with variation in target returns.

These numbers diagnose an update. None proves that a swarm learned to fly.

In the accepted synthetic probe, the second value clip fraction was 1.0 and
explained variance was strongly negative. That is evidence that the tiny
fixture's second critic step was aggressive. The implementation correctly
reported it rather than treating the checkpoint as a successful learned
policy.

## What is established now

ALiGn can compute finite recurrent actor and critic objectives, propagate
gradients through both LSTMs, clip gradient norms, change both parameter sets,
and exclude padding. It does so with the vendor CUDA runtime used by the future
trainer.

The probe uses constructed sequence data. It does not yet feed the accepted
live rollout into repeated training updates or evaluate a changed policy in
flight.

## What comes next

The [training recovery mechanism](17-training-recovery.md) now saves the actor,
critic, both optimizer states, normalization state, random-number states,
counters, immutable configuration, and checkpoint lineage. Its atomic write and
fallback checks passed in the vendor CUDA runtime.

The bounded [task-connected training loop](18-task-connected-training-loop.md) now
alternates live collection and recurrent PPO updates across two simulator
processes and saves recoverable checkpoints. Its critic diagnostics warn that
stable learning and meaningful reward trends remain unproven.
