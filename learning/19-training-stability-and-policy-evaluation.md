# Training stability and policy evaluation

## Why the working training loop is not enough

A finite PPO update proves that tensors, gradients, and the optimizer connect. It does not prove that repeated updates are well behaved. A critic can remain finite while changing so much that every stored value crosses the clip range. A reward can also change because the sampled trajectories changed, even when the policy did not improve.

The next check asks two narrower questions:

- Do several updates remain measurable and recoverable across more than one random initialization?
- Can a new process load the saved actor and measure its flight without changing it?

These are prerequisites for a longer experiment.

## Pre-update and post-update measurements

PPO collects a rollout with an old policy. During optimization it compares the current policy with the action probabilities saved in that rollout.

For an action with old log probability log p_old and current log probability log p, the importance ratio is

\[
r = \exp(\log p - \log p_{old}).
\]

Approximate KL and policy clip fraction summarize how far the policy moved. Value clip fraction plays a similar role for the critic.

Suppose one PPO epoch is configured. The loss for that epoch is calculated before optimizer.step(). At that point the just-loaded model still matches the rollout policy, so KL and clip fraction can be near zero. That does not measure the change made by the step. ALiGn now evaluates the same stored sequence chunks once more after the step, without gradients or another update. Those values are saved under post_update.

## Why the candidate optimizer is conservative

The earlier two-update integration used:

~~~text
critic learning rate: 0.001
PPO epochs:            2
~~~

One second-epoch batch had value clip fraction 1.0 and a pre-clipping critic gradient norm near 100. The calibration candidate changes these to:

~~~text
critic learning rate: 0.0001
PPO epochs:            1
~~~

This is a hypothesis: a smaller critic step and one pass may reduce movement away from rollout values. The calibration measures the hypothesis. It does not declare the settings optimal.

Gradient clipping still limits the applied global gradient norm to 0.5. The saved gradient norm before clip describes pressure from the batch; it is useful even though the actual optimizer step uses the clipped gradient.

## Why use two seeds

A policy seed controls network initialization and action sampling. One seed can look stable or unstable by chance. Seeds 41 and 73 are two independent checks, not enough for a publication result.

Each seed receives three updates:

~~~text
768 steps x 4 environments x 3 updates
    = 9,216 environment transitions

9,216 transitions x 4 drones
    = 36,864 agent transitions per seed
~~~

Across two seeds, the bounded run collects 18,432 environment transitions and 73,728 agent transitions.

Each update starts reset worlds and zero LSTM state, then restores the learner and RNG state from the previous checkpoint. Unfinished physical episodes are labeled abandoned. This preserves the established reset-mode recovery semantics while avoiding a hidden LSTM state from the previous batch.

## Why the longer rollout matters

The construction timing is:

~~~text
steps 0-49:    ground
steps 50-549:  takeoff
steps 550+:    formation
~~~

The earlier 64-step integration batches mainly checked wiring. A 768-step calibration batch provides an opportunity to reach formation control. A policy may terminate and reset before step 550, so configured opportunity and observed phase coverage are different facts. The evaluation report records phase row counts and whether formation was actually reached.

## What independent deterministic evaluation means

Training samples from the bounded Gaussian policy. Evaluation instead uses the transformed mean action:

~~~text
local observation -> actor + LSTM -> latent mean -> tanh -> bounded action
~~~

For each seed, the host ends the training container and starts a new evaluation container. The evaluator:

1. verifies the final checkpoint manifest and SHA-256 payload;
2. creates new actor, critic, and optimizer objects;
3. restores the saved learner state;
4. resets the simulator and LSTM memory;
5. uses deterministic actor actions for 800 steps;
6. performs no backward pass or optimizer step;
7. checks that every actor parameter is bitwise unchanged afterward.

The evaluator restores the optimizer because the complete recovery loader validates one coherent checkpoint. It does not use that optimizer.

## Guidance flags versus pass checks

A pass check establishes a hard engineering invariant, such as exact row count, finite tensors, bounded actions, or unchanged evaluation parameters.

A guidance flag compares a diagnostic with an initial threshold. These thresholds help find concerning updates, but they are not yet scientifically calibrated. A run can therefore pass its data-integrity contract while showing one or more guidance alerts. That result is useful: it says the measurement is trustworthy and the optimizer candidate needs more work.

## Data flow and files

For one seed:

~~~text
reset physics
  -> stochastic 768-step rollout
  -> recurrent chunks and GAE
  -> PPO update
  -> post-update diagnostics
  -> atomic checkpoint
  -> repeat three times
  -> end simulator process

new simulator process
  -> verify and load final checkpoint
  -> deterministic 800-step rollout
  -> raw evaluation.csv + outcome summary
~~~

rollout.csv supports inspection of the training inputs. updates.csv records the loss evaluation used for each optimizer epoch. Each update's metrics.json adds the post-update measurement and guidance. evaluation.csv is the raw independent trajectory table.

## How to interpret the result

First verify the run and per-seed phases passed. Next look for:

- increasing post-update KL or clip fractions across updates;
- critic gradients that repeatedly cross the guidance value;
- nonfinite data, which is always a hard failure;
- early safety outcomes that prevent formation phase coverage;
- large differences between the two seeds;
- deterministic formation error, minimum separation, and outcome counts.

Do not infer learning from a lower training reward or from these six updates. The result decides whether the optimizer profile is safe enough for a longer baseline or which diagnostic requires another controlled calibration.


## What the lab calibration taught us

The accepted run was 20260916T230826.067057IST-2534c4be. All six updates
were finite and recoverable, and both final actors loaded in separate processes.
That establishes the data path. It does not establish stable critic learning.

The policy diagnostics were controlled: post-update approximate KL stayed
between about 0.0037 and 0.0127, and at most about 20.3% of policy samples
crossed the PPO clip interval. The critic behaved differently. Every sample in
every update crossed the configured 0.2 value-change interval, so value clip
fraction was 1.0 six times. Explained variance stayed negative.

A useful interpretation is:

~~~text
small individual parameter change
    does not imply
small change in the network's output values
~~~

The largest critic parameter change was only about 0.0001 per update, but the
deep network can combine many small weight changes into a value-output shift
larger than 0.2 for every sample. Gradient clipping bounds the parameter update
mechanism; it does not guarantee a bound on predicted-value movement.

Both deterministic evaluations reached formation time without a safety
termination. Their assigned formation RMSE remained roughly 1.36-1.40 m, and
all four worlds per seed ended at the time limit with no success. This is
expected from only three updates and must not be presented as learned
formation behavior.

The four environments within one deterministic evaluation share the same
initial condition and deterministic actor, so their rows are parallel simulator
replicas rather than four independent policy seeds. The scientifically
independent units in this calibration are the two policy seeds. Future
performance evaluation needs declared environment/task variation and more
seeds.

The next controlled question concerns critic scale. Before long training,
measure the distributions of stored old values, bootstrap values, returns,
advantages, and post-update value deltas. Then compare a smaller critic step
and an explicitly documented value or return normalization choice on matched
seeds. The deterministic evaluator can remain unchanged and compare the
resulting checkpoints.
