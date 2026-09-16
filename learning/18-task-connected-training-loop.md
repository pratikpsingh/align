# How the task-connected training loop fits together

## The complete loop

Earlier components were checked separately. The task produced observations and
rewards, the collector produced recurrent sequences, PPO updated a synthetic
batch, and recovery reproduced a synthetic next update. The bounded trainer now
connects them with real drone physics:

~~~text
fresh/reset worlds
      |
local actor observations + centralized critic state
      |
bounded stochastic actions -> drone physics -> team reward
      |
time-ordered rollout + boundary masks + starting LSTM states
      |
GAE returns -> recurrent PPO epochs
      |
atomic learner checkpoint at the completed update boundary
~~~

The actor still receives only its bounded local observation. The centralized
critic state exists only in the training path. These 64-step acceptance
rollouts did not reach an episode boundary; the earlier collector check tested
terminal and truncation masks separately.

## Why two processes matter

Reloading into new model objects inside one Python process tests serialization,
but it does not prove the simulator launcher can restart. The acceptance run
uses two Docker processes. The second process rebuilds the scene, creates new
drones and controllers, resets them, loads the learner state, and begins with
zero actor and critic memory.

The policy and optimizer continue. The unfinished physical flight does not.
That is the practical meaning of reset-mode training recovery.

## Counting experience without duplication

Each attempt collects:

~~~text
64 steps x 4 environments = 256 environment transitions
256 transitions x 4 drones = 1,024 agent transitions
~~~

Attempt 1 ends at update 1. Attempt 2 restores those exact counters and adds the
same quantities once, ending at 512 environment transitions and 2,048 agent
transitions. The host compares the attempt-2 starting counters directly with
the attempt-1 ending counters.

Four physical episodes were still in progress when attempt 1 ended. Attempt 2
recorded those four as abandoned before starting new reset episodes. Their old
LSTM memory was not reused.

## Why there are three checkpoints for two updates

The first checkpoint stores the initialized networks, Adam state, RNG state,
and zero counters. It provides a recovery point before the first collection.
Update 1 points to that initialization payload, and update 2 points to update
1. This produces a verifiable three-link history:

~~~text
initialization (update 0) -> first update -> resumed second update
~~~

Each arrow is the SHA-256 of the parent payload, rather than only a filename.

## Reading the raw data

`training-rollout.csv` has one row per environment transition. It records team
reward, critic value, bootstrap value, action range, assigned and pairwise
formation error, minimum separation, and terminal flags. It therefore supports
post-run checks that do not trust only a final status field.

`training-updates.csv` has one row per PPO epoch. Useful columns include actor
and critic losses, approximate KL, clip fractions, entropy, explained variance,
and gradient norms before clipping.

The active training timer includes collection and optimization. The total host
duration also includes two expensive simulator startups, scene creation,
artifact copying, checkpoint writes, and shutdown.

## Why a passing loop may still be a poor learner

The acceptance rule asks whether the calculation is finite, connected, bounded,
and recoverable. It does not require reward to improve in two batches.

The first attempt illustrates the distinction. Its second critic epoch had a
large value loss and strongly negative explained variance. Negative explained
variance means the critic predictions were worse at explaining return variation
than a constant baseline for this small batch. A value clip fraction of 1.0
means every valid critic sample crossed the configured value-change limit.
Gradient clipping prevented an unbounded parameter step, but it did not make the
batch a good learning signal.

Those diagnostics are a reason to calibrate update scheduling and value/reward
scales before spending hours on training. Hiding them behind a single “passed”
flag would make the integration less useful.

## What comes next

The next bounded task is stability calibration over several updates and seeds.
It should compare declared scaling choices, record return and value
distributions, set explicit KL/value/gradient alert thresholds, and introduce a
separate deterministic evaluation rollout. Only after that check is stable
should the project begin longer formation learning or interpret reward trends.
