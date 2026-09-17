# Segmented training and process restarts

## The problem

A recurrent PPO learner has two kinds of state:

1. **Learning state**: network weights, optimizer momentum, normalization,
   random-number generators, and update counters.
2. **Flight state**: drone poses and velocities, controller state, contact
   state, episode progress, and LSTM memory for the current flight.

Keeping Isaac Sim alive for a long experiment makes one process failure more
expensive. Starting a new process is useful only if it continues the same
learning state without pretending that the old flight also continued.

ALiGn uses reset-mode recovery. A new process restores the complete learning
state and starts new physical episodes with zero recurrent memory.

## Data flow

For a 12-update budget split into groups of four:

~~~text
process A             process B             process C
updates 1..4          updates 5..8          updates 9..12
    |                     |                      |
commit checkpoint 4 -> load checkpoint 4
                          |
                    commit checkpoint 8 -> load checkpoint 8
                                                 |
                                           commit checkpoint 12
~~~

Every arrow is checked in two ways:

- checkpoint IDs must match;
- the payload SHA-256 hashes must match.

The hash check matters because a filename can be correct while its contents are
wrong or corrupt.

## Why segment bounds are not configuration changes

A checkpoint stores the resolved experiment configuration and validates it when
loaded. If process B changed `training.attempts` from 12 to 8, recovery would
correctly reject the checkpoint as belonging to a different experiment.

ALiGn therefore keeps the total budget at 12 in all processes. `start_update`
and `stop_update` are execution bounds supplied to the simulator entry point.
They say which part of the existing budget this process should perform. They do
not redefine the learner.

For example, segment 4-8 must enter with completed-update counter 4. It collects
four new rollouts and leaves with counter 8.

## Normalization must not restart

The centralized critic uses frozen statistics collected before update 1. If
every new process performed another warmup, later segments would see a
different input transform and the training lineage would silently change.

The first process records the warmup, freezes the state, and saves it in
checkpoint 0 and subsequent checkpoints. Processes B and C load that state and
must report zero new warmup transitions. The joined report also requires every
segment to contain the same serialized normalization state.

## What happens to recurrent memory

LSTM memory describes the observations preceding the current flight. It cannot
be carried into a newly reset simulated world because those observations no
longer describe the drone's current episode.

At a segment boundary ALiGn:

- commits the learner only after a complete PPO update;
- labels unfinished environment episodes as abandoned;
- creates reset environments in the new process;
- starts actor and critic recurrent memory at zero;
- restores optimizer and random-number generator state before the next update.

This loses at most an uncommitted rollout when a process fails. It does not lose
a completed, atomically published update.

## Observable evidence

The bounded acceptance uses two policy seeds and three training processes per
seed. A pass requires:

- update rows 1 through 12 exactly once per seed;
- three distinct segment container names;
- restored checkpoint ID and hash equality at both boundaries;
- final counters at update 12;
- one normalization warmup per seed;
- exact fresh-process evaluations at updates 0, 6, and 12.

These checks demonstrate process-level reset-mode continuation. They do not
show convergence, reliable formation completion, or exact simulator-trajectory
restoration.

## Relevant code

- `learning_curve_config.py` defines schema-2 segment ranges.
- `stability_training.py` executes one contiguous range while preserving the
  total configured budget.
- `vector_task.py` passes explicit range bounds into the simulator runner.
- `learning_curve_runtime.py` launches each range and audits the joined lineage.
- `run_segmented_learning.py` is the thin host entry point.

The operational command and artifacts are documented in
[segmented training across fresh simulator processes](../docs/26-segmented-training.md).

## What the lab run showed

The accepted run used three separate training processes per seed. Processes B
and C loaded the exact IDs and hashes committed by their predecessors. No
resumed process repeated the critic warmup, and both learner counters reached
12 with no missing or duplicated update.

The engineering mechanism worked, while task learning remained weak. Mean
assigned error improved from 1.423 m before training to 1.364 m at update 6,
then rose to 1.376 m at update 12. Pairwise shape error increased and minimum
separation decreased. Every evaluated episode timed out.

This separates two conclusions that are easy to confuse:

- the learner can now continue safely across simulator process boundaries;
- this 12-update policy still does not reliably complete the formation task.

The next research change can use segmented execution without attributing poor
flight results to lost optimizer state, repeated normalization, or broken
checkpoint lineage.
