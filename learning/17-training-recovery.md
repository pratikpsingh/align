# Training recovery with an environment reset

## What problem recovery solves

Training changes more than network weights. Adam remembers moving averages of
past gradients, normalizers remember data statistics, schedules remember where
they are, and random generators determine later samples. Loading only actor
weights after a failure starts a different learning process even if the policy
initially looks identical.

ALiGn therefore saves one coherent learner state after a completed optimizer
update. The checkpoint is the last durable boundary. Experience collected after
that boundary may be lost after a crash.

## What reset-mode resume means

Imagine training stopped halfway through a four-second flight. Restoring the
LSTM memory from that flight while resetting every drone to the launch pad would
combine a history from one physical situation with observations from another.
The memory would be misleading.

Reset-mode recovery restores learning history but starts a new episode:

~~~text
restore networks + Adam + statistics + RNG + counters
                         |
                         v
       discard partial rollout and old LSTM memory
                         |
                         v
        reset simulator, controller, task, and mission
~~~

This is why the mode is named
`training_resume_with_environment_reset`. It does not claim to rewind PhysX to
an exact interrupted timestep.

## Why Adam state matters

Adam keeps a first moment and second moment for every parameter. These act like
smoothed histories of gradient direction and size. If they are reset, the next
parameter update changes even when the model, batch, and random seed are the
same.

The acceptance probe performs one update, saves, and then compares two paths:

~~~text
uninterrupted: update 1 -> update 2
resumed:       update 1 -> save -> new models/optimizers -> restore -> update 2
~~~

Both final actor and critic parameter sets differed by exactly `0.0` in the
accepted CUDA run. Their Adam states and update diagnostics also matched. That
is stronger evidence than checking that a file exists.

## Why random-number state matters

The actor's transformed Gaussian samples actions during PPO entropy estimation.
A restored model with a different PyTorch RNG state can therefore take a
different update. Python, NumPy, PyTorch CPU, and every visible CUDA RNG state
are saved. Task-specific sampler progress is stored separately so future shape,
waypoint, or curriculum sampling can resume without silently repeating work.

## How an atomic checkpoint is published

A final filename should mean “complete and usable.” ALiGn first writes a hidden
temporary file on the same filesystem. It flushes and syncs the data, then
actually deserializes and checks the learner-state structure. Only then does it
compute the checksum and atomically rename the payload into place.

A JSON manifest is published afterward with the same safe-write helper. If the
process stops earlier, the old committed checkpoints remain. If the newest
payload is damaged later, its size or checksum fails and selection moves to the
next valid manifest.

The small `latest.json` file is a convenience pointer. Recovery trusts verified
immutable payload/manifest pairs, not the pointer by itself.

## Lineage and counters

A logical run can have several attempts. A resumed attempt records the hash of
the checkpoint it loaded. Update, environment-transition, agent-transition,
and active-training-time counters come from that checkpoint and advance once.
This prevents plots from presenting repeated work after a crash as new progress.

Future task-connected training also needs to label any metric rows after the
restored checkpoint as abandoned or superseded. The storage layer preserves the
needed identity; the full training reporter still has to apply that rule.

## What the failure tests tell us

The host tests terminate a child process in the middle of a write, simulate a
writer/disk error, leave a partial temporary file, reject a malformed payload
before publication, and damage the newest committed payload. In each case the
older valid state remains selectable.

The vendor CUDA probe also rejects a changed resolved configuration and loads
the fallback checkpoint in a separate Python process. That fresh process makes
sure evaluation does not accidentally rely on model objects left in memory by
the saver.

These checks establish the reset-mode learner recovery mechanism. They do not
establish exact simulator-state continuation or useful learned formation
behavior.

## What comes next

The bounded [task-connected training loop](18-task-connected-training-loop.md)
now joins the collector, recurrent PPO updater, and recovery store. It collects
real task sequences, commits completed updates, and resumes in a new simulator
process with environments and LSTM memory reset. Its raw diagnostics establish
operation and expose critic instability; they do not show that the policy has
learned the paper's reported performance.
