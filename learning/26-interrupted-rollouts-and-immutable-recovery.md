# Interrupted rollouts and immutable recovery

## The problem

Suppose four drones are collecting 256 time steps for PPO. The process stops at
step 64. The policy has not been updated, but the simulator and each LSTM have
already moved forward. Continuing from those in-memory values is impossible
after the process exits, and treating the 64 steps as a complete rollout would
change the training algorithm.

ALiGn uses the last checkpoint as the boundary between committed and unfinished
work. If update 4 is committed and update 5 stops during collection, recovery
loads update 4 and collects update 5 again in fresh environments.

## What is committed

A successful update publishes one atomic checkpoint containing:

- shared actor and centralized critic parameters;
- actor and critic Adam optimizer state;
- frozen critic normalization state;
- Python, NumPy, CPU Torch, and CUDA random-number generator states;
- completed-update and transition counters;
- parent-checkpoint lineage and configuration/source identities.

The publication happens only after PPO finishes and the temporary checkpoint
can be deserialized and verified. That makes the checkpoint comparable to a
database commit: earlier checkpoints are valid history, while an unfinished
rollout is uncommitted work.

## Worked example

The acceptance fault targets seed 41, update 5, rollout step 64:

1. A first simulator process commits updates 1 through 4.
2. A second process restores checkpoint 4 and starts update 5.
3. After 64 steps it flushes the raw trajectory and writes
   `interruption.json`.
4. It stops before computing PPO losses or publishing checkpoint 5.
5. The recovery runner copies checkpoints 0 through 4 into a new run and
   verifies their hashes.
6. A fresh process loads checkpoint 4, resets flight and recurrent state, and
   recollects update 5 from the beginning.

There are four environments, so the partial file contains `64 × 4 = 256` rows.
There are four drones per environment, so it represents `64 × 4 × 4 = 1,024`
agent transitions. Both counts remain in the report as discarded work.

## Why the environments and LSTMs reset

A learner checkpoint can restore everything that affects the next optimizer
update, but it cannot recreate an exact PhysX trajectory from the middle of a
rollout. The actor and critic LSTM states also describe that abandoned
trajectory. Recovery resets both the physical worlds and recurrent memory, then
recollects a full ordered sequence. This is
`training_resume_with_environment_reset`, not exact simulator continuation.

Frozen critic normalization does not restart. It is learner state, so it is
restored from checkpoint 4. Warmup must therefore appear once for seed 41 in
the source prefix and never repeat in its recovery segments. A seed that had
not started before the interruption performs its normal warmup in the recovery
run.

## Why the recovery run is separate

The source directory is evidence of what happened at interruption time.
Appending recovered output there would blur failure evidence with later work.
ALiGn instead hashes every source file before recovery, writes into a new run,
and hashes the source again afterward. Equal tree hashes establish that the
recovery process did not alter the source artifacts.

The new run records where every completed segment came from. Joined counters
must be exactly 1 through 12 for each seed. That catches duplicated update 4,
missing update 5, or accidental extra optimizer steps.

## Relevant code and artifacts

- `align.simulation.task_training` flushes the partial trajectory and stops
  before PPO.
- `align.runtime.learning_curve_runtime` constrains and records the deliberate
  interruption.
- `align.runtime.segmented_learning_recovery` audits the source, copies the
  checkpoint prefix, resumes missing work, and verifies immutability.
- `scripts/recover_segmented_learning.py` is the thin host entrypoint.
- `source-inventory-before.json` and `source-inventory-after.json` contain the
  full file hashes.
- `report.json` records discarded work, checkpoint copies, segment origins,
  counters, and evaluations.

## What the checks establish

The CPU tests verify contiguous source segments, exact checkpoint-prefix
copying, source mutation detection, raw partial-rollout row ordering, absence of
post-update artifacts, and explicit fault arguments. The full 197-test suite
passes under Python 3.12 and Python 3.10.

The live check produced interrupted source run
`20260917T195211.368696IST-61fd6fc0` and passing recovery run
`20260917T200314.721203IST-6c0eec2f`. The source preserved 64 partial rollout
steps without a PPO result or update-5 checkpoint. Recovery reused its exact
checkpoint-4 payload, completed both seeds with counters 1 through 12, loaded
checkpoints 0, 6, and 12 for evaluation, and left all 60 source files unchanged.
Normalization warmup occurred once per seed and did not repeat after a resume.

This validates recovery mechanics under live OmniDrones collection. The
resulting formation measurements remained non-monotonic, and no evaluation
episode achieved formation success. The run therefore does not establish
formation convergence or reproduce the paper's reported performance.
