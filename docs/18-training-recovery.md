# Training recovery and checkpoint integrity

## Supported recovery mode

ALiGn now supports `training_resume_with_environment_reset` for learner state.
It restores the shared actor, centralized critic, both Adam optimizers,
normalization state, schedules, progress counters, task-sampler state, resolved
configuration, and Python/NumPy/PyTorch CPU and CUDA random-number states.

Resume abandons an unfinished rollout. The simulator environments, controller
state, mission state, rollout buffer, and actor/critic recurrent memory must all
restart together. This continues learning from a committed optimizer boundary;
it does not recreate an interrupted physical trajectory.

Full simulator-state continuation and policy warm start are separate modes and
are not implemented by this command.

## Configuration

[training-recovery.json](../configs/training-recovery.json) declares:

- a 300-second target interval measured in active training time;
- at least three retained valid checkpoints;
- stop on a checkpoint write failure;
- the reset-mode recovery identifier.

The interval is a trainer policy to enforce at completed optimizer-update
boundaries. The acceptance probe validates the state and storage mechanisms; it
does not run for five minutes or profile a production save interval.

## Checkpoint contents

The PyTorch payload contains model and optimizer state, preprocessing state,
schedules, counters, resolved configuration, task sampling state, and RNG
state. The adjacent JSON manifest remains inspectable without loading PyTorch.
It records the logical run and attempt, immutable checkpoint ID, configuration
hash, source/runtime identities, parent-checkpoint hash, transition/update
counts, active training time, payload size and SHA-256, and UTC/IST save times.

Only load locally trusted PyTorch payloads. A matching checksum detects damage;
it does not establish that an untrusted file is safe to deserialize.

## Safe publication and fallback

`CheckpointStore` writes to a uniquely named temporary file on the checkpoint
filesystem. It flushes and syncs the bytes, rejects an empty file, and asks the
PyTorch layer to deserialize and validate the temporary payload. It then hashes
the bytes, atomically renames the payload, syncs the directory, atomically
publishes its manifest and advisory `latest.json` pointer, and syncs again.
Existing immutable checkpoints remain in place throughout.

Loading scans committed manifests from the highest completed update, validates
run/configuration identity, filenames, path containment, size, and SHA-256, and
falls back past an incomplete or corrupt candidate. A failed writer or failed
pre-publication validator removes its temporary file and leaves the previous
checkpoint selectable. The current store retains every checkpoint; a later
long-run retention worker may prune only after preserving the configured
minimum, milestones, and best-validation checkpoint.

## Run the acceptance check

Prepare and build a fresh immutable runtime context, then run:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&
uv run --locked python scripts/run_training_recovery.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The command uses the selected allocated GPU and the pinned vendor PyTorch. It
does not start Isaac Sim or perform sustained training. Results are saved under
`runs/training-recovery/<id>/`; `runs/` is ignored by Git and needs separate
backup.

The important artifacts are:

- `report.json`: host preflight, image/source identity, command, and final pass;
- `resolved-config.json`: exact policy, PPO, and recovery settings;
- `metrics.json`: every acceptance check and numerical difference;
- `checkpoints/*.pt` and `*.json`: immutable payload/manifest pairs;
- `fresh-load.json`: evaluation result from a separate Python process;
- `console.log`: complete container output.

## Accepted lab evidence

Run `20260916T214543.893138IST-10cdc57d` passed on GPU 0, an RTX A4000, with
Python 3.10.14, PyTorch 2.2.2+cu118, and CUDA 11.8. The host-observed duration
was 11.347 seconds and the recovery probe took 4.944 seconds.

All 18 checks passed. In particular:

- the uninterrupted and resumed second actor update differed by `0.0`;
- the uninterrupted and resumed second critic update differed by `0.0`;
- both Adam states and update diagnostics matched numerically;
- Python and NumPy RNG state, normalization, schedules, task-sampler state, and
  counters restored;
- progress advanced once from update 1 to update 2;
- checkpoint 4 was deliberately corrupted and selection fell back to valid
  checkpoint 3;
- three valid checkpoints remained after the corruption;
- a simulated write failure preserved the prior recovery point;
- an incompatible resolved configuration was rejected;
- a separate Python process loaded checkpoint 3, reset recurrence/environment
  state, and produced finite actor and critic outputs.

The earlier run `20260916T213828.102489IST-214d87e3` remains recorded as failed.
Its learner resume reached the comparison, but two acceptance-probe bugs used an
incorrect actor field and compared CPU/CUDA optimizer tensors without moving
them to a common device. The corrected evidence is a new run; the failure was
not rewritten.

## Limits and next work

This evidence uses a deterministic synthetic recurrent PPO fixture. It proves
learner-state recovery and storage failure behavior in the selected CUDA
runtime. It does not prove power-loss durability for every filesystem, exact
Isaac Sim continuation, or learned flight performance.

The next bounded task is a short task-connected training command that alternates
the accepted live collector and recurrent PPO updater, checkpoints only at
completed update boundaries, starts a new attempt on resume, labels abandoned
episodes, and writes raw per-update/task metrics. Only after that interruption
and resume exercise passes should a longer formation-learning experiment begin.
