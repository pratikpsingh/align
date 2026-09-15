# Training recovery and checkpoint integrity

Status: requirements for implementation and testing. No recovery mechanism exists in the current scaffold.

## Recovery promise

After a power cut or process failure, resume learning from the latest valid committed checkpoint. Work performed after that checkpoint may be lost. Saving every action is neither necessary nor sufficient for reliable recovery.

Start with a configurable target checkpoint interval of five minutes of active wall time, checked at completed optimizer-update boundaries. Also save at normal completion and requested safe shutdown. Record the actual interval, save duration, and resulting recovery window; an update lasting longer than five minutes can extend that window. Tune the interval after profiling.

A same-disk checkpoint protects against an interrupted process, not disk loss. Retained checkpoints and an independently verified backup serve different purposes.

## Three distinct loading modes

| Mode | What continues | What must be stated |
|---|---|---|
| Training resume with environment reset | Actor/critic, optimizer/schedule, normalization, completed update/sample history | Interrupted episodes are abandoned; environments, rollout buffer, and episode memory restart consistently |
| Full-state continuation | Training state plus the complete simulator/controller/mission/rollout state | Supported only after state coverage and continuation behavior are verified for the selected runtime |
| Policy warm start | Selected weights and explicitly selected preprocessing state | A new experiment/run, with optimizer/counters reset unless a documented transfer protocol says otherwise |

The practical minimum is training resume with environment reset. It continues the training process but does not recreate the exact unfinished physical flight. Label the recovery mode in every resumed attempt and report it in experiment limitations.

GPU simulation may not reproduce a trajectory bit for bit even with the same seeds and serialized visible state. Do not promise exact continuation without evidence. PyTorch also documents reproducibility limits across platforms and releases in its [randomness guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Checkpoint contents

| Category | Required state |
|---|---|
| Model | Actor and critic weights, learned distribution parameters, model schema |
| Optimization | Optimizer moments/step counts, learning-rate and entropy schedules, AMP scaler if used, completed update/epoch counters |
| Preprocessing | Observation/reward/value normalization statistics and configuration; evaluation freeze behavior |
| Randomness | Python, NumPy, PyTorch CPU/CUDA RNG states, dedicated generators, task/episode sampling RNGs |
| Research identity | Run and attempt IDs, parent checkpoint hash, resolved configuration/hash, source identity, runtime versions, checkpoint schema |
| Progress | Environment and agent transition counters, update count, curriculum position, total active training time, evaluation/best-checkpoint selection state |
| Episode/rollout state | Recurrent h/c or h, masks, current observations, unfinished sequence/rollout data if continuation needs them |
| Mission | Targets, assignments, waypoints, group memberships, transition progress, obstacle motion parameters, episode counters/timers |
| Simulator/control | Poses, velocities, actuator/controller internal state and any other state required by the validated serialization contract |

Not all state in the last three rows is restored in reset-mode recovery. The implementation must explicitly reset it, discard incompatible partial rollouts, and record the abandoned episodes. Never restore recurrent memory for an environment whose physical state has been reset.

For full-state continuation, visible pose/velocity is insufficient by itself: controller integrators, motor lag, physics internals, sensor/communication buffers, and per-environment randomness may matter. Document what the backend can and cannot serialize.

Save at a consistent learner boundary, not while optimizer/model tensors are being modified. If serialization is asynchronous, create an immutable coherent snapshot first. Prefer the simple synchronous implementation until profiling establishes a reason to complicate it. PyTorch's [checkpoint tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) explains why resuming requires optimizer state as well as weights.

## Safe write protocol

1. Create a uniquely named temporary checkpoint on the same filesystem as the final destination.
2. Serialize a coherent snapshot, flush buffers, sync file contents where supported, and compute/record an integrity checksum.
3. Validate that the serialized file and metadata are readable and consistent before publishing it.
4. Atomically rename the checkpoint into its immutable final name; publish its manifest and update the latest pointer using the same safe pattern. Sync directory metadata where supported.
5. Keep previous valid checkpoints until publication succeeds. On loading, validate pointers/manifests and fall back to the newest older valid checkpoint if necessary.

Atomic rename does not make arbitrary network filesystems or failing storage perfectly durable. Test on the actual output storage and document the filesystem assumptions. A manifest/checkpoint pair must tolerate interruption between publishing its members.

Retain at least the latest three valid checkpoints, a separately selected best-validation checkpoint, and configurable periodic milestones. Use storage quotas and retention rules. Never delete the only valid recovery point to make space for a new one.

On disk-full or failed writes, preserve the previous valid checkpoint, log the failure prominently, and follow a configured retry/stop policy. Avoid continuing expensive training indefinitely with broken persistence. Signal handlers may request a safe checkpoint; they cannot rescue a sudden power cut.

## Run lineage and logs after restart

One logical run may have several attempts. Every attempt has a new ID, start/end/heartbeat records, exit status, and a link to the checkpoint it loaded. Keep both logical learning progress and total compute time actually spent, including work later abandoned after recovery.

If metrics were logged beyond the restored checkpoint, retain those original records as evidence but mark them as superseded/abandoned for the resumed learning lineage. Reports must not concatenate duplicate update numbers as if training advanced monotonically. Link each metric to its attempt and update/checkpoint context.

Distinguish completed, failed, interrupted, and unknown/stale attempts. A missing normal-exit record is not proof of successful completion. Resume should detect that condition and record the inferred interruption explicitly.

Reject incompatible dimensions, observation/action schemas, checkpoint versions, or experiment-defining settings with a clear error. Do not silently load only matching layers and call it resume. Changed rewards/tasks/algorithms normally create a new run with parent provenance. Any explicitly supported migration needs its own tested conversion and record.

Load only trusted checkpoints. Save machine-readable metadata separately so basic inspection does not require executing a general object deserializer.

## Recovery checks required before long training

- Deterministic CPU learner comparison: interrupted/resumed updates preserve model, optimizer, normalization, counters, and appropriate RNG behavior.
- Abrupt subprocess termination: restart from the committed checkpoint rather than relying on graceful-exit hooks.
- Truncated temporary file and corrupt latest checkpoint: ignore incomplete state and load the previous valid checkpoint.
- Disk/write failure: preserve old data and expose failure status.
- Repeated resume: no duplicate active learning samples, misleading time totals, or double-counted evaluations in reports.
- Reset-mode recovery: all environment, recurrence, buffer, and mission state restarts consistently; abandoned episodes are labeled.
- GPU continuation, if supported: verify simulator/controller/state coverage and measure divergence rather than asserting bitwise identity.
- Fresh process loading: evaluate/export a saved actor without relying on in-memory objects from training.

These tests are acceptance requirements, not claims that this repository has passed them.
