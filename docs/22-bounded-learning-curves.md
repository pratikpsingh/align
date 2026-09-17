# Bounded learning curves and checkpoint evaluation

## Purpose

The learning-curve command extends the accepted three-update optimizer check
without starting an open-ended experiment. It trains two declared policy seeds
for ten recurrent PPO updates and evaluates three immutable checkpoints from
each seed in fresh simulator processes.

The default milestones are update 0, update 5, and update 10. Update 0 measures
the initialized policy before learning. Update 5 is the midpoint. Update 10 is
the final policy in this budget. Comparing these fixed checkpoints shows
whether critic diagnostics and physical task metrics move with additional
experience.

This remains a calibration-scale experiment. Ten updates do not establish
convergence, generalization, or reproduction of reported paper results.

## Exact configuration

The host budget is declared in
[learning-curve-baseline.json](../configs/learning-curve-baseline.json):

| Setting | Value |
|---|---:|
| Policy seeds | 41, 73 |
| Updates per seed | 10 |
| Evaluation milestones | 0, 5, 10 |
| Evaluation steps per environment | 800 |
| Parallel environments | 4 |
| Agents per environment | 4 |
| Rollout horizon per update | 768 |
| Critic learning rate | 1e-5 |

Each seed collects 30,720 environment transitions and 122,880 agent
transitions. Across both seeds, the training budget is 61,440 environment
transitions and 245,760 agent transitions.

Each checkpoint evaluation writes 3,200 raw environment rows. Six evaluations
produce 19,200 rows in total. The launcher runs eight containers sequentially:
one training and three evaluation containers for each of two seeds.

## Checkpoint selection contract

Evaluation receives an explicit completed-update number. The checkpoint store
verifies the exact manifest and payload checksum for that update. It does not
fall back to a newer or older checkpoint.

This distinction matters for a learning curve. If the update-0 evaluation
silently loaded update 10, every point could appear valid while the curve was
scientifically meaningless.

The training container retains the initialization checkpoint and every update
checkpoint. Its final checks also require contiguous parent lineage between
successive updates.

## Acceptance and diagnostic guidance

A run passes its engineering contract only when:

- both seeds complete exactly ten updates;
- counter totals and checkpoint counts match the declared budget;
- every update has contiguous checkpoint lineage;
- rollout tensors and PPO diagnostics are finite;
- each milestone loads the requested checkpoint in a fresh process;
- all evaluation observations, actions, and metrics are finite;
- deterministic evaluation does not change actor parameters;
- raw evaluation row and phase counts are exact;
- actions remain bounded and require no environment clipping.

KL, policy clipping, value clipping, and gradient limits remain diagnostic
guidance. Crossing them is retained as an experimental result instead of
deleting the run.

The summary reports training-wide ranges for post-update KL, policy clipping,
value clipping, explained variance, value loss, and critic gradient norm. For
each milestone it reports cross-seed ranges for reward, assigned formation
RMSE, pairwise RMSE, minimum separation, and termination reason counts.

## Run it

Prepare and build a fresh immutable image context, then run:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&

uv run --locked python scripts/run_learning_curve.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The launcher checks that the selected GPU is idle and disables container
networking. Each container has a default timeout of 1,800 seconds. During long
runs it refreshes an already-authorized sudo timestamp so a timed-out container
can still be removed. It never prompts for or stores a password.

## Recover an interrupted evaluation sequence

Training checkpoints and completed evaluations remain useful when a later
evaluation process stalls. Remove the stalled container first, then recover the
missing evaluations from the immutable source run:

~~~sh
sudo docker rm --force <container-name-from-source-report>
sudo -v
uv run --locked python scripts/recover_learning_curve.py \
  --source-run runs/learning-curve/<source-run-id> \
  --accept-eula \
  --gpu 0
~~~

Recovery is accepted only for a source run whose status is `failed`,
`timed_out`, or `interrupted`. It verifies both completed training phases and
the checksum of every requested checkpoint. It reuses valid completed
evaluations, starts only missing evaluations, mounts the source run read-only,
and writes a separate run under `runs/learning-curve-recovery/`. The final
report marks every evaluation as originating from the source or recovery run.
It never resumes or repeats training.

## Artifacts

Artifacts are saved under `runs/learning-curve/<run-id>/`:

- `report.json` contains runtime identity, exact commands, exits, per-seed
  training and evaluations, and the final summary;
- `config.json` contains the simulator-facing resolved configuration;
- `learning-curve-config.json` contains the host budget and milestone list;
- `learning-curve-summary.json` contains training diagnostic ranges,
  per-update training trends, and evaluation trends;
- `training-curve.csv` and `evaluation-curve.csv` provide compact long-form
  tables derived from the JSON summary;
- each seed has an immutable checkpoint directory;
- `train/update-XXXX/` contains raw rollout, update, and metric files;
- `evaluation-update-XXXX/evaluation.csv` contains raw deterministic
  trajectories for that milestone;
- simulator logs, events, scene, model assets, and probe results remain beside
  each container output.

The `runs/` directory is ignored by Git and requires separate backup.

## Interpretation

A falling assigned RMSE or rising success count across both seeds would support
a larger training experiment. Stable safety metrics and bounded actions are
required alongside task improvement.

Negative explained variance alone does not prove a broken critic early in
training. Its direction over updates matters. If it remains strongly negative
while value clipping stays controlled, the next experiment should introduce a
fully checkpointed normalization or critic-target change. If task metrics
improve despite weak explained variance, inspect the raw returns and value
predictions before changing the learner.

## Validation status

The host-safe configuration, exact checkpoint selector, bundle loading,
cross-seed summary, recovery selection, CLI imports, and failure cases pass
locally.

Run `20260917T001738.329486IST-5f61f50e` is retained as a timed-out attempt. Both
seeds completed ten updates. Seed 41 completed exact update 0, 5, and 10
evaluations. Seed 73's update-0 container stalled during Isaac Sim startup and
timed out after 1,800 seconds. The expired sudo timestamp prevented automatic
container removal in that attempt.

Recovery run `20260917T103636.493092IST-3c169a39` passed in 274.759 seconds. It
reused both training results and all three seed-41 evaluations, ran the three
missing seed-73 evaluations, refreshed sudo four times without failure, and
left GPU 0 idle. The combined result contains 20 optimizer updates and six
exact-checkpoint evaluations.

The two-seed mean assigned RMSE was 1.4230 m at update 0, 1.3563 m at update 5,
and 1.3756 m at update 10. Mean reward improved from -0.02291 to -0.02052 and
then regressed to -0.02124. Pairwise RMSE increased from 0.2767 m to 0.2899 m.
Minimum separation stayed between 0.8703 m and 0.9943 m across individual seed
summaries. Every one of the 24 evaluated episodes reached the time limit; none
met the formation success condition.

This is an accepted engineering learning curve, but it is not evidence of
convergence or reliable task learning. The update-5 improvement was not
monotonic, the pairwise shape metric worsened, and explained variance remained
negative from -3.0593 to -1.7757 across training updates.
