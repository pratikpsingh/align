# Bounded task-connected recurrent training

## Scope

ALiGn can now join the accepted vectorized OmniDrones task, local recurrent
actor, centralized recurrent critic, live rollout storage, GAE, recurrent PPO
update, and atomic recovery store. The acceptance command intentionally performs
only two optimizer updates in two separate simulator processes.

This is an integration and recovery check. It is not a training recipe, a
converged checkpoint, or evidence that the formation policy learned.

## Two-process contract

The launcher creates one logical run with two attempts:

1. Attempt 1 starts four fresh environments, creates an initialization
   checkpoint at update 0, collects 64 real physics steps per environment,
   performs one two-epoch recurrent PPO update, and commits update 1.
2. Attempt 1 exits and its Isaac Sim process ends.
3. Attempt 2 starts a new container and Isaac Sim application, verifies and
   restores update 1, labels unfinished episodes from the previous process as
   abandoned, resets all worlds and actor/critic LSTM memory, collects another
   64-step rollout, performs update 2, and commits a child checkpoint.
4. The host validates raw tables, counters, manifests, and the complete parent
   chain before the run can pass.

The environment is reset on resume because exact simulator continuation is not
supported. PPO consumes the completed rollout before checkpointing, so no
partial rollout is restored. Physical episodes that had not ended at the
process boundary are recorded as abandoned.

## Configuration

The bounded integration uses:

- four environments and four drones per environment;
- 64 collection steps per attempt;
- recurrent chunks of 16 timesteps;
- one optimizer update with two full-batch PPO epochs per attempt;
- the existing 55-value local actor observation and 80-value critic state;
- stochastic actions from the bounded transformed Gaussian policy;
- the reset-mode recovery contract and three retained checkpoints;
- declared feature scaling, with empirical observation normalization disabled.

The exact files are [task-training.json](../configs/task-training.json),
[recurrent-training-rollout.json](../configs/recurrent-training-rollout.json),
and [recurrent-training-task.json](../configs/recurrent-training-task.json),
plus the existing policy, PPO, formation, observation, reward, and recovery
configurations.

## Run it

Prepare and build a fresh immutable runtime context, then run:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&
uv run --locked python scripts/run_task_training.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The launcher checks that the selected GPU is idle before attempt 1. It uses the
same allocated device for both sequential containers and disables container
networking. The command takes roughly two simulator startups plus the two short
rollouts.

## Artifacts

Artifacts are saved under `runs/task-training/<run-id>/`:

- `report.json`: host/runtime identity, both commands and exits, totals, status;
- `config.json`: the complete resolved configuration and checkpoint hash basis;
- `attempt-0001/` and `attempt-0002/`: native logs, events, scene/assets,
  `probe-result.json`, `metrics.json`, `rollout.csv`, and `updates.csv`;
- `checkpoints/`: initialization, update-1, and update-2 payload/manifest pairs;
- `host-audit.json`: independent lineage, counter, and raw-table checks;
- `training-rollout.csv`: 512 combined environment-transition rows;
- `training-updates.csv`: four combined PPO epoch rows.

`runs/` is ignored by Git and requires separate backup.

## Accepted evidence

Run `20260916T220631.683824IST-5f89b3be` passed in 148.450 seconds on one RTX
A4000. Both container exits were zero and all 16 per-attempt checks plus all 17
host-audit checks passed.

The final counters were:

| Counter | Value |
|---|---:|
| Completed optimizer updates | 2 |
| Environment transitions | 512 |
| Agent transitions | 2,048 |
| Active collection/update time | 1.839 s |
| Valid checkpoints | 3 |
| Abandoned episodes labeled at resume | 4 |

Attempt 1 changed actor parameters by at most `0.00060020` and critic parameters
by `0.00199735`. Attempt 2 changed them by `0.00060054` and `0.00183413`.
Actions stayed within approximately `[-0.976, 0.974]`; the environment did not
clip them. Rewards, values, bootstraps, formation metrics, PPO diagnostics, and
all rollout tensors remained finite.

Both 64-step rollouts ended before the configured 80-step episode time limit;
no terminal or truncation boundary occurred inside these training rollouts.
Boundary bootstrapping and partial resets were validated by the earlier live
collector acceptance run, while this check establishes task-connected
optimization and process-level resume.

Attempt 2 loaded checkpoint
`000000000001-4b7c17151dfa46fea552605e60742ffe` and committed update-2 checkpoint
`000000000002-f6b07ab3ee864bb7b6ef990f155c8def`. The host verified that the
initial, update-1, and update-2 manifests form one SHA-256 parent chain.

## Stability warning and next work

Finite updates are necessary but do not show useful optimization. In attempt
1, PPO epoch 2 reported critic value loss `27.3542`, explained variance
`-95.1277`, value clip fraction `1.0`, and a pre-clipping critic gradient norm
of `99.8176`. Attempt 2 remained finite but still had negative explained
variance and value clip fraction `1.0` in epoch 2. The mean team rewards of the
two short rollouts were `-0.01179` and `-0.01188`; this is neither improvement
nor a statistically meaningful comparison.

Before sustained formation training, add empirical observation/value or reward
scale diagnostics as appropriate, establish minibatch/update scheduling, and
run a short multi-update stability calibration with frozen acceptance bounds.
Then add periodic evaluation that uses deterministic actions and frozen
normalization separately from training. Preserve all failures and choose
hyperparameters from controlled measurements rather than these two reward
means.
