# Feasible-schedule plane learning probe

The accepted [target-directed reference test](33-reference-control.md) succeeded in all four environments with a 15-second takeoff phase and 8-second formation phase. The earlier trained actor timed out even when replayed under that longer schedule. This probe trains a **new** plane-only actor under the same declared phase durations, so training and evaluation use one task contract. It does not reuse the old checkpoint.

The explicit configurations are `configs/feasible-plane-construction.json`, `configs/feasible-plane-task.json`, `configs/feasible-plane-rollout.json`, `configs/learning-curve-feasible-plane-probe.json`, and `configs/critic-normalization-feasible-plane.json`. They retain the baseline world, controller, reward, observations, policy, and optimizer. The critic-only frozen-normalization warmup also covers 2,304 steps, matching the rollout contract. Each 2,304-step rollout begins with a fresh environment, enters formation at step 1,551, and spans the reference's success step 2,104. A rollout ends before the 2,350-step task deadline, so an unfinished episode is handled as a rollout boundary. This is a small **four-update, two-seed integration and learning probe**, with exact checkpoint evaluations at updates 0, 2, and 4. It is not a sustained training budget.

## Lab command

From `align/`, prepare a new context after code/configuration changes:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
```

Use the printed `runs/runtime-build/<ID>` exactly once:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<ID> && \
uv run --locked python scripts/run_plane_baseline_training.py \
  --build-report runs/runtime-build/<ID>/report.json \
  --construction-config configs/feasible-plane-construction.json \
  --task-config configs/feasible-plane-task.json \
  --rollout-config configs/feasible-plane-rollout.json \
  --learning-curve-config configs/learning-curve-feasible-plane-probe.json \
  --critic-normalization-config configs/critic-normalization-feasible-plane.json \
  --accept-eula --gpu 0
```

The command creates a new ignored `runs/plane-baseline-training/<IST-run-id>/`; it does not replace the older specialist run. Follow each seed's train/evaluation `console.log`. The launcher saves resolved configurations, source/image identity, checkpoint lineage, raw update and evaluation tables, and the report. Back up ignored runs separately.

## Decision criteria

First require all process, checkpoint, finite-update, and host-audit checks to pass. Then inspect formation-phase exposure in each training update, frozen-policy success/reason counts, assigned and pairwise error, separation, and action direction. A passing launcher with 0 successes is a valid negative learning result. The reference is privileged and is only a feasibility bound; it is not an equal-information baseline. If this four-update probe is stable but unsuccessful, do not infer impossibility or launch a large blind training budget. Use its raw phase/action/reward evidence to choose a controlled reward or observation intervention, then compare equal-budget seeds.

This protocol is CPU-validated by `tests/test_feasible_plane_protocol.py`. The GPU run below has passed; its training behavior remains a negative mission result.

## Accepted bounded GPU result, 2026-09-18 IST

Run `20260918T172025.554059IST-f5bd6176` passed in 16.62 minutes on the pinned image and GPU 0. Both seeds completed four recurrent PPO updates and exact-checkpoint evaluations at updates 0, 2, and 4. Each training update collected 9,216 environment rows, including 3,016 formation rows, so the longer task **did expose** the learner to formation. All 24 evaluated environment episodes reached formation and timed out; none met success. The launcher, checkpoint lineage, finite update, and fresh-process evaluation contracts passed. This is a negative mission result, not a failed process.

| Seed | Update | First→last 50 formation assigned RMSE | First→last 50 pairwise RMSE | Formation steps below 0.55 m shaping margin | Minimum separation |
|---:|---:|---:|---:|---:|---:|
| 41 | 0 | 1.6737→1.6680 m | 0.8778→0.8709 m | 0% | 0.957 m |
| 41 | 2 | 1.4054→1.3462 m | 0.9160→0.9619 m | 0% | 0.577 m |
| 41 | 4 | 1.2911→1.3159 m | 0.9946→1.1287 m | 100% | 0.396 m |
| 73 | 0 | 1.6626→1.6595 m | 0.8501→0.8337 m | 0% | 0.884 m |
| 73 | 2 | 1.7128→1.7393 m | 0.8905→0.8993 m | 0% | 0.971 m |
| 73 | 4 | 1.6850→1.6969 m | 0.8861→0.8923 m | 0% | 0.883 m |

Each first/last value averages 50 formation rows per environment across four environments. The shaping margin is 0.55 m; the terminal separation threshold is 0.25 m. Thus seed 41 update 4 had a **near-separation problem throughout formation** but no terminal separation event. Positive critic explained variance (final post-update 0.574 and 0.684) did not imply mission success. The next diagnostic is a fresh replay of the update-4 frozen actors with per-drone targets, actions, realized motion, and weighted reward terms. Its result will guide a controlled intervention; simply increasing the update budget is not justified by these data.

## Frozen update-4 per-drone diagnosis

Both frozen update-4 checkpoints were replayed without training and passed raw geometry, reward-closure, action-conversion, and rotor-clamp audits. Seed 41 replay `20260918T173939.487962IST-84ef9cc6` and CPU analysis `20260918T174145.575220IST-99663d3e` found formation mean target distance 1.283 m, commanded speed 0.0553 m/s, realized speed 0.0510 m/s, and target-directed command fraction 45.8%. Seed 73 replay `20260918T174128.950622IST-c98db9a1` and analysis `20260918T174332.147922IST-c50382ca` found 1.682 m, 0.0131 m/s, 0.00847 m/s, and 25.0%, respectively. The inner controller approximately realized these small commands; the frozen actors seldom requested substantial target-directed motion. This is checkpoint-specific evidence and does not prove that the reward weight is the cause. The [one-variable formation-weight test](36-formation-reward-ablation.md) is prepared to examine one possible reward effect.
