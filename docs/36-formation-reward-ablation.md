# Formation-weight ablation on the feasible plane task

The accepted [feasible-schedule plane run](35-feasible-schedule-training.md) collected formation data in both seeds but achieved 0/24 frozen-evaluation successes. Its final-policy telemetry shows small, often misdirected commands. This experiment asks a narrower question: **does giving the pairwise formation term four times its baseline weight improve shape without worsening assigned-target error or safety?** It does not assume the reward weight is the cause of failure.

The treatment file `configs/task-reward-formation-4.json` differs from `configs/task-reward-baseline.json` only in `formation_weight: 1.0 → 4.0`. The pure reward test checks that every other weighted component is unchanged on an identical state/action example. The treatment uses the same simulator image, seeds 41/73, 2,350-step task, 2,304-step rollout and critic warmup, four PPO updates per seed, and exact checkpoint evaluations at updates 0/2/4. It starts new policies, not a warm start from the baseline.

## Run the treatment

From `align/`, after checking GPU 0 is idle:

```sh
sudo -v
uv run --locked python scripts/run_plane_baseline_training.py \
  --build-report runs/runtime-build/20260918T171602.229201IST-f0582262/report.json \
  --construction-config configs/feasible-plane-construction.json \
  --task-config configs/feasible-plane-task.json \
  --rollout-config configs/feasible-plane-rollout.json \
  --learning-curve-config configs/learning-curve-feasible-plane-probe.json \
  --critic-normalization-config configs/critic-normalization-feasible-plane.json \
  --reward-config configs/task-reward-formation-4.json \
  --accept-eula --gpu 0
```

This intentionally reuses the already built pinned simulator image, so there is no image-code change between arms. The host launcher creates a new immutable ignored `runs/plane-baseline-training/<IST-run-id>/`. After it passes, compare the two arms on CPU:

```sh
uv run --locked python scripts/compare_formation_reward.py \
  --baseline-run runs/plane-baseline-training/20260918T172025.554059IST-f5bd6176 \
  --treatment-run runs/plane-baseline-training/<TREATMENT-ID>
```

The comparison verifies image identity, identical resolved configuration except formation weight, equal seed/update schedule and training exposure, exact checkpoint selection, and every raw rollout/evaluation CSV against its metric summary. Its new `runs/formation-reward-comparison/<IST-run-id>/` contains `report.json` and `paired-checkpoints.csv` with successes, timeouts, safety terminations, assigned/pairwise errors, separation, and reset-safe first/last formation windows. Episode windows with fewer than 50 formation rows remain counted but have no first/last-window mean. Different reward scales make cross-arm team-reward magnitude **invalid as a performance comparison**.

## Interpretation

A useful treatment would improve final frozen-policy shape and assigned-target behavior across seeds without more near misses or safety events. One seed improving alone is not conclusive. If formation error improves while separation or target error worsens, report the trade-off. If both arms still fail, retain the negative result and investigate the command-selection objective rather than announcing formation learning. This experiment is a bounded causal ablation of a single weight, not a reproduction of the paper's reported results.

## Accepted paired result (2026-09-18 IST)

The treatment `20260918T175806.992858IST-5f316c32` passed on the same immutable image as the baseline. The CPU comparison `20260918T181524.100490IST-40b25b3b` passed its resolved-config and raw-row audits: each arm has 73,728 training rows in eight CSVs and 56,400 evaluation rows in six CSVs. Both arms used seeds 41/73 and update 0/2/4 evaluations. Update-0 metrics were identical for each paired seed, as expected before different reward weights influenced training.

| Seed, update 4 | Weight 1: assigned / pairwise RMSE; min separation | Weight 4: assigned / pairwise RMSE; min separation |
|---|---|---|
| 41 | 1.1704 / 0.4952 m; 0.3959 m | 1.1266 / 0.4775 m; 0.4816 m |
| 73 | 1.4934 / 0.3233 m; 0.8827 m | 1.4966 / 0.3396 m; 0.8629 m |

Seed 41 improved modestly on these whole-evaluation measures; seed 73 worsened modestly. **Both arms had 0/24 evaluation successes, 24/24 timeouts, and zero safety terminations.** In seed 41 the treatment still spent 99.19% of formation rows below the 0.55 m shaping margin at update 4. The experiment therefore did not establish reliable target acquisition, formation maintenance, or a safe policy advantage. The paired CSV records each episode's first/last formation window; reward magnitudes cannot be compared directly because the scales differ. Continue with frozen-policy command telemetry and a separately controlled action-selection intervention rather than merely increasing the same weight again.

## Frozen actor commands after the treatment

Read-only update-4 replays passed for treatment seeds 41 and 73 as `20260918T182039.223575IST-d1e8ed37` and `20260918T182234.970060IST-9ef97a09`. CPU analyses `20260918T182253.030074IST-32ddd5bf` and `20260918T182517.316291IST-0cc3793b` audited the raw per-drone rows. Compare them with baseline analyses `20260918T174145.575220IST-99663d3e` and `20260918T174332.147922IST-c50382ca`:

| Seed | Target-directed formation commands, weight 1 → 4 | Mean command speed, weight 1 → 4 | Mean target distance, weight 1 → 4 |
|---|---:|---:|---:|
| 41 | 45.8% → 61.3% | 0.0553 → 0.0526 m/s | 1.2832 → 1.1948 m |
| 73 | 25.0% → 25.0% | 0.0131 → 0.0207 m/s | 1.6817 → 1.6888 m |

These are descriptive replay measurements from four cloned environments per seed, not independent extra training seeds. Seed 41's target-directed fraction improved, but speed stayed low and no episode succeeded. Seed 73 moved a little faster but did not point toward its target more often. Its fourth action coordinate remained close to zero and changed sign in the treatment; because the decoder uses `abs(action[3])`, sign is redundant, but this observation alone does not establish it as the cause of failure. A later action-interface experiment must be a separately labeled protocol, not folded into the reward-weight result.
