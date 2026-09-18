# Frozen-policy episode timing comparison

This command evaluates the **same seed-41, update-12 checkpoint** twice in fresh simulator processes. The baseline uses its original 800-step episode (0.5 s ground, 5 s takeoff, 2.5 s available formation). The extended arm changes only evaluation timing: 0.5 s ground, 15 s takeoff, and 8 s formation, for 2,350 steps. The 2-second success dwell, speed ceiling, geometry, reward, observations, actor, controller, seed, and source checkpoint stay unchanged. The timing override is a separate file, never a mutation of the checkpoint's source configuration.

The two arms use the **same new derived image**, not the historical source image. Both mount the original training run read-only. Each saves an environment-step `evaluation.csv`, per-drone `policy-telemetry.csv`, metrics, probe, logs, and assets in its own `baseline/` or `extended/` directory. The host launcher rejects a checkpoint hash mismatch, wrong timing declaration, optimizer updates, incomplete row audit, or a failed simulator probe.

## Lab command

From `align/`, prepare the image context after any code change:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
```

Use the printed `runs/runtime-build/<ID>` once:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<ID> && \
uv run --locked python scripts/run_policy_timing.py \
  --source-run runs/plane-baseline-training/20260918T105406.233167IST-d74f8b47 \
  --build-report runs/runtime-build/<ID>/report.json \
  --timing-config configs/evaluation-timing-extended.json \
  --policy-seed 41 --evaluation-update 12 --accept-eula --gpu 0
```

The selected GPU must be allocated and idle. The script checks it before starting. The extended evaluation has more simulation steps and may take several minutes. Full artifacts go under `runs/policy-timing/<ID>/`; `report.json` contains source/build identities, exact commands, checkpoint and timing hashes, audit results, phase metrics, and success/timeout counts. Back up the ignored `runs/` source and comparison directories separately from Git.

## Reading the comparison

Use formation-phase assigned/pairwise RMSE, final per-environment windows, minimum separation, command alignment and speed, and episode outcomes. Compare the first formation-step altitude and the calculated command-budget bound in each trace. The arms have different durations, so whole-episode reward means and raw sums are **not** directly comparable as learning quality. Longer exposure may also cause the same checkpoint to see states outside its training distribution. This is an evaluation sensitivity check, not evidence that retraining with a longer task will succeed.

Both takeoff and formation durations change in the extended arm. A better result would establish a **combined timing effect**, not which duration caused it. A follow-up factorial comparison could change one duration at a time. A poor result would point to weak commands or task dynamics even after the original deadline is removed; inspect raw trajectories before changing rewards or controller gains. The accepted simulator outcome and its limits are recorded below.

## Accepted lab comparison

The two-arm GPU run `runs/policy-timing/20260918T132216.245829IST-f1dbf4b4` passed on 2026-09-18 IST. Both arms used image `sha256:b0cb5ee3ebbaf55b57971cf80499f317789634fa802e81befe3ea22515ca15da` and checkpoint `000000000012-020beda227f34911adaba9926198cfb4` (payload SHA-256 `fb98d68557e4a2f5a5ac7c97b8880f620a389073388cd33d744dfd213c199851`). Each simulator probe, raw telemetry audit, and zero-update check passed. The baseline logged 12,800 drone rows; the extended arm logged 37,600. All four worlds in each arm timed out with no success or terminal safety failure. These four cloned worlds are not independent policy seeds.

The additional CPU audit is reproducible:

```sh
uv run --locked python scripts/analyze_policy_timing.py \
  --source-run runs/policy-timing/20260918T132216.245829IST-f1dbf4b4
```

Its accepted report is `runs/policy-timing-analysis/20260918T132732.927020IST-f0da4632/report.json`. The first **8,800 drone rows (550 episode steps)** match exactly between arms, including actions and states. That confirms the timing intervention begins at the declared step 551. Mean drone height at formation entry rose from `0.237 m` (baseline) to `0.631 m` (extended), but average target-altitude deficit was still `0.869 m` in the extended arm.

| Formation-phase measurement | Baseline | Extended |
|---|---:|---:|
| Mean assigned RMSE | 1.582 m | 1.779 m |
| Mean pairwise RMSE | 0.992 m | 1.504 m |
| Minimum separation | 0.512 m | 0.470 m |
| Mean commanded speed | 0.0953 m/s | 0.0409 m/s |
| Commands pointing toward target | 45.8% | 34.0% |
| Successful / timed-out worlds | 0 / 4 | 0 / 4 |

The extended arm had enough **nominal command budget** to remove the earlier altitude-only deadline bound, yet this frozen actor did not use the extra time to converge. The result rules out the original short deadline as the **sole** explanation for this checkpoint's timeout. It does not establish that a policy trained with a longer task would fail, because the extended states are outside its original training schedule. No value of the formation reward weight has been ablated here. The next bounded test should use a deterministic target-directed reference controller under the same task, speed limit, and safety thresholds; that will distinguish a task/controller feasibility problem from this learned actor's behavior.
