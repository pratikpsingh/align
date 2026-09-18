# Target-directed reference control

This experiment asks whether the **current physical vector task** can reach its assigned formation using a simple controller before changing PPO. Each drone uses its exact task target and current position to request a world-frame velocity: `v = clamp_norm(position_gain × (target − position), max_speed)`. The existing pure `velocity_commands` and `velocity_actions` helpers encode that request as the same bounded four-value direction/speed action used by the learned policy. The pinned Lee inner controller, rotor action mapping, physics, phase targets, reward, termination, and safety thresholds are unchanged. Ground contacts, separation failures, and timeouts are measured rather than hidden.

The reference has **privileged exact target and pose access**. It is a diagnostic controller, not a decentralized policy or a fair performance competitor. It uses `position_gain_s_inv=0.8` and `max_speed_m_s=0.5` from the source construction configuration. It does not load a policy checkpoint, train, or run PPO. A passing software probe means finite bounded actions and reconciled logs; it does **not** mean the reference successfully formed a safe swarm.

## Lab command

From `align/`, prepare a new derived-image context after code changes:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
```

Use the printed `runs/runtime-build/<ID>` exactly once:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<ID> && \
uv run --locked python scripts/run_reference_control.py \
  --source-run runs/plane-baseline-training/20260918T105406.233167IST-d74f8b47 \
  --build-report runs/runtime-build/<ID>/report.json \
  --timing-config configs/evaluation-timing-extended.json \
  --policy-seed 41 --accept-eula --gpu 0
```

The source training run supplies only its resolved four-drone task configuration; it is mounted read-only. The launcher checks that GPU 0 is allocated and idle, then runs baseline 800-step and extended 2,350-step reference evaluations sequentially in the same image. It writes separate `evaluation.csv`, `policy-telemetry.csv`, `metrics.json`, probe and native logs under `runs/reference-control/<ID>/baseline/` and `extended/`. The host report audits every drone row against reward, geometry, command conversion, and rotor clipping, then checks the identical first 550 episode steps before schedules diverge. `report.json` stores commands, config and image identities, raw-row counts, outcomes, phase metrics, and first-formation altitude.

No previously accepted result is overwritten. `runs/` is ignored by Git and needs separate backup. This command and its source tests are implemented and CPU-checked; **live reference-control behavior remains unvalidated until the lab run finishes**.

## Interpreting the result

Inspect all three together: terminal reason codes (success=1, time limit=6, separation=2, airborne contact=3, envelope=4), formation assigned/pairwise RMSE, and minimum separation. If the reference succeeds safely in either schedule, the task and inner controller can support that mission under privileged target control; the learned actor still needs improvement. If the reference fails, inspect the raw trajectory: a collision during slot exchange suggests path/safety planning, whereas slow vertical progress suggests the high-level/inner-controller interface or deadline. A failed simple proportional rule alone does **not** prove the task impossible. Compare baseline and extended arms only within their declared durations; whole-episode reward means mix different phase exposure.

## Accepted lab result, 2026-09-18 IST

Build `20260918T170315.739230IST-dcdd8687` and reference run `20260918T170533.567052IST-8804df3e` passed the host and simulator audits on GPU 0. The baseline and extended arms had an identical first 8,800 drone telemetry rows (550 episode steps) before their schedules diverged. All actions were finite and bounded, the task did not clip them, and every active reference command pointed toward its assigned target.

| Schedule | First formation episode step | Outcomes across four environments | Assigned RMSE at episode end or success | Minimum separation |
|---|---:|---|---:|---:|
| Baseline, 800 steps | 551 | 0 success; 4 timeouts | 0.1651 m at step 800 | 0.7821 m |
| Extended, 2,350 steps | 1551 | 4 success; 0 safety terminations | 0.00251 m at success step 2104 | 0.7822 m |

The extended arm reached formation height before changing phase and completed the declared success dwell. This establishes **physical feasibility for this four-drone plane task with privileged exact-target proportional control and the extended schedule**. It does not establish decentralized-policy success or feasibility of every formation/template. The baseline timeout and the frozen-policy timing result remain separate negative evidence: more time alone did not rescue the previously trained actor. The next controlled training test must use a declared schedule consistently during training and evaluation, then compare target-directed actions, error, and safety against this reference. Do not compare whole-episode reward means between schedules as if they had identical phase exposure.
