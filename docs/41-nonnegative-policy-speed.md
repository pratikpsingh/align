# Nonnegative policy speed coordinate

The retained C03/C05 formation task is physically feasible with exact-target reference control, but the feasible-schedule trained actors have not reached the formation target. Their saved update-4 replays commanded only 0.0553 and 0.0131 m/s on average during formation, against a 0.5 m/s limit; target alignment was 45.8% and 25.0%. These data locate a policy-command problem but do not isolate its cause.

The submission declares a four-coordinate direction-and-speed action. The student's `VelocityAviary.py` and `BaseRLAviary.py` decode speed with `abs(action[3])`, and ALiGn preserved that rule in the calibrated controller interface. Both signs therefore yield the same physical speed. This experiment changes **only** the actor distribution's fourth lower action bound from `-1` to `0`; the controller decoder, reward, observation, optimizer, task, and maximum speed remain identical. Configs are `configs/recurrent-policy.json` and `configs/recurrent-policy-positive-speed.json`. The pure validator and [test](../tests/test_positive_speed_protocol.py) reject any other policy-field change.

With zero latent mean, the fourth action coordinate changes from 0 to 0.5 under the affine tanh map. If direction were unit length, this would request 0.25 m/s; if direction is zero, commanded velocity remains zero. Stochastic exploration and the initial deterministic policy therefore change. Update-0 results **cannot** be expected to match the baseline, and a post-update difference is not automatically evidence that PPO learned a better rule. The positive-speed coordinate can approach zero as its latent value becomes negative; stable settling still requires evaluation.

## Matched physical protocol

Use the rebuilt, phase-aware-contact image already accepted by the 3 m route and contact regression. Run both arms from fresh initialization with the same seeds 41/73, four updates, exact evaluations at 0/2/4, 2,350-step task, 2,304-step rollout, and critic normalization. From `align/`:

```sh
sudo -v
uv run --locked python scripts/run_plane_baseline_training.py \
  --build-report runs/runtime-build/20260918T202958.790600IST-4c23a0a2/report.json \
  --construction-config configs/feasible-plane-construction.json \
  --task-config configs/feasible-plane-task.json \
  --rollout-config configs/feasible-plane-rollout.json \
  --learning-curve-config configs/learning-curve-feasible-plane-probe.json \
  --critic-normalization-config configs/critic-normalization-feasible-plane.json \
  --policy-config configs/recurrent-policy.json \
  --accept-eula --gpu 0

uv run --locked python scripts/run_plane_baseline_training.py \
  --build-report runs/runtime-build/20260918T202958.790600IST-4c23a0a2/report.json \
  --construction-config configs/feasible-plane-construction.json \
  --task-config configs/feasible-plane-task.json \
  --rollout-config configs/feasible-plane-rollout.json \
  --learning-curve-config configs/learning-curve-feasible-plane-probe.json \
  --critic-normalization-config configs/critic-normalization-feasible-plane.json \
  --policy-config configs/recurrent-policy-positive-speed.json \
  --accept-eula --gpu 0
```

The tested raw comparison for these exact immutable runs is:

```sh
uv run --locked python scripts/compare_action_interface.py \
  --baseline-run runs/plane-baseline-training/20260918T214450.845559IST-7fde1830 \
  --treatment-run runs/plane-baseline-training/20260918T220124.368475IST-7a5bed66
```

Use a new run pair when the simulator image or other task settings change.

The comparison requires the same image, host package source, seeds, milestones, training exposure, and resolved configuration except the one action bound. It audits all raw rollout and evaluation CSVs, then writes paired checkpoint success, timeout, safety, assigned error, pairwise error, and separation. Read these task metrics together with frozen-policy action telemetry before deciding whether to keep the interface. A `passed` process with zero mission successes remains a negative flight result. This is an ALiGn intervention, not a claim that the student's original action support was `[0,1]`.

## Matched physical result

The same-image, two-seed arms completed on 2026-09-18 IST: baseline `20260918T214450.845559IST-7fde1830` and positive-speed treatment `20260918T220124.368475IST-7a5bed66`. Both process reports passed. Their image, host package SHA, all non-policy resolved settings, seeds, milestones, and per-seed training exposure match. A read-only audit recomputed all eight training and six evaluation CSVs per arm: 36,864 training rows per seed and 9,400 evaluation rows per seed/checkpoint.

**Neither arm achieved a formation success.** At update 4, baseline seed 41 had four contacts after 1,796 formation-world steps, while treatment seed 41 had four contacts at the first formation step, giving only four formation-world rows. Baseline seed 73 had four timeouts after 3,200 formation-world steps; treatment seed 73 had two contacts, two timeouts, and 1,991 formation-world rows. Earlier treatment checkpoints also had four contacts for each seed. These are worse safety and exposure outcomes; the nonnegative speed bound is not accepted as a formation solution.

All-step assigned-error averages are **not a fair tracking comparison** when one policy crashes just as formation begins. Direct raw-CSV phase filtering at update 4 gives formation-only assigned RMSE: baseline/treatment seed 41 `1.2064/1.6490 m` over `1,796/4` world-steps, and seed 73 `1.6690/3.3260 m` over `3,200/1,991` world-steps. Four rows are not enough to characterize sustained formation. The corrected host comparison `20260918T222308.300182IST-9536d525` passed and retains these phase exposures and formation-only means instead of rejecting unequal exposure. It writes `paired-checkpoints.csv` and independently audits 73,728 training plus 56,400 evaluation rows per arm. The one-variable intervention changed initialization and exploration as well as learning, so the result does not isolate a PPO mechanism.

A frozen-policy per-drone replay is pending to inspect height, contact force, commanded speed, and target direction through takeoff. Preserve these negative runs; any next controller or safety change needs a new image and a separate experiment lineage.

## Saved-command diagnosis before the intervention

A read-only count over each baseline update-4 frozen replay's 12,800 formation drone-steps found: seed 41 fourth-coordinate values were negative on 100% of rows, mean absolute scale 0.111, and 35% had magnitude below 0.1; seed 73 values were nonnegative on 100% of rows, mean absolute scale 0.026, and 100% had magnitude below 0.1. The respective mean commanded speeds were 0.0553 and 0.0131 m/s. The sign split is compatible with the many-to-one absolute-value decoder, while the very small magnitudes are the immediate reason for low requested speed. These are frozen deterministic outputs; stochastic training actions can have a different distribution. They motivate the experiment but do not establish its outcome.
