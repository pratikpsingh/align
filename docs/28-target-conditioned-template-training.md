# Target-conditioned multi-template training

## Purpose

ALiGn can now train one shared recurrent actor against cube, sphere, pyramid,
and plane targets in the same optimizer update. The actor is conditioned by its
assigned target displacement, which was already part of the declared local
55-value observation. It does not receive a global template name or the full
target layout.

This implements the task and reporting boundary needed for retained claim C05.
A short acceptance run can prove that all four targets reach live collection,
PPO, checkpoints, and frozen-policy evaluation. It cannot prove that four
updates produce a useful multi-template policy.

## Schedule contract

[`formation-schedule-four-templates.json`](../configs/formation-schedule-four-templates.json)
declares the exact template set, seed, and assignment rule. A SHA-256-derived
order makes the initial order independent of Python random-number-generator
versions. At each rollout or evaluation batch start, the four cloned environments
contain one template each. The mapping rotates with the completed-update
counter and is stored in the checkpoint task-sampler state. An individual
environment reset advances that environment independently, so instantaneous
counts can differ later in an episode; every raw row records the active
template.

At a resume boundary, ALiGn resets simulator and recurrent state, restores the
completed-update counter, and reconstructs the same next template batch. A
configuration mismatch still invalidates the checkpoint. Episode resets rotate
a selected environment to another declared template.

[`formation-schedule-plane.json`](../configs/formation-schedule-plane.json)
records the previous plane-only behavior explicitly. Commands that omit a
schedule remain backward compatible and construct this one-template schedule
from `construction.formation_kind`.

## Target and observation contract

The existing geometry library creates all four target layouts with 1.0 m
minimum slot spacing and the same world-frame center at `(0, 0, 1.5)` m. The
fixed Hungarian assignment is computed independently for each template. All
four-agent target altitudes lie inside the validated safety envelope.

The actor self-features remain:

```text
[normalized velocity xyz, normalized assigned-target-minus-position xyz]
```

Neighbor features and masks are unchanged. Thus the actor can distinguish
commands through a quantity each drone needs for control, without a global
one-hot shape label. The centralized critic receives the per-agent assigned
targets through its existing training-only state. Actor dimension remains 55;
critic dimension remains 80.

## Raw evidence and acceptance checks

Training `rollout.csv` and deterministic `evaluation.csv` now include
`formation_kind`. Each update and evaluation reports, per template:

- total rows and rows that reached the formation phase;
- mean team reward;
- assigned-position RMSE;
- pairwise-distance RMSE;
- minimum separation;
- terminal outcome counts during evaluation.

The host creates `template-training-curve.csv` and
`template-evaluation-curve.csv` across seeds and checkpoint milestones. A live
phase passes only if all declared templates appear and each reaches the
formation phase with finite, bounded task behavior. Existing PPO, checkpoint,
normalization, locality, and physics checks remain active.

## Bounded acceptance command

Prepare and build a new derived image containing the current source, then run:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<build-run-id> && \
uv run --locked python scripts/run_multi_template_training.py \
  --build-report runs/runtime-build/<build-run-id>/report.json \
  --critic-normalization-config configs/critic-normalization-active-groups-floor.json \
  --accept-eula \
  --gpu 0
```

The probe configuration uses two policy seeds, four PPO updates per seed, and
fresh-process deterministic evaluation at updates 0, 2, and 4. It writes to
`runs/multi-template-training/<run-id>/`. The run executes two training
containers and six evaluation containers. `runs/` is ignored and requires a
separate backup.

## Validation status

The deterministic schedule, all-template geometry/safety bounds, local target
conditioning, resolved configuration, and per-template host aggregation pass
202 CPU tests under Python 3.12 and the isolated Python 3.10 environment. Ruff,
formatting, compilation, and patch checks pass.

Live run `20260917T232002.612950IST-6bc40009` passed on GPU 0 with image
`sha256:9f6aea97eeb970351733cda0e3e50fd9d62af047d87b496987e2f15eafb3f029`.
It completed eight optimizer updates across seeds 41 and 73 and six
fresh-process evaluations at checkpoints 0, 2, and 4 in 717.39 seconds. An
independent raw-data audit found:

- eight training CSVs, each with 768 rows and 218 formation-phase rows for each
  of cube, plane, pyramid, and sphere;
- six evaluation CSVs, each with 800 rows and 250 formation-phase rows for each
  template;
- finite task behavior, valid normalization diagnostics, and exact requested
  checkpoint loads in every evaluation container.

Across seeds, evaluation assigned-position RMSE changed from 1.4355 m at
checkpoint 0 to 1.3969 m at checkpoint 4. Pairwise-distance RMSE changed from
0.2745 m to 0.2987 m, and minimum separation changed from 0.9747 m to 0.8485 m.
All 24 evaluated environment episodes ended at the time limit; none satisfied
the formation success contract. This run accepts the multi-template data,
training, checkpoint, and evaluation wiring. It does not establish reliable
multi-template learning or support a scientific performance claim.
