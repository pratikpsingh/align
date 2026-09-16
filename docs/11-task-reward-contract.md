# Formation-task reward contract

## Status and purpose

The reward mathematics is implemented without simulator imports in `align.tasks.reward`. The offline auditor applies it to saved post-physics trajectories and writes every raw component, weighted contribution, per-agent total, and team reward to CSV. It does not start Isaac Sim.

This contract corrects two problems in the student implementation:

- the formation weight is required to be positive instead of defaulting to zero;
- formation error uses an explicit mean, rather than a function documented as a mean that returns a sum.

The checked baseline weights are a starting configuration, not experimentally optimized values. This increment defines and audits the signal; it does not implement MAPPO or establish that the reward will produce a successful learned policy.

## Inputs and aggregation

At one policy decision, the pure function receives one row per agent:

- world position and assigned target in metres;
- world linear velocity in metres per second;
- the bounded four-value direction/speed action;
- measured contact-force magnitude in newtons;
- an explicit airborne flag;
- previous target distance and decoded velocity command, when the phase has not changed.

The action is decoded through the accepted velocity interface before smoothness or effort is measured. Inputs outside `[-1, 1]`, nonfinite inputs, malformed agent rows, and mismatched memory sizes are rejected.

Each agent receives a total assembled from named contributions. The cooperative team reward is the arithmetic **mean** of per-agent totals. It is not a sum, so adding agents does not multiply the reward scale merely because the swarm is larger.

## Components

Let `d_i` be agent `i`'s distance to its assigned target, `D` the target formation diameter, `s_i` its nearest-neighbor separation, `v_i` its speed, and `u_i` its decoded commanded world velocity. Baseline normalization uses `L = 1 m`, safe separation `s_safe = 0.55 m`, maximum speed `v_max = 0.5 m/s`, and control interval `dt = 0.01 s`.

| Component | Raw value | Meaning |
|---|---|---|
| Formation | negative mean all-pair squared distance distortion divided by `D²` | shared shape error, invariant to common translation and rotation |
| Tracking | `-(d_i/L)²` | assigned absolute target error |
| Progress | clipped `(d_previous - d_i)/L` | improvement since the prior decision |
| Separation | `-[max(0, (s_safe-s_i)/s_safe)]²` | continuous inter-UAV safety-margin intrusion |
| Contact | `-1` for measured airborne contact, otherwise `0` | physical contact distinct from geometric proximity |
| Settling | `-[max(0, 1-d_i/0.5 m)](v_i/v_max)²` | discourages flying through a nearby target |
| Smoothness | `-||u_i-u_previous||²/(3 v_max²)` | change in physical velocity command |
| Effort | `-||u_i||²/v_max²` | physical command magnitude |

Formation, tracking, separation, contact, settling, and effort are state costs. Their weighted values are multiplied by `dt`, making them time integrals. Progress and smoothness are transition terms and apply once per policy decision. The auditor resets progress and smoothness memory at episode reset and whenever the target phase changes; construction progress must not be credited for replacing the ground target with the takeoff target.

The component names, raw values, weights, and weighted values remain separate in artifacts. A report is rejected unless each total reconstructs from its weighted components and each team reward reconstructs as the agent mean.

## Configuration

The versioned baseline is [`configs/task-reward-baseline.json`](../configs/task-reward-baseline.json). Safety thresholds and the control interval must match the source task configuration. Unknown or missing configuration keys are errors.

A future reward change requires a new configuration and experiment lineage. Do not compare returns across reward definitions as though they had the same meaning.

## Audit a saved physical run

From the repository root:

```sh
uv run --locked python scripts/audit_task_rewards.py \
  runs/multi-drone/20260916T121136.980227IST-48963b76
```

The command creates `runs/task-reward/<id>/` containing:

- `report.json`: source identity and SHA-256, resolved reward configuration, checks, aggregation definitions, and repeat summaries;
- `reward.csv`: one row per agent and policy step with raw and weighted components;
- `reward.log` and `events.jsonl`: readable and structured completion records.

`runs/` is ignored and requires separate backup.

## Accepted-trajectory audit

Audit `20260916T122706.141515IST-e549a070` evaluated the accepted four-drone run `20260916T121136.980227IST-48963b76`:

- 8,840 finite agent-step records;
- 8/8 audit checks passed;
- formation weight was active and the formation component was nonzero;
- component totals and team means reconstructed exactly within `1e-15`;
- reward/task timestep and safety thresholds agreed;
- repeat team returns were `-0.05892` and `-0.05890`.

The deterministic controller is competent, so this trajectory has no separation intrusion or airborne contact penalty. Analytical tests separately exercise those failure terms. Similar repeat returns confirm audit repeatability for this trajectory; they do not validate learning.

## Current limitations

The reward is not yet connected to a vectorized OmniDrones learning environment. The baseline weights have not undergone sensitivity studies or ablations. Pairwise formation error is a centralized team signal even though the future actor observation will be local; centralized reward during training does not grant extra inputs to the deployed actor, but that separation must be enforced in the learner. Net base-link contact force does not identify the other collider. Terminal bonuses and failure penalties remain undefined until the vectorized termination contract is built.
