# Bounded local observation contract

## Status and purpose

ALiGn now has a simulator-independent observation builder for a shared decentralized actor and a separate centralized training critic. It corrects three problems in the student implementation:

- absolute position is removed from the actor input;
- radius filtering happens before the neighbor budget, with no minimum-neighbor filling outside the radius;
- fixed padding has an explicit validity mask.

The builder and offline auditor are implemented and CPU-tested. The accepted four-drone physical trajectory has been audited. The vectorized task now consumes the contract; its cloned-physics acceptance is documented separately. No neural policy consumes these observations yet.

## Measurement and coordinate assumptions

Positions, assigned targets, and linear velocities arrive in a shared right-handed, gravity-aligned world frame: +X and +Y are horizontal and +Z is up. The actor does not receive its absolute position. It receives its world velocity, relative target, and relative neighbor states in those shared axes.

This matches the validated world-velocity action interface, but assumes each vehicle has a consistent heading/world-axis estimate from simulation or a future estimator. It also assumes a mission layer supplies the agent's assigned target and a sensing or communication layer supplies relative neighbor state. The current contract does not claim that a physical radio, localization system, delay model, or state estimator has been implemented.

Attitude and angular velocity are omitted from the high-level actor input because the validated Lee controller owns attitude stabilization. A future controller-interface change requires reevaluating this observation choice.

## Actor observation

The baseline [`configs/local-observation-baseline.json`](../configs/local-observation-baseline.json) sets a 1.5 m neighbor radius and seven neighbor slots. Every actor receives exactly 55 bounded values:

| Flat indices | Count | Values | Scale |
|---|---:|---|---|
| 0–2 | 3 | own world linear velocity XYZ | 1.0 m/s |
| 3–5 | 3 | assigned target minus own position, world XYZ | 3.0 m |
| 6–47 | 42 | seven slots of relative position XYZ and relative velocity XYZ | 1.5 m and 2.0 m/s |
| 48–54 | 7 | one validity mask per neighbor slot | 0 or 1 |

Each physical value is divided by its configured scale and clipped to `[-1, 1]`. The observation records a saturation count so clipping is visible in reports. The mask distinguishes an empty zero-padded slot from a real neighbor whose relative state happens to be zero.

Neighbor identities and raw distances are audit metadata and are not included in the actor vector.

## Strict neighbor rule

For agent `i`:

1. Calculate Euclidean distance to every other group member from the unnormalized positions.
2. Keep only candidates whose distance is at most 1.5 m.
3. Sort by distance, then by stable agent identity to break an exact tie.
4. Keep at most seven.
5. Zero-pad remaining slots and mark their masks false.

There is no minimum neighbor count. If every other drone is 1.500001 m away, all slots remain masked. The radius is therefore a hard information boundary rather than a preference that can be overridden.

Radius filtering creates a directed observation topology after the budget is applied. With equal radii and no budget truncation it is geometrically symmetric, but a tight per-agent budget can make selected edges asymmetric.

## Separate centralized critic state

The training critic has a distinct 80-value input:

- eight agent slots;
- nine values per slot: absolute world position, world velocity, and absolute assigned target;
- one validity mask per slot.

The critic state is normalized and padded in stable identity order. It is available only during centralized training. It is never concatenated into an actor observation.

`build_actor_observations` has no critic-capacity restriction and preserves the same 55-value input for larger evaluated swarms. The combined training builder rejects more than eight agents instead of silently dropping global state. This lets a frozen decentralized actor be evaluated at larger sizes without requiring the training critic to change shape.

## Audit command and artifacts

Audit a saved physical trajectory without Isaac Sim:

```sh
uv run --locked python scripts/audit_local_observations.py \
  runs/multi-drone/20260916T121136.980227IST-48963b76
```

The command creates `runs/local-observation/<id>/` containing:

- `report.json`: source trajectory hash, implementation fingerprint, resolved configuration, feature contract, checks, dimensions, and topology metrics;
- `observations.csv`: one 55-value actor input per agent step plus masks, selected identities, raw distances, and saturation count;
- `observation.log` and `events.jsonl`: readable and structured completion records.

`runs/` is ignored and requires separate backup.

## Accepted-trajectory audit

Run `20260916T125326.859172IST-96b8931e` audited the accepted construction trajectory:

- 8,840 agent steps and 2,210 environment steps;
- 9/9 checks passed;
- actor dimension 55 and critic dimension 80 throughout;
- all observation values were finite and within `[-1, 1]`;
- no actor or critic values saturated;
- mean selected neighbors: 1.966;
- neighbor counts: 2,856 samples with one, 3,428 with two, and 2,556 with three;
- 17,380 directed in-radius candidates were selected;
- no physical sample reached the seven-neighbor budget.

The last item means the physical run validates changing radius-based topology and padding, while analytical tests establish budget truncation. Analytical tests also cover zero-neighbor observations, exact-radius inclusion, deterministic ties, translation invariance, input permutation, critic overflow, and 16-agent actor-only use.

## Communication accounting boundary

An observed directed edge means one actor was allowed to use another agent's state at that step. The 17,380 edges are not measured packets or bytes. They do not include discovery, headers, broadcast reuse, update frequency, retransmission, delay, loss, or radio energy. Future communication experiments must declare a traffic model and label modeled traffic separately from neighbor count.

## Current limitations

The [vectorized task environment](13-vectorized-task-environment.md) now returns these actor inputs, reward values, termination flags, and truncation flags. Its one-world and four-world GPU acceptance run passed.
