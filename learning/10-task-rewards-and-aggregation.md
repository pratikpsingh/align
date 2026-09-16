# Task rewards and aggregation

## What a reward does

A reinforcement-learning policy improves actions that lead to larger numerical rewards. That makes the reward part of the task definition, not just a diagnostic graph. If formation has weight zero, the learner has no direct reason to preserve shape even when a formation-error metric is logged.

ALiGn now has an active, inspectable formation-task reward contract. It is pure Python mathematics, so it can be tested without Isaac Sim and later called from the simulator environment and MAPPO rollout collector.

## Raw measurements, components, and weights

Keep three levels separate:

1. A **measurement** has physical units, such as target distance in metres.
2. A **raw component** converts that measurement to a normalized signed score.
3. A **weighted contribution** states how strongly that component affects the total.

The saved CSV retains all three. This is useful when a return changes: we can tell whether flight behavior changed or only a weight changed.

## A four-drone square example

Suppose the target is a one-metre square. All four drones form the correct square but the whole group is translated by 0.5 m in +X.

- Pairwise distances are still correct, so the formation component is zero.
- Each assigned-target distance is 0.5 m.
- With a one-metre distance scale, each raw tracking value is `-(0.5/1.0)² = -0.25`.
- At `dt = 0.01 s` and tracking weight 1, the per-step weighted tracking contribution is `-0.0025`.

This separation matters. Shape error asks whether the drones make the right geometry. Tracking asks whether that geometry is in the commanded place.

Now place two drones 0.275 m apart while the safety margin is 0.55 m:

```text
intrusion = (0.55 - 0.275) / 0.55 = 0.5
separation component = -(0.5)² = -0.25
```

Both drones receive this continuous margin penalty. A contact penalty is still zero unless the simulator reports physical airborne contact. Proximity and contact answer different questions.

## Why we use means

A sum grows when the number of drones grows. If eight agents each have the same error as four agents, a sum doubles even though average behavior is unchanged.

ALiGn uses:

- a mean over all drone pairs for formation distortion;
- a mean over agents for the cooperative team reward.

This does not make every swarm-size comparison automatically fair: the number and geometry of pairs still change. It removes the simplest size-dependent scale error and leaves the definition explicit.

## Why persistent costs use the timestep

Tracking error exists continuously. At 100 Hz, we observe it 100 times per second. At 50 Hz, we observe it 50 times. Summing the same per-step penalty without `dt` would make the objective depend on how often we sample it.

State costs are therefore multiplied by the control interval. Progress is already a change in distance, so it telescopes over time and is not multiplied by `dt`. Smoothness measures a change between consecutive commands and is also a transition term.

## Why smoothness uses decoded velocity

The action contains a direction and a speed scale. A unit direction has norm one even for a very slow command, and its coordinates can change abruptly near zero speed. Raw action-vector differences would therefore misrepresent physical command changes.

ALiGn decodes the action into world velocity first. Effort measures normalized command speed squared. Smoothness measures the normalized squared change in commanded velocity. The input action still has to be bounded to `[-1, 1]`; clipping a future sampled action remains the environment/action-distribution contract, not a hidden reward fallback.

## Memory and phase changes

Progress and smoothness need the previous step. Their memory is explicitly returned by the reward function and must be reset:

- at episode reset;
- when ground changes to takeoff;
- when takeoff changes to formation;
- later, when a waypoint or shape command changes.

Otherwise, replacing the target could create a fake progress penalty or bonus, and the first command of a new task could be compared to an unrelated old command.

## Code and evidence

- `align.tasks.reward` defines the configuration, component records, memory, and pure calculation.
- `align.tasks.reward_report` groups a raw trajectory by repeat and step, resets memory at phase boundaries, and writes auditable rows.
- `scripts/audit_task_rewards.py` is the thin command.
- `tests/test_task_reward.py` covers exact aggregation, active formation weight, shape versus tracking, safety versus contact, timestep scaling, bounded actions, memory reset, and configuration mismatch.

The accepted physical trajectory generated 8,840 agent-step reward rows and passed eight report checks. The two deterministic repeats had nearly identical returns. This establishes correct computation on real saved states. It does not show that MAPPO can learn from the signal.

## What comes next

The next task contract is the future actor observation: which own-state values are available, which neighbors are admitted by the hard radius and budget, how padding masks work, and what the centralized critic may see. After that contract is tested, the vectorized simulator environment can return observations, these rewards, termination flags, and truncation flags in a form suitable for recurrent MAPPO.
