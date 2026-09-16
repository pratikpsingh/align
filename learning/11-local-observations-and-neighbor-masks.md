# Local observations and neighbor masks

## What an observation is

A policy cannot act directly on the complete simulator. It receives an observation: a carefully defined set of numbers available at one decision time.

For decentralized execution, each UAV runs the same actor but builds its own observation. Drone 2 can use its own motion, its own assigned target, and selected nearby drones. It cannot inspect every world position simply because the simulator has them.

## Local does not mean body-frame here

ALiGn's high-level action commands velocity in shared world axes. The local actor observation therefore also uses shared gravity-aligned world axes:

```text
+X, +Y: horizontal
+Z: up
```

“Local” means that absolute position is absent and neighbor data is spatially restricted. It does not mean each vector is rotated into the drone body frame. This requires a common world-axis or heading estimate. The assumption is explicit because a real system would need localization or state estimation to supply it.

## The six self/task values

Each drone receives:

```text
own velocity: vx, vy, vz
relative target: target_x - x, target_y - y, target_z - z
```

Suppose a drone is at `(4, 2, 1)` and its slot is `(5, 2, 1.5)`. It sees the relative target `(1, 0, 0.5)`, not the absolute position `(4, 2, 1)`.

If the whole mission is translated by `(10, -3, 2)`, this actor observation remains the same. The centralized critic changes because it is allowed to know the global training state.

## Radius first, budget second

Assume a 1.5 m radius and a two-neighbor budget. Other drones are 0.4, 1.0, and 1.6 m away.

1. The 1.6 m drone is removed by the radius rule.
2. The 0.4 and 1.0 m drones are sorted by distance.
3. Both fit the two-slot budget.

If the budget were one, only the 0.4 m drone would remain. If every drone were 1.6 m away, every neighbor slot would be empty. ALiGn never reaches outside the radius to fill a minimum count.

This corrects the old implementation, which could include the closest drone even when it was beyond the declared communication radius.

## Why masks are necessary

Each neighbor contributes six relative values:

```text
relative position: dx, dy, dz
relative velocity: dvx, dvy, dvz
```

The actor always has seven slots, so absent neighbors are represented by zeroes. But a real neighbor could also have zero relative position and velocity in a synthetic or faulty sample. A separate mask removes the ambiguity:

```text
mask = 1: use this slot
mask = 0: ignore this padded slot
```

A future neighbor encoder must apply the mask rather than expecting the neural network to infer padding.

## Fixed size and larger swarms

The actor layout is:

```text
6 self/task values
7 × 6 neighbor values
7 masks
----------------------
55 values per drone
```

A four-drone swarm and a 128-drone swarm still give each actor 55 values. More drones can create more in-radius candidates, but the nearest seven are retained. This fixed capacity makes a shared frozen actor structurally usable at larger swarm sizes; it does not prove the policy will perform well there.

The training critic is different. It has eight padded global agent slots and is used only while training with the intended eight-agent setup. Actor-only construction remains available for larger evaluation swarms.

## Normalization and clipping

Neural networks train more reliably when inputs have comparable scales. ALiGn divides:

- own velocity by 1.0 m/s;
- target displacement by 3.0 m;
- neighbor displacement by the 1.5 m radius;
- relative neighbor velocity by 2.0 m/s.

Values are clipped to `[-1, 1]`, and every clipping event is counted. Silent saturation would hide an observation-distribution problem. The accepted trajectory had zero saturation, but randomized training scenarios still need measurement.

## What the audit established

The accepted physical trajectory created 8,840 actor observations. Each had exactly 55 finite bounded values. Neighbor counts changed between one, two, and three, which demonstrates a dynamic topology rather than a fixed all-to-all layout. Masks, padding, the radius, deterministic ordering, and the separate 80-value critic state all passed their checks.

The physical trajectory had only four drones, so it never exhausted seven slots. Analytical tests create more in-radius candidates and confirm that the nearest-budget rule truncates them. Other tests cover no-neighbor inputs and actor-only construction for 16 agents.

## Neighbor count is not communication traffic

An observation edge answers, “whose state was made available to this actor?” It does not answer how many radio packets were sent.

One broadcast might serve several neighbors. A packet has headers and may be retransmitted. Discovery messages, loss, delay, and update rate matter. ALiGn records the observation topology now and will add a separate declared traffic model before making communication-cost claims.

## What comes next

The next implemented task is a small vectorized OmniDrones environment contract. It combines:

- these actor and critic observations;
- the audited reward components;
- explicit termination and time-limit truncation;
- reset-safe per-environment memory;
- the validated velocity controller.

Its CPU contracts and cloned-physics acceptance pass. The next learning topic explains its state and reset behavior; the recurrent rollout topic then defines the checked temporal storage contract used before MAPPO updates.
