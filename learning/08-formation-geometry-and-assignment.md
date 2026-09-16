# Formation geometry and one-to-one assignment

## What we built

A formation controller needs a precise answer to three questions before it can learn:

1. Where should each slot be?
2. Which drone should track each slot?
3. How far is the current swarm from the requested formation?

The new `align.formations` package answers those questions with ordinary Python mathematics. Keeping it outside the simulator lets us test exact examples quickly and reuse the same definitions in task reset, reward calculation, evaluation, and plotting.

The operational contract and command are in the [formation geometry guide](../docs/09-formation-geometry.md).

## A shape is a set of offsets

Suppose a four-drone plane template contains:

```text
(-0.5, -0.5, 0)   (0.5, -0.5, 0)
(-0.5,  0.5, 0)   (0.5,  0.5, 0)
```

These are offsets around the formation center, not absolute flight positions. Their centroid is zero. If the desired center is `(10, 4, 2)`, adding that center puts every target at altitude 2 m near that point. A yaw rotation can turn the offsets around vertical before translation.

The generator measures the nearest pair and rescales the whole template. If the requested spacing is 1.5 m, the nearest target pair is exactly 1.5 m apart, apart from floating-point rounding. This makes “spacing” measurable instead of treating a shape radius or grid scale as an approximate substitute.

## Why the four generators differ

- A plane uses a two-dimensional square lattice.
- A cube uses a three-dimensional cubic lattice.
- A sphere distributes points across a spherical surface using a Fibonacci sequence.
- A pyramid uses a base and one apex.

For eight agents, the cube has eight exact vertices. The sphere deliberately has a different set of pair distances, so requesting cube versus sphere produces a real geometric change. When a lattice contains more locations than needed, deterministic farthest-point selection spreads the chosen points. Repeating the same input produces identical coordinates.

These constructions define ALiGn's task inputs. They do not show that one shape is best, that a particular neighbor count makes it rigid, or that drones can move between every pair of shapes without collision.

## Assignment avoids crossed destinations

Slots have no natural drone identity. Imagine two drones:

```text
drone 0 at x=9.9       slots at x=0 and x=10
drone 1 at x=0.1
```

Index-by-index assignment would send drone 0 toward 0 and drone 1 toward 10, making them cross. The minimum-distance assignment sends drone 0 to 10 and drone 1 to 0.

For many drones, choosing the nearest slot independently can assign two drones to the same target. The Hungarian algorithm solves the complete one-to-one problem: every agent receives exactly one distinct slot while minimizing the sum of squared travel distances. The implementation takes cubic time, written `O(N³)`, which is appropriate for centralized reset or mission planning at the swarm sizes under study.

Once assignment is chosen, the task should keep it fixed for the declared segment. If it recalculates every step, two nearby drones can exchange identities. Metrics may appear smooth while neither controller reliably reaches its own destination.

## Two errors answer different questions

Consider targets at `(0,0,0)` and `(1,0,0)`. Now translate both drones one metre upward:

```text
actual positions: (0,1,0), (1,1,0)
```

Each drone is one metre from its assigned target, so assigned mean squared error is 1 m² and assigned RMSE is 1 m. The drones remain exactly one metre apart, equal to the target distance, so pairwise shape error is zero.

This distinction is useful:

- **Assigned error** asks whether the formation is at the requested pose and whether each identity is at its slot.
- **Pairwise error** asks whether the internal shape has the requested inter-agent distances.

A controller can form the correct shape in the wrong place, or put the group center correctly while distorting the shape. Recording both reveals which happened.

## Why mean replaces the old raw sum

If every drone has a squared error of 1 m², a two-drone sum is 2 and an eight-drone sum is 8. The larger number comes only from counting more agents. The mean is 1 in both cases.

ALiGn retains the sum in reports for audit, but uses the mean and root mean square for interpretation. It also divides MSE by target diameter squared:

```text
normalized MSE = MSE / target_diameter²
```

That dimensionless value helps compare geometrically similar tasks at different physical scales. It does not remove every difference between swarm sizes: shape topology, crowding, dynamics, and controller interactions still change.

## Geometry edges are not communication

The pairwise metric examines all `N(N-1)/2` pairs. This is an evaluator with access to recorded states. A future actor will see only explicitly local neighbors within a configured range and capacity.

Counting metric pairs as messages would be wrong. Radio traffic depends on what payload is sent, how often it is sent, addressing or broadcast assumptions, retries, headers, delay, and loss. Those quantities will be implemented and reported separately.

Likewise, writing down `3N-6` edges does not by itself prove a three-dimensional formation is rigid or establish an optimal neighbor count. Rigidity depends on the actual geometry and graph.

## What the tests establish

The CPU suite checks:

- exact count, zero centroid, and measured minimum spacing for all four shapes;
- the eight cube vertices, planar Z coordinates, a pyramid apex, and a sphere distinct from the cube;
- deterministic generation and a known +Z yaw rotation;
- one-to-one assignment and equality with brute-force search on a small example;
- translation-sensitive assigned error versus translation-invariant pairwise error;
- mean-versus-sum behavior and known normalized errors;
- complete report construction.

These checks establish the mathematical contract. They do not create drones, execute a transition, test collisions, or prove reward-driven learning. The next bounded task must place several real drone instances safely, reset them reproducibly, and exercise these targets under deterministic control before MAPPO begins.
