# Formation geometry, assignment, and metrics

## Purpose and evidence status

This host-side component turns a shape name and swarm size into target offsets, places those offsets in the world, assigns agents to slots, and measures formation error. It contains no Isaac Sim, TorchRL, or policy imports. CPU tests establish its mathematical and serialization behavior; they do not establish that a physical swarm can reach or safely hold the targets.

The implementation supports `cube`, `sphere`, `pyramid`, and `plane`. It corrects the old raw-sum ambiguity by retaining the sum for audit while using explicit means and diameter-normalized means for comparison.

## Coordinate and template contract

All points are three-value tuples in metres in a right-handed world frame. +X and +Y are horizontal and +Z is up. A template is centred at the origin. `place_template` first rotates it by yaw about +Z using the right-hand rule, then adds a world-frame center.

For two or more slots, `minimum_spacing_m` is the actual minimum Euclidean distance between generated slots. Generation centres the raw shape, measures its closest pair, and scales the complete shape to the requested separation. For a one-agent template there is no pair, so its diameter and actual pair separation are zero.

| Shape | Construction |
|---|---|
| Cube | Regular cubic lattice; a deterministic farthest-point subset is used when the count is not a perfect cube. Eight agents give the eight cube vertices. |
| Sphere | Deterministic Fibonacci points on a spherical surface, recentered and rescaled. |
| Pyramid | `N-1` spatially spread points on a square base plus one apex. Five agents give four base corners and an apex. |
| Plane | Regular square lattice in the XY plane; a deterministic farthest-point subset fills incomplete grids. |

These definitions are concrete experiment inputs. Different counts can change a template's combinatorial layout, so reports retain every coordinate. The templates are not claims of optimal packing, rigidity, or dynamically feasible paths.

## Assignment contract

`assign_agents_to_slots` builds the squared Euclidean distance from every current agent position to every target and solves the one-to-one minimum-cost assignment with the Hungarian algorithm. It returns one slot index per agent plus total, mean, and root-mean-square assignment distance. Inputs and tie handling are deterministic, and runtime is `O(N³)`.

Assignment is centralized mission logic. It is not part of the future decentralized actor observation and does not imply inter-agent communication. For a task episode or commanded shape transition, compute assignment at the defined transition boundary and retain it while tracking those targets. Reassigning every control step can make identities switch and hide poor control.

## Metric contract

`evaluate_formation` receives current agent positions and assigned targets in the same agent order. It reports two complementary errors.

Assigned target error includes absolute translation and rotation:

```text
assigned MSE = (1/N) Σ_i ||position_i - target_i||²
```

All-pair shape distortion compares distances for fixed identities:

```text
pairwise MSE = (1/P) Σ_(i<j) (||position_i-position_j|| - ||target_i-target_j||)²
P = N(N-1)/2
```

A translated or rotated rigid copy can therefore have nonzero assigned error and zero pairwise error. Each MSE is also divided by the squared target diameter to create a dimensionless normalized value. The denominator is 1 only for a degenerate one-slot target. Reports also include target diameter and minimum actual agent separation.

All-pair evaluation is fixed independently of the actor's observed-neighbor graph. The `N(N-1)/2` evaluation pairs are neither radio messages nor rigidity edges. Communication accounting will use a separate declared graph and traffic model.

## Reproducible report

The checked configuration is [`configs/formations/eight-drone-suite.json`](../configs/formations/eight-drone-suite.json). Run:

```sh
uv run --locked python scripts/inspect_formations.py
```

The command creates a unique IST-named directory below `runs/formation-geometry/`. It atomically writes `report.json` and saves an IST-readable `formation.log` plus UTC/IST `events.jsonl`. The report records UTC and IST creation timestamps, elapsed time, a reproduction command, resolved configuration path, Python version, coordinate and aggregation contracts, all centred and placed coordinates, and checks for count, centroid, measured spacing, and reversed-slot assignment. A pass means every template satisfies those geometry checks and each reversed target set is recovered with zero assignment cost.

Use a different explicit configuration with:

```sh
uv run --locked python scripts/inspect_formations.py \
  --config configs/formations/eight-drone-suite.json \
  --output-dir runs/formation-geometry
```

`runs/` is ignored. Copy result directories needed for research evidence to the project backup.

## CPU result record

Run `20260916T103919.653203IST-b19ef72d` used CPython 3.12.14 and the checked eight-agent configuration. It passed every recorded template and assignment check for all four shapes: each measured nearest-slot spacing was exactly 1.0 m at stored precision and every reversed-order assignment cost was zero. Its readable log and JSONL event record are complete. This report contains no simulator action; it is mathematical evidence only.

## Code map

| File | Responsibility |
|---|---|
| `src/align/formations/geometry.py` | Pure templates, placement, assignment, and metrics |
| `src/align/formations/report.py` | Strict configuration loading and report construction |
| `scripts/inspect_formations.py` | Thin command entrypoint |
| `configs/formations/eight-drone-suite.json` | Explicit eight-agent example |
| `tests/test_formation_geometry.py` | Analytical geometry, brute-force assignment, metric, and report checks |

## Integration boundary

The next simulator task can consume placed targets without importing simulator code into this package. That task still needs group-aware state, collision-safe reset and takeoff, trajectory generation between shapes, terminations, reward components, and multi-drone controller validation. Minimum target spacing is an input geometry constraint; it does not prove safe separation during travel.
