"""Pure formation geometry used by simulation, training, and evaluation.

Coordinates use a right-handed world frame in metres: +X and +Y are horizontal
and +Z is up. Quaternions and simulator APIs deliberately do not appear here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

Point3 = tuple[float, float, float]
SUPPORTED_KINDS = ("cube", "sphere", "pyramid", "plane")
_EPSILON = 1e-12


@dataclass(frozen=True)
class FormationTemplate:
    """Centred target offsets whose nearest distinct slots have the requested spacing."""

    kind: str
    points_m: tuple[Point3, ...]
    minimum_spacing_m: float
    diameter_m: float


@dataclass(frozen=True)
class Assignment:
    """Minimum-distance mapping from each agent index to one target-slot index."""

    slot_for_agent: tuple[int, ...]
    total_squared_distance_m2: float
    mean_squared_distance_m2: float
    root_mean_squared_distance_m: float


@dataclass(frozen=True)
class FormationMetrics:
    """Auditable formation errors with explicit aggregation and normalization."""

    agent_count: int
    assigned_sum_squared_error_m2: float
    assigned_mean_squared_error_m2: float
    assigned_root_mean_squared_error_m: float
    assigned_normalized_mean_squared_error: float
    pairwise_mean_squared_error_m2: float
    pairwise_root_mean_squared_error_m: float
    pairwise_normalized_mean_squared_error: float
    target_diameter_m: float
    minimum_actual_separation_m: float


def generate_template(
    kind: str, num_agents: int, minimum_spacing_m: float = 1.0
) -> FormationTemplate:
    """Generate deterministic centred slots and enforce their actual minimum spacing."""
    if kind not in SUPPORTED_KINDS:
        choices = ", ".join(SUPPORTED_KINDS)
        raise ValueError(f"Unsupported formation kind {kind!r}; expected one of: {choices}")
    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents < 1:
        raise ValueError("num_agents must be a positive integer")
    if not math.isfinite(minimum_spacing_m) or minimum_spacing_m <= 0.0:
        raise ValueError("minimum_spacing_m must be finite and positive")

    generators = {
        "cube": _cube_points,
        "sphere": _sphere_points,
        "pyramid": _pyramid_points,
        "plane": _plane_points,
    }
    raw_points = generators[kind](num_agents)
    centred = _center(raw_points)
    scaled = _scale_to_minimum_spacing(centred, minimum_spacing_m)
    return FormationTemplate(
        kind=kind,
        points_m=scaled,
        minimum_spacing_m=minimum_spacing_m,
        diameter_m=_diameter(scaled),
    )


def place_template(
    template: FormationTemplate,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    yaw_rad: float = 0.0,
) -> tuple[Point3, ...]:
    """Rotate target offsets about world +Z, then translate them into the world frame."""
    center = _point(center_m, "center_m")
    if not math.isfinite(yaw_rad):
        raise ValueError("yaw_rad must be finite")
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return tuple(
        (
            center[0] + cosine * x - sine * y,
            center[1] + sine * x + cosine * y,
            center[2] + z,
        )
        for x, y, z in template.points_m
    )


def assign_agents_to_slots(
    agent_positions_m: Sequence[Sequence[float]],
    target_positions_m: Sequence[Sequence[float]],
) -> Assignment:
    """Find a one-to-one minimum-squared-distance assignment using Hungarian matching."""
    agents = _points(agent_positions_m, "agent_positions_m")
    targets = _points(target_positions_m, "target_positions_m")
    if len(agents) != len(targets):
        raise ValueError("agent and target counts must match")
    if not agents:
        raise ValueError("at least one agent and target are required")

    costs = [[_squared_distance(agent, target) for target in targets] for agent in agents]
    slot_for_agent = _hungarian(costs)
    total = math.fsum(costs[index][slot] for index, slot in enumerate(slot_for_agent))
    mean = total / len(agents)
    return Assignment(
        slot_for_agent=slot_for_agent,
        total_squared_distance_m2=total,
        mean_squared_distance_m2=mean,
        root_mean_squared_distance_m=math.sqrt(mean),
    )


def evaluate_formation(
    agent_positions_m: Sequence[Sequence[float]],
    assigned_target_positions_m: Sequence[Sequence[float]],
) -> FormationMetrics:
    """Measure tracking error and all-pair shape distortion for fixed agent identities.

    Target positions must already be ordered by agent identity. Assigned tracking
    measures absolute placement. Pairwise distortion ignores common translation
    and rotation because it compares inter-agent distances.
    """
    agents = _points(agent_positions_m, "agent_positions_m")
    targets = _points(assigned_target_positions_m, "assigned_target_positions_m")
    if len(agents) != len(targets):
        raise ValueError("agent and assigned-target counts must match")
    if not agents:
        raise ValueError("at least one agent and target are required")

    assigned_errors = [
        _squared_distance(agent, target) for agent, target in zip(agents, targets, strict=True)
    ]
    assigned_sum = math.fsum(assigned_errors)
    assigned_mean = assigned_sum / len(agents)
    target_diameter = _diameter(targets)
    normalization_scale_squared = target_diameter**2 if target_diameter > _EPSILON else 1.0

    pairwise_errors = [
        (_distance(agents[first], agents[second]) - _distance(targets[first], targets[second])) ** 2
        for first, second in combinations(range(len(agents)), 2)
    ]
    pairwise_mean = math.fsum(pairwise_errors) / len(pairwise_errors) if pairwise_errors else 0.0

    return FormationMetrics(
        agent_count=len(agents),
        assigned_sum_squared_error_m2=assigned_sum,
        assigned_mean_squared_error_m2=assigned_mean,
        assigned_root_mean_squared_error_m=math.sqrt(assigned_mean),
        assigned_normalized_mean_squared_error=assigned_mean / normalization_scale_squared,
        pairwise_mean_squared_error_m2=pairwise_mean,
        pairwise_root_mean_squared_error_m=math.sqrt(pairwise_mean),
        pairwise_normalized_mean_squared_error=pairwise_mean / normalization_scale_squared,
        target_diameter_m=target_diameter,
        minimum_actual_separation_m=_minimum_spacing(agents),
    )


def _plane_points(count: int) -> tuple[Point3, ...]:
    side = math.ceil(math.sqrt(count))
    candidates = tuple(
        (float(column), float(row), 0.0) for row in range(side) for column in range(side)
    )
    return _farthest_sample(candidates, count)


def _cube_points(count: int) -> tuple[Point3, ...]:
    side = math.ceil(count ** (1.0 / 3.0))
    while side**3 < count:
        side += 1
    candidates = tuple(
        (float(x), float(y), float(z))
        for z in range(side)
        for y in range(side)
        for x in range(side)
    )
    return _farthest_sample(candidates, count)


def _sphere_points(count: int) -> tuple[Point3, ...]:
    if count == 1:
        return ((0.0, 0.0, 0.0),)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(count):
        z = 1.0 - (2.0 * index + 1.0) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        azimuth = index * golden_angle
        points.append((radius * math.cos(azimuth), radius * math.sin(azimuth), z))
    return tuple(points)


def _pyramid_points(count: int) -> tuple[Point3, ...]:
    if count == 1:
        return ((0.0, 0.0, 0.0),)
    if count == 2:
        return ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))

    base_count = count - 1
    side = math.ceil(math.sqrt(base_count))
    half = (side - 1) / 2.0
    candidates = tuple(
        (float(column) - half, float(row) - half, 0.0)
        for row in range(side)
        for column in range(side)
    )
    base = _farthest_sample(candidates, base_count)
    apex = (0.0, 0.0, float(side - 1))
    return (*base, apex)


def _farthest_sample(candidates: Sequence[Point3], count: int) -> tuple[Point3, ...]:
    """Choose a deterministic, spatially spread subset from a regular lattice."""
    if count >= len(candidates):
        return tuple(candidates)
    centroid = _center(candidates)
    first = max(range(len(candidates)), key=lambda index: (_norm_squared(centroid[index]), index))
    selected = [first]
    remaining = set(range(len(candidates)))
    remaining.remove(first)
    while len(selected) < count:
        next_index = max(
            remaining,
            key=lambda index: (
                min(
                    _squared_distance(candidates[index], candidates[chosen]) for chosen in selected
                ),
                index,
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return tuple(candidates[index] for index in selected)


def _scale_to_minimum_spacing(
    points: Sequence[Point3], minimum_spacing_m: float
) -> tuple[Point3, ...]:
    if len(points) == 1:
        return tuple(points)
    current = _minimum_spacing(points)
    if current <= _EPSILON:
        raise ValueError("template contains coincident points")
    scale = minimum_spacing_m / current
    return tuple((x * scale, y * scale, z * scale) for x, y, z in points)


def _center(points: Sequence[Point3]) -> tuple[Point3, ...]:
    if not points:
        raise ValueError("cannot center an empty point set")
    inverse_count = 1.0 / len(points)
    center = (
        math.fsum(point[0] for point in points) * inverse_count,
        math.fsum(point[1] for point in points) * inverse_count,
        math.fsum(point[2] for point in points) * inverse_count,
    )
    return tuple(
        (point[0] - center[0], point[1] - center[1], point[2] - center[2]) for point in points
    )


def _points(values: Sequence[Sequence[float]], name: str) -> tuple[Point3, ...]:
    return tuple(_point(value, f"{name}[{index}]") for index, value in enumerate(values))


def _point(value: Sequence[float], name: str) -> Point3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    point = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{name} coordinates must be finite")
    return point


def _minimum_spacing(points: Sequence[Point3]) -> float:
    if len(points) < 2:
        return 0.0
    return min(_distance(first, second) for first, second in combinations(points, 2))


def _diameter(points: Sequence[Point3]) -> float:
    if len(points) < 2:
        return 0.0
    return max(_distance(first, second) for first, second in combinations(points, 2))


def _distance(first: Point3, second: Point3) -> float:
    return math.sqrt(_squared_distance(first, second))


def _squared_distance(first: Point3, second: Point3) -> float:
    return math.fsum((left - right) ** 2 for left, right in zip(first, second, strict=True))


def _norm_squared(point: Point3) -> float:
    return math.fsum(coordinate**2 for coordinate in point)


def _hungarian(costs: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Solve a square linear assignment problem in O(N^3), with deterministic ties."""
    size = len(costs)
    if any(len(row) != size for row in costs):
        raise ValueError("cost matrix must be square")

    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    matching = [0] * (size + 1)
    path = [0] * (size + 1)

    for row in range(1, size + 1):
        matching[0] = row
        column = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            active_row = matching[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    path[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[matching[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            previous = path[column]
            matching[column] = matching[previous]
            column = previous
            if column == 0:
                break

    assignment = [0] * size
    for column in range(1, size + 1):
        assignment[matching[column] - 1] = column - 1
    return tuple(assignment)
