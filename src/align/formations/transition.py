"""Identity-preserving destination assignment with analytic straight-line clearance.

This is a geometric planning check in world metres, not a physical collision
certificate. Drones can deviate from synchronous straight-line paths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations, permutations

from align.formations.geometry import SUPPORTED_KINDS

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class ShapeTransitionConfig:
    schema_version: int = 1
    source_kind: str = "plane"
    destination_kind: str = "pyramid"
    command_step: int = 1300
    minimum_planned_separation_m: float = 0.55
    maximum_search_agents: int = 8
    destination_center_m: Point3 | None = None
    destination_horizontal_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2, 3):
            raise ValueError("shape-transition schema_version must be 1, 2, or 3")
        if self.schema_version < 3 and self.destination_horizontal_scale != 1.0:
            raise ValueError("horizontal scale requires schema 3")
        if (
            type(self.destination_horizontal_scale) not in (int, float)
            or not math.isfinite(self.destination_horizontal_scale)
            or self.destination_horizontal_scale < 1.0
            or self.destination_horizontal_scale > 3.0
        ):
            raise ValueError("horizontal scale must be finite and in [1, 3]")
        if self.schema_version == 1 and self.destination_center_m is not None:
            raise ValueError("schema 1 uses the construction target center")
        if self.schema_version >= 2 and (
            self.destination_center_m is None
            or len(self.destination_center_m) != 3
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in self.destination_center_m
            )
        ):
            raise ValueError("schema 2/3 requires a finite destination XYZ center")
        if self.source_kind not in SUPPORTED_KINDS or self.destination_kind not in SUPPORTED_KINDS:
            raise ValueError("transition kinds must be supported formation templates")
        if self.source_kind == self.destination_kind:
            raise ValueError("a shape transition requires different source and destination kinds")
        if type(self.command_step) is not int or self.command_step < 1:
            raise ValueError("command_step must be a positive integer")
        if (
            not math.isfinite(self.minimum_planned_separation_m)
            or self.minimum_planned_separation_m <= 0
        ):
            raise ValueError("minimum planned separation must be finite and positive")
        if type(self.maximum_search_agents) is not int or not 2 <= self.maximum_search_agents <= 8:
            raise ValueError("maximum_search_agents must be in [2, 8]")

    @classmethod
    def from_dict(cls, value: dict) -> ShapeTransitionConfig:
        expected = set(cls.__dataclass_fields__)
        if value.get("schema_version") == 1:
            expected.remove("destination_center_m")
        if value.get("schema_version") in (1, 2):
            expected.remove("destination_horizontal_scale")
        if set(value) != expected:
            raise ValueError("shape-transition configuration keys differ")
        converted = dict(value)
        if "destination_center_m" in converted:
            if not isinstance(converted["destination_center_m"], list):
                raise ValueError("destination center must be a JSON list")
            converted["destination_center_m"] = tuple(converted["destination_center_m"])
        return cls(**converted)

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.schema_version == 1:
            result.pop("destination_center_m")
        else:
            result["destination_center_m"] = list(self.destination_center_m)
        if self.schema_version < 3:
            result.pop("destination_horizontal_scale")
        return result


@dataclass(frozen=True)
class TransitionAssignment:
    """One fixed-ID mapping and the synchronous linear-path clearance bound."""

    slot_for_agent: tuple[int, ...]
    assigned_destination_m: tuple[Point3, ...]
    minimum_interpolated_separation_m: float
    worst_alpha: float
    worst_pair: tuple[int, int]
    total_travel_m: float
    maximum_agent_travel_m: float
    searched_assignments: int
    feasible_assignments: int

    def to_dict(self) -> dict:
        return asdict(self)


def _points(values: Sequence[Sequence[float]], label: str) -> tuple[Point3, ...]:
    if not 2 <= len(values) <= 8:
        raise ValueError(f"{label} must contain 2–8 points")
    result = []
    for point in values:
        if len(point) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in point
        ):
            raise ValueError(f"{label} must contain finite XYZ triples")
        result.append(tuple(float(value) for value in point))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} has overlapping points")
    return tuple(result)


def linear_path_clearance(
    source_positions_m: Sequence[Sequence[float]],
    assigned_destinations_m: Sequence[Sequence[float]],
) -> tuple[float, float, tuple[int, int]]:
    """Return exact minimum pair distance during synchronized linear interpolation.

    For pair i,j, relative position is r(alpha)=r0+alpha*v. The squared
    distance is quadratic, whose minimum on alpha in [0,1] is at the clipped
    projection -dot(r0,v)/dot(v,v).
    """
    source = _points(source_positions_m, "source positions")
    destination = _points(assigned_destinations_m, "assigned destinations")
    if len(source) != len(destination):
        raise ValueError("source and destination counts differ")
    best = (math.inf, 0.0, (0, 1))
    for left, right in combinations(range(len(source)), 2):
        relative = tuple(source[left][axis] - source[right][axis] for axis in range(3))
        motion = tuple(
            destination[left][axis]
            - source[left][axis]
            - destination[right][axis]
            + source[right][axis]
            for axis in range(3)
        )
        motion_norm_sq = math.fsum(value * value for value in motion)
        alpha = (
            max(
                0.0,
                min(
                    1.0,
                    -math.fsum(a * b for a, b in zip(relative, motion, strict=True))
                    / motion_norm_sq,
                ),
            )
            if motion_norm_sq > 0
            else 0.0
        )
        distance = math.sqrt(
            math.fsum((a + alpha * b) ** 2 for a, b in zip(relative, motion, strict=True))
        )
        if distance < best[0]:
            best = (distance, alpha, (left, right))
    return best


def assign_transition_slots(
    source_positions_m: Sequence[Sequence[float]],
    destination_slots_m: Sequence[Sequence[float]],
    *,
    minimum_planned_separation_m: float,
    maximum_search_agents: int = 8,
) -> TransitionAssignment:
    """Select minimum-travel safe mapping among at most eight permutations.

    Exhaustive search is deliberate and capped: beyond eight agents, do not
    silently treat an unconstrained Hungarian assignment as collision-safe.
    """
    source = _points(source_positions_m, "source positions")
    slots = _points(destination_slots_m, "destination slots")
    if len(source) != len(slots):
        raise ValueError("source and destination counts differ")
    if type(maximum_search_agents) is not int or not 2 <= maximum_search_agents <= 8:
        raise ValueError("maximum_search_agents must be in [2, 8]")
    if len(source) > maximum_search_agents:
        raise ValueError("transition exceeds the declared exhaustive-search agent limit")
    if not math.isfinite(minimum_planned_separation_m) or minimum_planned_separation_m <= 0:
        raise ValueError("minimum planned separation must be finite and positive")
    if min(math.dist(source[i], source[j]) for i, j in combinations(range(len(source)), 2)) < (
        minimum_planned_separation_m - 1e-9
    ):
        raise ValueError("source positions already violate the planned margin")
    if min(math.dist(slots[i], slots[j]) for i, j in combinations(range(len(slots)), 2)) < (
        minimum_planned_separation_m - 1e-9
    ):
        raise ValueError("destination slots violate the planned margin")
    best = None
    feasible = 0
    searched = 0
    for slot_for_agent in permutations(range(len(source))):
        searched += 1
        assigned = tuple(slots[index] for index in slot_for_agent)
        separation, alpha, pair = linear_path_clearance(source, assigned)
        if separation + 1e-9 < minimum_planned_separation_m:
            continue
        feasible += 1
        distances = tuple(math.dist(a, b) for a, b in zip(source, assigned, strict=True))
        travel = math.fsum(distances)
        rank = (travel, slot_for_agent)
        if best is None or rank < best[0]:
            best = (
                rank,
                TransitionAssignment(
                    slot_for_agent=slot_for_agent,
                    assigned_destination_m=assigned,
                    minimum_interpolated_separation_m=separation,
                    worst_alpha=alpha,
                    worst_pair=pair,
                    total_travel_m=travel,
                    maximum_agent_travel_m=max(distances),
                    searched_assignments=0,
                    feasible_assignments=0,
                ),
            )
    if best is None:
        raise ValueError("no destination assignment meets the planned linear-path margin")
    return replace(best[1], searched_assignments=searched, feasible_assignments=feasible)
