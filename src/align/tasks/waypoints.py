"""Group-synchronous, fixed-ID waypoint targets and CPU progress contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations

from align.simulation.multi_drone_contract import MultiDroneConfig, build_group_layout
from align.tasks.environment import TaskEnvironmentConfig

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class WaypointRouteConfig:
    schema_version: int = 1
    command_step: int = 2600
    goal_displacement_m: Point3 = (3.0, 0.0, 0.0)
    maximum_leg_length_m: float = 0.75
    arrival_radius_m: float = 0.15
    maximum_arrival_speed_m_s: float = 0.2
    minimum_arrival_separation_m: float = 0.55
    arrival_dwell_steps: int = 20

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("waypoint route schema_version must be 1")
        if type(self.command_step) is not int or self.command_step < 1:
            raise ValueError("command_step must be a positive integer")
        if len(self.goal_displacement_m) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in self.goal_displacement_m
        ):
            raise ValueError("goal displacement must contain finite XYZ values")
        displacement_length = math.dist((0.0, 0.0, 0.0), self.goal_displacement_m)
        if not math.isfinite(displacement_length) or displacement_length <= 0:
            raise ValueError("goal displacement length must be finite and positive")
        for name in (
            "maximum_leg_length_m",
            "arrival_radius_m",
            "maximum_arrival_speed_m_s",
            "minimum_arrival_separation_m",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.arrival_dwell_steps) is not int or self.arrival_dwell_steps < 1:
            raise ValueError("arrival_dwell_steps must be positive")

    @classmethod
    def from_dict(cls, value: dict) -> WaypointRouteConfig:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("waypoint route configuration keys differ")
        converted = dict(value)
        if not isinstance(converted["goal_displacement_m"], list):
            raise ValueError("goal displacement must be a JSON list")
        converted["goal_displacement_m"] = tuple(converted["goal_displacement_m"])
        return cls(**converted)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["goal_displacement_m"] = list(self.goal_displacement_m)
        return result


@dataclass(frozen=True)
class WaypointPlan:
    source_center_m: Point3
    goal_center_m: Point3
    centers_m: tuple[Point3, ...]
    assigned_targets_m: tuple[tuple[Point3, ...], ...]
    path_length_m: float
    longest_leg_m: float
    command_step: int
    available_time_s: float
    ideal_speed_limited_lower_bound_s: float
    arrival_dwell_lower_bound_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def make_waypoint_plan(
    construction: MultiDroneConfig, task: TaskEnvironmentConfig, route: WaypointRouteConfig
) -> WaypointPlan:
    """Translate the same assigned formation through evenly spaced group goals."""
    if route.command_step <= construction.ground_steps + construction.takeoff_steps:
        raise ValueError("route must begin after formation construction")
    if route.command_step >= task.max_episode_steps:
        raise ValueError("route command must precede episode deadline")
    source_center = tuple(construction.target_center_m)
    displacement = route.goal_displacement_m
    distance = math.dist((0.0, 0.0, 0.0), displacement)
    leg_count = math.ceil(distance / route.maximum_leg_length_m)
    if leg_count > 128:
        raise ValueError("route exceeds the declared 128-leg CPU planning bound")
    layout = build_group_layout(construction)
    centers = tuple(
        tuple(source_center[axis] + displacement[axis] * leg / leg_count for axis in range(3))
        for leg in range(1, leg_count + 1)
    )
    targets = tuple(
        tuple(
            tuple(point[axis] + center[axis] - source_center[axis] for axis in range(3))
            for point in layout.assigned_target_positions_m
        )
        for center in centers
    )
    for leg in targets:
        for x, y, z in leg:
            if (
                abs(x) >= task.safety_xy_limit_m
                or abs(y) >= task.safety_xy_limit_m
                or not task.crash_height_m < z < task.safety_z_limit_m
            ):
                raise ValueError("route target lies outside the task safety envelope")
    min_target_separation = min(
        math.dist(leg[left], leg[right])
        for leg in targets
        for left, right in combinations(range(construction.num_agents), 2)
    )
    if min_target_separation < route.minimum_arrival_separation_m:
        raise ValueError("route targets violate minimum arrival separation")
    available = (task.max_episode_steps - route.command_step) * construction.physics_dt
    speed_lower_bound = distance / construction.max_speed_m_s
    dwell_lower_bound = leg_count * route.arrival_dwell_steps * construction.physics_dt
    if speed_lower_bound + dwell_lower_bound >= available:
        raise ValueError("route cannot meet deadline even at ideal maximum speed")
    return WaypointPlan(
        source_center_m=source_center,
        goal_center_m=tuple(source_center[i] + displacement[i] for i in range(3)),
        centers_m=centers,
        assigned_targets_m=targets,
        path_length_m=distance,
        longest_leg_m=distance / leg_count,
        command_step=route.command_step,
        available_time_s=available,
        ideal_speed_limited_lower_bound_s=speed_lower_bound,
        arrival_dwell_lower_bound_s=dwell_lower_bound,
    )


class WaypointProgress:
    """Advance one group waypoint only after every drone dwells near its slot."""

    def __init__(self, plan: WaypointPlan, config: WaypointRouteConfig) -> None:
        self.plan = plan
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.dwell_steps = 0
        self.complete = False

    @property
    def current_targets_m(self) -> tuple[Point3, ...]:
        return self.plan.assigned_targets_m[self.index]

    def observe(self, positions_m: tuple[Point3, ...], velocities_m_s: tuple[Point3, ...]) -> dict:
        if len(positions_m) != len(self.current_targets_m) or len(velocities_m_s) != len(
            positions_m
        ):
            raise ValueError("waypoint state has the wrong agent count")
        if any(
            len(row) != 3 or any(not math.isfinite(value) for value in row)
            for row in (*positions_m, *velocities_m_s)
        ):
            raise ValueError("waypoint state must contain finite XYZ triples")
        errors = tuple(
            math.dist(position, target)
            for position, target in zip(positions_m, self.current_targets_m, strict=True)
        )
        separation = min(
            math.dist(positions_m[left], positions_m[right])
            for left, right in combinations(range(len(positions_m)), 2)
        )
        max_speed = max(math.dist((0.0, 0.0, 0.0), velocity) for velocity in velocities_m_s)
        settled = (
            max(errors) <= self.config.arrival_radius_m
            and separation >= self.config.minimum_arrival_separation_m
            and max_speed <= self.config.maximum_arrival_speed_m_s
        )
        before = self.index
        if not self.complete:
            self.dwell_steps = self.dwell_steps + 1 if settled else 0
            if self.dwell_steps >= self.config.arrival_dwell_steps:
                if self.index == len(self.plan.centers_m) - 1:
                    self.complete = True
                else:
                    self.index += 1
                self.dwell_steps = 0
        return {
            "index_before": before,
            "index_after": self.index,
            "advanced": self.index != before,
            "complete": self.complete,
            "settled": settled,
            "dwell_steps": self.dwell_steps,
            "maximum_agent_error_m": max(errors),
            "minimum_separation_m": separation,
            "maximum_speed_m_s": max_speed,
        }
