"""Local bounded velocity intervention for a drone below a nearby neighbor."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WakeGuardConfig:
    schema_version: int = 1
    sensing_radius_m: float = 1.5
    wake_radius_m: float = 0.9
    minimum_overhead_m: float = 0.1
    maximum_overhead_m: float = 1.5
    prediction_seconds: float = 0.8
    maximum_command_speed_m_s: float = 0.5
    minimum_predicted_separation_m: float = 0.55
    airborne_height_m: float = 0.25

    def __post_init__(self):
        fields = (
            self.sensing_radius_m,
            self.wake_radius_m,
            self.minimum_overhead_m,
            self.maximum_overhead_m,
            self.prediction_seconds,
            self.maximum_command_speed_m_s,
            self.minimum_predicted_separation_m,
            self.airborne_height_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in fields):
            raise ValueError("wake guard values must be finite and positive")
        if not (self.schema_version == 1 and self.wake_radius_m < self.sensing_radius_m):
            raise ValueError("wake radius must fit inside the local sensing radius")
        if self.minimum_overhead_m >= self.maximum_overhead_m:
            raise ValueError("overhead interval is empty")


def guard_focal_command(
    positions: tuple[tuple[float, float, float], ...],
    requested: tuple[tuple[float, float, float], ...],
    focal_agent: int,
    config: WakeGuardConfig | None = None,
) -> tuple[tuple[float, float, float], dict]:
    """Choose a lateral escape command using only currently local relative poses.

    The focal agent alone may change. If every candidate predicts a separation
    violation, return the requested command and flag the unresolved conflict.
    """
    config = config or WakeGuardConfig()
    if len(positions) != len(requested) or not 0 <= focal_agent < len(positions):
        raise ValueError("position/command count or focal agent is invalid")
    if not all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in (*positions, *requested)):
        raise ValueError("positions and commands must be finite 3-vectors")
    if any(
        math.dist((0, 0, 0), cmd) > config.maximum_command_speed_m_s + 1e-5 for cmd in requested
    ):
        raise ValueError("requested speed exceeds the controller bound")
    focal = positions[focal_agent]
    near = [
        agent
        for agent, point in enumerate(positions)
        if agent != focal_agent and math.dist(focal, point) <= config.sensing_radius_m
    ]
    overhead = [
        agent
        for agent in near
        if config.minimum_overhead_m <= positions[agent][2] - focal[2] <= config.maximum_overhead_m
        and math.dist(focal[:2], positions[agent][:2]) < config.wake_radius_m
    ]
    details = {"active": False, "unresolved": False, "overhead_agents": overhead}
    if focal[2] < config.airborne_height_m or not overhead:
        return requested[focal_agent], details
    vz = requested[focal_agent][2]
    horizontal_speed = math.sqrt(max(0.0, config.maximum_command_speed_m_s**2 - vz**2))
    directions = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (2**-0.5, 2**-0.5),
        (2**-0.5, -(2**-0.5)),
        (-(2**-0.5), 2**-0.5),
        (-(2**-0.5), -(2**-0.5)),
    )
    best = None
    for direction in directions:
        command = (direction[0] * horizontal_speed, direction[1] * horizontal_speed, vz)
        predicted = tuple(focal[i] + config.prediction_seconds * command[i] for i in range(3))
        distances = {
            agent: math.dist(
                predicted,
                tuple(
                    positions[agent][i] + config.prediction_seconds * requested[agent][i]
                    for i in range(3)
                ),
            )
            for agent in near
        }
        if any(value < config.minimum_predicted_separation_m for value in distances.values()):
            continue
        wake_clearance = min(
            math.dist(
                predicted[:2],
                tuple(
                    positions[agent][i] + config.prediction_seconds * requested[agent][i]
                    for i in range(2)
                ),
            )
            for agent in overhead
        )
        score = (
            wake_clearance,
            min(distances.values()),
            sum(command[i] * requested[focal_agent][i] for i in range(3)),
        )
        if best is None or score > best[0]:
            best = (score, command)
    if best is None:
        details["unresolved"] = True
        return requested[focal_agent], details
    details.update(active=True, predicted_wake_clearance_m=best[0][0])
    return best[1], details
