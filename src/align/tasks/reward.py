"""Pure formation-task reward mathematics with explicit aggregation.

All position inputs use metres in the right-handed world frame. Linear
velocities use metres per second. Actions are the bounded four-value policy
commands. This module deliberately has no simulator, NumPy, or PyTorch import.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields

from align.formations import evaluate_formation
from align.simulation.contract import direction_speed

Point3 = tuple[float, float, float]
Action4 = tuple[float, float, float, float]
COMPONENT_NAMES = (
    "formation",
    "tracking",
    "progress",
    "separation",
    "contact",
    "settling",
    "smoothness",
    "effort",
)
TIME_INTEGRATED_COMPONENTS = frozenset(
    {"formation", "tracking", "separation", "contact", "settling", "effort"}
)


@dataclass(frozen=True)
class RewardConfig:
    """Versioned baseline weights and physical normalization scales."""

    schema_version: int = 1
    formation_weight: float = 1.0
    tracking_weight: float = 1.0
    progress_weight: float = 2.0
    separation_weight: float = 2.0
    contact_weight: float = 2.0
    settling_weight: float = 0.2
    smoothness_weight: float = 0.05
    effort_weight: float = 0.01
    control_dt_seconds: float = 0.01
    distance_scale_m: float = 1.0
    minimum_separation_m: float = 0.55
    contact_force_threshold_n: float = 0.01
    max_speed_m_s: float = 0.5
    settling_radius_m: float = 0.5

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        weights = {name: getattr(self, f"{name}_weight") for name in COMPONENT_NAMES}
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("reward weights must be finite and nonnegative")
        if self.formation_weight <= 0.0:
            raise ValueError("formation_weight must be positive")
        positive_scales = (
            self.control_dt_seconds,
            self.distance_scale_m,
            self.minimum_separation_m,
            self.max_speed_m_s,
            self.settling_radius_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_scales):
            raise ValueError("reward normalization scales must be finite and positive")
        if (
            not math.isfinite(self.contact_force_threshold_n)
            or self.contact_force_threshold_n < 0.0
        ):
            raise ValueError("contact_force_threshold_n must be finite and nonnegative")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RewardConfig:
        expected = {field.name for field in fields(cls)}
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"reward configuration keys mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RewardComponents:
    """Signed component values; better task behavior produces larger values."""

    formation: float
    tracking: float
    progress: float
    separation: float
    contact: float
    settling: float
    smoothness: float
    effort: float

    def total(self) -> float:
        return math.fsum(getattr(self, name) for name in COMPONENT_NAMES)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RewardMemory:
    """Previous policy-step values; create a fresh instance at reset or target change."""

    target_distances_m: tuple[float, ...] | None = None
    commanded_velocities_m_s: tuple[Point3, ...] | None = None


@dataclass(frozen=True)
class RewardStep:
    """Per-agent raw/weighted components and their swarm-size-invariant team mean."""

    raw_by_agent: tuple[RewardComponents, ...]
    weighted_by_agent: tuple[RewardComponents, ...]
    total_by_agent: tuple[float, ...]
    team_reward: float
    target_distances_m: tuple[float, ...]
    speeds_m_s: tuple[float, ...]
    minimum_separations_m: tuple[float, ...]
    airborne_contact: tuple[bool, ...]


def compute_step_reward(
    positions_m: Sequence[Sequence[float]],
    targets_m: Sequence[Sequence[float]],
    velocities_m_s: Sequence[Sequence[float]],
    actions: Sequence[Sequence[float]],
    contact_forces_n: Sequence[float],
    airborne: Sequence[bool],
    config: RewardConfig,
    memory: RewardMemory | None = None,
) -> tuple[RewardStep, RewardMemory]:
    """Compute one policy-step reward and the memory required by the next step.

    Formation uses mean all-pair distance distortion divided by target diameter
    squared. Per-agent totals are not summed across the swarm: ``team_reward`` is
    their arithmetic mean. The legacy ``airborne`` argument indicates when a
    contact is unsafe: once airborne or in the formation phase.
    """
    positions = _vectors(positions_m, 3, "positions_m")
    targets = _vectors(targets_m, 3, "targets_m")
    velocities = _vectors(velocities_m_s, 3, "velocities_m_s")
    action_values = _vectors(actions, 4, "actions")
    count = len(positions)
    if count == 0 or any(len(values) != count for values in (targets, velocities, action_values)):
        raise ValueError("positions, targets, velocities, and actions need one row per agent")
    if len(contact_forces_n) != count or len(airborne) != count:
        raise ValueError("contact forces and airborne flags need one value per agent")
    contacts = tuple(float(value) for value in contact_forces_n)
    if any(not math.isfinite(value) or value < 0.0 for value in contacts):
        raise ValueError("contact forces must be finite and nonnegative")
    if any(type(value) is not bool for value in airborne):
        raise ValueError("airborne flags must be booleans")
    if memory is None:
        memory = RewardMemory()
    if memory.target_distances_m is not None and len(memory.target_distances_m) != count:
        raise ValueError("reward distance memory has the wrong agent count")
    if any(abs(value) > 1.0 for action in action_values for value in action):
        raise ValueError("reward actions must be bounded to [-1, 1]")
    if (
        memory.commanded_velocities_m_s is not None
        and len(memory.commanded_velocities_m_s) != count
    ):
        raise ValueError("reward command memory has the wrong agent count")

    formation_metrics = evaluate_formation(positions, targets)
    shared_formation = -formation_metrics.pairwise_normalized_mean_squared_error
    distances = tuple(
        math.dist(position, target) for position, target in zip(positions, targets, strict=True)
    )
    speeds = tuple(
        math.sqrt(math.fsum(value * value for value in velocity)) for velocity in velocities
    )
    minimum_separations = tuple(
        min(
            (
                math.dist(position, other)
                for other_index, other in enumerate(positions)
                if other_index != index
            ),
            default=math.inf,
        )
        for index, position in enumerate(positions)
    )
    contact_flags = tuple(
        is_airborne and force > config.contact_force_threshold_n
        for is_airborne, force in zip(airborne, contacts, strict=True)
    )
    commanded_velocities = tuple(
        direction_speed(action, config.max_speed_m_s) for action in action_values
    )

    raw_by_agent = []
    weighted_by_agent = []
    totals = []
    for index in range(count):
        progress = 0.0
        if memory.target_distances_m is not None:
            progress = _clip(
                (memory.target_distances_m[index] - distances[index]) / config.distance_scale_m,
                -1.0,
                1.0,
            )
        separation_intrusion = max(
            0.0,
            (config.minimum_separation_m - minimum_separations[index])
            / config.minimum_separation_m,
        )
        proximity = max(0.0, 1.0 - distances[index] / config.settling_radius_m)
        smoothness = 0.0
        if memory.commanded_velocities_m_s is not None:
            smoothness = -math.fsum(
                (current - previous) ** 2
                for current, previous in zip(
                    commanded_velocities[index],
                    memory.commanded_velocities_m_s[index],
                    strict=True,
                )
            ) / (3.0 * config.max_speed_m_s**2)
        normalized_command_speed = (
            math.sqrt(math.fsum(value * value for value in commanded_velocities[index]))
            / config.max_speed_m_s
        )
        raw = RewardComponents(
            formation=shared_formation,
            tracking=-((distances[index] / config.distance_scale_m) ** 2),
            progress=progress,
            separation=-(separation_intrusion**2),
            contact=-1.0 if contact_flags[index] else 0.0,
            settling=-proximity * (speeds[index] / config.max_speed_m_s) ** 2,
            smoothness=smoothness,
            effort=-(normalized_command_speed**2),
        )
        weighted = RewardComponents(
            **{
                name: (
                    getattr(raw, name)
                    * getattr(config, f"{name}_weight")
                    * (config.control_dt_seconds if name in TIME_INTEGRATED_COMPONENTS else 1.0)
                )
                for name in COMPONENT_NAMES
            }
        )
        total = weighted.total()
        if not math.isfinite(total):
            raise ValueError("reward is nonfinite")
        raw_by_agent.append(raw)
        weighted_by_agent.append(weighted)
        totals.append(total)

    step = RewardStep(
        raw_by_agent=tuple(raw_by_agent),
        weighted_by_agent=tuple(weighted_by_agent),
        total_by_agent=tuple(totals),
        team_reward=math.fsum(totals) / count,
        target_distances_m=distances,
        speeds_m_s=speeds,
        minimum_separations_m=minimum_separations,
        airborne_contact=contact_flags,
    )
    next_memory = RewardMemory(
        target_distances_m=distances,
        commanded_velocities_m_s=commanded_velocities,
    )
    return step, next_memory


def _vectors(values: Sequence[Sequence[float]], width: int, name: str):
    result = []
    for index, value in enumerate(values):
        if len(value) != width:
            raise ValueError(f"{name}[{index}] must contain {width} values")
        vector = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in vector):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(vector)
    return tuple(result)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
