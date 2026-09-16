"""Bounded local actor and separate centralized critic observations.

The actor uses a shared gravity-aligned world-axis estimate, but never receives
absolute position. Neighbor states are relative, radius-filtered, budgeted, and
masked. This module has no simulator, NumPy, or PyTorch dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields

Point3 = tuple[float, float, float]
Neighbor6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ObservationConfig:
    """Versioned dimensions, physical locality, and normalization scales."""

    schema_version: int = 1
    max_neighbors: int = 7
    neighbor_radius_m: float = 1.5
    own_velocity_scale_m_s: float = 1.0
    relative_velocity_scale_m_s: float = 2.0
    target_distance_scale_m: float = 3.0
    critic_capacity: int = 8
    critic_position_scale_m: float = 5.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if type(self.max_neighbors) is not int or self.max_neighbors < 1:
            raise ValueError("max_neighbors must be a positive integer")
        if type(self.critic_capacity) is not int or self.critic_capacity < 1:
            raise ValueError("critic_capacity must be a positive integer")
        scales = (
            self.neighbor_radius_m,
            self.own_velocity_scale_m_s,
            self.relative_velocity_scale_m_s,
            self.target_distance_scale_m,
            self.critic_position_scale_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("observation scales must be finite and positive")

    @property
    def self_feature_count(self) -> int:
        return 6

    @property
    def neighbor_feature_count(self) -> int:
        return 6

    @property
    def actor_dimension(self) -> int:
        return self.self_feature_count + self.max_neighbors * (self.neighbor_feature_count + 1)

    @property
    def critic_agent_feature_count(self) -> int:
        return 9

    @property
    def critic_dimension(self) -> int:
        return self.critic_capacity * (self.critic_agent_feature_count + 1)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ObservationConfig:
        expected = {field.name for field in fields(cls)}
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"observation configuration keys mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActorObservation:
    """One fixed-capacity decentralized observation and audit metadata."""

    agent_id: int
    self_features: tuple[float, float, float, float, float, float]
    neighbor_features: tuple[Neighbor6, ...]
    neighbor_mask: tuple[bool, ...]
    neighbor_ids: tuple[int | None, ...]
    neighbor_distances_m: tuple[float | None, ...]
    candidate_count: int
    saturation_count: int

    def flat(self) -> tuple[float, ...]:
        """Return policy values followed by masks; audit IDs/distances are excluded."""
        return (
            *self.self_features,
            *(value for slot in self.neighbor_features for value in slot),
            *(1.0 if valid else 0.0 for valid in self.neighbor_mask),
        )


@dataclass(frozen=True)
class CriticObservation:
    """Padded global training state, unavailable to the decentralized actor."""

    agent_features: tuple[tuple[float, ...], ...]
    agent_mask: tuple[bool, ...]
    agent_ids: tuple[int | None, ...]
    saturation_count: int

    def flat(self) -> tuple[float, ...]:
        return (
            *(value for row in self.agent_features for value in row),
            *(1.0 if valid else 0.0 for valid in self.agent_mask),
        )


@dataclass(frozen=True)
class ObservationBatch:
    """Actor observations sorted by identity plus one distinct critic state."""

    actors: tuple[ActorObservation, ...]
    critic: CriticObservation


def build_observations(
    agent_ids: Sequence[int],
    positions_m: Sequence[Sequence[float]],
    velocities_m_s: Sequence[Sequence[float]],
    targets_m: Sequence[Sequence[float]],
    config: ObservationConfig,
) -> ObservationBatch:
    """Build actor and critic observations for a training batch."""
    return ObservationBatch(
        actors=build_actor_observations(agent_ids, positions_m, velocities_m_s, targets_m, config),
        critic=build_critic_observation(agent_ids, positions_m, velocities_m_s, targets_m, config),
    )


def build_actor_observations(
    agent_ids: Sequence[int],
    positions_m: Sequence[Sequence[float]],
    velocities_m_s: Sequence[Sequence[float]],
    targets_m: Sequence[Sequence[float]],
    config: ObservationConfig,
) -> tuple[ActorObservation, ...]:
    """Build actor inputs for any swarm size using radius before budget."""
    ordered = _validated_ordered(agent_ids, positions_m, velocities_m_s, targets_m)
    return tuple(_actor_observation(index, ordered, config) for index in range(len(ordered)))


def build_critic_observation(
    agent_ids: Sequence[int],
    positions_m: Sequence[Sequence[float]],
    velocities_m_s: Sequence[Sequence[float]],
    targets_m: Sequence[Sequence[float]],
    config: ObservationConfig,
) -> CriticObservation:
    """Build the fixed-capacity global state used only during training."""
    ordered = _validated_ordered(agent_ids, positions_m, velocities_m_s, targets_m)
    if len(ordered) > config.critic_capacity:
        raise ValueError("agent count exceeds the configured training critic capacity")
    return _critic_observation(ordered, config)


def _validated_ordered(agent_ids, positions_m, velocities_m_s, targets_m):
    identifiers = tuple(agent_ids)
    if not identifiers or any(type(value) is not int or value < 0 for value in identifiers):
        raise ValueError("agent_ids must contain nonnegative integers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("agent_ids must be unique")
    positions = _vectors(positions_m, "positions_m")
    velocities = _vectors(velocities_m_s, "velocities_m_s")
    targets = _vectors(targets_m, "targets_m")
    count = len(identifiers)
    if any(len(values) != count for values in (positions, velocities, targets)):
        raise ValueError("agent state arrays must match agent_ids")
    return tuple(
        sorted(
            zip(identifiers, positions, velocities, targets, strict=True), key=lambda item: item[0]
        )
    )


def _actor_observation(index, ordered, config: ObservationConfig) -> ActorObservation:
    agent_id, position, velocity, target = ordered[index]
    saturation_count = 0
    self_values = []
    for value in velocity:
        normalized, saturated = _normalized(value, config.own_velocity_scale_m_s)
        self_values.append(normalized)
        saturation_count += saturated
    for value in _subtract(target, position):
        normalized, saturated = _normalized(value, config.target_distance_scale_m)
        self_values.append(normalized)
        saturation_count += saturated

    candidates = []
    for other_id, other_position, other_velocity, _ in ordered:
        if other_id == agent_id:
            continue
        relative_position = _subtract(other_position, position)
        distance = math.sqrt(math.fsum(value * value for value in relative_position))
        if distance <= config.neighbor_radius_m:
            candidates.append(
                (
                    distance,
                    other_id,
                    relative_position,
                    _subtract(other_velocity, velocity),
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[: config.max_neighbors]

    neighbor_features: list[Neighbor6] = []
    neighbor_mask = []
    neighbor_ids: list[int | None] = []
    neighbor_distances: list[float | None] = []
    for distance, other_id, relative_position, relative_velocity in selected:
        features = []
        for value in relative_position:
            normalized, saturated = _normalized(value, config.neighbor_radius_m)
            features.append(normalized)
            saturation_count += saturated
        for value in relative_velocity:
            normalized, saturated = _normalized(value, config.relative_velocity_scale_m_s)
            features.append(normalized)
            saturation_count += saturated
        neighbor_features.append(tuple(features))
        neighbor_mask.append(True)
        neighbor_ids.append(other_id)
        neighbor_distances.append(distance)
    while len(neighbor_features) < config.max_neighbors:
        neighbor_features.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        neighbor_mask.append(False)
        neighbor_ids.append(None)
        neighbor_distances.append(None)

    return ActorObservation(
        agent_id=agent_id,
        self_features=tuple(self_values),
        neighbor_features=tuple(neighbor_features),
        neighbor_mask=tuple(neighbor_mask),
        neighbor_ids=tuple(neighbor_ids),
        neighbor_distances_m=tuple(neighbor_distances),
        candidate_count=len(candidates),
        saturation_count=saturation_count,
    )


def _critic_observation(ordered, config: ObservationConfig) -> CriticObservation:
    rows = []
    mask = []
    identifiers: list[int | None] = []
    saturation_count = 0
    for agent_id, position, velocity, target in ordered:
        features = []
        for values, scale in (
            (position, config.critic_position_scale_m),
            (velocity, config.own_velocity_scale_m_s),
            (target, config.critic_position_scale_m),
        ):
            for value in values:
                normalized, saturated = _normalized(value, scale)
                features.append(normalized)
                saturation_count += saturated
        rows.append(tuple(features))
        mask.append(True)
        identifiers.append(agent_id)
    while len(rows) < config.critic_capacity:
        rows.append((0.0,) * config.critic_agent_feature_count)
        mask.append(False)
        identifiers.append(None)
    return CriticObservation(
        agent_features=tuple(rows),
        agent_mask=tuple(mask),
        agent_ids=tuple(identifiers),
        saturation_count=saturation_count,
    )


def _vectors(values: Sequence[Sequence[float]], name: str) -> tuple[Point3, ...]:
    result = []
    for index, value in enumerate(values):
        if len(value) != 3:
            raise ValueError(f"{name}[{index}] must contain three values")
        vector = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in vector):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(vector)
    return tuple(result)


def _subtract(left: Point3, right: Point3) -> Point3:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _normalized(value: float, scale: float) -> tuple[float, int]:
    scaled = value / scale
    return min(1.0, max(-1.0, scaled)), int(abs(scaled) > 1.0)
