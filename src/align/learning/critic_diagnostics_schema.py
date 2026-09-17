"""Host-safe semantic layout for the centralized critic's fixed state vector."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from align.tasks.observation import ObservationConfig


@dataclass(frozen=True)
class CriticFeature:
    """One indexed scalar in the flattened centralized critic state."""

    index: int
    name: str
    slot: int
    group: str
    axis: str | None
    is_mask: bool

    def to_dict(self) -> dict:
        return asdict(self)


def critic_feature_layout(config: ObservationConfig) -> tuple[CriticFeature, ...]:
    """Describe position, velocity, target, and mask scalars in flatten order."""
    groups = (
        ("position", "x"),
        ("position", "y"),
        ("position", "z"),
        ("velocity", "x"),
        ("velocity", "y"),
        ("velocity", "z"),
        ("target", "x"),
        ("target", "y"),
        ("target", "z"),
    )
    features = []
    for slot in range(config.critic_capacity):
        for offset, (group, axis) in enumerate(groups):
            index = slot * config.critic_agent_feature_count + offset
            features.append(
                CriticFeature(index, f"agent_{slot}/{group}_{axis}", slot, group, axis, False)
            )
    mask_start = config.critic_capacity * config.critic_agent_feature_count
    for slot in range(config.critic_capacity):
        features.append(
            CriticFeature(mask_start + slot, f"agent_{slot}/mask", slot, "mask", None, True)
        )
    result = tuple(features)
    if tuple(item.index for item in result) != tuple(range(config.critic_dimension)):
        raise RuntimeError("critic feature layout does not cover the configured state exactly")
    return result
