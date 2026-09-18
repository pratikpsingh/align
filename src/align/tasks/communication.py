"""Counterfactual communication costs for a saved local-observation topology.

These numbers describe declared packet schemes, not measured radio traffic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from align.tasks.observation import ActorObservation


@dataclass(frozen=True)
class CommunicationConfig:
    schema_version: int = 1
    position_components: int = 3
    velocity_components: int = 3
    bytes_per_component: int = 4
    packet_overhead_bytes: int = 16
    budgets: tuple[int, ...] = (1, 2, 3, 7)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for name in (
            "position_components",
            "velocity_components",
            "bytes_per_component",
            "packet_overhead_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.position_components + self.velocity_components:
            raise ValueError("at least one state component is required")
        if not self.bytes_per_component:
            raise ValueError("bytes_per_component must be positive")
        if not self.budgets or any(type(value) is not int or value < 1 for value in self.budgets):
            raise ValueError("budgets must contain positive integers")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be unique and ascending")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CommunicationConfig:
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ValueError("communication configuration keys mismatch")
        return cls(**{**value, "budgets": tuple(value["budgets"])})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def packet_bytes(self) -> int:
        return (
            self.position_components + self.velocity_components
        ) * self.bytes_per_component + self.packet_overhead_bytes


def account_snapshot(
    actors: tuple[ActorObservation, ...], config: CommunicationConfig
) -> dict[str, object]:
    """Count directed unicast edges and unique broadcast senders for one snapshot.

    A sender broadcasts once per snapshot if at least one selected receiver uses it.
    This is an ideal lower-cost broadcast scheme; it assumes perfect discovery and
    ignores retries, contention, acknowledgements, and routing.
    """
    identities = [actor.agent_id for actor in actors]
    if len(set(identities)) != len(identities):
        raise ValueError("actor identities must be unique")
    edges: list[tuple[int, int]] = []
    for actor in actors:
        if len(actor.neighbor_mask) != len(actor.neighbor_ids):
            raise ValueError("neighbor mask and identifiers disagree")
        for valid, sender in zip(actor.neighbor_mask, actor.neighbor_ids, strict=True):
            if valid:
                if sender is None or sender == actor.agent_id or sender not in identities:
                    raise ValueError("invalid selected neighbor identity")
                edges.append((sender, actor.agent_id))
            elif sender is not None:
                raise ValueError("padded slot has an identity")
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate directed edge")
    senders = {sender for sender, _ in edges}
    return {
        "selected_directed_edges": len(edges),
        "unique_broadcast_senders": len(senders),
        "unicast_proxy_bytes": len(edges) * config.packet_bytes,
        "ideal_broadcast_proxy_bytes": len(senders) * config.packet_bytes,
        "directed_edges": tuple(sorted(edges)),
    }
