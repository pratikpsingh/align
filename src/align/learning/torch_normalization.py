"""Frozen active-slot group normalization for centralized critic inputs."""

from __future__ import annotations

import math

import torch

from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.tasks.observation import ObservationConfig

NORMALIZATION_SCHEMA_VERSION = 1
GROUP_SLICES = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "target": slice(6, 9),
}


def disabled_normalization_state() -> dict:
    """Return the complete checkpoint state for the declared-scale baseline."""
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "enabled": False,
        "contract": "declared_feature_scaling",
        "frozen": True,
    }


def validate_normalization_state(state: object) -> dict:
    """Validate a normalization checkpoint without importing simulator code."""
    if not isinstance(state, dict):
        raise ValueError("normalization state must be a dictionary")
    common = {"schema_version", "enabled", "contract", "frozen"}
    if state.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("normalization state schema is incompatible")
    if type(state.get("enabled")) is not bool or type(state.get("frozen")) is not bool:
        raise ValueError("normalization enabled and frozen fields must be bool")
    if not state["frozen"]:
        raise ValueError("checkpoint normalization state must be frozen")
    if not state["enabled"]:
        if set(state) != common or state["contract"] != "declared_feature_scaling":
            raise ValueError("disabled normalization state fields are invalid")
        return state

    expected = common | {
        "epsilon",
        "clip",
        "warmup_steps",
        "environment_samples",
        "active_agent_samples",
        "groups",
    }
    if set(state) != expected:
        raise ValueError("enabled normalization state fields are incomplete or unexpected")
    if state["contract"] != "frozen_active_critic_group_standardization":
        raise ValueError("enabled normalization contract is invalid")
    for name in ("epsilon", "clip"):
        value = state[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"normalization {name} must be finite and positive")
    for name in ("warmup_steps", "environment_samples", "active_agent_samples"):
        if type(state[name]) is not int or state[name] < 1:
            raise ValueError(f"normalization {name} must be a positive integer")
    groups = state["groups"]
    if not isinstance(groups, dict) or set(groups) != set(GROUP_SLICES):
        raise ValueError("normalization group names are invalid")
    for name, values in groups.items():
        if not isinstance(values, dict) or set(values) != {"count", "mean", "variance"}:
            raise ValueError(f"normalization group state is invalid: {name}")
        if type(values["count"]) is not int or values["count"] < 1:
            raise ValueError(f"normalization group count is invalid: {name}")
        if values["count"] != state["active_agent_samples"] * 3:
            raise ValueError(f"normalization group count does not match active samples: {name}")
        if any(
            type(values[field]) not in (int, float) or not math.isfinite(values[field])
            for field in ("mean", "variance")
        ):
            raise ValueError(f"normalization group moments are not finite: {name}")
        if values["variance"] < 0:
            raise ValueError(f"normalization group variance is negative: {name}")
    return state


def normalization_matches_config(state: dict, config: CriticNormalizationConfig) -> bool:
    """Return whether a frozen state implements the requested experiment contract."""
    validate_normalization_state(state)
    if state["enabled"] != config.enabled:
        return False
    if not config.enabled:
        return True
    return (
        state["warmup_steps"] == config.warmup_steps
        and state["epsilon"] == config.epsilon
        and state["clip"] == config.clip
    )


class CriticGroupAccumulator:
    """Accumulate scalar moments from active critic position/velocity/target slots."""

    def __init__(self, observation: ObservationConfig, expected_agents: int, device) -> None:
        if not 1 <= expected_agents <= observation.critic_capacity:
            raise ValueError("expected_agents must fit the critic capacity")
        self.observation = observation
        self.expected_agents = expected_agents
        self.device = torch.device(device)
        self.counts = {
            name: torch.zeros((), dtype=torch.int64, device=self.device) for name in GROUP_SLICES
        }
        self.sums = {
            name: torch.zeros((), dtype=torch.float64, device=self.device) for name in GROUP_SLICES
        }
        self.squares = {
            name: torch.zeros((), dtype=torch.float64, device=self.device) for name in GROUP_SLICES
        }
        self.environment_samples = 0
        self.active_agent_samples = 0

    def _parts(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 2 or states.shape[-1] != self.observation.critic_dimension:
            raise ValueError("critic warmup state shape differs from the observation contract")
        if states.device != self.device:
            raise ValueError("critic warmup states are on the wrong device")
        if not bool(torch.isfinite(states).all()):
            raise ValueError("critic warmup states contain nonfinite values")
        capacity = self.observation.critic_capacity
        features = states[:, : capacity * 9].reshape(-1, capacity, 9)
        masks = states[:, capacity * 9 :]
        if not bool(((masks == 0) | (masks == 1)).all()):
            raise ValueError("critic masks must be binary")
        if not bool((masks.sum(dim=-1) == self.expected_agents).all()):
            raise ValueError("critic active-slot count differs from the swarm size")
        inactive = masks == 0
        if not bool((features[inactive] == 0).all()):
            raise ValueError("inactive critic slots must contain zero padding")
        if not bool((features.abs() <= 1.0).all()):
            raise ValueError("declared-scale critic features must remain in [-1, 1]")
        return features, masks.bool()

    def update(self, states: torch.Tensor) -> dict[str, dict[str, float | int]]:
        """Add one vectorized state batch and return its auditable reductions."""
        features, active = self._parts(states)
        reductions = {}
        active_count = int(active.sum().item())
        self.environment_samples += states.shape[0]
        self.active_agent_samples += active_count
        for name, group_slice in GROUP_SLICES.items():
            values = features[..., group_slice][active].to(torch.float64).reshape(-1)
            count = values.numel()
            total = values.sum()
            squares = values.square().sum()
            self.counts[name] += count
            self.sums[name] += total
            self.squares[name] += squares
            reductions[name] = {
                "count": count,
                "sum": float(total.item()),
                "sum_squares": float(squares.item()),
            }
        return reductions

    def freeze(self, config: CriticNormalizationConfig, warmup_steps: int) -> dict:
        """Convert device accumulators to a validated immutable checkpoint payload."""
        if not config.enabled or warmup_steps != config.warmup_steps:
            raise ValueError("normalization warmup does not match its configuration")
        groups = {}
        for name in GROUP_SLICES:
            count = int(self.counts[name].item())
            mean = float((self.sums[name] / count).item())
            variance = float((self.squares[name] / count).item()) - mean * mean
            groups[name] = {
                "count": count,
                "mean": mean,
                "variance": max(0.0, variance),
            }
        state = {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "enabled": True,
            "contract": "frozen_active_critic_group_standardization",
            "frozen": True,
            "epsilon": config.epsilon,
            "clip": config.clip,
            "warmup_steps": warmup_steps,
            "environment_samples": self.environment_samples,
            "active_agent_samples": self.active_agent_samples,
            "groups": groups,
        }
        return validate_normalization_state(state)


class FrozenCriticGroupNormalizer:
    """Apply checkpointed moments to active critic features only."""

    def __init__(self, state: dict, observation: ObservationConfig) -> None:
        self.state = validate_normalization_state(state)
        self.observation = observation

    def apply(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.observation.critic_dimension:
            raise ValueError("critic state shape differs from the normalization contract")
        if not bool(torch.isfinite(states).all()):
            raise ValueError("critic states contain nonfinite values")
        result = states.clone()
        if not self.state["enabled"]:
            return result
        capacity = self.observation.critic_capacity
        features = result[..., : capacity * 9].reshape(*result.shape[:-1], capacity, 9)
        masks = result[..., capacity * 9 :]
        if not bool(((masks == 0) | (masks == 1)).all()):
            raise ValueError("critic masks must remain binary")
        active = masks.bool()
        for name, group_slice in GROUP_SLICES.items():
            moments = self.state["groups"][name]
            denominator = math.sqrt(max(moments["variance"], self.state["epsilon"] ** 2))
            group = features[..., group_slice]
            transformed = ((group - moments["mean"]) / denominator).clamp(
                -self.state["clip"], self.state["clip"]
            )
            group.copy_(torch.where(active.unsqueeze(-1), transformed, torch.zeros_like(group)))
        if not bool(torch.isfinite(result).all()):
            raise ValueError("normalized critic states contain nonfinite values")
        if not torch.equal(result[..., capacity * 9 :], states[..., capacity * 9 :]):
            raise ValueError("critic normalization changed mask features")
        return result
