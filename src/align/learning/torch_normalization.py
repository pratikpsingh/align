"""Frozen active-slot group normalization for centralized critic inputs."""

from __future__ import annotations

import math

import torch

from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.learning.critic_normalization_state import (
    normalization_standard_deviation,
    validate_normalization_state,
)
from align.tasks.observation import ObservationConfig

GROUP_SLICES = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "target": slice(6, 9),
}


def disabled_normalization_state() -> dict:
    """Return the complete checkpoint state for the declared-scale baseline."""
    return {
        "schema_version": 1,
        "enabled": False,
        "contract": "declared_feature_scaling",
        "frozen": True,
    }


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
        and float(state.get("minimum_standard_deviation", 0.0)) == config.minimum_standard_deviation
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
            "schema_version": config.schema_version,
            "enabled": True,
            "contract": (
                "frozen_active_critic_group_standardization_with_floor"
                if config.schema_version == 2
                else "frozen_active_critic_group_standardization"
            ),
            "frozen": True,
            "epsilon": config.epsilon,
            "clip": config.clip,
            "warmup_steps": warmup_steps,
            "environment_samples": self.environment_samples,
            "active_agent_samples": self.active_agent_samples,
            "groups": groups,
        }
        if config.schema_version == 2:
            state["minimum_standard_deviation"] = config.minimum_standard_deviation
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
            _, denominator = normalization_standard_deviation(self.state, name)
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


def summarize_critic_rollout_distribution(
    raw_states: torch.Tensor,
    normalized_states: torch.Tensor,
    normalization: dict,
    observation: ObservationConfig,
    expected_agents: int,
) -> list[dict]:
    """Measure active-group drift and pre-clamp exceedances on one PPO rollout."""
    if raw_states.ndim != 3 or raw_states.shape != normalized_states.shape:
        raise ValueError("raw and normalized critic rollouts must have the same [T, E, D] shape")
    if raw_states.shape[-1] != observation.critic_dimension:
        raise ValueError("critic rollout feature dimension differs from observation config")
    normalizer = FrozenCriticGroupNormalizer(normalization, observation)
    expected = normalizer.apply(raw_states)
    if not torch.equal(expected, normalized_states):
        raise ValueError("stored critic rollout differs from the frozen normalization contract")
    flat = raw_states.reshape(-1, observation.critic_dimension)
    validator = CriticGroupAccumulator(observation, expected_agents, flat.device)
    features, active = validator._parts(flat)
    rows = []
    for name, group_slice in GROUP_SLICES.items():
        values = features[..., group_slice][active].to(torch.float64).reshape(-1)
        if values.numel() != raw_states.shape[0] * raw_states.shape[1] * expected_agents * 3:
            raise ValueError("critic rollout active scalar count is inconsistent")
        mean = float(values.mean().item())
        standard_deviation = float(values.std(unbiased=False).item())
        if normalization["enabled"]:
            moments = normalization["groups"][name]
            warmup_std, denominator = normalization_standard_deviation(normalization, name)
            before_clip = (values - moments["mean"]) / denominator
            clipped = before_clip.abs() > normalization["clip"]
            warmup_mean = moments["mean"]
            mean_shift = (mean - warmup_mean) / warmup_std
            std_ratio = standard_deviation / warmup_std
            effective_mean_shift = (mean - warmup_mean) / denominator
            effective_std_ratio = standard_deviation / denominator
        else:
            before_clip = values
            clipped = torch.zeros_like(values, dtype=torch.bool)
            warmup_mean = 0.0
            warmup_std = 1.0
            denominator = 1.0
            mean_shift = 0.0
            std_ratio = 1.0
            effective_mean_shift = 0.0
            effective_std_ratio = 1.0
        transformed = (
            before_clip.clamp(-normalization["clip"], normalization["clip"])
            if normalization["enabled"]
            else before_clip
        )
        clipped_count = int(clipped.sum().item())
        row = {
            "group": name,
            "count": values.numel(),
            "raw_mean": mean,
            "raw_standard_deviation": standard_deviation,
            "raw_minimum": float(values.min().item()),
            "raw_maximum": float(values.max().item()),
            "warmup_mean": warmup_mean,
            "warmup_standard_deviation": warmup_std,
            "normalization_standard_deviation": denominator,
            "mean_shift_warmup_standard_deviations": mean_shift,
            "raw_standard_deviation_ratio": std_ratio,
            "mean_shift_normalization_standard_deviations": effective_mean_shift,
            "raw_standard_deviation_normalization_ratio": effective_std_ratio,
            "normalized_mean": float(transformed.mean().item()),
            "normalized_standard_deviation": float(transformed.std(unbiased=False).item()),
            "normalized_minimum": float(transformed.min().item()),
            "normalized_maximum": float(transformed.max().item()),
            "clipped_count": clipped_count,
            "clipped_fraction": clipped_count / values.numel(),
        }
        if not all(math.isfinite(value) for key, value in row.items() if key != "group"):
            raise ValueError("critic rollout distribution contains nonfinite statistics")
        rows.append(row)
    return rows
