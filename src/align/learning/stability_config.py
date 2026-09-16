"""Host-safe configuration for bounded multi-seed stability calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class StabilityConfig:
    """Calibration budget and diagnostic alert thresholds, not training claims."""

    schema_version: int
    policy_seeds: tuple[int, ...]
    updates_per_seed: int
    evaluation_steps: int
    max_post_update_approximate_kl: float
    max_post_update_policy_clip_fraction: float
    max_post_update_value_clip_fraction: float
    max_actor_gradient_norm_before_clip: float
    max_critic_gradient_norm_before_clip: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if len(self.policy_seeds) < 2 or len(set(self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must contain at least two unique seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.policy_seeds):
            raise ValueError("policy seeds must be nonnegative integers")
        if type(self.updates_per_seed) is not int or self.updates_per_seed < 2:
            raise ValueError("updates_per_seed must be at least two")
        if type(self.evaluation_steps) is not int or self.evaluation_steps <= 0:
            raise ValueError("evaluation_steps must be a positive integer")
        for name in (
            "max_post_update_approximate_kl",
            "max_post_update_policy_clip_fraction",
            "max_post_update_value_clip_fraction",
            "max_actor_gradient_norm_before_clip",
            "max_critic_gradient_norm_before_clip",
        ):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "max_post_update_policy_clip_fraction",
            "max_post_update_value_clip_fraction",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} cannot exceed one")

    @classmethod
    def from_dict(cls, values: dict) -> StabilityConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "stability configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not isinstance(values["policy_seeds"], list):
            raise ValueError("policy_seeds must be a JSON list")
        return cls(**{**values, "policy_seeds": tuple(values["policy_seeds"])})

    def to_dict(self) -> dict:
        return {
            field.name: list(self.policy_seeds)
            if field.name == "policy_seeds"
            else getattr(self, field.name)
            for field in fields(self)
        }
