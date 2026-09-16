"""Explicit configuration for the recurrent actor and centralized critic."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrentPolicyConfig:
    """Dimensions and distribution limits for the first ALiGn policy."""

    schema_version: int = 1
    actor_observation_dim: int = 55
    critic_state_dim: int = 80
    action_dim: int = 4
    encoder_hidden_size: int = 256
    recurrent_hidden_size: int = 256
    recurrent_layers: int = 1
    log_std_initial: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    squash_epsilon: float = 1e-6
    action_low: tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0)
    action_high: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for name in (
            "actor_observation_dim",
            "critic_state_dim",
            "action_dim",
            "encoder_hidden_size",
            "recurrent_hidden_size",
            "recurrent_layers",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("log_std_initial", "log_std_min", "log_std_max", "squash_epsilon"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.log_std_min <= self.log_std_initial <= self.log_std_max:
            raise ValueError("log_std_initial must be within the configured limits")
        if not 0.0 < self.squash_epsilon < 0.01:
            raise ValueError("squash_epsilon must be in (0, 0.01)")
        if len(self.action_low) != self.action_dim or len(self.action_high) != self.action_dim:
            raise ValueError("action bounds must match action_dim")
        for low, high in zip(self.action_low, self.action_high, strict=True):
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("each action lower bound must be finite and below its upper bound")

    @classmethod
    def from_dict(cls, values: dict) -> RecurrentPolicyConfig:
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "policy configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        converted = dict(values)
        converted["action_low"] = tuple(converted["action_low"])
        converted["action_high"] = tuple(converted["action_high"])
        return cls(**converted)

    def to_dict(self) -> dict:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["action_low"] = list(self.action_low)
        result["action_high"] = list(self.action_high)
        return result

    def validate_task_dimensions(
        self, *, actor_observation_dim: int, critic_state_dim: int, action_dim: int
    ) -> None:
        actual = {
            "actor_observation_dim": actor_observation_dim,
            "critic_state_dim": critic_state_dim,
            "action_dim": action_dim,
        }
        mismatches = {
            name: (getattr(self, name), value)
            for name, value in actual.items()
            if getattr(self, name) != value
        }
        if mismatches:
            raise ValueError(f"policy dimensions do not match task contract: {mismatches}")
