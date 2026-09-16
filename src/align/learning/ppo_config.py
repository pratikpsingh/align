"""Host-safe configuration for recurrent MAPPO optimizer updates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class RecurrentPPOConfig:
    """Explicit hyperparameters for one shared-actor/centralized-critic update."""

    schema_version: int = 1
    actor_learning_rate: float = 0.0003
    critic_learning_rate: float = 0.001
    adam_epsilon: float = 1e-5
    policy_clip_ratio: float = 0.2
    value_clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    update_epochs: int = 2
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if type(self.update_epochs) is not int or self.update_epochs <= 0:
            raise ValueError("update_epochs must be a positive integer")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("normalize_advantages must be bool")
        finite_positive = (
            "actor_learning_rate",
            "critic_learning_rate",
            "adam_epsilon",
            "policy_clip_ratio",
            "value_clip_range",
            "value_loss_coefficient",
            "max_gradient_norm",
            "advantage_epsilon",
        )
        for name in finite_positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be finite and nonnegative")
        if self.policy_clip_ratio >= 1:
            raise ValueError("policy_clip_ratio must be below 1")

    @classmethod
    def from_dict(cls, values: dict) -> RecurrentPPOConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "PPO configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
