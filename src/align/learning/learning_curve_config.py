"""Host-safe configuration for bounded checkpoint learning curves."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from align.learning.stability_config import StabilityConfig


@dataclass(frozen=True)
class LearningCurveConfig:
    """Training budget, evaluation milestones, and diagnostic guidance."""

    schema_version: int
    policy_seeds: tuple[int, ...]
    updates_per_seed: int
    evaluation_milestones: tuple[int, ...]
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
        if type(self.updates_per_seed) is not int or self.updates_per_seed < 4:
            raise ValueError("updates_per_seed must be at least four")
        milestones = self.evaluation_milestones
        if (
            len(milestones) < 3
            or any(type(value) is not int for value in milestones)
            or tuple(sorted(set(milestones))) != milestones
            or milestones[0] != 0
            or milestones[-1] != self.updates_per_seed
        ):
            raise ValueError(
                "evaluation_milestones must be sorted unique integers "
                "including zero and final update"
            )
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
    def from_dict(cls, values: dict) -> LearningCurveConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "learning curve configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not isinstance(values["policy_seeds"], list) or not isinstance(
            values["evaluation_milestones"], list
        ):
            raise ValueError("policy_seeds and evaluation_milestones must be JSON lists")
        return cls(
            **{
                **values,
                "policy_seeds": tuple(values["policy_seeds"]),
                "evaluation_milestones": tuple(values["evaluation_milestones"]),
            }
        )

    def to_stability_config(self) -> StabilityConfig:
        """Build the simulator-facing training and diagnostic configuration."""
        return StabilityConfig(
            schema_version=1,
            policy_seeds=self.policy_seeds,
            updates_per_seed=self.updates_per_seed,
            evaluation_steps=self.evaluation_steps,
            max_post_update_approximate_kl=self.max_post_update_approximate_kl,
            max_post_update_policy_clip_fraction=self.max_post_update_policy_clip_fraction,
            max_post_update_value_clip_fraction=self.max_post_update_value_clip_fraction,
            max_actor_gradient_norm_before_clip=self.max_actor_gradient_norm_before_clip,
            max_critic_gradient_norm_before_clip=self.max_critic_gradient_norm_before_clip,
        )

    def to_dict(self) -> dict:
        return {
            field.name: (
                list(getattr(self, field.name))
                if field.name in {"policy_seeds", "evaluation_milestones"}
                else getattr(self, field.name)
            )
            for field in fields(self)
        }
