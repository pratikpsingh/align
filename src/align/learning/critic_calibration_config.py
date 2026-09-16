"""Host-safe configuration for matched-rollout critic step calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class CriticCalibrationConfig:
    """Candidate critic steps and diagnostic acceptance for one frozen batch."""

    schema_version: int
    policy_seeds: tuple[int, ...]
    candidate_critic_learning_rates: tuple[float, ...]
    max_value_clip_fraction: float
    pre_prediction_tolerance: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if len(self.policy_seeds) < 2 or len(set(self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must contain at least two unique seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.policy_seeds):
            raise ValueError("policy seeds must be nonnegative integers")
        rates = self.candidate_critic_learning_rates
        if len(rates) < 2 or len(set(rates)) != len(rates):
            raise ValueError("candidate rates must contain at least two unique values")
        if any(
            not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
            for rate in rates
        ):
            raise ValueError("candidate critic learning rates must be finite and positive")
        if any(left <= right for left, right in zip(rates[:-1], rates[1:], strict=True)):
            raise ValueError("candidate critic learning rates must be strictly decreasing")
        if (
            not math.isfinite(self.max_value_clip_fraction)
            or not 0 <= self.max_value_clip_fraction <= 1
        ):
            raise ValueError("max_value_clip_fraction must be finite and in [0, 1]")
        if not math.isfinite(self.pre_prediction_tolerance) or self.pre_prediction_tolerance <= 0:
            raise ValueError("pre_prediction_tolerance must be finite and positive")

    @classmethod
    def from_dict(cls, values: dict) -> CriticCalibrationConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "critic calibration configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not isinstance(values["policy_seeds"], list) or not isinstance(
            values["candidate_critic_learning_rates"], list
        ):
            raise ValueError("policy seeds and candidate rates must be JSON lists")
        return cls(
            **{
                **values,
                "policy_seeds": tuple(values["policy_seeds"]),
                "candidate_critic_learning_rates": tuple(values["candidate_critic_learning_rates"]),
            }
        )

    def to_dict(self) -> dict:
        return {
            field.name: (
                list(getattr(self, field.name))
                if field.name in {"policy_seeds", "candidate_critic_learning_rates"}
                else getattr(self, field.name)
            )
            for field in fields(self)
        }
