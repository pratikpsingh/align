"""Host-safe contract for frozen critic-only group normalization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class CriticNormalizationConfig:
    """Warmup and transform settings for centralized critic inputs only."""

    schema_version: int = 1
    enabled: bool = False
    warmup_steps: int = 0
    epsilon: float = 1e-6
    clip: float = 5.0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be bool")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.enabled and self.warmup_steps < 1:
            raise ValueError("enabled critic normalization requires warmup_steps")
        if not self.enabled and self.warmup_steps != 0:
            raise ValueError("disabled critic normalization requires zero warmup_steps")
        if not math.isfinite(self.epsilon) or not 0.0 < self.epsilon <= 0.1:
            raise ValueError("epsilon must be finite and in (0, 0.1]")
        if not math.isfinite(self.clip) or not 1.0 <= self.clip <= 20.0:
            raise ValueError("clip must be finite and in [1, 20]")

    @classmethod
    def from_dict(cls, values: dict) -> CriticNormalizationConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "critic normalization configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
