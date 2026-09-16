"""Host-safe configuration for reset-mode training recovery."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

from align.learning.checkpoint_store import RECOVERY_MODE


@dataclass(frozen=True)
class RecoveryConfig:
    """Persistence policy for one logical training run."""

    schema_version: int = 1
    recovery_mode: str = RECOVERY_MODE
    checkpoint_interval_active_seconds: float = 300.0
    minimum_retained_checkpoints: int = 3
    write_failure_policy: str = "stop"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.recovery_mode != RECOVERY_MODE:
            raise ValueError(f"recovery_mode must be {RECOVERY_MODE!r}")
        if not math.isfinite(self.checkpoint_interval_active_seconds) or not (
            self.checkpoint_interval_active_seconds > 0
        ):
            raise ValueError("checkpoint interval must be finite and positive")
        if (
            type(self.minimum_retained_checkpoints) is not int
            or self.minimum_retained_checkpoints < 3
        ):
            raise ValueError("at least three checkpoints must be retained")
        if self.write_failure_policy != "stop":
            raise ValueError("the only supported write_failure_policy is 'stop'")

    @classmethod
    def from_dict(cls, values: dict) -> RecoveryConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "recovery configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
