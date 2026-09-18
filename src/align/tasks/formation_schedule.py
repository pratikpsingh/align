"""Host-safe deterministic formation-template scheduling."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields

from align.formations.geometry import SUPPORTED_KINDS


@dataclass(frozen=True)
class FormationScheduleConfig:
    """Balance declared templates across environments at each rollout boundary."""

    schema_version: int = 1
    kinds: tuple[str, ...] = ("plane",)
    seed: int = 23
    assignment: str = "seeded_balanced_rotation"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("formation schedule schema_version must be 1")
        if not self.kinds or len(set(self.kinds)) != len(self.kinds):
            raise ValueError("formation kinds must be nonempty and unique")
        if any(kind not in SUPPORTED_KINDS for kind in self.kinds):
            raise ValueError(f"formation kinds must come from {SUPPORTED_KINDS}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("formation schedule seed must be a nonnegative integer")
        if self.assignment != "seeded_balanced_rotation":
            raise ValueError("unsupported formation assignment rule")

    @classmethod
    def from_dict(cls, values: dict) -> FormationScheduleConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "formation schedule configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not isinstance(values["kinds"], list):
            raise ValueError("formation schedule kinds must be a JSON list")
        return cls(**{**values, "kinds": tuple(values["kinds"])})

    def to_dict(self) -> dict:
        values = asdict(self)
        values["kinds"] = list(self.kinds)
        return values

    @property
    def ordered_kinds(self) -> tuple[str, ...]:
        """Return a seed-specific order without relying on Python RNG versions."""
        return tuple(
            sorted(
                self.kinds,
                key=lambda kind: hashlib.sha256(f"{self.seed}:{kind}".encode()).digest(),
            )
        )

    def assignments(self, num_envs: int, batch_index: int) -> tuple[str, ...]:
        """Assign every kind equally, rotating the environment mapping by batch."""
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(batch_index) is not int or batch_index < 0:
            raise ValueError("batch_index must be a nonnegative integer")
        if num_envs % len(self.kinds):
            raise ValueError("num_envs must be divisible by the number of formation kinds")
        ordered = self.ordered_kinds
        return tuple(ordered[(env_id + batch_index) % len(ordered)] for env_id in range(num_envs))
