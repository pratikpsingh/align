"""Host-safe configuration for the bounded task-connected training check."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class TaskTrainingConfig:
    """Small integration run limits; this is not an experiment budget."""

    schema_version: int = 1
    attempts: int = 2
    updates_per_attempt: int = 1
    policy_seed: int = 41
    stochastic_actions: bool = True
    normalization_enabled: bool = False
    require_parameter_change: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if type(self.attempts) is not int or self.attempts <= 0:
            raise ValueError("attempts must be a positive integer")
        if self.updates_per_attempt != 1:
            raise ValueError("the acceptance contract requires one update per attempt")
        if type(self.policy_seed) is not int or self.policy_seed < 0:
            raise ValueError("policy_seed must be a nonnegative integer")
        for name in (
            "stochastic_actions",
            "normalization_enabled",
            "require_parameter_change",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")

    @classmethod
    def from_dict(cls, values: dict) -> TaskTrainingConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            raise ValueError(
                "training configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
