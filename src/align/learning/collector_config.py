"""Host-safe configuration for the live recurrent rollout collector check."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


def select_episode_boundary_memory(memory, boundary):
    """Select post-step memory where a time/environment boundary occurred.

    Memory is stored with time, layer, and environment leading axes while the
    mask has time and environment axes. Moving environment before layer aligns
    the leading dimensions for boolean indexing. swapaxes works for both NumPy
    arrays in host tests and Torch tensors in the simulator.
    """
    return memory[1:].swapaxes(1, 2)[boundary]


@dataclass(frozen=True)
class CollectorProbeConfig:
    schema_version: int = 1
    policy_seed: int = 29
    forced_termination_step: int = 30
    forced_termination_environment: int = 1
    stochastic_actions: bool = True
    require_true_termination: bool = True
    require_truncation: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for name in ("policy_seed", "forced_termination_step", "forced_termination_environment"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("stochastic_actions", "require_true_termination", "require_truncation"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")

    @classmethod
    def from_dict(cls, value: dict) -> CollectorProbeConfig:
        expected = {field.name for field in fields(cls)}
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "collector configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self, *, horizon: int, num_envs: int) -> None:
        if self.forced_termination_step >= horizon:
            raise ValueError("forced_termination_step must be inside the rollout horizon")
        if self.forced_termination_environment >= num_envs:
            raise ValueError("forced_termination_environment is outside the environment batch")
