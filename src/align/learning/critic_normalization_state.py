"""Host-safe validation for frozen critic normalization checkpoint state."""

from __future__ import annotations

import math

NORMALIZATION_SCHEMA_VERSION = 2
SUPPORTED_NORMALIZATION_SCHEMA_VERSIONS = (1, 2)
NORMALIZED_GROUPS = ("position", "velocity", "target")


def validate_normalization_state(state: object) -> dict:
    """Validate current and historical normalization checkpoints."""
    if not isinstance(state, dict):
        raise ValueError("normalization state must be a dictionary")
    common = {"schema_version", "enabled", "contract", "frozen"}
    version = state.get("schema_version")
    if version not in SUPPORTED_NORMALIZATION_SCHEMA_VERSIONS:
        raise ValueError("normalization state schema is incompatible")
    if type(state.get("enabled")) is not bool or type(state.get("frozen")) is not bool:
        raise ValueError("normalization enabled and frozen fields must be bool")
    if not state["frozen"]:
        raise ValueError("checkpoint normalization state must be frozen")
    if not state["enabled"]:
        if version != 1 or set(state) != common or state["contract"] != "declared_feature_scaling":
            raise ValueError("disabled normalization state fields are invalid")
        return state

    expected = common | {
        "epsilon",
        "clip",
        "warmup_steps",
        "environment_samples",
        "active_agent_samples",
        "groups",
    }
    contract = "frozen_active_critic_group_standardization"
    if version == 2:
        expected.add("minimum_standard_deviation")
        contract += "_with_floor"
    if set(state) != expected:
        raise ValueError("enabled normalization state fields are incomplete or unexpected")
    if state["contract"] != contract:
        raise ValueError("enabled normalization contract is invalid")
    for name in ("epsilon", "clip"):
        value = state[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"normalization {name} must be finite and positive")
    if version == 2:
        floor = state["minimum_standard_deviation"]
        if type(floor) not in (int, float) or not math.isfinite(floor) or not 0 < floor <= 1:
            raise ValueError("normalization minimum_standard_deviation is invalid")
    for name in ("warmup_steps", "environment_samples", "active_agent_samples"):
        if type(state[name]) is not int or state[name] < 1:
            raise ValueError(f"normalization {name} must be a positive integer")
    groups = state["groups"]
    if not isinstance(groups, dict) or set(groups) != set(NORMALIZED_GROUPS):
        raise ValueError("normalization group names are invalid")
    for name, values in groups.items():
        if not isinstance(values, dict) or set(values) != {"count", "mean", "variance"}:
            raise ValueError(f"normalization group state is invalid: {name}")
        if type(values["count"]) is not int or values["count"] < 1:
            raise ValueError(f"normalization group count is invalid: {name}")
        if values["count"] != state["active_agent_samples"] * 3:
            raise ValueError(f"normalization group count does not match active samples: {name}")
        if any(
            type(values[field]) not in (int, float) or not math.isfinite(values[field])
            for field in ("mean", "variance")
        ):
            raise ValueError(f"normalization group moments are not finite: {name}")
        if values["variance"] < 0:
            raise ValueError(f"normalization group variance is negative: {name}")
    return state


def normalization_standard_deviation(state: dict, group: str) -> tuple[float, float]:
    """Return raw warmup spread and the effective checkpointed denominator."""
    validate_normalization_state(state)
    if not state["enabled"]:
        return 1.0, 1.0
    if group not in NORMALIZED_GROUPS:
        raise ValueError(f"unknown critic normalization group: {group}")
    raw = math.sqrt(max(state["groups"][group]["variance"], state["epsilon"] ** 2))
    effective = max(raw, float(state.get("minimum_standard_deviation", 0.0)))
    return raw, effective
