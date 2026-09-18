"""CPU-only contract for an isolated nonnegative speed-coordinate policy arm."""

from __future__ import annotations

import math

from align.policies.config import RecurrentPolicyConfig


def validate_positive_speed_treatment(
    baseline: RecurrentPolicyConfig,
    treatment: RecurrentPolicyConfig,
    *,
    max_speed_m_s: float,
) -> dict:
    """Require only the fourth action lower bound to change from -1 to 0."""
    if not math.isfinite(max_speed_m_s) or max_speed_m_s <= 0:
        raise ValueError("maximum commanded speed must be finite and positive")
    before = baseline.to_dict()
    after = treatment.to_dict()
    changed = {key for key in before if before[key] != after[key]}
    if changed != {"action_low"}:
        raise ValueError("policy arms differ outside action_low")
    if (
        baseline.action_dim != 4
        or baseline.action_low != (-1.0, -1.0, -1.0, -1.0)
        or treatment.action_low != (-1.0, -1.0, -1.0, 0.0)
        or baseline.action_high != (1.0, 1.0, 1.0, 1.0)
        or treatment.action_high != baseline.action_high
    ):
        raise ValueError("expected the exact one-coordinate speed-bound change")
    return {
        "changed_configuration_field": "action_low[3]",
        "baseline_speed_action_bounds": [-1.0, 1.0],
        "treatment_speed_action_bounds": [0.0, 1.0],
        "baseline_zero_latent_speed_coordinate": 0.0,
        "treatment_zero_latent_speed_coordinate": 0.5,
        "speed_if_direction_is_unit_m_s": 0.5 * max_speed_m_s,
        "zero_latent_direction_has_zero_commanded_velocity": True,
        "controller_decoder_unchanged": "max_speed_m_s * abs(action[3]) * normalized_direction",
        "trained_checkpoint_portable_between_arms": False,
    }
