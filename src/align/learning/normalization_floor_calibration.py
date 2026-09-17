"""Host-safe selection contract for a critic normalization denominator floor."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class NormalizationFloorCalibrationConfig:
    """Candidate floors and conservative historical-distribution gates."""

    schema_version: int
    policy_seeds: tuple[int, ...]
    expected_updates_per_seed: int
    candidate_minimum_standard_deviations: tuple[float, ...]
    max_absolute_effective_mean_shift: float
    max_effective_standard_deviation_ratio: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if len(self.policy_seeds) < 2 or len(set(self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must contain at least two unique seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.policy_seeds):
            raise ValueError("policy seeds must be nonnegative integers")
        if type(self.expected_updates_per_seed) is not int or self.expected_updates_per_seed < 1:
            raise ValueError("expected_updates_per_seed must be a positive integer")
        candidates = self.candidate_minimum_standard_deviations
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ValueError("candidate floors must contain at least two unique values")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1
            for value in candidates
        ):
            raise ValueError("candidate floors must be finite and in (0, 1]")
        if any(left >= right for left, right in zip(candidates[:-1], candidates[1:], strict=True)):
            raise ValueError("candidate floors must be strictly increasing")
        for name in (
            "max_absolute_effective_mean_shift",
            "max_effective_standard_deviation_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_dict(cls, values: dict) -> NormalizationFloorCalibrationConfig:
        expected = {field.name for field in fields(cls)}
        if not isinstance(values, dict) or set(values) != expected:
            supplied = set(values) if isinstance(values, dict) else set()
            raise ValueError(
                "normalization floor calibration keys mismatch; "
                f"missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}"
            )
        if not isinstance(values["policy_seeds"], list) or not isinstance(
            values["candidate_minimum_standard_deviations"], list
        ):
            raise ValueError("policy seeds and candidate floors must be JSON lists")
        return cls(
            **{
                **values,
                "policy_seeds": tuple(values["policy_seeds"]),
                "candidate_minimum_standard_deviations": tuple(
                    values["candidate_minimum_standard_deviations"]
                ),
            }
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "policy_seeds": list(self.policy_seeds),
            "expected_updates_per_seed": self.expected_updates_per_seed,
            "candidate_minimum_standard_deviations": list(
                self.candidate_minimum_standard_deviations
            ),
            "max_absolute_effective_mean_shift": self.max_absolute_effective_mean_shift,
            "max_effective_standard_deviation_ratio": (self.max_effective_standard_deviation_ratio),
        }


def summarize_floor_candidates(
    seed_data: list[dict], config: NormalizationFloorCalibrationConfig
) -> dict:
    """Select the smallest floor passing exact extrema and effective-scale gates."""
    if [item.get("policy_seed") for item in seed_data] != list(config.policy_seeds):
        raise ValueError("floor calibration seed data does not match configured order")
    candidates = []
    passing = []
    for floor in config.candidate_minimum_standard_deviations:
        rows = []
        other_group_floor_inactive = True
        for item in seed_data:
            normalization = item["normalization"]
            if (
                not normalization.get("enabled")
                or normalization.get("minimum_standard_deviation", 0.0) != 0.0
            ):
                raise ValueError("source run must use enabled, unfloored normalization")
            if len(item["velocity_updates"]) != config.expected_updates_per_seed:
                raise ValueError("source run has the wrong number of velocity updates")
            for group in ("position", "target"):
                raw_std = math.sqrt(
                    max(
                        normalization["groups"][group]["variance"],
                        normalization["epsilon"] ** 2,
                    )
                )
                other_group_floor_inactive &= floor <= raw_std
            velocity = normalization["groups"]["velocity"]
            warmup_std = math.sqrt(max(velocity["variance"], normalization["epsilon"] ** 2))
            denominator = max(warmup_std, floor)
            for update in item["velocity_updates"]:
                maximum_absolute = max(
                    abs(update["raw_minimum"] - velocity["mean"]),
                    abs(update["raw_maximum"] - velocity["mean"]),
                )
                rows.append(
                    {
                        "policy_seed": item["policy_seed"],
                        "completed_update": update["completed_update"],
                        "maximum_absolute_preclamp": maximum_absolute / denominator,
                        "absolute_effective_mean_shift": abs(
                            (update["raw_mean"] - velocity["mean"]) / denominator
                        ),
                        "effective_standard_deviation_ratio": (
                            update["raw_standard_deviation"] / denominator
                        ),
                    }
                )
        maximum_preclamp = max(row["maximum_absolute_preclamp"] for row in rows)
        maximum_mean_shift = max(row["absolute_effective_mean_shift"] for row in rows)
        maximum_std_ratio = max(row["effective_standard_deviation_ratio"] for row in rows)
        zero_clip = maximum_preclamp <= seed_data[0]["normalization"]["clip"]
        passes = (
            zero_clip
            and other_group_floor_inactive
            and maximum_mean_shift <= config.max_absolute_effective_mean_shift
            and maximum_std_ratio <= config.max_effective_standard_deviation_ratio
        )
        if passes:
            passing.append(float(floor))
        candidates.append(
            {
                "minimum_standard_deviation": float(floor),
                "zero_velocity_clipping_guaranteed_by_observed_extrema": zero_clip,
                "position_and_target_denominators_unchanged": other_group_floor_inactive,
                "maximum_absolute_velocity_preclamp": maximum_preclamp,
                "maximum_absolute_effective_velocity_mean_shift": maximum_mean_shift,
                "maximum_effective_velocity_standard_deviation_ratio": maximum_std_ratio,
                "passes_all_gates": passes,
            }
        )
    selected = passing[0] if passing else None
    return {
        "source_seed_count": len(seed_data),
        "source_updates_per_seed": config.expected_updates_per_seed,
        "source_velocity_rows": len(seed_data) * config.expected_updates_per_seed,
        "candidates": candidates,
        "passing_minimum_standard_deviations": passing,
        "selected_minimum_standard_deviation": selected,
        "selection_rule": "smallest_candidate_passing_all_gates",
        "selection_is_historical_calibration_not_training_evidence": True,
    }
