"""Host-safe checks for conservative critic normalization-floor selection."""

import json
import math
import sys
import unittest
from pathlib import Path

from align.learning.normalization_floor_calibration import (
    NormalizationFloorCalibrationConfig,
    summarize_floor_candidates,
)


class NormalizationFloorCalibrationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.values = json.loads(
            (root / "configs/critic-normalization-floor-calibration.json").read_text()
        )
        self.config = NormalizationFloorCalibrationConfig.from_dict(self.values)

    @staticmethod
    def _normalization(velocity_std: float) -> dict:
        return {
            "schema_version": 1,
            "enabled": True,
            "contract": "frozen_active_critic_group_standardization",
            "frozen": True,
            "epsilon": 1e-6,
            "clip": 5.0,
            "warmup_steps": 768,
            "environment_samples": 3072,
            "active_agent_samples": 12288,
            "groups": {
                "position": {"count": 36864, "mean": 0.0, "variance": 0.13**2},
                "velocity": {
                    "count": 36864,
                    "mean": 0.0,
                    "variance": velocity_std**2,
                },
                "target": {"count": 36864, "mean": 0.09, "variance": 0.18**2},
            },
        }

    def _seed_data(self):
        result = []
        for seed, warmup_std in ((41, 0.01349), (73, 0.01257)):
            updates = []
            for update in range(1, 11):
                scale = update / 10
                updates.append(
                    {
                        "completed_update": update,
                        "raw_mean": (0.031 if seed == 41 else -0.005) * scale,
                        "raw_standard_deviation": ((0.034 if seed == 41 else 0.025) * scale),
                        "raw_minimum": (-0.065 if seed == 41 else -0.08) * scale,
                        "raw_maximum": (0.117 if seed == 41 else 0.09) * scale,
                    }
                )
            result.append(
                {
                    "policy_seed": seed,
                    "normalization": self._normalization(warmup_std),
                    "velocity_updates": updates,
                }
            )
        return result

    def test_config_is_exact_host_safe_and_strict(self):
        self.assertEqual(self.config.to_dict(), self.values)
        self.assertNotIn("torch", sys.modules)
        for values in (
            {key: value for key, value in self.values.items() if key != "policy_seeds"},
            {**self.values, "unknown": True},
            {**self.values, "policy_seeds": [41]},
            {**self.values, "candidate_minimum_standard_deviations": [0.025, 0.02]},
            {**self.values, "candidate_minimum_standard_deviations": [0.02, math.nan]},
            {**self.values, "max_effective_standard_deviation_ratio": 0.0},
        ):
            with self.assertRaises(ValueError):
                NormalizationFloorCalibrationConfig.from_dict(values)

    def test_smallest_candidate_passing_every_seed_is_selected(self):
        summary = summarize_floor_candidates(self._seed_data(), self.config)
        self.assertEqual(summary["source_velocity_rows"], 20)
        self.assertEqual(summary["selected_minimum_standard_deviation"], 0.025)
        by_floor = {row["minimum_standard_deviation"]: row for row in summary["candidates"]}
        self.assertFalse(by_floor[0.02]["passes_all_gates"])
        self.assertTrue(by_floor[0.025]["passes_all_gates"])
        self.assertTrue(by_floor[0.025]["position_and_target_denominators_unchanged"])
        self.assertLessEqual(by_floor[0.025]["maximum_absolute_velocity_preclamp"], 5.0)
        self.assertLessEqual(by_floor[0.025]["maximum_absolute_effective_velocity_mean_shift"], 1.5)
        self.assertLessEqual(
            by_floor[0.025]["maximum_effective_velocity_standard_deviation_ratio"], 1.5
        )

    def test_changed_seed_order_update_count_or_floored_source_is_rejected(self):
        seed_data = self._seed_data()
        with self.assertRaisesRegex(ValueError, "seed data"):
            summarize_floor_candidates(list(reversed(seed_data)), self.config)
        seed_data = self._seed_data()
        seed_data[0]["velocity_updates"].pop()
        with self.assertRaisesRegex(ValueError, "wrong number"):
            summarize_floor_candidates(seed_data, self.config)
        seed_data = self._seed_data()
        seed_data[0]["normalization"]["minimum_standard_deviation"] = 0.01
        with self.assertRaisesRegex(ValueError, "unfloored"):
            summarize_floor_candidates(seed_data, self.config)


if __name__ == "__main__":
    unittest.main()
