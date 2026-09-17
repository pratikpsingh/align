"""Host-safe configuration checks for frozen critic group normalization."""

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.runtime.stability_runtime import valid_normalization_warmup_csv


class CriticNormalizationConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_enabled_and_disabled_configs_are_exact_and_host_safe(self):
        disabled_values = json.loads(
            (self.root / "configs/critic-normalization-disabled.json").read_text()
        )
        enabled_values = json.loads(
            (self.root / "configs/critic-normalization-active-groups.json").read_text()
        )
        disabled = CriticNormalizationConfig.from_dict(disabled_values)
        enabled = CriticNormalizationConfig.from_dict(enabled_values)
        self.assertEqual(disabled.to_dict(), disabled_values)
        self.assertEqual(enabled.to_dict(), enabled_values)
        self.assertFalse(disabled.enabled)
        self.assertEqual(enabled.warmup_steps, 768)
        self.assertNotIn("torch", sys.modules)

    def test_invalid_keys_ranges_and_warmup_contract_are_rejected(self):
        baseline = json.loads(
            (self.root / "configs/critic-normalization-active-groups.json").read_text()
        )
        for values in (
            {key: value for key, value in baseline.items() if key != "clip"},
            {**baseline, "unknown": True},
            {**baseline, "enabled": 1},
            {**baseline, "warmup_steps": 0},
            {**baseline, "epsilon": 0.0},
            {**baseline, "clip": 0.5},
            {**baseline, "schema_version": 2},
            {**baseline, "enabled": False},
        ):
            with self.assertRaises(ValueError):
                CriticNormalizationConfig.from_dict(values)


class CriticNormalizationWarmupAuditTests(unittest.TestCase):
    def test_raw_warmup_reductions_have_exact_active_counts(self):
        fields = (
            "warmup_step",
            "position_count",
            "position_sum",
            "position_sum_squares",
            "velocity_count",
            "velocity_sum",
            "velocity_sum_squares",
            "target_count",
            "target_sum",
            "target_sum_squares",
            "action_min",
            "action_max",
            "terminated_count",
            "truncated_count",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "normalization-warmup.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for step in range(2):
                    writer.writerow(
                        {
                            "warmup_step": step,
                            "position_count": 24,
                            "position_sum": 1.0,
                            "position_sum_squares": 2.0,
                            "velocity_count": 24,
                            "velocity_sum": 0.1,
                            "velocity_sum_squares": 0.2,
                            "target_count": 24,
                            "target_sum": 3.0,
                            "target_sum_squares": 4.0,
                            "action_min": -0.5,
                            "action_max": 0.5,
                            "terminated_count": 0,
                            "truncated_count": 0,
                        }
                    )
            self.assertTrue(
                valid_normalization_warmup_csv(path, warmup_steps=2, num_envs=2, num_agents=4)
            )
            self.assertFalse(
                valid_normalization_warmup_csv(path, warmup_steps=3, num_envs=2, num_agents=4)
            )


if __name__ == "__main__":
    unittest.main()
