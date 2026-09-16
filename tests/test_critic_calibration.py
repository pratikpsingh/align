"""Host-safe contracts for matched-rollout critic calibration."""

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.learning.critic_calibration_config import CriticCalibrationConfig
from align.runtime.critic_calibration_runtime import (
    _raw_csv_is_valid,
    summarize_critic_results,
    valid_critic_result,
)


class CriticCalibrationConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.values = json.loads((self.root / "configs/critic-step-calibration.json").read_text())

    def test_exact_configuration_is_host_safe(self):
        config = CriticCalibrationConfig.from_dict(self.values)
        self.assertEqual(config.to_dict(), self.values)
        self.assertEqual(config.policy_seeds, (41, 73))
        self.assertEqual(config.candidate_critic_learning_rates[0], 0.0001)
        self.assertNotIn("torch", sys.modules)

    def test_missing_duplicate_unsorted_and_invalid_values_are_rejected(self):
        for values in (
            {key: value for key, value in self.values.items() if key != "policy_seeds"},
            {**self.values, "unknown": True},
            {**self.values, "policy_seeds": [41]},
            {**self.values, "policy_seeds": [41, 41]},
            {**self.values, "candidate_critic_learning_rates": [0.0001]},
            {
                **self.values,
                "candidate_critic_learning_rates": [0.00001, 0.0001],
            },
            {
                **self.values,
                "candidate_critic_learning_rates": [0.0001, 0.0],
            },
            {**self.values, "max_value_clip_fraction": 1.1},
            {**self.values, "pre_prediction_tolerance": 0.0},
        ):
            with self.assertRaises(ValueError):
                CriticCalibrationConfig.from_dict(values)


class CriticCalibrationBundleTests(unittest.TestCase):
    def test_bundle_loads_exact_critic_calibration_section(self):
        from align.simulation.vector_task import load_bundle

        root = Path(__file__).resolve().parents[1]
        names = {
            "construction": "multi-drone-construction.json",
            "observation": "local-observation-baseline.json",
            "reward": "task-reward-baseline.json",
            "task": "recurrent-stability-task.json",
            "policy": "recurrent-policy.json",
            "rollout": "recurrent-stability-rollout.json",
            "ppo": "recurrent-ppo-calibration.json",
            "recovery": "training-recovery.json",
            "training": "task-training.json",
            "critic_calibration": "critic-step-calibration.json",
        }
        values = {
            section: json.loads((root / "configs" / filename).read_text())
            for section, filename in names.items()
        }
        values["training"]["attempts"] = 1
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(values))
            bundle = load_bundle(path)
            self.assertEqual(bundle[7].critic_learning_rate, 0.0001)
            self.assertEqual(bundle[9].attempts, 1)
            self.assertEqual(bundle[10], values)
            self.assertIsNone(bundle[11])
            self.assertEqual(
                bundle[12].candidate_critic_learning_rates,
                (0.0001, 0.00003, 0.00001),
            )


class CriticCalibrationRuntimeTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = CriticCalibrationConfig.from_dict(
            json.loads((root / "configs/critic-step-calibration.json").read_text())
        )

    @staticmethod
    def _item(seed, clip_by_rate):
        candidates = []
        for rate, clip in clip_by_rate.items():
            candidates.append(
                {
                    "critic_learning_rate": rate,
                    "post_value_clip_fraction": clip,
                    "pre_prediction_max_abs_error": 1e-7,
                    "critic_gradient_norm_before_clip": 4.0,
                    "critic_parameter_max_abs_change": rate,
                    "absolute_value_delta": {"p95": rate * 1000},
                    "guidance": {
                        "pre_predictions_reproduce_rollout_values": True,
                        "post_value_clip_fraction_within_guidance": clip <= 0.5,
                    },
                }
            )
        distribution = {"mean": -0.2, "standard_deviation": 0.3}
        return {
            "policy_seed": seed,
            "metrics": {
                "critic_calibration": {
                    "candidates": candidates,
                    "input_distributions": {
                        "returns": distribution,
                        "old_values": distribution,
                    },
                }
            },
        }

    def test_summary_selects_largest_rate_passing_both_seeds(self):
        rates = self.config.candidate_critic_learning_rates
        items = [
            self._item(41, {rates[0]: 1.0, rates[1]: 0.4, rates[2]: 0.1}),
            self._item(73, {rates[0]: 1.0, rates[1]: 0.5, rates[2]: 0.2}),
        ]
        summary = summarize_critic_results(items, self.config)
        self.assertEqual(summary["seed_count"], 2)
        self.assertEqual(
            summary["largest_rate_passing_value_clip_guidance_on_all_seeds"],
            0.00003,
        )
        self.assertFalse(summary["candidates"][0]["passes_value_clip_guidance_on_all_seeds"])
        self.assertTrue(summary["candidates"][1]["passes_value_clip_guidance_on_all_seeds"])

    def test_runtime_gate_requires_physics_and_calibration_checks(self):
        probe = {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
            "critic_calibration_tested": True,
        }
        metrics = {
            "status": "passed",
            "critic_calibration": {
                "status": "passed",
                "checks": {"matched": True},
            },
        }
        self.assertTrue(valid_critic_result(0, probe, metrics))
        self.assertFalse(valid_critic_result(1, probe, metrics))
        probe["critic_calibration_tested"] = False
        self.assertFalse(valid_critic_result(0, probe, metrics))

    def test_raw_csv_audit_requires_exact_finite_rows(self):
        fields = (
            "critic_learning_rate",
            "team_reward",
            "old_value",
            "return",
            "advantage",
            "pre_prediction",
            "post_prediction",
            "value_delta",
            "absolute_value_delta",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({name: 0.1 for name in fields})
                writer.writerow({name: 0.2 for name in fields})
            self.assertTrue(_raw_csv_is_valid(path, 2))
            self.assertFalse(_raw_csv_is_valid(path, 3))
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[1]["return"] = "nan"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.assertFalse(_raw_csv_is_valid(path, 2))


if __name__ == "__main__":
    unittest.main()
