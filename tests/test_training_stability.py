"""Host-safe contracts for bounded stability calibration and evaluation."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.learning.stability_config import StabilityConfig
from align.learning.training_config import TaskTrainingConfig
from align.runtime.stability_runtime import (
    summarize_seed_results,
    valid_phase,
    validate_stability_resolved_config,
)


class StabilityConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.values = json.loads((self.root / "configs/training-stability.json").read_text())

    def test_exact_multi_seed_configuration_is_host_safe(self):
        config = StabilityConfig.from_dict(self.values)
        self.assertEqual(config.to_dict(), self.values)
        self.assertGreaterEqual(len(config.policy_seeds), 2)
        self.assertGreaterEqual(config.updates_per_seed, 2)
        self.assertNotIn("torch", sys.modules)

    def test_bad_seed_budget_thresholds_and_keys_are_rejected(self):
        for values in (
            {key: value for key, value in self.values.items() if key != "policy_seeds"},
            {**self.values, "unknown": True},
            {**self.values, "policy_seeds": [41]},
            {**self.values, "policy_seeds": [41, 41]},
            {**self.values, "policy_seeds": [41, -1]},
            {**self.values, "updates_per_seed": 1},
            {**self.values, "evaluation_steps": 0},
            {**self.values, "max_post_update_policy_clip_fraction": 1.1},
        ):
            with self.assertRaises(ValueError):
                StabilityConfig.from_dict(values)

    def test_task_training_attempt_count_supports_calibration(self):
        values = json.loads((self.root / "configs/task-training.json").read_text())
        values["attempts"] = self.values["updates_per_seed"]
        config = TaskTrainingConfig.from_dict(values)
        self.assertEqual(config.attempts, 3)


class StabilityBundleTests(unittest.TestCase):
    def test_stability_bundle_loads_exact_section(self):
        from align.simulation.vector_task import load_bundle

        root = Path(__file__).resolve().parents[1]
        names = {
            "construction": "multi-drone-construction.json",
            "observation": "local-observation-baseline.json",
            "reward": "task-reward-baseline.json",
            "task": "recurrent-stability-task.json",
            "policy": "recurrent-policy.json",
            "rollout": "recurrent-stability-rollout.json",
            "ppo": "recurrent-ppo-selected.json",
            "recovery": "training-recovery.json",
            "training": "task-training.json",
            "stability": "training-stability.json",
            "critic_normalization": "critic-normalization-active-groups.json",
        }
        values = {
            section: json.loads((root / "configs" / name).read_text())
            for section, name in names.items()
        }
        values["training"]["attempts"] = values["stability"]["updates_per_seed"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(values))
            bundle = load_bundle(path)
            self.assertEqual(bundle[7].update_epochs, 1)
            self.assertEqual(bundle[7].critic_learning_rate, 1e-5)
            self.assertEqual(bundle[9].attempts, 3)
            self.assertEqual(bundle[10], values)
            self.assertEqual(bundle[11].evaluation_steps, 800)
            self.assertTrue(bundle[13].enabled)


class StabilityRuntimeTests(unittest.TestCase):
    def test_resolved_serialized_construction_derives_formation_start(self):
        root = Path(__file__).resolve().parents[1]
        resolved = {
            "construction": json.loads(
                (root / "configs/multi-drone-construction.json").read_text()
            ),
            "task": json.loads((root / "configs/recurrent-stability-task.json").read_text()),
            "rollout": json.loads((root / "configs/recurrent-stability-rollout.json").read_text()),
            "critic_normalization": json.loads(
                (root / "configs/critic-normalization-active-groups.json").read_text()
            ),
        }
        stability = StabilityConfig.from_dict(
            json.loads((root / "configs/training-stability.json").read_text())
        )
        self.assertNotIn("ground_steps", resolved["construction"])
        self.assertEqual(validate_stability_resolved_config(resolved, stability), 550)

        resolved["rollout"]["horizon"] = 550
        with self.assertRaisesRegex(ValueError, "formation phase"):
            validate_stability_resolved_config(resolved, stability)

    def test_phase_gate_requires_fresh_physics_evidence(self):
        probe = {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
            "training_stability_tested": True,
        }
        metrics = {"status": "passed", "checks": {"finite": True}}
        self.assertTrue(valid_phase(0, probe, metrics, "training_stability_tested"))
        self.assertFalse(valid_phase(1, probe, metrics, "training_stability_tested"))
        self.assertFalse(valid_phase(0, probe, metrics, "deterministic_evaluation_tested"))

    def test_summary_keeps_seed_ranges_and_update_count(self):
        def item(seed, offset):
            measurements = {
                "team_reward_mean": 1.0 + offset,
                "assigned_rmse_mean_m": 2.0 + offset,
                "pairwise_rmse_mean_m": 3.0 + offset,
                "minimum_separation_m": 4.0 + offset,
            }
            update = {
                "post_approximate_kl": 0.01 + offset,
                "post_policy_clip_fraction": 0.02 + offset,
                "post_value_clip_fraction": 0.03 + offset,
                "max_critic_gradient_norm_before_clip": 5.0 + offset,
            }
            return {
                "policy_seed": seed,
                "train": {
                    "metrics": {
                        "measurements": [update, update],
                        "critic_normalization": {"enabled": True},
                        "critic_normalization_warmup_environment_transitions": 3072,
                    }
                },
                "evaluation": {"metrics": {"measurements": measurements}},
            }

        summary = summarize_seed_results([item(41, 0.0), item(73, 0.1)])
        self.assertEqual(summary["seed_count"], 2)
        self.assertEqual(summary["update_count"], 4)
        self.assertEqual(summary["seeds"], [41, 73])
        self.assertAlmostEqual(summary["evaluation_team_reward_mean"]["mean"], 1.05)


if __name__ == "__main__":
    unittest.main()
