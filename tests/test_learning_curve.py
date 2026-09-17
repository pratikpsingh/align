"""Host-safe contracts for bounded multi-seed learning curves."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.learning.learning_curve_config import LearningCurveConfig
from align.runtime.learning_curve_runtime import (
    summarize_learning_curve,
    write_learning_curve_tables,
)


class LearningCurveConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.values = json.loads((self.root / "configs/learning-curve-baseline.json").read_text())

    def test_exact_config_converts_to_simulator_stability_contract(self):
        config = LearningCurveConfig.from_dict(self.values)
        self.assertEqual(config.to_dict(), self.values)
        self.assertEqual(config.evaluation_milestones, (0, 5, 10))
        stability = config.to_stability_config()
        self.assertEqual(stability.updates_per_seed, 10)
        self.assertEqual(stability.policy_seeds, (41, 73))
        self.assertEqual(stability.evaluation_steps, 800)
        self.assertNotIn("torch", sys.modules)

    def test_invalid_budgets_milestones_and_keys_are_rejected(self):
        cases = (
            {key: value for key, value in self.values.items() if key != "evaluation_milestones"},
            {**self.values, "unknown": True},
            {**self.values, "policy_seeds": [41]},
            {**self.values, "updates_per_seed": 3},
            {**self.values, "evaluation_milestones": [1, 5, 10]},
            {**self.values, "evaluation_milestones": [0, 10]},
            {**self.values, "evaluation_milestones": [0, 6, 5, 10]},
            {**self.values, "evaluation_milestones": [0, 5, 9]},
            {**self.values, "evaluation_milestones": [0, 5, 5, 10]},
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    LearningCurveConfig.from_dict(values)

    def test_converted_bundle_loads_with_exact_training_budget(self):
        from align.simulation.vector_task import load_bundle

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
        }
        values = {
            section: json.loads((self.root / "configs" / name).read_text())
            for section, name in names.items()
        }
        config = LearningCurveConfig.from_dict(self.values)
        values["training"]["attempts"] = config.updates_per_seed
        values["stability"] = config.to_stability_config().to_dict()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(values))
            bundle = load_bundle(path)
        self.assertEqual(bundle[7].critic_learning_rate, 1e-5)
        self.assertEqual(bundle[9].attempts, 10)
        self.assertEqual(bundle[11].updates_per_seed, 10)


class LearningCurveSummaryTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = LearningCurveConfig.from_dict(
            json.loads((root / "configs/learning-curve-baseline.json").read_text())
        )

    @staticmethod
    def _evaluation(seed, milestone):
        offset = seed / 1000 + milestone / 100
        return {
            "completed_update": milestone,
            "metrics": {
                "completed_updates": milestone,
                "checkpoint_id": f"{seed}-{milestone}",
                "raw_rows": 3200,
                "formation_phase_reached": True,
                "outcome_counts": {
                    "1": int(milestone == 10),
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 0,
                    "6": 4,
                },
                "measurements": {
                    "team_reward_mean": 1.0 + offset,
                    "assigned_rmse_mean_m": 2.0 + offset,
                    "pairwise_rmse_mean_m": 3.0 + offset,
                    "minimum_separation_m": 4.0 + offset,
                },
            },
        }

    def _item(self, seed):
        measurements = [
            {
                "completed_update": update,
                "post_approximate_kl": 0.01,
                "post_policy_clip_fraction": 0.02,
                "post_value_clip_fraction": 0.0,
                "post_explained_variance": -1.0 + update / 10,
                "post_value_loss": 0.2 - update / 100,
                "team_reward_mean": -0.03 + update / 10000,
                "max_critic_gradient_norm_before_clip": 5.0,
            }
            for update in range(1, 11)
        ]
        return {
            "policy_seed": seed,
            "train": {"metrics": {"measurements": measurements}},
            "evaluations": [
                self._evaluation(seed, milestone) for milestone in self.config.evaluation_milestones
            ],
        }

    def test_summary_keeps_update_order_outcomes_and_ranges(self):
        result = summarize_learning_curve([self._item(41), self._item(73)], self.config)
        self.assertEqual(result["training_update_count"], 20)
        self.assertEqual(result["evaluation_container_count"], 6)
        self.assertEqual(len(result["training_trends"]), 10)
        self.assertEqual(result["training_trends"][0]["completed_update"], 1)
        self.assertEqual(
            [row["completed_update"] for row in result["evaluation_trends"]],
            [0, 5, 10],
        )
        self.assertEqual(result["evaluation_trends"][-1]["outcome_counts"]["1"], 2)
        self.assertTrue(result["evaluation_trends"][0]["formation_phase_reached_all_seeds"])

    def test_summary_tables_have_declared_long_form_rows(self):
        result = summarize_learning_curve([self._item(41), self._item(73)], self.config)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_learning_curve_tables(output, result)
            training = (output / "training-curve.csv").read_text().splitlines()
            evaluation = (output / "evaluation-curve.csv").read_text().splitlines()
        self.assertEqual(len(training), 1 + 10 * 6)
        self.assertEqual(len(evaluation), 1 + 3 * 4)

    def test_summary_rejects_missing_or_wrong_milestone(self):
        items = [self._item(41), self._item(73)]
        items[1]["evaluations"].pop()
        with self.assertRaisesRegex(ValueError, "exactly one evaluation"):
            summarize_learning_curve(items, self.config)


if __name__ == "__main__":
    unittest.main()
