"""Guard the single-variable reward comparison and formation-only windows."""

import csv
import tempfile
import unittest
from pathlib import Path

from align.runtime.reward_ablation import _formation_window, validate_single_weight_change


class FormationRewardComparisonTests(unittest.TestCase):
    def test_rejects_hidden_task_or_reward_changes(self):
        baseline = {
            "task": {"max_episode_steps": 2350},
            "reward": {"formation_weight": 1.0, "tracking_weight": 1.0},
        }
        treatment = {
            "task": {"max_episode_steps": 2350},
            "reward": {"formation_weight": 4.0, "tracking_weight": 1.0},
        }
        self.assertEqual(validate_single_weight_change(baseline, treatment), (1.0, 4.0))
        for change in (
            {"task": {"max_episode_steps": 2400}},
            {"reward": {"tracking_weight": 2.0}},
            {"reward": {"formation_weight": 1.0}},
        ):
            changed = {key: dict(value) for key, value in treatment.items()}
            for group, values in change.items():
                changed[group].update(values)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_single_weight_change(baseline, changed)

    def test_episode_reset_does_not_join_two_formation_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "env_id",
                        "episode_step",
                        "phase",
                        "assigned_rmse_m",
                        "pairwise_rmse_m",
                        "minimum_separation_m",
                    ),
                )
                writer.writeheader()
                for value in (2.0, 0.5):
                    for step in range(50):
                        writer.writerow(
                            {
                                "env_id": 0,
                                "episode_step": step + 1,
                                "phase": "formation",
                                "assigned_rmse_m": value,
                                "pairwise_rmse_m": value,
                                "minimum_separation_m": 0.7,
                            }
                        )
            result = _formation_window(path, margin_m=0.55)
        self.assertEqual(result["formation_episodes"], 2)
        self.assertEqual(result["windowed_formation_episodes"], 2)
        self.assertEqual(result["first_assigned_rmse_m"], 1.25)
        self.assertEqual(result["last_assigned_rmse_m"], 1.25)

    def test_first_and_last_formation_windows_are_per_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "env_id",
                        "episode_step",
                        "phase",
                        "assigned_rmse_m",
                        "pairwise_rmse_m",
                        "minimum_separation_m",
                    ),
                )
                writer.writeheader()
                for env in (0, 1):
                    for step in range(100):
                        writer.writerow(
                            {
                                "env_id": env,
                                "episode_step": step + 1,
                                "phase": "formation",
                                "assigned_rmse_m": 2.0 if step < 50 else 1.0,
                                "pairwise_rmse_m": 0.4 if step < 50 else 0.6,
                                "minimum_separation_m": 0.7 if step < 50 else 0.5,
                            }
                        )
            result = _formation_window(path, margin_m=0.55)
        self.assertEqual(result["formation_rows"], 200)
        self.assertEqual(result["formation_episodes"], 2)
        self.assertEqual(result["first_assigned_rmse_m"], 2.0)
        self.assertEqual(result["last_assigned_rmse_m"], 1.0)
        self.assertEqual(result["first_pairwise_rmse_m"], 0.4)
        self.assertEqual(result["last_pairwise_rmse_m"], 0.6)
        self.assertEqual(result["formation_minimum_separation_m"], 0.5)
        self.assertEqual(result["below_margin_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
