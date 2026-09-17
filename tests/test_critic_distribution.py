"""Host-safe checks for active critic distribution tables and learning trends."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.learning.critic_distribution_schema import (
    COLUMNS,
    GROUPS,
    read_valid_critic_distribution,
    validate_critic_distribution_rows,
)
from align.learning.learning_curve_config import LearningCurveConfig
from align.runtime.learning_curve_runtime import (
    summarize_learning_curve,
    write_learning_curve_tables,
)


def row(group: str, *, update: int = 1, count: int = 48, clipped: int = 0) -> dict:
    return {
        "completed_update": update,
        "group": group,
        "count": count,
        "raw_mean": 0.1,
        "raw_standard_deviation": 0.2,
        "raw_minimum": -0.3,
        "raw_maximum": 0.5,
        "warmup_mean": 0.0,
        "warmup_standard_deviation": 0.2,
        "normalization_standard_deviation": 0.25,
        "mean_shift_warmup_standard_deviations": 0.5,
        "raw_standard_deviation_ratio": 1.0,
        "mean_shift_normalization_standard_deviations": 0.4,
        "raw_standard_deviation_normalization_ratio": 0.8,
        "normalized_mean": 0.5,
        "normalized_standard_deviation": 1.0,
        "normalized_minimum": -1.5,
        "normalized_maximum": 2.5,
        "clipped_count": clipped,
        "clipped_fraction": clipped / count,
    }


class CriticDistributionSchemaTests(unittest.TestCase):
    def test_exact_three_active_groups_are_parsed(self):
        rows = [row(group) for group in GROUPS]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "critic-distribution.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            loaded = read_valid_critic_distribution(
                path, update=1, scalar_count=48, enabled=True, clip=5.0
            )
        self.assertIsNotNone(loaded)
        self.assertEqual({item["group"] for item in loaded}, set(GROUPS))

    def test_missing_group_wrong_count_and_unbounded_values_fail(self):
        rows = [row(group) for group in GROUPS]
        self.assertFalse(
            validate_critic_distribution_rows(
                rows[:2], update=1, scalar_count=48, enabled=True, clip=5.0
            )
        )
        rows[0]["count"] = 47
        self.assertFalse(
            validate_critic_distribution_rows(
                rows, update=1, scalar_count=48, enabled=True, clip=5.0
            )
        )
        rows[0]["count"] = 48
        rows[0]["normalized_maximum"] = 5.1
        self.assertFalse(
            validate_critic_distribution_rows(
                rows, update=1, scalar_count=48, enabled=True, clip=5.0
            )
        )


class CriticDistributionTrendTests(unittest.TestCase):
    def test_two_seed_update_trends_and_csv_are_exact(self):
        root = Path(__file__).resolve().parents[1]
        config = LearningCurveConfig.from_dict(
            json.loads((root / "configs/learning-curve-baseline.json").read_text())
        )
        items = []
        for seed in config.policy_seeds:
            measurements = []
            for update in range(1, config.updates_per_seed + 1):
                distribution = [
                    {
                        "group": group,
                        "clipped_fraction": update / 100 + seed / 10000,
                        "mean_shift_warmup_standard_deviations": update / 10,
                        "raw_standard_deviation_ratio": 1 + update / 100,
                        "mean_shift_normalization_standard_deviations": update / 20,
                        "raw_standard_deviation_normalization_ratio": 0.5 + update / 100,
                    }
                    for group in GROUPS
                ]
                measurements.append(
                    {
                        "completed_update": update,
                        "post_approximate_kl": 0.01,
                        "post_policy_clip_fraction": 0.02,
                        "post_value_clip_fraction": 0.0,
                        "post_explained_variance": 0.1,
                        "post_value_loss": 0.2,
                        "team_reward_mean": -0.03,
                        "max_critic_gradient_norm_before_clip": 5.0,
                        "critic_distribution": distribution,
                    }
                )
            evaluations = []
            for milestone in config.evaluation_milestones:
                evaluations.append(
                    {
                        "completed_update": milestone,
                        "metrics": {
                            "completed_updates": milestone,
                            "checkpoint_id": f"{seed}-{milestone}",
                            "raw_rows": 3200,
                            "formation_phase_reached": True,
                            "outcome_counts": {str(code): 0 for code in range(1, 7)},
                            "measurements": {
                                "team_reward_mean": -0.02,
                                "assigned_rmse_mean_m": 1.0,
                                "pairwise_rmse_mean_m": 0.3,
                                "minimum_separation_m": 0.9,
                            },
                        },
                    }
                )
            items.append(
                {
                    "policy_seed": seed,
                    "train": {"metrics": {"measurements": measurements}},
                    "evaluations": evaluations,
                }
            )
        summary = summarize_learning_curve(items, config)
        self.assertEqual(len(summary["critic_distribution_trends"]), 30)
        self.assertEqual(summary["critic_distribution_trends"][0]["group"], "position")
        self.assertAlmostEqual(
            summary["critic_distribution_trends"][0]["clipped_fraction"]["mean"],
            0.0157,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_learning_curve_tables(output, summary)
            lines = (output / "critic-distribution-curve.csv").read_text().splitlines()
        self.assertEqual(len(lines), 1 + 10 * 3 * 5)
        self.assertAlmostEqual(
            summary["critic_distribution_trends"][0][
                "mean_shift_normalization_standard_deviations"
            ]["mean"],
            0.05,
        )

        del items[1]["train"]["metrics"]["measurements"][0]["critic_distribution"]
        with self.assertRaisesRegex(ValueError, "missing from some updates"):
            summarize_learning_curve(items, config)


if __name__ == "__main__":
    unittest.main()
