"""Paired replay comparison refuses confounded arms and audits raw outcomes."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from align.tasks.policy_guard_comparison import summarize_evaluation_rows, validate_pair


class PolicyGuardComparisonTests(unittest.TestCase):
    def test_pair_requires_same_checkpoint_and_image(self):
        identity = {
            "source_run_id": "training",
            "source_image_id": "training-image",
            "image_id": "evaluation-image",
            "policy_seed": 41,
            "requested_completed_update": 4,
            "checkpoint_id": "checkpoint",
            "checkpoint_sha256": "checksum",
            "source_config_sha256": "config",
            "training_performed": False,
            "optimizer_updates": 0,
        }
        baseline = {**identity, "wake_guard_enabled": False}
        guarded = {**identity, "wake_guard_enabled": True}
        validate_pair(baseline, guarded)
        with self.assertRaisesRegex(ValueError, "image_id"):
            validate_pair(baseline, {**guarded, "image_id": "other"})
        with self.assertRaisesRegex(ValueError, "did not enable"):
            validate_pair(baseline, {**guarded, "wake_guard_enabled": False})
        with self.assertRaisesRegex(ValueError, "frozen actor"):
            validate_pair({**baseline, "optimizer_updates": 1}, guarded)

    def test_phase_and_success_are_read_from_raw_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            fields = (
                "evaluation_step",
                "phase",
                "assigned_rmse_m",
                "pairwise_rmse_m",
                "minimum_separation_m",
                "reason_code",
            )
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        dict(zip(fields, (0, "ground", 2, 1, 1, 0), strict=True)),
                        dict(zip(fields, (1, "takeoff", 1, 0.5, 0.7, 0), strict=True)),
                        dict(zip(fields, (2, "formation", 0.4, 0.3, 0.65, 1), strict=True)),
                    ]
                )
            result = summarize_evaluation_rows(path)
            self.assertEqual(result["environment_rows"], 3)
            self.assertEqual(result["outcome_counts"]["1"], 1)
            self.assertEqual(result["first_formation_evaluation_step"], 2)
            self.assertAlmostEqual(result["formation_assigned_rmse_mean_m"], 0.4)
            self.assertAlmostEqual(result["minimum_nonground_separation_m"], 0.65)


if __name__ == "__main__":
    unittest.main()
