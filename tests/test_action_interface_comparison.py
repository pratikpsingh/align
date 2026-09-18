"""Phase exposure must remain visible when matched policies end at different times."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from align.runtime.action_interface_comparison import summarize_formation_csv


class ActionInterfaceComparisonTests(unittest.TestCase):
    def test_formation_metrics_exclude_ground_and_takeoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "phase",
                        "assigned_rmse_m",
                        "pairwise_rmse_m",
                        "minimum_separation_m",
                        "reason_code",
                        "episode_step",
                    ),
                )
                writer.writeheader()
                for row in (
                    ("ground", 0.1, 0.1, 1.0, 0, 1),
                    ("takeoff", 0.2, 0.2, 0.9, 0, 2),
                    ("formation", 2.0, 0.4, 0.8, 0, 3),
                    ("formation", 4.0, 0.6, 0.7, 3, 4),
                ):
                    writer.writerow(dict(zip(writer.fieldnames, row, strict=True)))
            result = summarize_formation_csv(path)
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["formation_rows"], 2)
        self.assertEqual(result["formation_assigned_rmse_mean_m"], 3.0)
        self.assertEqual(result["formation_pairwise_rmse_mean_m"], 0.5)
        self.assertEqual(result["formation_minimum_separation_m"], 0.7)
        self.assertEqual(result["terminal_episode_steps"], [4])

    def test_no_formation_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            path.write_text(
                "phase,assigned_rmse_m,pairwise_rmse_m,minimum_separation_m,reason_code,episode_step\n"
                "ground,0,0,1,3,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lacks evaluation or terminal formation"):
                summarize_formation_csv(path)


if __name__ == "__main__":
    unittest.main()
