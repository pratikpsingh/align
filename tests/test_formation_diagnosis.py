"""Meaningful phase-window and episode-boundary checks for saved evaluations."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.runtime.formation_diagnosis import _episodes, analyze_episode


class FormationDiagnosisTests(unittest.TestCase):
    @staticmethod
    def _row(step, phase, error, pairwise, separation, action_max=0.2):
        return {
            "env_id": "0",
            "episode_step": str(step),
            "formation_kind": "plane",
            "phase": phase,
            "team_reward": "-0.1",
            "assigned_rmse_m": str(error),
            "pairwise_rmse_m": str(pairwise),
            "minimum_separation_m": str(separation),
            "action_min": "-0.2",
            "action_max": str(action_max),
            "reason_code": "6" if step == 5 else "0",
        }

    def test_formation_windows_separate_error_from_early_phases(self):
        rows = [
            self._row(1, "ground", 0.0, 0.0, 1.0),
            self._row(2, "takeoff", 0.1, 0.1, 1.0),
            self._row(3, "formation", 1.5, 0.8, 0.6),
            self._row(4, "formation", 1.4, 0.9, 0.5, 0.99),
            self._row(5, "formation", 1.3, 1.0, 0.4),
            self._row(6, "formation", 1.2, 1.1, 0.3),
        ]
        result = analyze_episode(rows, window_steps=2, separation_margin_m=0.55, action_bound=0.98)
        self.assertEqual(result["ground_rows"], 1)
        self.assertEqual(result["takeoff_rows"], 1)
        self.assertEqual(result["formation_rows"], 4)
        self.assertAlmostEqual(result["formation_first_assigned_rmse_m"], 1.45)
        self.assertAlmostEqual(result["formation_last_assigned_rmse_m"], 1.25)
        self.assertAlmostEqual(result["formation_last_minus_first_pairwise_rmse_m"], 0.2)
        self.assertEqual(result["separation_below_margin_rows"], 3)
        self.assertEqual(result["action_near_bound_rows"], 1)
        self.assertAlmostEqual(result["max_abs_action"], 0.99)

    def test_nonconsecutive_steps_and_short_formation_are_rejected(self):
        rows = [self._row(i, "formation", 1.0, 0.2, 0.8) for i in (1, 3, 4, 5)]
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            analyze_episode(rows, window_steps=2, separation_margin_m=0.55, action_bound=0.98)
        short_rows = [self._row(i, "formation", 1.0, 0.2, 0.8) for i in (1, 2, 3)]
        with self.assertRaisesRegex(ValueError, "shorter"):
            analyze_episode(short_rows, window_steps=2, separation_margin_m=0.55, action_bound=0.98)

    def test_csv_episode_split_keeps_environment_and_reset_boundaries(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            rows = [
                self._row(1, "ground", 0.0, 0.0, 1.0),
                self._row(2, "formation", 1.0, 0.2, 0.8),
                self._row(1, "ground", 0.0, 0.0, 1.0),
            ]
            rows.append({**self._row(1, "ground", 0.0, 0.0, 1.0), "env_id": "1"})
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            groups = _episodes(path)
        self.assertEqual(
            [(env, index, len(values)) for env, index, values in groups],
            [(0, 0, 2), (0, 1, 1), (1, 0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
