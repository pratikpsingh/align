"""Contracts for the one-variable formation-reward ablation."""

import json
import unittest
from pathlib import Path

from align.tasks.reward import RewardConfig, compute_step_reward

ROOT = Path(__file__).resolve().parents[1] / "configs"


class FormationRewardAblationTests(unittest.TestCase):
    def test_only_formation_weight_changes_and_weighted_effect_is_exact(self):
        baseline = json.loads((ROOT / "task-reward-baseline.json").read_text())
        treatment = json.loads((ROOT / "task-reward-formation-4.json").read_text())
        self.assertEqual(
            {key: value for key, value in treatment.items() if key != "formation_weight"},
            {key: value for key, value in baseline.items() if key != "formation_weight"},
        )
        self.assertEqual(baseline["formation_weight"], 1.0)
        self.assertEqual(treatment["formation_weight"], 4.0)
        positions = ((0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 1.0, 1.0))
        targets = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))
        args = (
            positions,
            targets,
            ((0.0, 0.0, 0.0),) * 3,
            ((0.0,) * 4,) * 3,
            (0.0,) * 3,
            (True,) * 3,
        )
        control, _ = compute_step_reward(*args, RewardConfig.from_dict(baseline))
        experiment, _ = compute_step_reward(*args, RewardConfig.from_dict(treatment))
        for old, new in zip(control.weighted_by_agent, experiment.weighted_by_agent, strict=True):
            self.assertAlmostEqual(new.formation, 4 * old.formation)
            for name in (
                "tracking",
                "progress",
                "separation",
                "contact",
                "settling",
                "smoothness",
                "effort",
            ):
                self.assertEqual(getattr(new, name), getattr(old, name))


if __name__ == "__main__":
    unittest.main()
