"""Positive-speed action arm changes exactly one policy bound."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from align.policies.action_interface import validate_positive_speed_treatment
from align.policies.config import RecurrentPolicyConfig
from align.simulation.contract import direction_speed


class PositiveSpeedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "configs"
        cls.baseline = RecurrentPolicyConfig.from_dict(
            json.loads((root / "recurrent-policy.json").read_text())
        )
        cls.treatment = RecurrentPolicyConfig.from_dict(
            json.loads((root / "recurrent-policy-positive-speed.json").read_text())
        )

    def test_only_speed_lower_bound_changes(self):
        result = validate_positive_speed_treatment(self.baseline, self.treatment, max_speed_m_s=0.5)
        self.assertEqual(result["changed_configuration_field"], "action_low[3]")
        self.assertEqual(result["speed_if_direction_is_unit_m_s"], 0.25)
        self.assertEqual(direction_speed((1.0, 0.0, 0.0, 0.5), 0.5), (0.25, 0.0, 0.0))
        self.assertEqual(direction_speed((0.0, 0.0, 0.0, 0.5), 0.5), (0.0, 0.0, 0.0))

    def test_extra_policy_changes_are_rejected(self):
        changed = self.treatment.to_dict()
        changed["log_std_initial"] = -0.2
        with self.assertRaisesRegex(ValueError, "outside action_low"):
            validate_positive_speed_treatment(
                self.baseline, RecurrentPolicyConfig.from_dict(changed), max_speed_m_s=0.5
            )
        changed = self.treatment.to_dict()
        changed["action_low"][3] = -0.1
        with self.assertRaisesRegex(ValueError, "exact one-coordinate"):
            validate_positive_speed_treatment(
                self.baseline, RecurrentPolicyConfig.from_dict(changed), max_speed_m_s=0.5
            )


if __name__ == "__main__":
    unittest.main()
