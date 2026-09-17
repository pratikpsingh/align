"""Host-safe checks for the centralized critic diagnostic feature schema."""

import json
import sys
import unittest
from pathlib import Path

from align.learning.critic_diagnostics_schema import critic_feature_layout
from align.tasks.observation import ObservationConfig


class CriticDiagnosticSchemaTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = ObservationConfig.from_dict(
            json.loads((root / "configs/local-observation-baseline.json").read_text())
        )

    def test_layout_labels_every_configured_scalar_once(self):
        layout = critic_feature_layout(self.config)
        self.assertEqual(len(layout), self.config.critic_dimension)
        self.assertEqual([item.index for item in layout], list(range(80)))
        self.assertEqual(layout[0].name, "agent_0/position_x")
        self.assertEqual(layout[8].name, "agent_0/target_z")
        self.assertEqual(layout[9].name, "agent_1/position_x")
        self.assertEqual(layout[72].name, "agent_0/mask")
        self.assertEqual(layout[-1].name, "agent_7/mask")

    def test_layout_has_declared_groups_and_is_host_safe(self):
        layout = critic_feature_layout(self.config)
        counts = {
            group: sum(item.group == group for item in layout)
            for group in ("position", "velocity", "target", "mask")
        }
        self.assertEqual(counts, {"position": 24, "velocity": 24, "target": 24, "mask": 8})
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
