"""CPU contracts for deterministic target-conditioned template scheduling."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

from align.simulation.multi_drone_contract import MultiDroneConfig, build_group_layout
from align.tasks.formation_schedule import FormationScheduleConfig
from align.tasks.observation import ObservationConfig, build_observations


class FormationScheduleTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = FormationScheduleConfig.from_dict(
            json.loads((root / "configs/formation-schedule-four-templates.json").read_text())
        )

    def test_every_batch_is_balanced_and_rotation_is_reproducible(self):
        first = self.config.assignments(4, 0)
        second = self.config.assignments(4, 1)
        self.assertEqual(set(first), {"cube", "sphere", "pyramid", "plane"})
        self.assertEqual(set(second), set(first))
        self.assertEqual(second, first[1:] + first[:1])
        self.assertEqual(first, self.config.assignments(4, 0))

    def test_invalid_schedule_is_rejected(self):
        for changes in (
            {"kinds": ()},
            {"kinds": ("plane", "plane")},
            {"kinds": ("unknown",)},
            {"seed": -1},
            {"assignment": "random"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FormationScheduleConfig(**changes)
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.config.assignments(5, 0)

    def test_all_templates_fit_safety_envelope_and_change_local_targets(self):
        construction = MultiDroneConfig()
        observation = ObservationConfig()
        actor_targets = {}
        for kind in self.config.kinds:
            layout = build_group_layout(replace(construction, formation_kind=kind))
            self.assertGreaterEqual(
                min(point[2] for point in layout.assigned_target_positions_m),
                construction.airborne_height_m,
            )
            self.assertLessEqual(
                max(point[2] for point in layout.assigned_target_positions_m),
                construction.safety_z_limit_m,
            )
            batch = build_observations(
                layout.agent_ids,
                layout.ground_positions_m,
                tuple((0.0, 0.0, 0.0) for _ in layout.agent_ids),
                layout.assigned_target_positions_m,
                observation,
            )
            actor_targets[kind] = tuple(tuple(actor.self_features[3:6]) for actor in batch.actors)
        self.assertEqual(len(set(actor_targets.values())), len(self.config.kinds))


if __name__ == "__main__":
    unittest.main()
