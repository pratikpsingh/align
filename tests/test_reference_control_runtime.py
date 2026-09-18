"""CPU checks for the reference action contract and read-only lab command."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from align.runtime.reference_control_runtime import reference_command
from align.simulation.contract import direction_speed
from align.simulation.multi_drone_contract import (
    MultiDroneConfig,
    build_group_layout,
    velocity_actions,
    velocity_commands,
)


class ReferenceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.config = MultiDroneConfig.from_dict(
            json.loads((root / "configs/multi-drone-construction.json").read_text())
        )
        cls.layout = build_group_layout(cls.config)

    def test_takeoff_commands_use_world_z_and_action_speed_limit(self):
        commands = velocity_commands(
            self.layout.ground_positions_m,
            self.layout.takeoff_positions_m,
            position_gain_s_inv=self.config.position_gain_s_inv,
            max_speed_m_s=self.config.max_speed_m_s,
        )
        actions = velocity_actions(commands, self.config.max_speed_m_s)
        for command, action in zip(commands, actions, strict=True):
            self.assertAlmostEqual(command[0], 0.0)
            self.assertAlmostEqual(command[1], 0.0)
            self.assertAlmostEqual(command[2], 0.5)
            self.assertEqual(action, (0.0, 0.0, 1.0, 1.0))
            self.assertEqual(direction_speed(action, 0.5), command)
        formation_commands = velocity_commands(
            self.layout.takeoff_positions_m,
            self.layout.assigned_target_positions_m,
            position_gain_s_inv=self.config.position_gain_s_inv,
            max_speed_m_s=self.config.max_speed_m_s,
        )
        self.assertTrue(all(math.dist(command, (0, 0, 0)) <= 0.5 for command in formation_commands))

    def test_reference_arms_use_same_source_and_no_checkpoint(self):
        common = dict(
            source=Path("/source"),
            run=Path("/output"),
            image_id="sha256:test",
            gpu=0,
            seed=41,
            num_envs=4,
        )
        baseline = reference_command(["docker"], **common, arm="baseline")
        extended = reference_command(["docker"], **common, arm="extended")
        for command in (baseline, extended):
            self.assertIn("type=bind,src=/source,dst=/source,readonly", command)
            self.assertEqual(command[command.index("--scenario") + 1], "reference")
            self.assertNotIn("--checkpoint-directory", command)
        self.assertNotIn("--evaluation-timing-config", baseline)
        self.assertEqual(
            extended[-2:], ["--evaluation-timing-config", "/output/timing-config.json"]
        )


if __name__ == "__main__":
    unittest.main()
