"""A feasible eight-drone reference before any learned-policy size claim."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from align.formations.transition import linear_path_clearance
from align.simulation.multi_drone_contract import MultiDroneConfig, build_group_layout
from align.tasks.observation import ObservationConfig, build_observations


class EightDroneReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "configs/eight-drone-reference-plane.json"
        cls.config = MultiDroneConfig.from_dict(json.loads(path.read_text()))
        cls.layout = build_group_layout(cls.config)

    def test_launch_and_destination_have_nominal_clearance(self):
        cfg = self.config
        layout = self.layout
        self.assertEqual(cfg.num_agents, 8)
        self.assertTrue(
            all(
                abs(x) < cfg.safety_xy_limit_m and abs(y) < cfg.safety_xy_limit_m
                for x, y, _ in (*layout.ground_positions_m, *layout.assigned_target_positions_m)
            )
        )
        clearance, _, _ = linear_path_clearance(
            layout.takeoff_positions_m, layout.assigned_target_positions_m
        )
        self.assertGreaterEqual(clearance, cfg.minimum_separation_m)
        self.assertAlmostEqual(clearance, math.sqrt(0.5), places=6)

    def test_deadline_covers_ideal_saturated_proportional_motion_and_dwell(self):
        cfg = self.config
        layout = self.layout
        longest = max(
            math.dist(start, end)
            for start, end in zip(
                layout.takeoff_positions_m, layout.assigned_target_positions_m, strict=True
            )
        )
        switch = cfg.max_speed_m_s / cfg.position_gain_s_inv
        ideal_seconds = (
            max(0.0, longest - switch) / cfg.max_speed_m_s
            + math.log(switch / cfg.formation_rmse_tolerance_m) / cfg.position_gain_s_inv
        )
        self.assertLess(ideal_seconds + cfg.dwell_seconds, cfg.formation_timeout_seconds)

    def test_local_actor_shape_stays_fixed_at_eight(self):
        cfg = self.config
        layout = self.layout
        observation = ObservationConfig()
        positions = layout.ground_positions_m
        velocities = ((0.0, 0.0, 0.0),) * cfg.num_agents
        batch = build_observations(
            layout.agent_ids, positions, velocities, layout.assigned_target_positions_m, observation
        )
        self.assertEqual((observation.actor_dimension, observation.critic_dimension), (55, 80))
        self.assertEqual(len(batch.actors), cfg.num_agents)
        self.assertTrue(all(len(actor.flat()) == 55 for actor in batch.actors))
        self.assertEqual(len(batch.critic.flat()), 80)


if __name__ == "__main__":
    unittest.main()
