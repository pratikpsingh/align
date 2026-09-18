"""Waypoint plan, fixed identities, and all-drone advancement gates."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.waypoints import WaypointProgress, WaypointRouteConfig, make_waypoint_plan


class WaypointRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "configs"
        cls.construction = MultiDroneConfig.from_dict(
            json.loads((root / "feasible-plane-construction.json").read_text())
        )
        cls.task = TaskEnvironmentConfig.from_dict(
            json.loads((root / "shape-transition-task.json").read_text())
        )
        cls.route = WaypointRouteConfig.from_dict(
            json.loads((root / "waypoint-route-3m.json").read_text())
        )
        cls.direct = WaypointRouteConfig.from_dict(
            json.loads((root / "waypoint-route-3m-direct.json").read_text())
        )

    def test_route_preserves_assignment_and_reaches_exact_goal(self):
        plan = make_waypoint_plan(self.construction, self.task, self.route)
        self.assertEqual(len(plan.centers_m), 4)
        self.assertEqual(plan.centers_m[-1], plan.goal_center_m)
        self.assertEqual(plan.goal_center_m, (3.0, 0.0, 1.5))
        self.assertAlmostEqual(plan.longest_leg_m, 0.75)
        for center, leg in zip(plan.centers_m, plan.assigned_targets_m, strict=True):
            offsets = tuple(tuple(point[i] - center[i] for i in range(3)) for point in leg)
            self.assertEqual(
                offsets, ((-0.5, -0.5, 0.0), (-0.5, 0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0))
            )
        self.assertLess(
            plan.ideal_speed_limited_lower_bound_s + plan.arrival_dwell_lower_bound_s,
            plan.available_time_s,
        )

    def test_direct_arm_changes_only_maximum_leg_length(self):
        changed = {
            key
            for key, value in self.route.to_dict().items()
            if self.direct.to_dict()[key] != value
        }
        self.assertEqual(changed, {"maximum_leg_length_m"})
        self.assertEqual(
            len(make_waypoint_plan(self.construction, self.task, self.direct).centers_m), 1
        )

    def test_group_gate_requires_every_drone_and_resets_dwell(self):
        plan = make_waypoint_plan(self.construction, self.task, self.route)
        progress = WaypointProgress(plan, self.route)
        still = tuple((0.0, 0.0, 0.0) for _ in range(self.construction.num_agents))
        far = list(progress.current_targets_m)
        far[0] = (far[0][0] + 0.2, far[0][1], far[0][2])
        for _ in range(self.route.arrival_dwell_steps - 1):
            self.assertFalse(progress.observe(progress.current_targets_m, still)["advanced"])
        self.assertEqual(progress.dwell_steps, self.route.arrival_dwell_steps - 1)
        self.assertFalse(progress.observe(tuple(far), still)["settled"])
        self.assertEqual(progress.dwell_steps, 0)
        for index in range(len(plan.centers_m)):
            targets = progress.current_targets_m
            for _ in range(self.route.arrival_dwell_steps):
                progress.observe(targets, still)
            self.assertEqual(progress.index, min(index + 1, len(plan.centers_m) - 1))
        self.assertTrue(progress.complete)
        progress.reset()
        self.assertEqual(progress.index, 0)
        self.assertFalse(progress.complete)
        self.assertEqual(progress.dwell_steps, 0)

    def test_group_gate_rejects_close_drone_pair_or_excess_speed(self):
        plan = make_waypoint_plan(self.construction, self.task, self.route)
        progress = WaypointProgress(plan, self.route)
        positions = list(progress.current_targets_m)
        positions[1] = positions[0]
        still = tuple((0.0, 0.0, 0.0) for _ in positions)
        unsafe = progress.observe(tuple(positions), still)
        self.assertFalse(unsafe["settled"])
        self.assertEqual(unsafe["minimum_separation_m"], 0.0)
        moving = ((0.3, 0.0, 0.0), *still[1:])
        too_fast = progress.observe(progress.current_targets_m, moving)
        self.assertFalse(too_fast["settled"])
        self.assertGreater(too_fast["maximum_speed_m_s"], self.route.maximum_arrival_speed_m_s)

    def test_nonfinite_displacement_length_is_rejected(self):
        values = {**self.route.to_dict(), "goal_displacement_m": [1.79e308, 1.79e308, 1.79e308]}
        with self.assertRaisesRegex(ValueError, "displacement length"):
            WaypointRouteConfig.from_dict(values)

    def test_envelope_and_impossible_deadline_are_rejected(self):
        values = {**self.route.to_dict(), "goal_displacement_m": [5.0, 0.0, 0.0]}
        with self.assertRaisesRegex(ValueError, "safety envelope"):
            make_waypoint_plan(self.construction, self.task, WaypointRouteConfig.from_dict(values))
        values = {**self.route.to_dict(), "goal_displacement_m": [4.0, 0.0, 0.0]}
        short = TaskEnvironmentConfig.from_dict({**self.task.to_dict(), "max_episode_steps": 2650})
        with self.assertRaisesRegex(ValueError, "deadline"):
            make_waypoint_plan(self.construction, short, WaypointRouteConfig.from_dict(values))


if __name__ == "__main__":
    unittest.main()
