"""Matched waypoint/direct report comparison guards."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from align.runtime.waypoint_comparison import compare_waypoint_reports
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.waypoints import WaypointRouteConfig, make_waypoint_plan


class WaypointComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configs = Path(__file__).resolve().parents[1] / "configs"
        construction = MultiDroneConfig.from_dict(
            json.loads((configs / "feasible-plane-construction.json").read_text())
        )
        task = TaskEnvironmentConfig.from_dict(
            json.loads((configs / "waypoint-task-48s.json").read_text())
        )
        cls.waypoint = WaypointRouteConfig.from_dict(
            json.loads((configs / "waypoint-route-3m.json").read_text())
        )
        cls.direct = WaypointRouteConfig.from_dict(
            json.loads((configs / "waypoint-route-3m-direct.json").read_text())
        )
        cls.waypoint_plan = make_waypoint_plan(construction, task, cls.waypoint)
        cls.direct_plan = make_waypoint_plan(construction, task, cls.direct)

    def _arm(self, route, plan, outcome_step):
        return {
            "status": "passed",
            "image_id": "sha256:matching",
            "config_sha256": "bundle-matching",
            "host_gpu_index": 0,
            "training_performed": False,
            "checkpoint_loaded": False,
            "input_sha256": {
                "construction": "same",
                "task": "same",
                "observation": "same",
                "reward": "same",
                "route": str(route.maximum_leg_length_m),
            },
            "route_config": route.to_dict(),
            "plan": plan.to_dict(),
            "audit": {
                "status": "passed",
                "leg_count": len(plan.centers_m),
                "episodes": [
                    {
                        "env_id": env_id,
                        "route_complete": True,
                        "success": True,
                        "terminal_reason_code": 1,
                        "time_from_command_to_outcome_s": outcome_step,
                        "minimum_post_command_separation_m": 0.8,
                        "post_command_pairwise_rmse_mean_m": 0.06,
                        "post_command_assigned_rmse_mean_m": 0.08,
                    }
                    for env_id in range(4)
                ],
            },
        }

    def test_only_leg_length_may_differ(self):
        waypoint = self._arm(self.waypoint, self.waypoint_plan, 10.0)
        direct = self._arm(self.direct, self.direct_plan, 8.0)
        result = compare_waypoint_reports(waypoint, direct)
        self.assertEqual(result["world_count"], 4)
        self.assertEqual(result["waypoint_legs"], 4)
        self.assertEqual(result["paired_rows"][0]["waypoint_minus_direct_outcome_time_s"], 2.0)
        changed = copy.deepcopy(direct)
        changed["route_config"]["arrival_radius_m"] = 0.5
        with self.assertRaisesRegex(ValueError, "beyond maximum_leg_length_m"):
            compare_waypoint_reports(waypoint, changed)
        changed = copy.deepcopy(direct)
        changed["image_id"] = "sha256:different"
        with self.assertRaisesRegex(ValueError, "different simulator images"):
            compare_waypoint_reports(waypoint, changed)


if __name__ == "__main__":
    unittest.main()
