"""Exact linear-path clearance and fixed-identity shape-command contracts."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from align.formations.downwash import vertical_downwash_exposure
from align.formations.transition import (
    ShapeTransitionConfig,
    assign_transition_slots,
    linear_path_clearance,
)
from align.formations.transition_report import make_transition_report
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig


class ShapeTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.construction = MultiDroneConfig.from_dict(
            json.loads((root / "configs/feasible-plane-construction.json").read_text())
        )
        cls.task = TaskEnvironmentConfig.from_dict(
            json.loads((root / "configs/shape-transition-task.json").read_text())
        )
        cls.command = ShapeTransitionConfig.from_dict(
            json.loads((root / "configs/shape-transition-plane-pyramid.json").read_text())
        )

    def test_analytic_clearance_finds_midpoint_crossing(self):
        distance, alpha, pair = linear_path_clearance(
            ((-1, 0, 1), (1, 0, 1)), ((1, 0, 1), (-1, 0, 1))
        )
        self.assertEqual(distance, 0.0)
        self.assertEqual(alpha, 0.5)
        self.assertEqual(pair, (0, 1))

    def test_assignment_avoids_naive_plane_to_pyramid_crossing(self):
        report = make_transition_report(self.construction, self.task, self.command)
        assignment = report["assignment"]
        self.assertEqual(report["command_step"], 2600)
        self.assertEqual(report["source_kind"], "plane")
        self.assertEqual(report["destination_kind"], "pyramid")
        self.assertTrue(report["formation_start_step"] < report["command_step"])
        self.assertTrue(report["command_step"] < report["latest_final_dwell_start_step"])
        self.assertAlmostEqual(
            report["naive_ground_assigned_minimum_interpolated_separation_m"], 0.0
        )
        self.assertEqual(sorted(assignment["slot_for_agent"]), list(range(4)))
        self.assertEqual(assignment["searched_assignments"], math.factorial(4))
        self.assertGreater(assignment["feasible_assignments"], 0)
        self.assertGreaterEqual(assignment["minimum_interpolated_separation_m"], 0.55)
        self.assertEqual(
            set(map(tuple, assignment["assigned_destination_m"])),
            set(map(tuple, report["destination_slots_m"])),
        )
        self.assertGreater(report["speed_limited_transition_lower_bound_s"], 0)
        self.assertLess(
            report["speed_limited_transition_lower_bound_s"]
            + self.task.success_dwell_steps * self.construction.physics_dt,
            report["post_command_time_available_s"],
        )
        self.assertFalse(report["physical_transition_tested"])

    def test_raised_pyramid_preserves_base_height_and_report_center(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs/shape-transition-plane-pyramid-raised.json"
        )
        command = ShapeTransitionConfig.from_dict(json.loads(path.read_text()))
        self.assertEqual(command.schema_version, 2)
        self.assertEqual(command.to_dict()["destination_center_m"], [0.0, 0.0, 1.75])
        report = make_transition_report(self.construction, self.task, command)
        self.assertEqual(report["destination_center_m"], (0.0, 0.0, 1.75))
        assignment = report["assignment"]
        self.assertGreaterEqual(assignment["minimum_interpolated_separation_m"], 0.55)
        self.assertGreaterEqual(
            min(point[2] for point in assignment["assigned_destination_m"]),
            min(point[2] for point in report["source_positions_m"]),
        )
        self.assertAlmostEqual(
            report["naive_ground_assigned_minimum_interpolated_separation_m"], 0.0
        )

    def test_wide_pyramid_reduces_nominal_downwash_exposure(self):
        root = Path(__file__).resolve().parents[1]
        raised = ShapeTransitionConfig.from_dict(
            json.loads((root / "configs/shape-transition-plane-pyramid-raised.json").read_text())
        )
        wide = ShapeTransitionConfig.from_dict(
            json.loads((root / "configs/shape-transition-plane-pyramid-wide.json").read_text())
        )
        self.assertEqual(wide.to_dict()["destination_horizontal_scale"], 2.0)
        raised_report = make_transition_report(self.construction, self.task, raised)
        wide_report = make_transition_report(self.construction, self.task, wide)
        self.assertGreaterEqual(
            wide_report["assignment"]["minimum_interpolated_separation_m"], 0.55
        )
        self.assertLess(
            wide_report["nominal_downwash_proxy"]["maximum_nominal_exposure_fraction"],
            raised_report["nominal_downwash_proxy"]["maximum_nominal_exposure_fraction"] / 10,
        )
        self.assertFalse(wide_report["nominal_downwash_proxy"]["physical_force_measured"])
        with self.assertRaisesRegex(ValueError, "horizontal scale"):
            ShapeTransitionConfig.from_dict({**wide.to_dict(), "destination_horizontal_scale": 0.5})

    def test_wide_transition_leaves_time_for_long_hold_check(self):
        root = Path(__file__).resolve().parents[1]
        task = TaskEnvironmentConfig.from_dict(
            json.loads((root / "configs/shape-transition-hold-task.json").read_text())
        )
        command = ShapeTransitionConfig.from_dict(
            json.loads((root / "configs/shape-transition-plane-pyramid-wide.json").read_text())
        )
        report = make_transition_report(self.construction, task, command)
        self.assertEqual(task.success_dwell_steps, 1000)
        self.assertGreater(report["latest_final_dwell_start_step"], report["command_step"])
        self.assertGreater(report["post_command_time_available_s"], 10.0)

    def test_long_hold_gain_probe_changes_only_reference_position_gain(self):
        root = Path(__file__).resolve().parents[1]
        new = MultiDroneConfig.from_dict(
            json.loads((root / "configs/shape-transition-gain-1p2-construction.json").read_text())
        )
        differing = {
            name
            for name, value in self.construction.to_dict().items()
            if new.to_dict()[name] != value
        }
        self.assertEqual(differing, {"position_gain_s_inv"})
        self.assertEqual(new.position_gain_s_inv, 1.2)

    def test_downwash_proxy_is_directional_and_zero_at_equal_heights(self):
        self.assertEqual(vertical_downwash_exposure(((0, 0, 1), (0, 0, 1))), (0, 0))
        exposure = vertical_downwash_exposure(((0, 0, 1), (0, 0, 2)))
        self.assertGreater(exposure[0], 0)
        self.assertEqual(exposure[1], 0)
        self.assertLess(vertical_downwash_exposure(((0, 0, 1), (1, 0, 2)))[0], exposure[0])

    def test_rejects_invalid_margin_or_deadline(self):
        with self.assertRaisesRegex(ValueError, "planned margin"):
            assign_transition_slots(
                ((0, 0, 0), (1, 0, 0)),
                ((0, 0, 0), (1, 0, 0)),
                minimum_planned_separation_m=1.1,
            )
        value = {**self.command.to_dict(), "command_step": 4700}
        with self.assertRaisesRegex(ValueError, "final dwell window"):
            make_transition_report(
                self.construction, self.task, ShapeTransitionConfig.from_dict(value)
            )

    def test_rejects_search_above_declared_limit(self):
        with self.assertRaisesRegex(ValueError, "search agent limit"):
            assign_transition_slots(
                ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                minimum_planned_separation_m=0.5,
                maximum_search_agents=2,
            )


if __name__ == "__main__":
    unittest.main()
