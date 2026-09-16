"""Analytical checks for multi-drone construction; no simulator evidence."""

import copy
import csv
import json
import math
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.simulation.contract import direction_speed
from align.simulation.multi_drone import COLUMNS
from align.simulation.multi_drone_contract import (
    MultiDroneConfig,
    build_group_layout,
    evaluate,
    phase_at,
    targets_for_phase,
    velocity_actions,
    velocity_commands,
)
from align.simulation.multi_drone_report import assess_saved_run


def short_config():
    return MultiDroneConfig(
        ground_settle_seconds=0.02,
        takeoff_seconds=0.04,
        formation_timeout_seconds=0.08,
        dwell_seconds=0.02,
    )


def analytical_run(config):
    layout = build_group_layout(config)
    rows = []
    for repeat in range(config.repetitions):
        for step in range(config.maximum_steps):
            phase = phase_at(config, step)
            positions = targets_for_phase(layout, phase)
            contact = 1.0 if phase == "ground" else 0.0
            for agent, (position, target) in enumerate(zip(positions, positions, strict=True)):
                row = {
                    "repeat": repeat,
                    "step": step,
                    "t": (step + 1) * config.physics_dt,
                    "phase": phase,
                    "agent_id": agent,
                    "slot_index": layout.assignment.slot_for_agent[agent],
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                    "qw": 1.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vz": 0.0,
                    "wx": 0.0,
                    "wy": 0.0,
                    "wz": 0.0,
                    "target_x": target[0],
                    "target_y": target[1],
                    "target_z": target[2],
                    "target_vx": 0.0,
                    "target_vy": 0.0,
                    "target_vz": 0.0,
                    "a0": 0.0,
                    "a1": 0.0,
                    "a2": 0.0,
                    "a3": 0.0,
                    "raw_u0": 0.0,
                    "raw_u1": 0.0,
                    "raw_u2": 0.0,
                    "raw_u3": 0.0,
                    "u0": 0.0,
                    "u1": 0.0,
                    "u2": 0.0,
                    "u3": 0.0,
                    "rotor0": 0.0,
                    "rotor1": 0.0,
                    "rotor2": 0.0,
                    "rotor3": 0.0,
                    "contact_force_n": contact,
                }
                rows.append(row)
    resets = [{"repeat": repeat, "max_reset_error": 0.0} for repeat in range(config.repetitions)]
    episodes = [
        {
            "repeat": repeat,
            "steps": config.maximum_steps,
            "termination_reason": "success",
        }
        for repeat in range(config.repetitions)
    ]
    return rows, resets, episodes


class ConfigurationTests(unittest.TestCase):
    def test_schedule_uses_exact_steps_and_phases(self):
        config = short_config()
        self.assertEqual(
            (config.ground_steps, config.takeoff_steps, config.formation_steps),
            (2, 4, 8),
        )
        self.assertEqual(
            [phase_at(config, step) for step in range(config.maximum_steps)],
            ["ground"] * 2 + ["takeoff"] * 4 + ["formation"] * 8,
        )

    def test_invalid_or_unsupported_configuration_is_rejected(self):
        for changes in (
            {"num_agents": 1},
            {"formation_kind": "cube"},
            {"control_decimation": 2},
            {"dwell_seconds": 8.0},
            {"minimum_separation_m": 1.0},
            {"target_center_m": (0.0, 0.0, math.inf)},
            {"calibration_status": "frozen"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MultiDroneConfig(**changes)


class LayoutAndCommandTests(unittest.TestCase):
    def test_layout_has_stable_identity_assignment_and_safe_endpoints(self):
        config = short_config()
        layout = build_group_layout(config)
        self.assertEqual(layout.group_id, 0)
        self.assertEqual(layout.agent_ids, tuple(range(4)))
        self.assertEqual(sorted(layout.assignment.slot_for_agent), list(range(4)))
        self.assertTrue(
            all(point[2] == config.ground_height_m for point in layout.ground_positions_m)
        )
        self.assertTrue(
            all(point[2] == config.target_center_m[2] for point in layout.takeoff_positions_m)
        )
        for points, spacing in (
            (layout.ground_positions_m, config.ground_spacing_m),
            (layout.assigned_target_positions_m, config.target_spacing_m),
        ):
            nearest = min(
                math.dist(first, second)
                for index, first in enumerate(points)
                for second in points[index + 1 :]
            )
            self.assertAlmostEqual(nearest, spacing)

    def test_outer_loop_saturates_vector_norm_and_action_round_trips(self):
        commands = velocity_commands(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ((3.0, 4.0, 0.0), (1.0, 1.0, 1.0)),
            position_gain_s_inv=2.0,
            max_speed_m_s=0.5,
        )
        self.assertAlmostEqual(math.dist(commands[0], (0.0, 0.0, 0.0)), 0.5)
        self.assertEqual(commands[1], (0.0, 0.0, 0.0))
        actions = velocity_actions(commands, 0.5)
        for velocity, action in zip(commands, actions, strict=True):
            self.assertLessEqual(max(map(abs, action)), 1.0)
            decoded = direction_speed(action, 0.5)
            self.assertLess(math.dist(decoded, velocity), 1e-12)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = short_config()
        self.rows, self.resets, self.episodes = analytical_run(self.config)

    def test_complete_safe_construction_passes(self):
        result = evaluate(self.rows, self.resets, self.episodes, self.config)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))
        self.assertAlmostEqual(result["metrics"]["repeat/0"]["minimum_separation_m"], 1.0)

    def test_float32_saved_targets_and_commands_preserve_contract(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            for key in (
                "target_x",
                "target_y",
                "target_z",
                "target_vx",
                "target_vy",
                "target_vz",
                "a0",
                "a1",
                "a2",
                "a3",
            ):
                row[key] = struct.unpack("f", struct.pack("f", row[key]))[0]
        self.assertEqual(
            evaluate(rows, self.resets, self.episodes, self.config)["status"], "passed"
        )

    def test_airborne_contact_and_separation_are_distinct_failures(self):
        rows = copy.deepcopy(self.rows)
        airborne = next(row for row in rows if row["phase"] == "formation")
        airborne["contact_force_n"] = 1.0
        result = evaluate(rows, self.resets, self.episodes, self.config)
        self.assertFalse(result["checks"]["repeat/0/no_airborne_contact"])
        self.assertTrue(result["checks"]["repeat/0/separation"])

        rows = copy.deepcopy(self.rows)
        formation = [
            row
            for row in rows
            if row["repeat"] == 0
            and row["phase"] == "formation"
            and row["step"] == self.config.ground_steps + self.config.takeoff_steps
        ]
        for axis in "xyz":
            formation[1][axis] = formation[0][axis]
        result = evaluate(rows, self.resets, self.episodes, self.config)
        self.assertFalse(result["checks"]["repeat/0/separation"])
        self.assertTrue(result["checks"]["repeat/0/no_airborne_contact"])

    def test_timeout_or_reset_leakage_fails(self):
        episodes = copy.deepcopy(self.episodes)
        episodes[0]["termination_reason"] = "timeout"
        self.assertFalse(
            evaluate(self.rows, self.resets, episodes, self.config)["checks"][
                "repeat/0/termination_success"
            ]
        )

        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["repeat"] == 1:
                row["rotor0"] += 0.01
        result = evaluate(rows, self.resets, self.episodes, self.config)
        self.assertFalse(result["checks"]["reset_repeat"])
        self.assertFalse(result["checks"]["reset_repeat/rotor_state"])

    def test_schema_one_config_migrates_for_historical_recomputation(self):
        legacy = self.config.to_dict()
        legacy["schema_version"] = 1
        for key in tuple(legacy):
            if key.startswith("repeat_"):
                legacy.pop(key)
        legacy["repeat_trajectory_tolerance"] = 0.001
        migrated = MultiDroneConfig.from_dict(legacy)
        self.assertEqual(migrated.schema_version, 2)
        self.assertEqual(migrated.repeat_position_tolerance_m, 0.0002)

    def test_missing_agent_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.rows[:-1], self.resets, self.episodes, self.config)

    def test_saved_csv_is_recomputed_without_simulator_imports(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config.json").write_text(json.dumps(self.config.to_dict()))
            (run / "resets.json").write_text(json.dumps({"resets": self.resets}))
            (run / "episodes.json").write_text(json.dumps({"episodes": self.episodes}))
            with (run / "trajectory.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(self.rows)
            result, loaded_rows = assess_saved_run(run)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(loaded_rows), len(self.rows))


if __name__ == "__main__":
    unittest.main()
