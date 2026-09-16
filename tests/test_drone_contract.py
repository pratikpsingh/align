"""Analytical fixtures test reporting rules, never serve as flight evidence."""

import copy
import math
import unittest

from align.runtime.drone_runtime import valid_result
from align.simulation.contract import DroneCheckConfig, cases, command_at, direction_speed, evaluate


def analytical_rows(config):
    rows, resets = [], []
    for repeat in range(config.repetitions):
        for name, axis, yaw in cases():
            resets.append({"case": name, "repeat": repeat, "max_reset_error": 0.0})
            duration = (
                config.hover_seconds
                if axis is None
                else (config.settle_seconds + config.command_seconds + config.brake_seconds)
            )
            pos = [0.0, 0.0, config.initial_height_m]
            for step in range(round(duration / config.physics_dt)):
                _, velocity, active = command_at(config, axis, step)
                pos = [p + v * config.physics_dt for p, v in zip(pos, velocity, strict=True)]
                row = dict(
                    case=name,
                    repeat=repeat,
                    step=step,
                    t=(step + 1) * config.physics_dt,
                    command_active=int(active),
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                    vx=velocity[0],
                    vy=velocity[1],
                    vz=velocity[2],
                    qw=math.cos(yaw / 2),
                    qx=0.0,
                    qy=0.0,
                    qz=math.sin(yaw / 2),
                    wx=0.0,
                    wy=0.0,
                    wz=0.0,
                )
                row.update({f"u{i}": 0.0 for i in range(4)})
                row.update({f"rotor{i}": 0.5 for i in range(4)})
                rows.append(row)
    return rows, resets


class CommandTests(unittest.TestCase):
    def test_diagonal_speed_is_norm_bounded(self):
        v = direction_speed([2, 2, 2, -3], 0.5)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 0.5)
        self.assertTrue(all(x > 0 for x in v))

    def test_zero_direction_and_negative_speed_match_legacy_mapping(self):
        self.assertEqual(direction_speed([0, 0, 0, 1], 0.5), (0, 0, 0))
        self.assertEqual(direction_speed([0, -1, 0, -0.5], 0.5), (0, -0.25, 0))

    def test_nonfinite_and_invalid_dimensions_rejected(self):
        for value in ([0, 0, 0], [math.nan, 0, 0, 1], [0, 0, 0, math.inf]):
            with self.assertRaises(ValueError):
                direction_speed(value, 0.5)

    def test_timing_uses_exact_physics_intervals(self):
        config = DroneCheckConfig()
        active = [step for step in range(600) if command_at(config, 0, step)[2]]
        self.assertEqual(active, list(range(100, 400)))
        self.assertFalse(command_at(config, None, 200)[2])

    def test_invalid_configuration(self):
        for changes in (
            {"physics_dt": math.nan},
            {"physics_dt": 0.03},
            {"seed": -1},
            {"repetitions": 1},
            {"control_decimation": 2},
            {"command_speed_m_s": 1.0},
            {"brake_seconds": 1.005},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DroneCheckConfig(**changes)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DroneCheckConfig()
        cls.rows, cls.resets = analytical_rows(cls.config)

    def test_analytical_fixture_passes_reporting_rules(self):
        result = evaluate(self.rows, self.resets, self.config)
        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(result["metrics"]["x/0"]["axis_displacement_m"], 0.75)

    def test_wrong_axis_fails_even_with_finite_motion(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["case"] == "x":
                row["x"], row["y"] = row["y"], row["x"]
                row["vx"], row["vy"] = row["vy"], row["vx"]
        result = evaluate(rows, self.resets, self.config)
        self.assertFalse(result["checks"]["x/0/axis"])
        self.assertFalse(result["checks"]["x/0/cross_axis"])

    def test_controller_memory_leak_fails_repeat_comparison(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["case"] == "hover" and row["repeat"] == 1:
                row["rotor2"] += 0.1
        result = evaluate(rows, self.resets, self.config)
        self.assertFalse(result["checks"]["hover/reset_repeat"])

    def test_missing_or_reordered_samples_are_not_success(self):
        for rows in (self.rows[:-1], [self.rows[1], self.rows[0], *self.rows[2:]]):
            with self.assertRaises(ValueError):
                evaluate(rows, self.resets, self.config)

    def test_nonfinite_or_out_of_bounds_actuation(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["u0"] = 1.1
        self.assertFalse(
            evaluate(rows, self.resets, self.config)["checks"]["hover/0/actuator_bounds"]
        )
        rows[0]["u0"] = math.nan
        with self.assertRaises(ValueError):
            evaluate(rows, self.resets, self.config)

    def test_reset_pose_error_is_not_masked_by_good_flight(self):
        resets = copy.deepcopy(self.resets)
        resets[0]["max_reset_error"] = 0.01
        self.assertFalse(evaluate(self.rows, resets, self.config)["checks"]["reset_state"])

    def test_fast_shutdown_needs_both_physics_checks_and_zero_exit(self):
        probe = {"status": "passed", "phase": "before_close", "drone_physics_tested": True}
        metrics = {"status": "passed", "checks": {"axis": True}}
        self.assertTrue(valid_result(0, probe, metrics))
        self.assertFalse(valid_result(1, probe, metrics))
        self.assertFalse(valid_result(0, None, metrics))
        self.assertFalse(valid_result(0, probe, {"status": "passed", "checks": {}}))
        self.assertFalse(valid_result(0, probe, {"status": "failed", "checks": {"axis": False}}))


if __name__ == "__main__":
    unittest.main()
