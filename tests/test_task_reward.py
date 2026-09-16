"""Analytical checks for the simulator-independent formation-task reward."""

import math
import unittest

from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.reward import RewardConfig, RewardMemory, compute_step_reward
from align.tasks.reward_report import audit_rows


def square_state():
    positions = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    velocities = ((0.0, 0.0, 0.0),) * 4
    actions = ((0.0, 0.0, 0.0, 0.0),) * 4
    contacts = (0.0,) * 4
    airborne = (True,) * 4
    return positions, velocities, actions, contacts, airborne


class RewardConfigurationTests(unittest.TestCase):
    def test_formation_weight_must_be_active(self):
        with self.assertRaisesRegex(ValueError, "formation_weight must be positive"):
            RewardConfig(formation_weight=0.0)

    def test_config_dict_requires_exact_schema(self):
        values = RewardConfig().to_dict()
        self.assertEqual(RewardConfig.from_dict(values), RewardConfig())
        values["invented"] = 1.0
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            RewardConfig.from_dict(values)


class RewardMathematicsTests(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig()
        self.positions, self.velocities, self.actions, self.contacts, self.airborne = square_state()

    def compute(
        self,
        *,
        positions=None,
        targets=None,
        velocities=None,
        actions=None,
        contacts=None,
        airborne=None,
        memory=None,
    ):
        return compute_step_reward(
            positions or self.positions,
            targets or self.positions,
            velocities or self.velocities,
            actions or self.actions,
            contacts or self.contacts,
            airborne or self.airborne,
            self.config,
            memory,
        )

    def test_perfect_stationary_formation_has_zero_reward(self):
        result, memory = self.compute()
        self.assertEqual(result.team_reward, 0.0)
        self.assertTrue(all(value == 0.0 for value in result.total_by_agent))
        self.assertEqual(memory.target_distances_m, (0.0,) * 4)

    def test_translation_preserves_shape_but_not_target_tracking(self):
        translated = tuple((x + 0.5, y, z) for x, y, z in self.positions)
        result, _ = self.compute(positions=translated, targets=self.positions)
        self.assertTrue(all(item.formation == 0.0 for item in result.raw_by_agent))
        self.assertTrue(all(math.isclose(item.tracking, -0.25) for item in result.raw_by_agent))
        self.assertLess(result.team_reward, 0.0)

    def test_shape_distortion_is_mean_normalized_and_shared(self):
        distorted = list(self.positions)
        distorted[3] = (1.5, 1.0, 1.0)
        result, _ = self.compute(positions=tuple(distorted), targets=self.positions)
        values = {item.formation for item in result.raw_by_agent}
        self.assertEqual(len(values), 1)
        self.assertLess(values.pop(), 0.0)

    def test_safety_margin_and_airborne_contact_are_separate(self):
        close = list(self.positions)
        close[1] = (0.275, 0.0, 1.0)
        result, _ = self.compute(positions=tuple(close))
        self.assertAlmostEqual(result.raw_by_agent[0].separation, -0.25)
        self.assertAlmostEqual(result.raw_by_agent[1].separation, -0.25)
        self.assertEqual(result.raw_by_agent[0].contact, 0.0)

        result, _ = self.compute(contacts=(1.0, 0.0, 0.0, 0.0))
        self.assertEqual(result.raw_by_agent[0].separation, 0.0)
        self.assertEqual(result.raw_by_agent[0].contact, -1.0)

        result, _ = self.compute(
            contacts=(1.0, 0.0, 0.0, 0.0),
            airborne=(False, True, True, True),
        )
        self.assertEqual(result.raw_by_agent[0].contact, 0.0)

    def test_progress_and_smoothness_use_explicit_reset_memory(self):
        previous = RewardMemory(
            target_distances_m=(1.0,) * 4,
            commanded_velocities_m_s=((0.5, 0.0, 0.0),) * 4,
        )
        result, _ = self.compute(memory=previous)
        self.assertTrue(all(item.progress == 1.0 for item in result.raw_by_agent))
        self.assertTrue(
            all(math.isclose(item.smoothness, -1.0 / 3.0) for item in result.raw_by_agent)
        )

        reset_result, _ = self.compute(memory=RewardMemory())
        self.assertTrue(all(item.progress == 0.0 for item in reset_result.raw_by_agent))
        self.assertTrue(all(item.smoothness == 0.0 for item in reset_result.raw_by_agent))

    def test_weighted_sum_and_team_mean_reconstruct_exactly(self):
        velocities = ((0.1, 0.0, 0.0),) * 4
        actions = ((1.0, 0.0, 0.0, 0.2),) * 4
        result, _ = self.compute(velocities=velocities, actions=actions)
        for weighted, total in zip(result.weighted_by_agent, result.total_by_agent, strict=True):
            self.assertEqual(weighted.total(), total)
        self.assertEqual(
            result.team_reward,
            math.fsum(result.total_by_agent) / len(result.total_by_agent),
        )

    def test_persistent_costs_are_scaled_by_control_timestep(self):
        translated = tuple((x + 0.5, y, z) for x, y, z in self.positions)
        first, _ = self.compute(positions=translated, targets=self.positions)
        slower_config = RewardConfig(control_dt_seconds=0.02)
        second, _ = compute_step_reward(
            translated,
            self.positions,
            self.velocities,
            self.actions,
            self.contacts,
            self.airborne,
            slower_config,
        )
        self.assertAlmostEqual(
            second.weighted_by_agent[0].tracking,
            2.0 * first.weighted_by_agent[0].tracking,
        )

    def test_nonfinite_and_wrong_shape_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.compute(actions=((0.0, 0.0, 0.0),) * 4)
        positions = list(self.positions)
        positions[0] = (math.nan, 0.0, 1.0)
        with self.assertRaises(ValueError):
            self.compute(positions=tuple(positions))
        with self.assertRaisesRegex(ValueError, "bounded"):
            self.compute(actions=((2.0, 0.0, 0.0, 1.0),) * 4)


class RewardAuditTests(unittest.TestCase):
    @staticmethod
    def trajectory_rows():
        xy = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        rows = []
        for step, (phase, height, target_height) in enumerate(
            (("ground", 0.06, 0.06), ("takeoff", 0.10, 1.0), ("takeoff", 0.20, 1.0))
        ):
            for agent, (x, y) in enumerate(xy):
                rows.append(
                    {
                        "repeat": 0,
                        "step": step,
                        "t": (step + 1) * 0.01,
                        "phase": phase,
                        "agent_id": agent,
                        "x": x,
                        "y": y,
                        "z": height,
                        "target_x": x,
                        "target_y": y,
                        "target_z": target_height,
                        "vx": 0.0,
                        "vy": 0.0,
                        "vz": 0.1 if phase == "takeoff" else 0.0,
                        "a0": 0.0,
                        "a1": 0.0,
                        "a2": 1.0 if phase == "takeoff" else 0.0,
                        "a3": 0.2 if phase == "takeoff" else 0.0,
                        "contact_force_n": 0.0,
                    }
                )
        return rows

    def test_phase_change_resets_progress_and_command_memory(self):
        report, rows = audit_rows(
            self.trajectory_rows(),
            MultiDroneConfig(),
            RewardConfig(),
        )
        step_one = [row for row in rows if row["step"] == 1]
        step_two = [row for row in rows if row["step"] == 2]
        self.assertTrue(all(row["raw_progress"] == 0.0 for row in step_one))
        self.assertTrue(all(math.isclose(row["raw_progress"], 0.1) for row in step_two))
        self.assertTrue(all(row["raw_smoothness"] == 0.0 for row in step_one))
        self.assertTrue(report["checks"]["control_timestep_matches_source"])

    def test_audit_detects_task_reward_contract_mismatch(self):
        report, _ = audit_rows(
            self.trajectory_rows(),
            MultiDroneConfig(),
            RewardConfig(control_dt_seconds=0.02, minimum_separation_m=0.6),
        )
        self.assertFalse(report["checks"]["control_timestep_matches_source"])
        self.assertFalse(report["checks"]["safety_thresholds_match_source"])


if __name__ == "__main__":
    unittest.main()
