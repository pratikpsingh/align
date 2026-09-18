"""Meaningful host-only checks for frozen-policy telemetry auditing."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from align.runtime.policy_telemetry_runtime import (
    replay_command,
    valid_early_contact_capture,
)
from align.tasks.policy_telemetry import (
    TELEMETRY_COLUMNS,
    audit_airborne_contacts,
    audit_telemetry,
    summarize_telemetry,
)


class PolicyTelemetryTests(unittest.TestCase):
    def test_recomputes_rewards_geometry_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation.csv"
            telemetry = root / "policy-telemetry.csv"
            with evaluation.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "evaluation_step",
                        "env_id",
                        "episode_step",
                        "phase",
                        "formation_kind",
                        "team_reward",
                        "assigned_rmse_m",
                        "pairwise_rmse_m",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "evaluation_step": 0,
                        "env_id": 0,
                        "episode_step": 1,
                        "phase": "formation",
                        "formation_kind": "plane",
                        "team_reward": -0.01,
                        "assigned_rmse_m": 2**-0.5,
                        "pairwise_rmse_m": 1.0,
                    }
                )
            rows = []
            for agent_id in range(2):
                row = dict.fromkeys(TELEMETRY_COLUMNS, 0.0)
                row.update(
                    evaluation_step=0,
                    env_id=0,
                    episode_step=1,
                    phase="formation",
                    formation_kind="plane",
                    agent_id=agent_id,
                    qw=1.0,
                    x_m=2.0 * agent_id,
                    target_x_m=float(agent_id),
                    pre_x_m=2.0 * agent_id,
                    action_0=1.0,
                    action_3=0.5,
                    command_vx_m_s=0.25,
                    vx_m_s=0.1,
                    raw_formation=-1.0,
                    weighted_formation=-0.01,
                    agent_reward=-0.01,
                    team_reward=-0.01,
                )
                rows.append(row)
            with telemetry.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TELEMETRY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            result = audit_telemetry(
                telemetry,
                evaluation,
                num_agents=2,
                max_speed_m_s=0.5,
                reward_config={
                    "formation_weight": 1.0,
                    "control_dt_seconds": 0.01,
                    **{
                        f"{name}_weight": 1.0
                        for name in (
                            "tracking",
                            "progress",
                            "separation",
                            "contact",
                            "settling",
                            "smoothness",
                            "effort",
                        )
                    },
                },
            )
            self.assertEqual(result["drone_rows"], 2)
            self.assertEqual(result["target_aligned_command_fraction"], 0.0)
            self.assertEqual(result["positive_velocity_response_fraction"], 1.0)
            rows[0]["weighted_formation"] = -0.02
            with telemetry.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TELEMETRY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "reward weighting"):
                audit_telemetry(
                    telemetry,
                    evaluation,
                    num_agents=2,
                    max_speed_m_s=0.5,
                    reward_config={
                        "formation_weight": 1.0,
                        "control_dt_seconds": 0.01,
                        **{
                            f"{name}_weight": 1.0
                            for name in (
                                "tracking",
                                "progress",
                                "separation",
                                "contact",
                                "settling",
                                "smoothness",
                                "effort",
                            )
                        },
                    },
                )

    def test_dwell_deadline_exceeds_commanded_altitude_reach(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = Path(directory) / "policy-telemetry.csv"
            with telemetry.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TELEMETRY_COLUMNS)
                writer.writeheader()
                for agent_id in range(2):
                    row = dict.fromkeys(TELEMETRY_COLUMNS, 0.0)
                    row.update(
                        evaluation_step=0,
                        env_id=0,
                        episode_step=551,
                        phase="formation",
                        formation_kind="plane",
                        agent_id=agent_id,
                        z_m=0.2,
                        target_z_m=1.5,
                    )
                    writer.writerow(row)
            result = summarize_telemetry(
                telemetry,
                max_speed_m_s=0.5,
                control_dt_seconds=0.01,
                max_episode_steps=800,
                success_dwell_steps=200,
            )
            self.assertEqual(result["latest_dwell_start_episode_step"], 601)
            self.assertEqual(result["remaining_command_steps_at_first_formation"][0], 50)
            self.assertAlmostEqual(result["maximum_commanded_travel_before_dwell_m"][0], 0.25)
            self.assertAlmostEqual(
                result["altitude_only_assigned_rmse_lower_bound_at_dwell_m"][0], 1.05
            )

    def test_airborne_contact_audit_preserves_per_world_reset_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation.csv"
            telemetry = root / "policy-telemetry.csv"
            states = {
                0: [
                    (1, "ground", 0.06, 0.10, 0, False),
                    (2, "takeoff", 0.30, 0.00, 0, False),
                    (3, "takeoff", 0.06, 0.50, 0, False),
                    (4, "formation", 0.06, 0.50, 3, True),
                    (1, "ground", 0.06, 0.10, 0, False),
                ],
                1: [
                    (1, "ground", 0.06, 0.10, 0, False),
                    (2, "takeoff", 0.30, 0.00, 0, False),
                    (3, "takeoff", 0.30, 0.00, 0, False),
                    (4, "takeoff", 0.30, 0.00, 0, False),
                    (5, "takeoff", 0.06, 0.40, 0, False),
                ],
            }
            with (
                evaluation.open("w", newline="", encoding="utf-8") as eval_stream,
                telemetry.open("w", newline="", encoding="utf-8") as drone_stream,
            ):
                eval_writer = csv.DictWriter(
                    eval_stream,
                    fieldnames=(
                        "evaluation_step",
                        "env_id",
                        "episode_step",
                        "phase",
                        "reason_code",
                        "terminated",
                    ),
                )
                drone_writer = csv.DictWriter(drone_stream, fieldnames=TELEMETRY_COLUMNS)
                eval_writer.writeheader()
                drone_writer.writeheader()
                for evaluation_step in range(5):
                    for env_id in range(2):
                        step, phase, height, force, reason, terminated = states[env_id][
                            evaluation_step
                        ]
                        eval_writer.writerow(
                            {
                                "evaluation_step": evaluation_step,
                                "env_id": env_id,
                                "episode_step": step,
                                "phase": phase,
                                "reason_code": reason,
                                "terminated": terminated,
                            }
                        )
                        for agent_id in range(2):
                            row = dict.fromkeys(TELEMETRY_COLUMNS, 0)
                            row.update(
                                evaluation_step=evaluation_step,
                                env_id=env_id,
                                episode_step=step,
                                phase=phase,
                                formation_kind="plane",
                                agent_id=agent_id,
                                z_m=height if agent_id == 0 else 0.06,
                                contact_force_n=force if agent_id == 0 else 0.0,
                                command_vz_m_s=0.2,
                            )
                            drone_writer.writerow(row)
            result = audit_airborne_contacts(
                telemetry,
                evaluation,
                airborne_height_m=0.25,
                contact_force_threshold_n=0.01,
            )
            self.assertEqual(result["first_post_airborne_contact_count"], 2)
            first, second = result["first_post_airborne_contacts"]
            self.assertEqual(
                (first["env_id"], first["episode_index"], first["agent_id"]),
                (0, 0, 0),
            )
            self.assertEqual(first["first_airborne_episode_step"], 2)
            self.assertEqual(first["first_contact_episode_step"], 3)
            self.assertEqual(first["reason_code_at_contact"], 0)
            self.assertEqual(
                (second["env_id"], second["episode_index"], second["first_contact_episode_step"]),
                (1, 0, 5),
            )

    def test_preformation_safety_replay_can_be_analyzed(self):
        probe = {
            "status": "failed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
            "deterministic_evaluation_tested": True,
        }
        metrics = {
            "status": "failed",
            "formation_phase_reached": False,
            "outcome_counts": {"3": 4},
            "checks": {
                "all_declared_templates_reached_formation_phase": False,
                "raw_row_count_is_exact": True,
            },
        }
        self.assertTrue(valid_early_contact_capture(1, probe, metrics))
        self.assertTrue(valid_early_contact_capture(0, probe, metrics))
        probe["error"] = "unexpected simulator exception"
        self.assertFalse(valid_early_contact_capture(0, probe, metrics))
        probe.pop("error")
        metrics["checks"]["raw_row_count_is_exact"] = False
        self.assertFalse(valid_early_contact_capture(1, probe, metrics))
        metrics["checks"]["raw_row_count_is_exact"] = True
        metrics["outcome_counts"]["3"] = 0
        self.assertFalse(valid_early_contact_capture(1, probe, metrics))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy-telemetry.csv"
            row = dict.fromkeys(TELEMETRY_COLUMNS, 0)
            row.update(
                evaluation_step=0,
                env_id=0,
                episode_step=1,
                phase="takeoff",
                formation_kind="plane",
                agent_id=0,
            )
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TELEMETRY_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
            summary = summarize_telemetry(
                path,
                max_speed_m_s=0.5,
                control_dt_seconds=0.01,
                max_episode_steps=100,
                success_dwell_steps=10,
            )
            self.assertFalse(summary["formation_phase_reached"])
            self.assertEqual(summary["remaining_command_steps_at_first_formation"], {})

    def test_source_checkpoint_is_read_only_and_trace_is_explicit(self):
        command = replay_command(
            ["docker"],
            source=Path("/source-run"),
            run=Path("/new-run"),
            image_id="sha256:test",
            gpu=0,
            seed=41,
            update=12,
            num_envs=4,
            source_identity="source",
        )
        self.assertIn("type=bind,src=/source-run,dst=/source,readonly", command)
        self.assertIn("--policy-telemetry", command)
        self.assertEqual(
            command[command.index("--checkpoint-directory") + 1],
            "/source/seed-0000000041/checkpoints",
        )
        self.assertEqual(command[command.index("--scenario") + 1], "evaluation")


if __name__ == "__main__":
    unittest.main()
