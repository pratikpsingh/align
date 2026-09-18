"""Meaningful host-only checks for frozen-policy telemetry auditing."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from align.runtime.policy_telemetry_runtime import replay_command
from align.tasks.policy_telemetry import TELEMETRY_COLUMNS, audit_telemetry, summarize_telemetry


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
