"""Guarded frozen-policy telemetry preserves requested and applied commands."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from align.runtime.policy_telemetry_runtime import replay_command
from align.tasks.policy_guard_comparison import summarize_guard_applicability
from align.tasks.policy_telemetry import GUARDED_TELEMETRY_COLUMNS, audit_telemetry
from align.tasks.wake_guard import guard_focal_command


class GuardedPolicyTelemetryTests(unittest.TestCase):
    def test_guarded_command_and_flags_are_independently_recomputed(self):
        positions = ((-1.5, 0.0, 0.8), (-0.5, 0.0, 0.8), (0.5, 0.0, 0.7), (1.2, 0.0, 0.95))
        requested = ((0.1, 0.0, 0.14),) * 4
        guarded, details = guard_focal_command(positions, requested, 2)
        self.assertTrue(details["active"])
        magnitude = math.dist((0, 0, 0), requested[0])
        direction = tuple(value / magnitude for value in requested[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            evaluation = path / "evaluation.csv"
            telemetry = path / "policy-telemetry.csv"
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
                        "phase": "takeoff",
                        "formation_kind": "plane",
                        "team_reward": 0,
                        "assigned_rmse_m": 0,
                        "pairwise_rmse_m": 0,
                    }
                )
            rows = []
            for agent_id, position in enumerate(positions):
                row = dict.fromkeys(GUARDED_TELEMETRY_COLUMNS, 0.0)
                row.update(
                    evaluation_step=0,
                    env_id=0,
                    episode_step=1,
                    phase="takeoff",
                    formation_kind="plane",
                    agent_id=agent_id,
                    qw=1.0,
                    action_0=direction[0],
                    action_1=direction[1],
                    action_2=direction[2],
                    action_3=magnitude / 0.5,
                    wake_guard_active=int(agent_id == 2),
                )
                for axis, value in zip("xyz", position, strict=True):
                    row[f"pre_{axis}_m"] = value
                    row[f"{axis}_m"] = value
                    row[f"target_{axis}_m"] = value
                for axis, value in zip("xyz", requested[agent_id], strict=True):
                    row[f"requested_v{axis}_m_s"] = value
                command = guarded if agent_id == 2 else requested[agent_id]
                for axis, value in zip("xyz", command, strict=True):
                    row[f"command_v{axis}_m_s"] = value
                rows.append(row)

            def write_rows():
                with telemetry.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=GUARDED_TELEMETRY_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)

            write_rows()
            audit = audit_telemetry(
                telemetry, evaluation, num_agents=4, max_speed_m_s=0.5, wake_guard=True
            )
            self.assertEqual(audit["wake_guard_active_steps"], 1)
            self.assertEqual(audit["wake_guard_unresolved_steps"], 0)
            applicability = summarize_guard_applicability(telemetry)
            self.assertEqual(applicability["takeoff"]["candidate_active_steps"], 1)
            self.assertEqual(applicability["formation"]["environment_steps"], 0)
            rows[2]["command_vx_m_s"] = requested[2][0]
            write_rows()
            with self.assertRaisesRegex(ValueError, "action-to-command"):
                audit_telemetry(
                    telemetry, evaluation, num_agents=4, max_speed_m_s=0.5, wake_guard=True
                )
            rows[2]["command_vx_m_s"] = guarded[0]
            rows[2]["requested_vz_m_s"] = 0
            write_rows()
            with self.assertRaisesRegex(ValueError, "action-to-request"):
                audit_telemetry(
                    telemetry, evaluation, num_agents=4, max_speed_m_s=0.5, wake_guard=True
                )

    def test_launcher_flag_is_opt_in(self):
        kwargs = dict(
            source=Path("/tmp/source"),
            run=Path("/tmp/attempt"),
            image_id="sha256:identity",
            gpu=0,
            seed=41,
            update=4,
            num_envs=4,
            source_identity="source",
        )
        standard = replay_command(["docker"], **kwargs)
        guarded = replay_command(["docker"], **kwargs, wake_guard=True)
        self.assertNotIn("--wake-guard", standard)
        self.assertEqual(guarded.count("--wake-guard"), 1)


if __name__ == "__main__":
    unittest.main()
