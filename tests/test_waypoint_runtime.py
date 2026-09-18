"""Raw waypoint-route audit and launcher contracts without simulator imports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from align.runtime.waypoint_runtime import waypoint_command
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.waypoint_audit import audit_waypoint_route
from align.tasks.waypoints import WaypointRouteConfig, make_waypoint_plan


class WaypointRuntimeTests(unittest.TestCase):
    def test_command_mounts_route_and_uses_reference_control(self):
        command = waypoint_command(
            ["docker"], run=Path("/run/example"), image_id="sha256:test", gpu=0, num_envs=4
        )
        self.assertEqual(
            command[command.index("--waypoint-route-config") + 1],
            "/output/waypoint-route-config.json",
        )
        self.assertEqual(command[command.index("--scenario") + 1], "reference")
        self.assertEqual(command[command.index("--gpus") + 1], "device=0")
        self.assertNotIn("--checkpoint-directory", command)

    def test_audit_requires_command_gate_and_exact_fixed_id_targets(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        construction = MultiDroneConfig.from_dict(
            json.loads((configs / "feasible-plane-construction.json").read_text())
        )
        task = TaskEnvironmentConfig.from_dict(
            json.loads((configs / "shape-transition-task.json").read_text())
        )
        direct = WaypointRouteConfig.from_dict(
            json.loads((configs / "waypoint-route-3m-direct.json").read_text())
        )
        plan = make_waypoint_plan(construction, task, direct)
        route = replace(direct, command_step=1, arrival_dwell_steps=2)
        plan = replace(plan, command_step=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation.csv"
            progress = root / "waypoint-progress.csv"
            telemetry = root / "policy-telemetry.csv"
            with evaluation.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "env_id",
                        "episode_step",
                        "reason_code",
                        "formation_kind",
                        "pairwise_rmse_m",
                        "assigned_rmse_m",
                    ),
                )
                writer.writeheader()
                for step in (1, 2, 3):
                    writer.writerow(
                        dict(
                            env_id=0,
                            episode_step=step,
                            reason_code=1 if step == 3 else 0,
                            formation_kind="plane",
                            pairwise_rmse_m=0.02,
                            assigned_rmse_m=0.02,
                        )
                    )
            with progress.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "env_id",
                        "episode_step",
                        "reason_code",
                        "active",
                        "index_before",
                        "index_after",
                        "settled",
                        "dwell_steps",
                        "advanced",
                        "complete",
                        "maximum_agent_error_m",
                        "minimum_separation_m",
                        "maximum_speed_m_s",
                    ),
                )
                writer.writeheader()
                for step in (1, 2, 3):
                    writer.writerow(
                        dict(
                            env_id=0,
                            episode_step=step,
                            reason_code=1 if step == 3 else 0,
                            active=step > 1,
                            index_before=0,
                            index_after=0,
                            settled=step > 1,
                            dwell_steps=1 if step == 2 else 0,
                            advanced=False,
                            complete=step == 3,
                            maximum_agent_error_m=0.02,
                            minimum_separation_m=1.0,
                            maximum_speed_m_s=0.01,
                        )
                    )
            fields = (
                "env_id",
                "episode_step",
                "agent_id",
                "target_x_m",
                "target_y_m",
                "target_z_m",
            )
            with telemetry.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for step in (2, 3):
                    for agent_id, target in enumerate(plan.assigned_targets_m[0]):
                        writer.writerow(
                            dict(
                                env_id=0,
                                episode_step=step,
                                agent_id=agent_id,
                                target_x_m=target[0],
                                target_y_m=target[1],
                                target_z_m=target[2],
                            )
                        )
            result = audit_waypoint_route(
                evaluation,
                telemetry,
                progress,
                plan,
                route,
                control_dt_seconds=0.01,
                formation_kind="plane",
            )
            self.assertEqual(result["success_count"], 1)
            self.assertEqual(result["route_completion_count"], 1)
            self.assertEqual(result["episodes"][0]["leg_completion_steps"], [3])
            with telemetry.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    dict(
                        env_id=0,
                        episode_step=2,
                        agent_id=0,
                        target_x_m=999,
                        target_y_m=0,
                        target_z_m=1.5,
                    )
                )
            with self.assertRaisesRegex(ValueError, "target differs"):
                audit_waypoint_route(
                    evaluation,
                    telemetry,
                    progress,
                    plan,
                    route,
                    control_dt_seconds=0.01,
                    formation_kind="plane",
                )


if __name__ == "__main__":
    unittest.main()
