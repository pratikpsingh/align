"""Host contracts for the explicit in-flight shape reference and raw target audit."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from align.runtime.shape_transition_runtime import transition_command
from align.tasks.shape_transition_audit import audit_shape_transition


class ShapeTransitionRuntimeTests(unittest.TestCase):
    def test_command_mounts_resolved_contract_and_declares_reference(self):
        command = transition_command(
            ["docker"],
            run=Path("/run/20260918T000000IST-test"),
            image_id="sha256:test",
            gpu=0,
            num_envs=4,
        )
        self.assertIn("--shape-transition-config", command)
        self.assertEqual(
            command[command.index("--shape-transition-config") + 1],
            "/output/transition-config.json",
        )
        self.assertEqual(command[command.index("--scenario") + 1], "reference")
        self.assertEqual(command[command.index("--gpus") + 1], "device=0")
        self.assertNotIn("--checkpoint-directory", command)

    def test_raw_audit_requires_a_real_first_episode_target_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation.csv"
            telemetry = root / "telemetry.csv"
            with evaluation.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "env_id",
                        "episode_step",
                        "formation_kind",
                        "reason_code",
                        "minimum_separation_m",
                        "assigned_rmse_m",
                    ),
                )
                writer.writeheader()
                for step in range(1, 5):
                    writer.writerow(
                        dict(
                            env_id=0,
                            episode_step=step,
                            formation_kind="plane" if step <= 2 else "pyramid",
                            reason_code=1 if step == 4 else 0,
                            minimum_separation_m=0.8,
                            assigned_rmse_m=0.2,
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
                for agent_id, x in enumerate((0.0, 1.0)):
                    writer.writerow(
                        dict(
                            env_id=0,
                            episode_step=3,
                            agent_id=agent_id,
                            target_x_m=x,
                            target_y_m=0,
                            target_z_m=1,
                        )
                    )
            plan = dict(
                command_step=2,
                control_dt_seconds=0.01,
                source_kind="plane",
                destination_kind="pyramid",
                assignment=dict(
                    assigned_destination_m=((0, 0, 1), (1, 0, 1)),
                    minimum_interpolated_separation_m=0.8,
                ),
            )
            result = audit_shape_transition(evaluation, telemetry, plan)
            self.assertEqual(result["success_count"], 1)
            self.assertEqual(result["safety_termination_count"], 0)
            self.assertEqual(result["episodes"][0]["time_from_command_to_outcome_s"], 0.02)
            with telemetry.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    dict(
                        env_id=0,
                        episode_step=3,
                        agent_id=0,
                        target_x_m=0,
                        target_y_m=0,
                        target_z_m=1,
                    )
                )
                writer.writerow(
                    dict(
                        env_id=0,
                        episode_step=3,
                        agent_id=1,
                        target_x_m=9,
                        target_y_m=0,
                        target_z_m=1,
                    )
                )
            with self.assertRaisesRegex(ValueError, "destination target differs"):
                audit_shape_transition(evaluation, telemetry, plan)


if __name__ == "__main__":
    unittest.main()
