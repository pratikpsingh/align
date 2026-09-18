"""CPU contracts for evaluation-only timing changes and matched replay commands."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from align.learning.stability_config import StabilityConfig
from align.runtime.policy_telemetry_runtime import replay_command
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.evaluation_timing import (
    EvaluationTimingConfig,
    apply_evaluation_timing,
    compare_shared_prefix,
    formation_switch_altitude,
    summarize_evaluation_phases,
)


class EvaluationTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "configs"
        cls.construction = MultiDroneConfig.from_dict(
            json.loads((root / "multi-drone-construction.json").read_text())
        )
        cls.task = TaskEnvironmentConfig.from_dict(
            json.loads((root / "recurrent-stability-task.json").read_text())
        )
        cls.stability = StabilityConfig.from_dict(
            json.loads((root / "training-stability.json").read_text())
        )

    def test_extends_only_phase_duration_and_episode_length(self):
        timing = EvaluationTimingConfig.from_dict(
            {"schema_version": 1, "takeoff_seconds": 15.0, "formation_seconds": 8.0}
        )
        construction, task, stability, details = apply_evaluation_timing(
            timing, self.construction, self.task, self.stability
        )
        self.assertEqual((construction.ground_steps, construction.takeoff_steps), (50, 1500))
        self.assertEqual((task.max_episode_steps, stability.evaluation_steps), (2350, 2350))
        self.assertEqual(task.success_dwell_steps, self.task.success_dwell_steps)
        self.assertEqual(construction.max_speed_m_s, self.construction.max_speed_m_s)
        self.assertEqual(self.task.max_episode_steps, 800)
        self.assertEqual(details["source_evaluation_steps"], 800)
        self.assertEqual(details["effective_formation_steps"], 800)

    def test_rejects_shortenings_and_nonintegral_steps(self):
        for takeoff, formation in ((4.0, 8.0), (15.0, 2.0), (15.001, 8.0)):
            with self.subTest(takeoff=takeoff, formation=formation):
                with self.assertRaises(ValueError):
                    apply_evaluation_timing(
                        EvaluationTimingConfig(1, takeoff, formation),
                        self.construction,
                        self.task,
                        self.stability,
                    )

    def test_matched_commands_change_only_output_and_timing_flag(self):
        common = dict(
            source=Path("/source"),
            run=Path("/output"),
            image_id="sha256:test",
            gpu=0,
            seed=41,
            update=12,
            num_envs=4,
            source_identity="source",
        )
        baseline = replay_command(["docker"], **common, output_label="baseline")
        extended = replay_command(
            ["docker"],
            **common,
            output_label="extended",
            timing_config="/output/timing-config.json",
        )
        self.assertNotIn("--evaluation-timing-config", baseline)
        self.assertEqual(
            extended[-2:], ["--evaluation-timing-config", "/output/timing-config.json"]
        )
        for command in (baseline, extended):
            self.assertIn("type=bind,src=/source,dst=/source,readonly", command)
            self.assertEqual(
                command[command.index("--checkpoint-directory") + 1],
                "/source/seed-0000000041/checkpoints",
            )
            self.assertIn("--policy-telemetry", command)

    def test_shared_prefix_and_switch_height_from_raw_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "evaluation_step",
                "env_id",
                "episode_step",
                "phase",
                "agent_id",
                "z_m",
                "target_z_m",
            )
            rows = [
                dict(
                    evaluation_step=step,
                    env_id=env_id,
                    episode_step=step + 1,
                    phase="takeoff",
                    agent_id=agent_id,
                    z_m=0.2,
                    target_z_m=1.5,
                )
                for step in range(2)
                for env_id in range(2)
                for agent_id in range(2)
            ]
            baseline, extended = root / "baseline.csv", root / "extended.csv"
            for path, extra in ((baseline, []), (extended, [])):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows + extra)
            result = compare_shared_prefix(
                baseline, extended, divergence_step=2, num_envs=2, num_agents=2
            )
            self.assertEqual(result["identical_drone_rows"], 8)
            with extended.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                changed = [dict(row) for row in rows]
                changed[1]["z_m"] = 0.3
                writer.writerows(changed)
            with self.assertRaisesRegex(ValueError, "diverged early"):
                compare_shared_prefix(
                    baseline, extended, divergence_step=2, num_envs=2, num_agents=2
                )
            with baseline.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(
                    [dict(row, phase="formation", episode_step=551) for row in rows[:4]]
                )
            switch = formation_switch_altitude(baseline)
            self.assertEqual(switch["first_formation_episode_step"], 551)
            self.assertAlmostEqual(switch["mean_target_altitude_deficit_m"], 1.3)

    def test_phase_tail_uses_each_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            columns = (
                "phase",
                "env_id",
                "team_reward",
                "assigned_rmse_m",
                "pairwise_rmse_m",
                "minimum_separation_m",
            )
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for env_id in (0, 1):
                    for error in (1.0, 2.0) if env_id == 0 else (3.0, 4.0):
                        writer.writerow(
                            dict(
                                phase="formation",
                                env_id=env_id,
                                team_reward=-1,
                                assigned_rmse_m=error,
                                pairwise_rmse_m=0.5,
                                minimum_separation_m=0.6,
                            )
                        )
            summary = summarize_evaluation_phases(path, final_window=1)["formation"]
            self.assertEqual(summary["environment_rows"], 4)
            self.assertEqual(summary["mean_assigned_rmse_m"], 2.5)
            self.assertEqual(summary["last_window_environment_rows"], 2)
            self.assertEqual(summary["last_window_assigned_rmse_m"], 3.0)


if __name__ == "__main__":
    unittest.main()
