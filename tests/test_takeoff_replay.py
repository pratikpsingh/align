"""CPU checks for immutable command extraction and three-arm comparison."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from align.runtime.takeoff_replay_runtime import replay_command, validate_baseline_report
from align.tasks.takeoff_replay import (
    ARMS,
    REPLAY_COLUMNS,
    WAKE_COLUMNS,
    assess_replays,
    assess_wake_guard,
    load_trace,
    prepare_trace,
    save_replay_plot,
)
from align.tasks.wake_guard import guard_focal_command


class TakeoffReplayTests(unittest.TestCase):
    def _source(self, path: Path):
        fields = (
            "env_id",
            "episode_step",
            "phase",
            "agent_id",
            "x_m",
            "y_m",
            "z_m",
            "vz_m_s",
            "contact_force_n",
            "command_vx_m_s",
            "command_vy_m_s",
            "command_vz_m_s",
        )
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for _episode in range(2):
                for step in range(3):
                    for env_id in (0, 1):
                        for agent in range(4):
                            writer.writerow(
                                {
                                    "env_id": env_id,
                                    "episode_step": step + 1,
                                    "phase": "ground" if step == 0 else "takeoff",
                                    "agent_id": agent,
                                    "x_m": agent - 1.5,
                                    "y_m": 0,
                                    "z_m": (0.06, 0.3, 0.06)[step] if agent == 2 else 0.06,
                                    "vz_m_s": 0,
                                    "contact_force_n": 0.5 if agent == 2 and step == 2 else 0,
                                    "command_vx_m_s": 0,
                                    "command_vy_m_s": 0,
                                    "command_vz_m_s": 0.2 if agent == 2 else 0,
                                }
                            )

    def _arm(self, path: Path, arm: str):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REPLAY_COLUMNS)
            writer.writeheader()
            for step in range(3):
                for agent in range(4):
                    row = dict.fromkeys(REPLAY_COLUMNS, 0)
                    row.update(
                        step=step,
                        phase="ground" if step == 0 else "takeoff",
                        agent_id=agent,
                        qw=1.0,
                        command_vz_m_s=0.2 if agent == 2 else 0,
                        z_m=(0.06, 0.3, 0.06 if arm == "four" else 0.4)[step]
                        if agent == 2
                        else 0.06,
                        contact_force_n=0.5 if agent == 2 and step == 2 and arm == "four" else 0,
                        model_force_z_n=-0.8 if agent == 2 and arm == "four" else 0,
                        rotor_thrust_sum_n=7.0,
                    )
                    writer.writerow(row)

    def test_extract_first_episode_and_support_only_when_baseline_reproduces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root / "source.csv")
            metadata = prepare_trace(root / "source.csv", root / "commands.csv")
            self.assertEqual(metadata["steps"], 3)
            self.assertEqual(len(load_trace(root / "commands.csv")), 3)
            outputs = {arm: root / f"{arm}.csv" for arm in ARMS}
            for arm, path in outputs.items():
                self._arm(path, arm)
            report = assess_replays(root / "commands.csv", outputs)
            self.assertTrue(report["baseline_reproduced_with_declared_tolerance"])
            self.assertTrue(report["modeled_downwash_causal_support"])
            save_replay_plot(root / "commands.csv", outputs, root / "replay.svg")
            self.assertIn("<svg", (root / "replay.svg").read_text())
            with outputs["four"].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[2]["command_vz_m_s"] = "-0.2"
            with outputs["four"].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=REPLAY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "command differs"):
                assess_replays(root / "commands.csv", outputs)

    def test_wake_guard_audit_recomputes_commands_and_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root / "source.csv")
            prepare_trace(root / "source.csv", root / "commands.csv")
            trace = load_trace(root / "commands.csv")
            positions = ((-1.5, 0, 0.8), (-0.5, 0, 0.8), (0.5, 0, 0.7), (1.2, 0, 0.95))
            guard_path = root / "wake.csv"
            with guard_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=WAKE_COLUMNS)
                writer.writeheader()
                for step, block in enumerate(trace):
                    requested = tuple(
                        tuple(float(row[f"command_v{axis}_m_s"]) for axis in "xyz") for row in block
                    )
                    guarded, details = (
                        guard_focal_command(positions, requested, 2)
                        if block[0]["phase"] == "takeoff"
                        else (requested[2], {"active": False, "unresolved": False})
                    )
                    for agent in range(4):
                        row = dict.fromkeys(WAKE_COLUMNS, 0)
                        row.update(step=step, phase=block[agent]["phase"], agent_id=agent, qw=1.0)
                        for index, axis in enumerate("xyz"):
                            row[f"pre_{axis}_m"] = positions[agent][index]
                            row[f"{axis}_m"] = positions[agent][index]
                            row[f"requested_v{axis}_m_s"] = requested[agent][index]
                            row[f"command_v{axis}_m_s"] = (
                                guarded[index] if agent == 2 else requested[agent][index]
                            )
                        row["guard_active"] = int(details["active"]) if agent == 2 else 0
                        row["guard_unresolved"] = int(details["unresolved"]) if agent == 2 else 0
                        writer.writerow(row)
            audit = assess_wake_guard(root / "commands.csv", guard_path)
            self.assertEqual(audit["guard_active_steps"], 2)
            self.assertTrue(audit["candidate_accepted"])
            with guard_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[6]["command_vx_m_s"] = "0.0"
            with guard_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=WAKE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "command differs"):
                assess_wake_guard(root / "commands.csv", guard_path)

    def test_missing_step_and_wrong_phase_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root / "source.csv")
            prepare_trace(root / "source.csv", root / "commands.csv")
            with (root / "commands.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows.pop()
            with (root / "commands.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_trace(root / "commands.csv")

    def test_reused_baseline_requires_exact_source_and_trace(self):
        report = {
            "status": "passed",
            "source_run_id": "source",
            "source_telemetry_sha256": "telemetry",
            "config_sha256": "config",
            "source_checkpoint_id": "checkpoint",
            "command_trace_sha256": "trace",
            "comparison": {"baseline_reproduced_with_declared_tolerance": True},
        }
        arguments = {
            "source_run_id": "source",
            "source_telemetry_sha256": "telemetry",
            "config_sha256": "config",
            "checkpoint_id": "checkpoint",
            "trace_sha256": "trace",
        }
        validate_baseline_report(report, **arguments)
        with self.assertRaisesRegex(ValueError, "baseline run"):
            validate_baseline_report(report, **{**arguments, "trace_sha256": "different"})
        report["comparison"]["baseline_reproduced_with_declared_tolerance"] = False
        with self.assertRaisesRegex(ValueError, "baseline run"):
            validate_baseline_report(report, **arguments)

    def test_container_command_isolated_and_arm_specific(self):
        command = replay_command(
            ["sudo", "-n", "docker"],
            run=Path("/tmp/replay"),
            arm="no_downwash",
            image_id="sha256:test",
            gpu=0,
        )
        self.assertIn("--network=none", command)
        self.assertEqual(command[-3:], ["--replay-arm", "no_downwash", "--allow-root"])
        self.assertIn("/output/commands.csv", command)
        with self.assertRaisesRegex(ValueError, "unknown"):
            replay_command([], run=Path("/tmp/replay"), arm="other", image_id="x", gpu=0)


if __name__ == "__main__":
    unittest.main()
