"""Reconstruct downwash exposure from complete saved physical snapshots."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from align.tasks.downwash_audit import audit_downwash


class DownwashAuditTests(unittest.TestCase):
    def test_upper_drone_exposes_lower_drone_in_saved_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.csv"
            fields = (
                "evaluation_step",
                "env_id",
                "episode_step",
                "agent_id",
                "x_m",
                "y_m",
                "z_m",
                "target_z_m",
                "vz_m_s",
                "command_vz_m_s",
                "contact_force_n",
            )
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for evaluation_step in range(2):
                    for agent_id, z in enumerate((1.0, 1.0 + evaluation_step)):
                        writer.writerow(
                            dict(
                                evaluation_step=evaluation_step,
                                env_id=0,
                                episode_step=evaluation_step + 1,
                                agent_id=agent_id,
                                x_m=0,
                                y_m=0,
                                z_m=z,
                                target_z_m=z,
                                vz_m_s=0,
                                command_vz_m_s=0,
                                contact_force_n=0,
                            )
                        )
            report, rows = audit_downwash(path, {0: 2})
            self.assertEqual(report["snapshots"], 2)
            self.assertEqual(report["raw_drone_rows"], 4)
            self.assertEqual([row["vertical_thrust_exposure_proxy"] for row in rows[:2]], [0, 0])
            self.assertGreater(rows[2]["vertical_thrust_exposure_proxy"], 0)
            self.assertEqual(rows[3]["vertical_thrust_exposure_proxy"], 0)
            self.assertFalse(report["physical_force_measured"])


if __name__ == "__main__":
    unittest.main()
