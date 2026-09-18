"""Host command checks for a critic-free, fresh-process actor artifact load."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from align.runtime.actor_export_runtime import export_command, verify_command


class ActorExportRuntimeTests(unittest.TestCase):
    def test_export_mounts_trusted_training_run_read_only_and_uses_cpu(self):
        with patch("align.runtime.actor_export_runtime.os.getuid", return_value=1001):
            with patch("align.runtime.actor_export_runtime.os.getgid", return_value=1002):
                command = export_command(
                    ["docker"],
                    source=Path("/training"),
                    output=Path("/export"),
                    image_id="sha256:test",
                    seed=41,
                    manifest={
                        "payload": {"file": "checkpoint-0004.pt", "sha256": "a" * 64},
                        "checkpoint_id": "0004-test",
                        "completed_updates": 4,
                    },
                    repeats=200,
                    container_name="align-export-test",
                )
        self.assertIn("type=bind,src=/training,dst=/source,readonly", command)
        self.assertEqual(command[command.index("--checkpoint-sha256") + 1], "a" * 64)
        self.assertEqual(command[command.index("--user") + 1], "1001:1002")
        self.assertEqual(command[command.index("-m") + 1], "align.policies.actor_export")
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--runtime", command)

    def test_verifier_receives_only_actor_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            command = verify_command(
                ["docker"],
                output=output,
                image_id="sha256:test",
                container_name="align-verify-test",
            )
            self.assertIn(f"type=bind,src={output / 'artifact'},dst=/artifact,readonly", command)
            self.assertEqual(command[command.index("-m") + 1], "align.policies.actor_inference")
            self.assertNotIn("--gpus", command)
            self.assertFalse(any("/source" in item for item in command))


if __name__ == "__main__":
    unittest.main()
