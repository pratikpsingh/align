"""Host-safe checks for immutable learning-curve evaluation recovery."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.runtime.learning_curve_recovery import _command, _valid_source_evaluation


class LearningCurveRecoveryTests(unittest.TestCase):
    def test_source_evaluation_requires_exact_requested_checkpoint(self):
        probe = {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
            "deterministic_evaluation_tested": True,
        }
        metrics = {
            "status": "passed",
            "checks": {"exact": True},
            "completed_updates": 5,
        }
        item = {
            "completed_update": 5,
            "exit_code": 0,
            "probe": probe,
            "metrics": metrics,
        }
        self.assertTrue(_valid_source_evaluation(item, 5))
        self.assertFalse(_valid_source_evaluation(item, 10))
        item["metrics"]["completed_updates"] = 4
        self.assertFalse(_valid_source_evaluation(item, 5))

    def test_recovery_command_mounts_source_read_only_and_requests_update(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-run"
            recovery = root / "recovery-run"
            seed = source / "seed-0000000073"
            output = recovery / "seed-0000000073/evaluation-update-0005"
            output.mkdir(parents=True)
            command = _command(
                ["sudo", "-n", "docker"],
                source_run=source,
                recovery_run=recovery,
                source_seed=seed,
                output_directory=output,
                image_id="sha256:abc",
                gpu=0,
                num_envs=4,
                source_identity="git:abc:package:def",
                milestone=5,
            )
        self.assertIn(f"type=bind,src={source},dst=/source,readonly", command)
        self.assertEqual(command[command.index("--evaluation-update") + 1], "5")
        self.assertEqual(
            command[command.index("--checkpoint-directory") + 1],
            "/source/seed-0000000073/checkpoints",
        )


if __name__ == "__main__":
    unittest.main()
