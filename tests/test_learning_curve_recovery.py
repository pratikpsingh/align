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


class SegmentedLearningRecoveryTests(unittest.TestCase):
    def setUp(self):
        import json

        from align.learning.learning_curve_config import LearningCurveConfig

        self.root = Path(__file__).resolve().parents[1]
        self.curve = LearningCurveConfig.from_dict(
            json.loads((self.root / "configs/learning-curve-segmented.json").read_text())
        )

    @staticmethod
    def _probe():
        return {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
            "training_stability_tested": True,
        }

    def test_source_segments_must_be_valid_contiguous_prefix(self):
        from align.runtime.segmented_learning_recovery import completed_source_segments

        records = []
        for start, stop in self.curve.segment_ranges():
            records.append(
                {
                    "start_update": start,
                    "stop_update": stop,
                    "exit_code": 0,
                    "probe": self._probe(),
                    "metrics": {"status": "passed", "checks": {"exact": True}},
                }
            )
        item = {"train": {"segments": records[:1] + [{**records[1], "exit_code": 1}]}}
        completed = completed_source_segments(item, self.curve)
        self.assertEqual([(row["start_update"], row["stop_update"]) for row in completed], [(0, 4)])
        self.assertEqual(completed[0]["artifact_origin"], "source_run")

        item["train"]["segments"].append(records[2])
        with self.assertRaisesRegex(ValueError, "valid training after"):
            completed_source_segments(item, self.curve)

    def test_checkpoint_prefix_copy_reverifies_exact_boundary(self):
        import hashlib

        from align.learning.checkpoint_store import CheckpointStore
        from align.runtime.segmented_learning_recovery import copy_checkpoint_prefix

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            config_hash = hashlib.sha256(b"config").hexdigest()
            store = CheckpointStore(
                source, run_id="source-seed-0000000041", config_sha256=config_hash
            )
            for update in range(5):
                store.save(
                    lambda stream, value=update: stream.write(f"state-{value}".encode()),
                    attempt_id=f"update-{update:04d}",
                    completed_updates=update,
                    environment_transitions=update * 4,
                    agent_transitions=update * 16,
                    active_training_seconds=float(update),
                    source_identity="source",
                    runtime_identity="runtime",
                )
            result = copy_checkpoint_prefix(
                source,
                destination,
                logical_run_id="source-seed-0000000041",
                config_sha256=config_hash,
                through_update=4,
            )
            copied = CheckpointStore(
                destination,
                run_id="source-seed-0000000041",
                config_sha256=config_hash,
            )
            latest, path = copied.latest_valid()
            payload = path.read_bytes()
        self.assertEqual(result["checkpoint_count"], 5)
        self.assertEqual(latest["completed_updates"], 4)
        self.assertEqual(payload, b"state-4")

    def test_source_inventory_detects_any_artifact_change(self):
        from align.runtime.segmented_learning_recovery import source_inventory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text("{}")
            before = source_inventory(root)
            (root / "report.json").write_text('{"status":"changed"}')
            after = source_inventory(root)
        self.assertNotEqual(before["tree_sha256"], after["tree_sha256"])
        self.assertEqual(before["file_count"], after["file_count"])

    def test_partial_rollout_audit_requires_exact_rows_and_no_update_artifacts(self):
        import csv
        import json

        from align.runtime.segmented_learning_recovery import validate_interrupted_rollout

        interruption = {
            "schema_version": 1,
            "kind": "injected_mid_rollout_interruption",
            "attempt_id": "update-0005",
            "start_checkpoint_id": "checkpoint-4",
            "start_checkpoint_sha256": "a" * 64,
            "completed_updates_before": 4,
            "partial_rollout_steps": 2,
            "discarded_environment_transitions": 8,
            "discarded_agent_transitions": 32,
            "ppo_update_started": False,
            "checkpoint_committed": False,
            "policy_seed": 41,
            "discard_policy": "discard_partial_rollout_and_reset_environment_and_recurrent_memory",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            update = root / "seed-0000000041/train-segment-0004-0008/update-0005"
            update.mkdir(parents=True)
            saved = {
                key: value
                for key, value in interruption.items()
                if key not in {"policy_seed", "discard_policy"}
            }
            (update / "interruption.json").write_text(json.dumps(saved))
            with (update / "rollout.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("rollout_step", "env_id"))
                writer.writeheader()
                for step in range(2):
                    for env_id in range(4):
                        writer.writerow({"rollout_step": step, "env_id": env_id})
            result = validate_interrupted_rollout(
                root,
                interruption=interruption,
                curve=self.curve,
                resolved={"rollout": {"num_envs": 4}},
            )
            self.assertEqual(result["row_count"], 8)
            (update / "updates.csv").write_text("should not exist")
            with self.assertRaisesRegex(ValueError, "post-update artifacts"):
                validate_interrupted_rollout(
                    root,
                    interruption=interruption,
                    curve=self.curve,
                    resolved={"rollout": {"num_envs": 4}},
                )


if __name__ == "__main__":
    unittest.main()
