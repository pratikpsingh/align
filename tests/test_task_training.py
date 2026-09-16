"""Host-safe configuration and result-gating checks for bounded training."""

import json
import sys
import unittest
from pathlib import Path

from align.learning.rollout import RolloutConfig
from align.learning.training_config import TaskTrainingConfig
from align.tasks.environment import TaskEnvironmentConfig


class TaskTrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.values = json.loads((self.root / "configs/task-training.json").read_text())

    def test_checked_configuration_is_exact_and_host_safe(self):
        config = TaskTrainingConfig.from_dict(self.values)
        rollout = RolloutConfig.from_dict(
            json.loads((self.root / "configs/recurrent-training-rollout.json").read_text())
        )
        task = TaskEnvironmentConfig.from_dict(
            json.loads((self.root / "configs/recurrent-training-task.json").read_text())
        )
        self.assertEqual(config.to_dict(), self.values)
        self.assertEqual(config.attempts, 2)
        self.assertEqual(config.updates_per_attempt, 1)
        self.assertLess(rollout.horizon, task.max_episode_steps)
        self.assertNotIn("torch", sys.modules)

    def test_invalid_attempt_update_seed_and_keys_are_rejected(self):
        for values in (
            {key: value for key, value in self.values.items() if key != "attempts"},
            {**self.values, "unknown": True},
            {**self.values, "attempts": 1},
            {**self.values, "updates_per_attempt": 2},
            {**self.values, "policy_seed": -1},
            {**self.values, "stochastic_actions": 1},
        ):
            with self.assertRaises(ValueError):
                TaskTrainingConfig.from_dict(values)


if __name__ == "__main__":
    unittest.main()


class TaskTrainingBundleTests(unittest.TestCase):
    def test_training_bundle_loads_exact_sections_without_simulator_imports(self):
        from tempfile import TemporaryDirectory

        from align.simulation.vector_task import load_bundle

        root = Path(__file__).resolve().parents[1]
        names = {
            "construction": "multi-drone-construction.json",
            "observation": "local-observation-baseline.json",
            "reward": "task-reward-baseline.json",
            "task": "recurrent-training-task.json",
            "policy": "recurrent-policy.json",
            "rollout": "recurrent-training-rollout.json",
            "ppo": "recurrent-ppo.json",
            "recovery": "training-recovery.json",
            "training": "task-training.json",
        }
        values = {
            section: json.loads((root / "configs" / filename).read_text())
            for section, filename in names.items()
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(values))
            bundle = load_bundle(path)
            self.assertIsNone(bundle[6])
            self.assertEqual(bundle[7].update_epochs, 2)
            self.assertEqual(bundle[8].minimum_retained_checkpoints, 3)
            self.assertEqual(bundle[9].attempts, 2)
            self.assertEqual(bundle[10], values)

            values.pop("recovery")
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "optional_sections"):
                load_bundle(path)


class TaskTrainingRuntimeTests(unittest.TestCase):
    def test_pass_requires_both_physics_attempts_and_host_audit(self):
        from align.runtime.training_runtime import valid_training_result

        probe = {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
            "task_training_tested": True,
        }
        metrics = {"status": "passed", "checks": {"finite": True}}
        attempts = [
            {"exit_code": 0, "probe": probe, "metrics": metrics},
            {"exit_code": 0, "probe": probe, "metrics": metrics},
        ]
        audit = {"status": "passed", "checks": {"lineage": True}}
        self.assertTrue(valid_training_result(attempts, audit))
        self.assertFalse(valid_training_result(attempts[:1], audit))
        attempts[1] = {**attempts[1], "exit_code": 1}
        self.assertFalse(valid_training_result(attempts, audit))

    def test_host_audit_checks_lineage_counters_and_raw_tables(self):
        import csv
        import hashlib
        from tempfile import TemporaryDirectory

        from align.learning.checkpoint_store import CheckpointStore
        from align.learning.training_report import audit_training_run

        with TemporaryDirectory() as directory:
            run = Path(directory) / "logical-run"
            run.mkdir()
            config = {
                "construction": {},
                "observation": {},
                "reward": {},
                "task": {},
                "policy": {},
                "rollout": {"horizon": 1, "num_envs": 2, "num_agents": 4},
                "ppo": {"update_epochs": 1},
                "recovery": {},
                "training": {},
            }
            config_path = run / "config.json"
            config_path.write_text(json.dumps(config))
            store = CheckpointStore(
                run / "checkpoints",
                run_id=run.name,
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            )

            def save(update, parent):
                return store.save(
                    lambda stream: stream.write(f"state-{update}".encode()),
                    attempt_id="attempt-0001" if update < 2 else "attempt-0002",
                    completed_updates=update,
                    environment_transitions=update * 2,
                    agent_transitions=update * 8,
                    active_training_seconds=float(update),
                    source_identity="git:abc",
                    runtime_identity="image:abc",
                    parent_checkpoint_sha256=parent,
                )

            initial = save(0, None)
            first = save(1, initial["payload"]["sha256"])
            second = save(2, first["payload"]["sha256"])
            counters = [
                {
                    "completed_updates": value,
                    "environment_transitions": value * 2,
                    "agent_transitions": value * 8,
                    "active_training_seconds": float(value),
                }
                for value in range(3)
            ]
            for number, start, end, unfinished in (
                (1, initial, first, 2),
                (2, first, second, 1),
            ):
                attempt = run / f"attempt-{number:04d}"
                attempt.mkdir()
                metrics = {
                    "status": "passed",
                    "checks": {"finite": True},
                    "resumed": number == 2,
                    "loaded_checkpoint_id": first["checkpoint_id"] if number == 2 else None,
                    "start_checkpoint_id": start["checkpoint_id"],
                    "committed_checkpoint_id": end["checkpoint_id"],
                    "start_counters": counters[number - 1],
                    "end_counters": counters[number],
                    "abandoned_environment_episodes": 2 if number == 2 else 0,
                    "unfinished_environment_count": unfinished,
                }
                (attempt / "metrics.json").write_text(json.dumps(metrics))
                with (attempt / "rollout.csv").open("w", newline="") as stream:
                    fields = (
                        "attempt_id",
                        "rollout_step",
                        "env_id",
                        "team_reward",
                        "value",
                        "bootstrap_value",
                        "action_min",
                        "action_max",
                        "assigned_rmse_m",
                        "pairwise_rmse_m",
                        "minimum_separation_m",
                    )
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for env_id in range(2):
                        writer.writerow(
                            {
                                "attempt_id": f"attempt-{number:04d}",
                                "rollout_step": 0,
                                "env_id": env_id,
                                "team_reward": 0.1,
                                "value": 0.2,
                                "bootstrap_value": 0.3,
                                "action_min": -0.5,
                                "action_max": 0.5,
                                "assigned_rmse_m": 1.0,
                                "pairwise_rmse_m": 0.1,
                                "minimum_separation_m": 0.8,
                            }
                        )
                with (attempt / "updates.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=("attempt_id", "completed_update", "loss")
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "attempt_id": f"attempt-{number:04d}",
                            "completed_update": number,
                            "loss": 0.25,
                        }
                    )
            result = audit_training_run(run)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(all(result["checks"].values()))
