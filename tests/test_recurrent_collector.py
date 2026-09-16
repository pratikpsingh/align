"""Host-safe configuration checks for live recurrent collection."""

import json
import unittest
from pathlib import Path

from align.learning.collector_config import (
    CollectorProbeConfig,
    select_episode_boundary_memory,
)
from align.learning.rollout import RolloutConfig
from align.policies.config import RecurrentPolicyConfig
from align.tasks.environment import TaskEnvironmentConfig


class RecurrentCollectorConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_configs_form_one_explicit_tensor_contract(self):
        collector = CollectorProbeConfig.from_dict(
            json.loads((self.root / "configs/recurrent-collector-probe.json").read_text())
        )
        task = TaskEnvironmentConfig.from_dict(
            json.loads((self.root / "configs/recurrent-collector-task.json").read_text())
        )
        rollout = RolloutConfig.from_dict(
            json.loads((self.root / "configs/recurrent-rollout.json").read_text())
        )
        policy = RecurrentPolicyConfig.from_dict(
            json.loads((self.root / "configs/recurrent-policy.json").read_text())
        )
        collector.validate(horizon=rollout.horizon, num_envs=rollout.num_envs)
        self.assertEqual(task.num_envs, rollout.num_envs)
        self.assertLess(task.max_episode_steps, rollout.horizon)
        self.assertEqual(policy.actor_observation_dim, rollout.actor_observation_dim)
        self.assertEqual(policy.critic_state_dim, rollout.critic_state_dim)
        self.assertEqual(policy.action_dim, rollout.action_dim)
        self.assertEqual(policy.recurrent_hidden_size, rollout.recurrent_hidden_size)
        self.assertEqual(policy.recurrent_layers, rollout.recurrent_layers)

    def test_exact_keys_and_forced_event_bounds(self):
        values = json.loads((self.root / "configs/recurrent-collector-probe.json").read_text())
        values.pop("policy_seed")
        with self.assertRaisesRegex(ValueError, "missing=.*policy_seed"):
            CollectorProbeConfig.from_dict(values)
        with self.assertRaisesRegex(ValueError, "inside the rollout horizon"):
            CollectorProbeConfig(forced_termination_step=8).validate(horizon=8, num_envs=4)
        with self.assertRaisesRegex(ValueError, "outside the environment batch"):
            CollectorProbeConfig(forced_termination_environment=4).validate(horizon=128, num_envs=4)

    def test_boundary_memory_selection_matches_time_environment_mask(self):
        calls = []

        class LayoutProbe:
            def __getitem__(self, key):
                calls.append(("index", key))
                return self

            def swapaxes(self, left, right):
                calls.append(("swapaxes", left, right))
                return self

        boundary = object()
        selected = select_episode_boundary_memory(LayoutProbe(), boundary)

        self.assertIsInstance(selected, LayoutProbe)
        self.assertEqual(calls[0], ("index", slice(1, None, None)))
        self.assertEqual(calls[1], ("swapaxes", 1, 2))
        self.assertEqual(calls[2], ("index", boundary))


if __name__ == "__main__":
    unittest.main()


class CollectorRuntimeTests(unittest.TestCase):
    def test_pass_requires_live_physics_metrics_and_host_audit(self):
        from align.runtime.collector_runtime import valid_collector_result

        probe = {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
            "recurrent_collector_tested": True,
        }
        metrics = {"status": "passed", "checks": {"device": True}}
        audit = {"status": "passed", "checks": {"raw": True}}
        self.assertTrue(valid_collector_result(0, probe, metrics, audit))
        self.assertFalse(valid_collector_result(1, probe, metrics, audit))
        self.assertFalse(
            valid_collector_result(
                0, {**probe, "recurrent_collector_tested": False}, metrics, audit
            )
        )
        self.assertFalse(
            valid_collector_result(
                0, probe, {"status": "failed", "checks": {"device": False}}, audit
            )
        )

    def test_saved_collector_audit_checks_rows_and_hashes(self):
        import csv
        import hashlib
        from tempfile import TemporaryDirectory

        from align.learning.collector_report import audit_collector_run

        collector_columns = (
            "global_step",
            "env_id",
            "agent_id",
            "episode_step",
            "action_0",
            "action_1",
            "action_2",
            "action_3",
            "old_log_prob",
            "team_reward",
            "value",
            "bootstrap_value",
            "terminated",
            "truncated",
            "reason_code",
        )

        with TemporaryDirectory() as directory:
            run = Path(directory)
            config = {
                "construction": {},
                "observation": {},
                "reward": {},
                "task": {},
                "policy": {},
                "rollout": {
                    "horizon": 1,
                    "num_envs": 1,
                    "num_agents": 1,
                    "action_dim": 4,
                },
                "collector": {},
            }
            (run / "config.json").write_text(json.dumps(config))
            raw = run / "rollout.npz"
            checkpoint = run / "initial-policy.pt"
            raw.write_bytes(b"raw")
            checkpoint.write_bytes(b"checkpoint")

            def artifact(path):
                return {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            metrics = {
                "status": "passed",
                "checks": {"container": True},
                "raw_rollout": artifact(raw),
                "policy_checkpoint": artifact(checkpoint),
            }
            (run / "metrics.json").write_text(json.dumps(metrics))
            with (run / "collector.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=collector_columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "global_step": 0,
                        "env_id": 0,
                        "agent_id": 0,
                        "episode_step": 1,
                        "action_0": 0.1,
                        "action_1": 0.2,
                        "action_2": 0.3,
                        "action_3": 0.4,
                        "old_log_prob": -1.0,
                        "team_reward": 0.5,
                        "value": 0.2,
                        "bootstrap_value": 0.0,
                        "terminated": True,
                        "truncated": False,
                        "reason_code": 4,
                    }
                )
            # Add a truncation row by reusing the same one-step slot only after
            # validating all other fields; the audit deliberately requires both.
            result = audit_collector_run(run)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["checks"]["truncation_rows_present"])

            config["rollout"]["horizon"] = 2
            (run / "config.json").write_text(json.dumps(config))
            with (run / "collector.csv").open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=collector_columns)
                writer.writerow(
                    {
                        "global_step": 1,
                        "env_id": 0,
                        "agent_id": 0,
                        "episode_step": 1,
                        "action_0": 0.0,
                        "action_1": 0.0,
                        "action_2": 0.0,
                        "action_3": 0.0,
                        "old_log_prob": -1.0,
                        "team_reward": 0.0,
                        "value": 0.0,
                        "bootstrap_value": 0.1,
                        "terminated": False,
                        "truncated": True,
                        "reason_code": 6,
                    }
                )
            result = audit_collector_run(run)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(all(result["checks"].values()))
