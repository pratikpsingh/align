"""Host-safe checks for policy configuration and runtime result validation."""

import json
import sys
import unittest
from pathlib import Path

from align.policies import RecurrentPolicyConfig
from align.runtime.policy_runtime import valid_policy_result


class RecurrentPolicyConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.values = json.loads((self.root / "configs/recurrent-policy.json").read_text())

    def test_checked_configuration_matches_task_and_submission_dimensions(self):
        config = RecurrentPolicyConfig.from_dict(self.values)
        config.validate_task_dimensions(
            actor_observation_dim=55,
            critic_state_dim=80,
            action_dim=4,
        )
        self.assertEqual(config.encoder_hidden_size, 256)
        self.assertEqual(config.recurrent_hidden_size, 256)
        self.assertEqual(config.action_low, (-1.0,) * 4)
        self.assertEqual(config.action_high, (1.0,) * 4)
        self.assertEqual(config.to_dict(), self.values)

    def test_configuration_is_exact_and_rejects_invalid_bounds(self):
        missing = dict(self.values)
        missing.pop("log_std_min")
        with self.assertRaisesRegex(ValueError, "missing=.*log_std_min"):
            RecurrentPolicyConfig.from_dict(missing)

        invalid = dict(self.values)
        invalid["action_low"] = [-1.0, -1.0, -1.0, 2.0]
        with self.assertRaisesRegex(ValueError, "lower bound"):
            RecurrentPolicyConfig.from_dict(invalid)

    def test_host_safe_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_torch_module_is_isolated_from_host_imports(self):
        package = (self.root / "src/align/policies/__init__.py").read_text()
        config = (self.root / "src/align/policies/config.py").read_text()
        self.assertNotIn("import torch", package)
        self.assertNotIn("import torch", config)


class PolicyRuntimeResultTests(unittest.TestCase):
    def test_result_requires_exit_success_and_every_check(self):
        metrics = {"status": "passed", "checks": {"bounded": True, "finite": True}}
        self.assertTrue(valid_policy_result(0, metrics))
        self.assertFalse(valid_policy_result(1, metrics))
        self.assertFalse(valid_policy_result(0, {"status": "passed", "checks": {"bounded": False}}))
        self.assertFalse(valid_policy_result(0, {"status": "passed", "checks": {}}))


if __name__ == "__main__":
    unittest.main()
