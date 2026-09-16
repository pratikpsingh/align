"""Host-safe checks for recurrent MAPPO configuration and runtime gating."""

import json
import sys
import unittest
from pathlib import Path

from align.learning.ppo_config import RecurrentPPOConfig


class RecurrentPPOConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.values = json.loads((self.root / "configs/recurrent-ppo.json").read_text())

    def test_checked_configuration_is_exact_and_host_safe(self):
        config = RecurrentPPOConfig.from_dict(self.values)
        self.assertEqual(config.to_dict(), self.values)
        self.assertEqual(config.update_epochs, 2)
        self.assertTrue(config.normalize_advantages)
        self.assertNotIn("torch", sys.modules)

    def test_missing_unknown_and_invalid_values_are_rejected(self):
        missing = dict(self.values)
        missing.pop("policy_clip_ratio")
        with self.assertRaisesRegex(ValueError, "missing=.*policy_clip_ratio"):
            RecurrentPPOConfig.from_dict(missing)

        unknown = {**self.values, "minibatch_size": 4}
        with self.assertRaisesRegex(ValueError, "unknown=.*minibatch_size"):
            RecurrentPPOConfig.from_dict(unknown)

        for key, value in (
            ("actor_learning_rate", 0.0),
            ("policy_clip_ratio", 1.0),
            ("entropy_coefficient", -0.1),
            ("update_epochs", 0),
        ):
            invalid = {**self.values, key: value}
            with self.assertRaises(ValueError):
                RecurrentPPOConfig.from_dict(invalid)

    def test_torch_updater_is_isolated_from_host_imports(self):
        package = (self.root / "src/align/learning/__init__.py").read_text()
        updater = self.root / "src/align/learning/torch_ppo.py"
        self.assertNotIn("torch_ppo", package)
        self.assertFalse(updater.exists() and "torch_ppo" in package)


class RecurrentPPORuntimeResultTests(unittest.TestCase):
    def test_pass_requires_zero_exit_and_every_check(self):
        from align.runtime.ppo_runtime import valid_ppo_result

        metrics = {"status": "passed", "checks": {"finite": True, "changed": True}}
        self.assertTrue(valid_ppo_result(0, metrics))
        self.assertFalse(valid_ppo_result(1, metrics))
        self.assertFalse(valid_ppo_result(0, {"status": "passed", "checks": {"finite": False}}))
        self.assertFalse(valid_ppo_result(0, {"status": "passed", "checks": {}}))


if __name__ == "__main__":
    unittest.main()
