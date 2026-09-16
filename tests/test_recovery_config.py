"""Host-safe checks for recovery policy and runtime result gating."""

import json
import sys
import unittest
from pathlib import Path

from align.learning.recovery_config import RecoveryConfig


class RecoveryConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.values = json.loads((self.root / "configs/training-recovery.json").read_text())

    def test_checked_configuration_is_exact_and_host_safe(self):
        config = RecoveryConfig.from_dict(self.values)
        self.assertEqual(config.to_dict(), self.values)
        self.assertGreaterEqual(config.minimum_retained_checkpoints, 3)
        self.assertNotIn("torch", sys.modules)

    def test_missing_unknown_and_invalid_values_are_rejected(self):
        for values in (
            {key: value for key, value in self.values.items() if key != "recovery_mode"},
            {**self.values, "unknown": True},
            {**self.values, "checkpoint_interval_active_seconds": 0},
            {**self.values, "minimum_retained_checkpoints": 2},
            {**self.values, "write_failure_policy": "ignore"},
        ):
            with self.assertRaises(ValueError):
                RecoveryConfig.from_dict(values)

    def test_runtime_pass_requires_zero_exit_and_every_check(self):
        from align.runtime.recovery_runtime import valid_recovery_result

        metrics = {"status": "passed", "checks": {"exact": True, "fallback": True}}
        self.assertTrue(valid_recovery_result(0, metrics))
        self.assertFalse(valid_recovery_result(1, metrics))
        self.assertFalse(valid_recovery_result(0, {"status": "passed", "checks": {"exact": False}}))
        self.assertFalse(valid_recovery_result(0, {"status": "passed", "checks": {}}))

    def test_torch_recovery_is_isolated_from_host_imports(self):
        package = (self.root / "src/align/learning/__init__.py").read_text()
        self.assertNotIn("torch_recovery", package)


if __name__ == "__main__":
    unittest.main()
