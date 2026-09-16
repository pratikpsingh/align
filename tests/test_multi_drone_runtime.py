"""Host pass criteria require explicit multi-drone physics evidence."""

import unittest

from align.runtime.multi_drone_runtime import metrics_match, valid_result


class MultiDroneRuntimeTests(unittest.TestCase):
    def test_pass_requires_physics_metrics_and_zero_exit(self):
        probe = {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "multi_drone_physics_tested": True,
        }
        metrics = {"status": "passed", "checks": {"formation": True, "separation": True}}
        self.assertTrue(valid_result(0, probe, metrics))
        self.assertFalse(valid_result(1, probe, metrics))
        self.assertFalse(valid_result(0, {**probe, "multi_drone_physics_tested": False}, metrics))
        self.assertFalse(valid_result(0, probe, {"status": "failed", "checks": {"x": False}}))
        self.assertFalse(valid_result(0, None, metrics))

    def test_host_comparison_accounts_for_json_tuple_conversion(self):
        recomputed = {"layout": {"agent_ids": (0, 1), "points": ((0.0, 0.0, 0.0),)}}
        saved = {"layout": {"agent_ids": [0, 1], "points": [[0.0, 0.0, 0.0]]}}
        self.assertTrue(metrics_match(saved, recomputed))


if __name__ == "__main__":
    unittest.main()
