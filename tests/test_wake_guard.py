"""Local wake guard chooses bounded escape without changing unrelated commands."""

from __future__ import annotations

import math
import unittest

from align.tasks.wake_guard import WakeGuardConfig, guard_focal_command


class WakeGuardTests(unittest.TestCase):
    def test_below_upper_drone_chooses_local_bounded_escape(self):
        positions = ((-1.5, 0, 0.8), (-0.5, 0, 0.8), (0.5, 0, 0.7), (1.2, 0, 0.95))
        commands = ((0.1, 0, 0.14),) * 4
        guarded, details = guard_focal_command(positions, commands, 2)
        self.assertTrue(details["active"])
        self.assertEqual(details["overhead_agents"], [3])
        self.assertLessEqual(math.dist((0, 0, 0), guarded), 0.500001)
        self.assertLess(guarded[0], commands[2][0])
        self.assertEqual(guarded[2], commands[2][2])

    def test_no_upper_neighbor_or_not_airborne_preserves_request(self):
        positions = ((-1.5, 0, 0.06), (-0.5, 0, 0.06), (0.5, 0, 0.06), (1.2, 0, 0.30))
        commands = ((0, 0, 0.14),) * 4
        guarded, details = guard_focal_command(positions, commands, 2)
        self.assertEqual(guarded, commands[2])
        self.assertFalse(details["active"])
        positions = ((-1.5, 0, 0.8), (-0.5, 0, 0.8), (0.5, 0, 0.7), (2.0, 0, 1.0))
        guarded, details = guard_focal_command(positions, commands, 2)
        self.assertEqual(guarded, commands[2])
        self.assertFalse(details["active"])

    def test_invalid_speed_and_unsafe_configuration_are_rejected(self):
        positions = ((0, 0, 0.5), (1, 0, 0.8))
        with self.assertRaisesRegex(ValueError, "speed"):
            guard_focal_command(positions, ((0, 0, 0), (0, 0, 0.6)), 1)
        with self.assertRaisesRegex(ValueError, "wake radius"):
            WakeGuardConfig(wake_radius_m=2.0)


if __name__ == "__main__":
    unittest.main()
