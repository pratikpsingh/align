"""Analytical tests for communication proxies and offline budget sweeps."""

import unittest

from align.tasks.communication import CommunicationConfig, account_snapshot
from align.tasks.communication_report import sweep_rows
from align.tasks.observation import ObservationConfig, build_actor_observations


class CommunicationAccountingTests(unittest.TestCase):
    def test_bidirectional_pair_counts_two_unicasts_and_two_broadcasts(self):
        config = CommunicationConfig(budgets=(1, 2))
        actors = build_actor_observations(
            (0, 1),
            ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
            ((0.0, 0.0, 0.0),) * 2,
            ((0.0, 0.0, 0.0),) * 2,
            ObservationConfig(max_neighbors=1),
        )
        result = account_snapshot(actors, config)
        self.assertEqual(config.packet_bytes, 40)
        self.assertEqual(result["directed_edges"], ((0, 1), (1, 0)))
        self.assertEqual(result["unicast_proxy_bytes"], 80)
        self.assertEqual(result["ideal_broadcast_proxy_bytes"], 80)

    def test_broadcast_sender_deduplicates_receivers(self):
        actors = build_actor_observations(
            (0, 1, 2),
            ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)),
            ((0.0, 0.0, 0.0),) * 3,
            ((0.0, 0.0, 0.0),) * 3,
            ObservationConfig(max_neighbors=1, neighbor_radius_m=0.75),
        )
        result = account_snapshot(actors, CommunicationConfig())
        self.assertEqual(result["directed_edges"], ((0, 1), (0, 2), (1, 0)))
        self.assertEqual(result["unicast_proxy_bytes"], 120)
        self.assertEqual(result["ideal_broadcast_proxy_bytes"], 80)

    def test_budget_sweep_preserves_saved_trajectory_and_radius(self):
        rows = [
            {
                "repeat": 0,
                "step": 0,
                "t": 0.0,
                "phase": "formation",
                "agent_id": agent,
                "x": x,
                "y": 0.0,
                "z": 1.0,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "target_x": x,
                "target_y": 0.0,
                "target_z": 1.0,
            }
            for agent, x in enumerate((0.0, 0.5, 1.0, 5.0))
        ]
        topology, summaries = sweep_rows(
            rows,
            ObservationConfig(neighbor_radius_m=1.0),
            CommunicationConfig(budgets=(1, 2, 3)),
        )
        self.assertEqual([row["selected_directed_edges"] for row in topology], [3, 6, 6])
        self.assertEqual([row["candidate_directed_edges"] for row in topology], [6, 6, 6])
        self.assertEqual([row["unicast_proxy_bytes"] for row in topology], [120, 240, 240])
        self.assertEqual([row["agent_steps"] for row in summaries], [4, 4, 4])
        self.assertTrue(all("3>" not in row["directed_edges"] for row in topology))

    def test_rejects_invalid_packet_and_budget_contracts(self):
        for changes in (
            {"bytes_per_component": 0},
            {"packet_overhead_bytes": -1},
            {"budgets": (2, 1)},
            {"budgets": (1, 1)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CommunicationConfig(**changes)
        config = CommunicationConfig()
        self.assertEqual(CommunicationConfig.from_dict(config.to_dict()), config)


if __name__ == "__main__":
    unittest.main()
