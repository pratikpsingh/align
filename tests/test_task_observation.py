"""Analytical checks for bounded local actor observations."""

import math
import unittest

from align.tasks.observation import (
    ObservationConfig,
    build_actor_observations,
    build_observations,
)
from align.tasks.observation_report import audit_rows


def build(ids, positions, *, velocities=None, targets=None, config=None):
    count = len(ids)
    if velocities is None:
        velocities = ((0.0, 0.0, 0.0),) * count
    if targets is None:
        targets = positions
    return build_observations(
        ids,
        positions,
        velocities,
        targets,
        config or ObservationConfig(),
    )


class ObservationConfigurationTests(unittest.TestCase):
    def test_fixed_dimensions_include_explicit_masks(self):
        config = ObservationConfig()
        self.assertEqual(config.actor_dimension, 55)
        self.assertEqual(config.critic_dimension, 80)
        self.assertEqual(ObservationConfig.from_dict(config.to_dict()), config)

    def test_configuration_requires_exact_valid_schema(self):
        for changes in (
            {"max_neighbors": 0},
            {"neighbor_radius_m": 0.0},
            {"critic_capacity": 0},
            {"target_distance_scale_m": math.inf},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ObservationConfig(**changes)
        values = ObservationConfig().to_dict()
        values["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            ObservationConfig.from_dict(values)


class LocalActorObservationTests(unittest.TestCase):
    def test_radius_is_hard_and_does_not_fill_a_minimum(self):
        config = ObservationConfig(max_neighbors=3, neighbor_radius_m=1.0)
        batch = build((0, 1), ((0.0, 0.0, 0.0), (1.000001, 0.0, 0.0)), config=config)
        first = batch.actors[0]
        self.assertEqual(first.neighbor_mask, (False, False, False))
        self.assertEqual(first.neighbor_ids, (None, None, None))
        self.assertTrue(all(slot == (0.0,) * 6 for slot in first.neighbor_features))

    def test_boundary_neighbor_is_included(self):
        config = ObservationConfig(max_neighbors=1, neighbor_radius_m=1.0)
        first = build((0, 1), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), config=config).actors[0]
        self.assertEqual(first.neighbor_mask, (True,))
        self.assertEqual(first.neighbor_ids, (1,))
        self.assertEqual(first.neighbor_distances_m, (1.0,))

    def test_budget_keeps_nearest_with_identity_tie_break(self):
        config = ObservationConfig(max_neighbors=2, neighbor_radius_m=2.0)
        batch = build(
            (10, 30, 20, 40),
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
            config=config,
        )
        center = next(item for item in batch.actors if item.agent_id == 10)
        self.assertEqual(center.neighbor_ids, (40, 20))
        self.assertEqual(center.neighbor_mask, (True, True))

    def test_actor_is_translation_invariant_but_critic_is_not(self):
        ids = (0, 1)
        positions = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))
        targets = ((0.5, 0.0, 1.0), (1.5, 0.0, 1.0))
        original = build(ids, positions, targets=targets)
        translated_positions = tuple((x + 2.0, y - 1.0, z + 0.5) for x, y, z in positions)
        translated_targets = tuple((x + 2.0, y - 1.0, z + 0.5) for x, y, z in targets)
        translated = build(ids, translated_positions, targets=translated_targets)
        self.assertEqual(
            tuple(item.flat() for item in original.actors),
            tuple(item.flat() for item in translated.actors),
        )
        self.assertNotEqual(original.critic.flat(), translated.critic.flat())

    def test_input_order_does_not_change_identity_observations(self):
        ids = (9, 2, 5)
        positions = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
        first = build(ids, positions)
        second = build(tuple(reversed(ids)), tuple(reversed(positions)))
        self.assertEqual(first, second)

    def test_actor_dimension_is_independent_of_swarm_size(self):
        config = ObservationConfig(max_neighbors=3)
        one = build((0,), ((0.0, 0.0, 0.0),), config=config)
        four = build(
            (0, 1, 2, 3),
            ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0)),
            config=config,
        )
        self.assertTrue(all(len(item.flat()) == config.actor_dimension for item in one.actors))
        self.assertTrue(all(len(item.flat()) == config.actor_dimension for item in four.actors))

    def test_actor_only_api_is_independent_of_training_critic_capacity(self):
        config = ObservationConfig(critic_capacity=8)
        count = 16
        ids = tuple(range(count))
        positions = tuple((2.0 * index, 0.0, 1.0) for index in ids)
        velocities = ((0.0, 0.0, 0.0),) * count
        actors = build_actor_observations(ids, positions, velocities, positions, config)
        self.assertEqual(len(actors), count)
        self.assertTrue(all(len(actor.flat()) == config.actor_dimension for actor in actors))
        with self.assertRaisesRegex(ValueError, "critic capacity"):
            build_observations(ids, positions, velocities, positions, config)

    def test_values_are_bounded_and_saturation_is_counted(self):
        config = ObservationConfig(
            max_neighbors=1,
            neighbor_radius_m=2.0,
            own_velocity_scale_m_s=1.0,
            relative_velocity_scale_m_s=1.0,
            target_distance_scale_m=1.0,
        )
        batch = build(
            (0, 1),
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            velocities=((2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
            targets=((3.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            config=config,
        )
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in batch.actors[0].flat()))
        self.assertGreater(batch.actors[0].saturation_count, 0)

    def test_actor_and_critic_padding_are_explicit(self):
        config = ObservationConfig(max_neighbors=2, critic_capacity=3)
        batch = build((4,), ((0.0, 0.0, 0.0),), config=config)
        actor = batch.actors[0]
        self.assertEqual(actor.neighbor_mask, (False, False))
        self.assertEqual(batch.critic.agent_mask, (True, False, False))
        self.assertEqual(batch.critic.agent_ids, (4, None, None))
        self.assertEqual(len(actor.flat()), config.actor_dimension)
        self.assertEqual(len(batch.critic.flat()), config.critic_dimension)

    def test_invalid_state_and_critic_overflow_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build((0, 0), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        with self.assertRaisesRegex(ValueError, "critic capacity"):
            build(
                (0, 1),
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                config=ObservationConfig(critic_capacity=1),
            )
        with self.assertRaises(ValueError):
            build((0,), ((math.nan, 0.0, 0.0),))


class ObservationAuditTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for agent_id, x in ((10, 0.0), (20, 0.4), (30, 0.8)):
            rows.append(
                {
                    "repeat": 0,
                    "step": 0,
                    "t": 0.01,
                    "phase": "formation",
                    "agent_id": agent_id,
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
            )
        return rows

    def test_audit_records_fixed_bounded_budgeted_observations(self):
        config = ObservationConfig(max_neighbors=1, neighbor_radius_m=1.0)
        report, rows = audit_rows(self.rows(), config)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["metrics"]["agent_steps"], 3)
        self.assertEqual(report["metrics"]["budget_limited_agent_steps"], 3)
        self.assertEqual(report["metrics"]["selected_directed_edges"], 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(
            all(len([key for key in row if key.startswith("obs_")]) == 13 for row in rows)
        )

    def test_audit_rejects_duplicate_identity_in_one_step(self):
        rows = self.rows()
        rows[1]["agent_id"] = rows[0]["agent_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            audit_rows(rows, ObservationConfig())


if __name__ == "__main__":
    unittest.main()
