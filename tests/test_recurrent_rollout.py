"""Checks for temporal rollout order, recurrent resets, and return masks."""

import json
import math
import unittest
from pathlib import Path

from align.learning.rollout import (
    RecurrentFrame,
    RecurrentRollout,
    RolloutConfig,
    RolloutTransition,
)


def config(**changes):
    values = RolloutConfig(
        horizon=3,
        num_envs=1,
        num_agents=1,
        actor_observation_dim=2,
        critic_state_dim=3,
        action_dim=1,
        recurrent_layers=1,
        recurrent_hidden_size=2,
        chunk_length=2,
        gamma=0.5,
        gae_lambda=1.0,
    ).to_dict()
    values.update(changes)
    return RolloutConfig.from_dict(values)


def memory(cfg, marker):
    return tuple(
        tuple(
            tuple(
                tuple(
                    float(marker + env * 100 + agent * 10 + layer)
                    for _ in range(cfg.recurrent_hidden_size)
                )
                for layer in range(cfg.recurrent_layers)
            )
            for agent in range(cfg.num_agents)
        )
        for env in range(cfg.num_envs)
    )


def frame(cfg, marker, *, zero_memory=False):
    state = memory(cfg, 0 if zero_memory else marker)
    return RecurrentFrame(
        actor_observations=tuple(
            tuple(
                tuple(float(marker + agent) for _ in range(cfg.actor_observation_dim))
                for agent in range(cfg.num_agents)
            )
            for _ in range(cfg.num_envs)
        ),
        critic_states=tuple(
            tuple(float(1000 + marker) for _ in range(cfg.critic_state_dim))
            for _ in range(cfg.num_envs)
        ),
        actor_hidden=state,
        actor_cell=state,
        critic_hidden=state,
        critic_cell=state,
    )


def transition(cfg, reward, bootstrap, *, terminated=False, truncated=False):
    return RolloutTransition(
        actions=tuple(tuple((0.25,) for _ in range(cfg.num_agents)) for _ in range(cfg.num_envs)),
        old_log_probs=tuple(
            tuple(-0.5 for _ in range(cfg.num_agents)) for _ in range(cfg.num_envs)
        ),
        rewards=tuple(
            tuple(float(reward) for _ in range(cfg.num_agents)) for _ in range(cfg.num_envs)
        ),
        values=tuple(tuple(0.0 for _ in range(cfg.num_agents)) for _ in range(cfg.num_envs)),
        bootstrap_values=tuple(
            tuple(float(bootstrap) for _ in range(cfg.num_agents)) for _ in range(cfg.num_envs)
        ),
        terminated=tuple(terminated for _ in range(cfg.num_envs)),
        truncated=tuple(truncated for _ in range(cfg.num_envs)),
    )


class RecurrentRolloutTests(unittest.TestCase):
    def test_gae_bootstraps_truncation_but_stops_cross_episode_trace(self):
        cfg = config()
        rollout = RecurrentRollout(cfg, frame(cfg, 10))
        rollout.append(transition(cfg, 1, 2), frame(cfg, 11))
        rollout.append(
            transition(cfg, 3, 0, terminated=True),
            frame(cfg, 20, zero_memory=True),
        )
        rollout.append(
            transition(cfg, 5, 4, truncated=True),
            frame(cfg, 30, zero_memory=True),
        )

        advantages = rollout.compute_gae()
        self.assertEqual([advantages[t][0][0] for t in range(3)], [3.5, 3.0, 7.0])
        self.assertEqual([rollout.returns[t][0][0] for t in range(3)], [3.5, 3.0, 7.0])

    def test_chunks_keep_order_and_never_cross_episode_boundary(self):
        cfg = config()
        rollout = RecurrentRollout(cfg, frame(cfg, 10))
        rollout.append(transition(cfg, 1, 2), frame(cfg, 11))
        rollout.append(
            transition(cfg, 3, 0, terminated=True),
            frame(cfg, 20, zero_memory=True),
        )
        rollout.append(
            transition(cfg, 5, 4, truncated=True),
            frame(cfg, 30, zero_memory=True),
        )
        rollout.compute_gae()

        chunks = rollout.sequence_chunks()
        self.assertEqual(
            [(item.start_step, item.valid_length) for item in chunks], [(0, 2), (2, 1)]
        )
        self.assertEqual(chunks[0].actor_observations, ((10.0, 10.0), (11.0, 11.0)))
        self.assertEqual(chunks[0].critic_states[0], (1010.0, 1010.0, 1010.0))
        self.assertNotIn(1010.0, chunks[0].actor_observations[0])
        self.assertEqual(chunks[1].valid_mask, (1.0, 0.0))
        self.assertEqual(chunks[1].actor_observations[1], (0.0, 0.0))
        self.assertEqual(chunks[1].initial_actor_hidden, ((0.0, 0.0),))

    def test_done_transition_requires_zero_next_recurrent_state(self):
        cfg = config(horizon=1, chunk_length=1)
        rollout = RecurrentRollout(cfg, frame(cfg, 1))
        with self.assertRaisesRegex(ValueError, "recurrent state must reset"):
            rollout.append(
                transition(cfg, 1, 0, terminated=True),
                frame(cfg, 2),
            )

    def test_true_termination_rejects_nonzero_bootstrap(self):
        cfg = config(horizon=1, chunk_length=1)
        rollout = RecurrentRollout(cfg, frame(cfg, 1))
        with self.assertRaisesRegex(ValueError, "zero bootstrap"):
            rollout.append(
                transition(cfg, 1, 2, terminated=True),
                frame(cfg, 2, zero_memory=True),
            )

    def test_mutually_exclusive_boundaries_and_finite_values(self):
        cfg = config(horizon=1, chunk_length=1)
        rollout = RecurrentRollout(cfg, frame(cfg, 1))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            rollout.append(
                transition(cfg, 1, 0, terminated=True, truncated=True),
                frame(cfg, 2, zero_memory=True),
            )

        bad = transition(cfg, math.nan, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            rollout.append(bad, frame(cfg, 2))

    def test_checked_config_matches_accepted_task_dimensions(self):
        root = Path(__file__).resolve().parents[1]
        values = json.loads((root / "configs/recurrent-rollout.json").read_text())
        cfg = RolloutConfig.from_dict(values)
        cfg.validate_dimensions(
            num_envs=4,
            num_agents=4,
            actor_observation_dim=55,
            critic_state_dim=80,
            action_dim=4,
        )

    def test_configuration_requires_every_explicit_key(self):
        values = config().to_dict()
        values.pop("gamma")
        with self.assertRaisesRegex(ValueError, "missing=.*gamma"):
            RolloutConfig.from_dict(values)

    def test_out_of_range_executed_action_is_rejected(self):
        cfg = config(horizon=1, chunk_length=1)
        item = transition(cfg, 1, 1)
        bad = RolloutTransition(
            actions=(((1.01,),),),
            old_log_probs=item.old_log_probs,
            rewards=item.rewards,
            values=item.values,
            bootstrap_values=item.bootstrap_values,
            terminated=item.terminated,
            truncated=item.truncated,
        )
        rollout = RecurrentRollout(cfg, frame(cfg, 1))
        with self.assertRaisesRegex(ValueError, "bounded"):
            rollout.append(bad, frame(cfg, 2))

    def test_shuffle_changes_chunk_order_only(self):
        cfg = config(horizon=3, num_envs=2, num_agents=2)
        rollout = RecurrentRollout(cfg, frame(cfg, 1))
        for step in range(3):
            rollout.append(transition(cfg, step, step + 1), frame(cfg, step + 2))
        rollout.compute_gae()
        original = rollout.sequence_chunks()
        first = rollout.shuffled_minibatches(3, seed=7)
        second = rollout.shuffled_minibatches(3, seed=7)
        self.assertEqual(first, second)
        self.assertCountEqual(
            [(item.environment_id, item.agent_id, item.start_step) for item in original],
            [
                (item.environment_id, item.agent_id, item.start_step)
                for batch in first
                for item in batch
            ],
        )


if __name__ == "__main__":
    unittest.main()
