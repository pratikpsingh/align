"""CPU checks for vectorized episode state, outcomes, and reset boundaries."""

import json
import math
import unittest
from pathlib import Path

from align.simulation.multi_drone_contract import (
    MultiDroneConfig,
    targets_for_phase,
)
from align.tasks.environment import BatchedTaskEnvironment, TaskEnvironmentConfig
from align.tasks.observation import ObservationConfig
from align.tasks.reward import RewardConfig


def construction(**changes):
    values = MultiDroneConfig().to_dict()
    values.update(changes)
    return MultiDroneConfig.from_dict(values)


def task_config(base, **changes):
    values = TaskEnvironmentConfig(
        safety_xy_limit_m=base.safety_xy_limit_m,
        safety_z_limit_m=base.safety_z_limit_m,
        airborne_height_m=base.airborne_height_m,
        contact_force_threshold_n=base.contact_force_threshold_n,
        formation_rmse_tolerance_m=base.formation_rmse_tolerance_m,
        pairwise_rmse_tolerance_m=base.pairwise_rmse_tolerance_m,
        speed_tolerance_m_s=base.speed_tolerance_m_s,
    ).to_dict()
    values.update(changes)
    return TaskEnvironmentConfig.from_dict(values)


def environment(num_envs=4, max_episode_steps=160, *, base=None, **task_changes):
    base = base or construction()
    task = task_config(
        base,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        **task_changes,
    )
    return BatchedTaskEnvironment(task, base, ObservationConfig(), RewardConfig())


def repeated(value, count):
    return tuple(value for _ in range(count))


class TaskEnvironmentConfigurationTests(unittest.TestCase):
    def test_checked_config_file_matches_runtime_contract(self):
        root = Path(__file__).resolve().parents[1]
        values = json.loads((root / "configs/vector-task-probe.json").read_text())
        config = TaskEnvironmentConfig.from_dict(values)
        self.assertEqual(config.num_envs, 4)
        self.assertEqual(config.max_episode_steps, 160)
        config.validate_compatibility(
            construction(),
            ObservationConfig(),
            RewardConfig(),
        )

    def test_cross_component_mismatches_are_rejected(self):
        base = construction()
        with self.assertRaisesRegex(ValueError, "critic capacity"):
            task_config(base).validate_compatibility(
                base,
                ObservationConfig(critic_capacity=2),
                RewardConfig(),
            )
        with self.assertRaisesRegex(ValueError, "terminal separation"):
            task_config(base, terminal_separation_m=0.55).validate_compatibility(
                base,
                ObservationConfig(),
                RewardConfig(),
            )
        with self.assertRaisesRegex(ValueError, "timestep"):
            task_config(base).validate_compatibility(
                base,
                ObservationConfig(),
                RewardConfig(control_dt_seconds=0.02),
            )


class BatchedTaskEnvironmentTests(unittest.TestCase):
    def test_observation_shapes_are_fixed_and_finite(self):
        env = environment()
        positions = repeated(env.layout.ground_positions_m, 4)
        velocities = repeated(repeated((0.0, 0.0, 0.0), 4), 4)
        observations = env.observe(positions, velocities)
        self.assertEqual(len(observations), 4)
        for item in observations:
            self.assertEqual(len(item.actors), 4)
            self.assertTrue(all(len(actor.flat()) == 55 for actor in item.actors))
            self.assertEqual(len(item.critic.flat()), 80)
            self.assertTrue(
                all(math.isfinite(value) for actor in item.actors for value in actor.flat())
            )
            self.assertTrue(all(math.isfinite(value) for value in item.critic.flat()))

    def test_true_termination_and_time_limit_truncation_are_distinct(self):
        env = environment(num_envs=2, max_episode_steps=1)
        safe = env.layout.ground_positions_m
        collision = list(safe)
        collision[1] = collision[0]
        positions = (tuple(collision), safe)
        velocities = repeated(repeated((0.0, 0.0, 0.0), 4), 2)
        actions = repeated(repeated((0.0, 0.0, 0.0, 0.0), 4), 2)
        contacts = repeated((0.0,) * 4, 2)
        result = env.step(positions, velocities, actions, contacts)
        self.assertEqual(result.terminated, (True, False))
        self.assertEqual(result.truncated, (False, True))
        self.assertEqual(result.done, (True, True))
        self.assertEqual(result.reasons, ("separation_violation", "time_limit"))

    def test_phase_change_resets_reward_memory(self):
        base = construction(
            ground_settle_seconds=0.01,
            takeoff_seconds=0.01,
            formation_timeout_seconds=0.04,
            dwell_seconds=0.01,
        )
        env = environment(num_envs=1, max_episode_steps=5, base=base)
        velocities = (repeated((0.0, 0.0, 0.0), 4),)
        actions = (repeated((0.0, 0.0, 0.0, 0.0), 4),)
        contacts = ((0.0,) * 4,)
        first = env.step((env.layout.ground_positions_m,), velocities, actions, contacts)
        second = env.step((env.layout.takeoff_positions_m,), velocities, actions, contacts)
        self.assertEqual(first.reward_phases, ("ground",))
        self.assertEqual(first.observation_phases, ("takeoff",))
        self.assertEqual(second.reward_phases, ("takeoff",))
        self.assertTrue(
            all(component.progress == 0.0 for component in second.rewards[0].raw_by_agent)
        )

    def test_partial_reset_clears_only_selected_temporal_state(self):
        env = environment(num_envs=2, max_episode_steps=10)
        offset = tuple((x, y, z + 0.1) for x, y, z in env.layout.ground_positions_m)
        velocities = repeated(repeated((0.0, 0.0, 0.0), 4), 2)
        zero_actions = repeated(repeated((0.0, 0.0, 0.0, 0.0), 4), 2)
        contacts = repeated((0.0,) * 4, 2)
        env.step((offset, offset), velocities, zero_actions, contacts)

        outside = tuple((x + 6.0, y, z) for x, y, z in env.layout.ground_positions_m)
        moving = repeated((1.0, 0.0, 0.0, 0.2), 4)
        result = env.step(
            (outside, env.layout.ground_positions_m),
            velocities,
            (zero_actions[0], moving),
            contacts,
        )
        self.assertEqual(result.reasons, ("safety_envelope", None))
        self.assertEqual(result.episode_steps, (2, 2))

        env.reset((0,))
        result = env.step(
            (env.layout.ground_positions_m, env.layout.ground_positions_m),
            velocities,
            zero_actions,
            contacts,
        )
        self.assertEqual(result.episode_steps, (1, 3))
        self.assertTrue(
            all(component.progress == 0.0 for component in result.rewards[0].raw_by_agent)
        )
        self.assertTrue(
            all(component.smoothness == 0.0 for component in result.rewards[0].raw_by_agent)
        )
        self.assertTrue(
            all(component.smoothness < 0.0 for component in result.rewards[1].raw_by_agent)
        )

    def test_success_requires_consecutive_formation_dwell(self):
        base = construction(
            ground_settle_seconds=0.01,
            takeoff_seconds=0.01,
            formation_timeout_seconds=0.05,
            dwell_seconds=0.02,
        )
        env = environment(
            num_envs=1,
            max_episode_steps=7,
            base=base,
            success_dwell_steps=2,
        )
        velocities = (repeated((0.0, 0.0, 0.0), 4),)
        actions = (repeated((0.0, 0.0, 0.0, 0.0), 4),)
        contacts = ((0.0,) * 4,)
        for phase in ("ground", "takeoff", "formation"):
            result = env.step(
                (targets_for_phase(env.layout, phase),),
                velocities,
                actions,
                contacts,
            )
        self.assertEqual(result.success_dwell_steps, (1,))
        result = env.step(
            (targets_for_phase(env.layout, "formation"),),
            velocities,
            actions,
            contacts,
        )
        self.assertEqual(result.terminated, (True,))
        self.assertEqual(result.truncated, (False,))
        self.assertEqual(result.reasons, ("success",))

    def test_done_environment_must_be_reset_before_reuse(self):
        env = environment(num_envs=1, max_episode_steps=1)
        positions = (env.layout.ground_positions_m,)
        velocities = (repeated((0.0, 0.0, 0.0), 4),)
        actions = (repeated((0.0, 0.0, 0.0, 0.0), 4),)
        contacts = ((0.0,) * 4,)
        env.step(positions, velocities, actions, contacts)
        with self.assertRaisesRegex(RuntimeError, "must be reset"):
            env.step(positions, velocities, actions, contacts)
        env.reset((0,))
        self.assertFalse(env.step(positions, velocities, actions, contacts).terminated[0])


if __name__ == "__main__":
    unittest.main()
