"""CPU-only episode contract joining observations, rewards, and task outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields

from align.formations import evaluate_formation
from align.simulation.multi_drone_contract import (
    MultiDroneConfig,
    build_group_layout,
    phase_at,
    targets_for_phase,
)
from align.tasks.observation import ObservationBatch, ObservationConfig, build_observations
from align.tasks.reward import RewardConfig, RewardMemory, RewardStep, compute_step_reward


@dataclass(frozen=True)
class TaskEnvironmentConfig:
    """Episode limits and true terminal conditions for a vectorized task."""

    schema_version: int = 1
    num_envs: int = 4
    max_episode_steps: int = 160
    terminal_separation_m: float = 0.25
    crash_height_m: float = -0.05
    safety_xy_limit_m: float = 5.0
    safety_z_limit_m: float = 3.0
    airborne_height_m: float = 0.25
    contact_force_threshold_n: float = 0.01
    formation_rmse_tolerance_m: float = 0.10
    pairwise_rmse_tolerance_m: float = 0.08
    speed_tolerance_m_s: float = 0.08
    success_dwell_steps: int = 200

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for name in ("num_envs", "max_episode_steps", "success_dwell_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        positive = (
            self.terminal_separation_m,
            self.safety_xy_limit_m,
            self.safety_z_limit_m,
            self.airborne_height_m,
            self.contact_force_threshold_n,
            self.formation_rmse_tolerance_m,
            self.pairwise_rmse_tolerance_m,
            self.speed_tolerance_m_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("task thresholds must be finite and positive")
        if not math.isfinite(self.crash_height_m):
            raise ValueError("crash_height_m must be finite")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TaskEnvironmentConfig:
        expected = {field.name for field in fields(cls)}
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"task environment configuration keys mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate_compatibility(
        self,
        construction: MultiDroneConfig,
        observation: ObservationConfig,
        reward: RewardConfig,
    ) -> None:
        """Reject configuration combinations that silently change task meaning."""
        if observation.critic_capacity < construction.num_agents:
            raise ValueError("critic capacity is smaller than the task agent count")
        if not math.isclose(reward.control_dt_seconds, construction.physics_dt):
            raise ValueError("reward control timestep must equal the physics timestep")
        if not math.isclose(reward.max_speed_m_s, construction.max_speed_m_s):
            raise ValueError("reward and controller maximum speeds must match")
        if not math.isclose(reward.minimum_separation_m, construction.minimum_separation_m):
            raise ValueError("reward and construction safety margins must match")
        if self.terminal_separation_m >= reward.minimum_separation_m:
            raise ValueError("terminal separation must be below the shaped safety margin")
        pairs = (
            (self.safety_xy_limit_m, construction.safety_xy_limit_m, "XY safety limit"),
            (self.safety_z_limit_m, construction.safety_z_limit_m, "Z safety limit"),
            (self.airborne_height_m, construction.airborne_height_m, "airborne height"),
            (
                self.contact_force_threshold_n,
                construction.contact_force_threshold_n,
                "contact threshold",
            ),
            (
                self.formation_rmse_tolerance_m,
                construction.formation_rmse_tolerance_m,
                "formation tolerance",
            ),
            (
                self.pairwise_rmse_tolerance_m,
                construction.pairwise_rmse_tolerance_m,
                "pairwise tolerance",
            ),
            (self.speed_tolerance_m_s, construction.speed_tolerance_m_s, "speed tolerance"),
        )
        for task_value, construction_value, label in pairs:
            if not math.isclose(task_value, construction_value):
                raise ValueError(f"task and construction {label} must match")


@dataclass(frozen=True)
class TaskStep:
    """Results from one simultaneous transition in every active environment."""

    actor_observations: tuple[tuple[tuple[float, ...], ...], ...]
    critic_observations: tuple[tuple[float, ...], ...]
    rewards: tuple[RewardStep, ...]
    reward_phases: tuple[str, ...]
    observation_phases: tuple[str, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    done: tuple[bool, ...]
    reasons: tuple[str | None, ...]
    episode_steps: tuple[int, ...]
    success_dwell_steps: tuple[int, ...]


class BatchedTaskEnvironment:
    """Reference episode ledger with independent state for each environment."""

    def __init__(
        self,
        task: TaskEnvironmentConfig,
        construction: MultiDroneConfig,
        observation: ObservationConfig,
        reward: RewardConfig,
    ) -> None:
        task.validate_compatibility(construction, observation, reward)
        self.task = task
        self.construction = construction
        self.observation = observation
        self.reward = reward
        self.layout = build_group_layout(construction)
        self.agent_ids = self.layout.agent_ids
        self.episode_steps = [0] * task.num_envs
        self.success_dwell_steps = [0] * task.num_envs
        self._reward_memories = [RewardMemory() for _ in range(task.num_envs)]
        self._reward_phases: list[str | None] = [None] * task.num_envs
        self.active = [True] * task.num_envs

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear only selected episode ledgers and temporal reward memory."""
        selected = range(self.task.num_envs) if env_ids is None else env_ids
        identifiers = tuple(selected)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("reset environment IDs must be unique")
        for env_id in identifiers:
            if type(env_id) is not int or not 0 <= env_id < self.task.num_envs:
                raise ValueError("reset environment ID is outside the batch")
            self.episode_steps[env_id] = 0
            self.success_dwell_steps[env_id] = 0
            self._reward_memories[env_id] = RewardMemory()
            self._reward_phases[env_id] = None
            self.active[env_id] = True

    def observe(
        self,
        positions_m: Sequence[Sequence[Sequence[float]]],
        velocities_m_s: Sequence[Sequence[Sequence[float]]],
    ) -> tuple[ObservationBatch, ...]:
        positions, velocities = self._validate_state_batch(positions_m, velocities_m_s)
        observations = []
        for env_id, (env_positions, env_velocities) in enumerate(
            zip(positions, velocities, strict=True)
        ):
            phase = self._phase_for_observation(env_id)
            observations.append(
                build_observations(
                    self.agent_ids,
                    env_positions,
                    env_velocities,
                    targets_for_phase(self.layout, phase),
                    self.observation,
                )
            )
        return tuple(observations)

    def step(
        self,
        positions_m: Sequence[Sequence[Sequence[float]]],
        velocities_m_s: Sequence[Sequence[Sequence[float]]],
        actions: Sequence[Sequence[Sequence[float]]],
        contact_forces_n: Sequence[Sequence[float]],
    ) -> TaskStep:
        """Account for one post-physics transition in every active environment."""
        if not all(self.active):
            inactive = [index for index, active in enumerate(self.active) if not active]
            raise RuntimeError(f"done environments must be reset before stepping: {inactive}")
        positions, velocities = self._validate_state_batch(positions_m, velocities_m_s)
        if len(actions) != self.task.num_envs or len(contact_forces_n) != self.task.num_envs:
            raise ValueError("actions and contact forces need one row per environment")

        rewards = []
        reward_phases = []
        observation_phases = []
        terminated = []
        truncated = []
        reasons: list[str | None] = []
        for env_id in range(self.task.num_envs):
            phase = phase_at(self.construction, self.episode_steps[env_id])
            reward_phases.append(phase)
            if self._reward_phases[env_id] != phase:
                self._reward_memories[env_id] = RewardMemory()
                self._reward_phases[env_id] = phase
            targets = targets_for_phase(self.layout, phase)
            airborne = tuple(point[2] > self.task.airborne_height_m for point in positions[env_id])
            reward_step, memory = compute_step_reward(
                positions[env_id],
                targets,
                velocities[env_id],
                actions[env_id],
                contact_forces_n[env_id],
                airborne,
                self.reward,
                self._reward_memories[env_id],
            )
            self._reward_memories[env_id] = memory
            rewards.append(reward_step)
            reason = self._true_terminal_reason(
                phase,
                positions[env_id],
                velocities[env_id],
                targets,
                reward_step,
                env_id,
            )
            self.episode_steps[env_id] += 1
            is_terminated = reason is not None
            is_truncated = (
                not is_terminated and self.episode_steps[env_id] >= self.task.max_episode_steps
            )
            if is_truncated:
                reason = "time_limit"
            if is_terminated or is_truncated:
                self.active[env_id] = False
            terminated.append(is_terminated)
            truncated.append(is_truncated)
            reasons.append(reason)
            observation_phases.append(self._phase_for_observation(env_id))

        observations = self.observe(positions, velocities)
        return TaskStep(
            actor_observations=tuple(
                tuple(actor.flat() for actor in batch.actors) for batch in observations
            ),
            critic_observations=tuple(batch.critic.flat() for batch in observations),
            rewards=tuple(rewards),
            reward_phases=tuple(reward_phases),
            observation_phases=tuple(observation_phases),
            terminated=tuple(terminated),
            truncated=tuple(truncated),
            done=tuple(left or right for left, right in zip(terminated, truncated, strict=True)),
            reasons=tuple(reasons),
            episode_steps=tuple(self.episode_steps),
            success_dwell_steps=tuple(self.success_dwell_steps),
        )

    def _phase_for_observation(self, env_id: int) -> str:
        step = min(self.episode_steps[env_id], self.construction.maximum_steps - 1)
        return phase_at(self.construction, step)

    def _true_terminal_reason(
        self,
        phase,
        positions,
        velocities,
        targets,
        reward_step: RewardStep,
        env_id: int,
    ) -> str | None:
        if min(reward_step.minimum_separations_m) < self.task.terminal_separation_m:
            self.success_dwell_steps[env_id] = 0
            return "separation_violation"
        if any(reward_step.airborne_contact):
            self.success_dwell_steps[env_id] = 0
            return "airborne_contact"
        if any(
            abs(x) > self.task.safety_xy_limit_m
            or abs(y) > self.task.safety_xy_limit_m
            or z < self.task.crash_height_m
            or z > self.task.safety_z_limit_m
            for x, y, z in positions
        ):
            self.success_dwell_steps[env_id] = 0
            return "safety_envelope"

        metrics = evaluate_formation(positions, targets)
        max_speed = max(math.dist((0.0, 0.0, 0.0), velocity) for velocity in velocities)
        settled = (
            phase == "formation"
            and metrics.assigned_root_mean_squared_error_m <= self.task.formation_rmse_tolerance_m
            and metrics.pairwise_root_mean_squared_error_m <= self.task.pairwise_rmse_tolerance_m
            and max_speed <= self.task.speed_tolerance_m_s
        )
        self.success_dwell_steps[env_id] = self.success_dwell_steps[env_id] + 1 if settled else 0
        if self.success_dwell_steps[env_id] >= self.task.success_dwell_steps:
            return "success"
        return None

    def _validate_state_batch(self, positions_m, velocities_m_s):
        if len(positions_m) != self.task.num_envs or len(velocities_m_s) != self.task.num_envs:
            raise ValueError("state needs one row per environment")
        positions = tuple(self._vectors(env, "positions_m") for env in positions_m)
        velocities = tuple(self._vectors(env, "velocities_m_s") for env in velocities_m_s)
        expected = self.construction.num_agents
        if any(len(env) != expected for env in (*positions, *velocities)):
            raise ValueError("state needs one row per configured agent")
        return positions, velocities

    @staticmethod
    def _vectors(values, name):
        result = []
        for vector in values:
            if len(vector) != 3:
                raise ValueError(f"{name} vectors must contain three values")
            row = tuple(float(value) for value in vector)
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"{name} vectors must be finite")
            result.append(row)
        return tuple(result)
