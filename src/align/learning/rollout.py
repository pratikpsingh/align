"""CPU reference contract for temporally correct recurrent MAPPO rollouts."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TypeAlias

Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[Vector, ...]
AgentTensor: TypeAlias = tuple[Matrix, ...]
ActorRecurrentTensor: TypeAlias = tuple[tuple[Matrix, ...], ...]
CriticRecurrentTensor: TypeAlias = tuple[Matrix, ...]


def _finite(values, name: str) -> None:
    def visit(value):
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif not math.isfinite(float(value)):
            raise ValueError(f"{name} must contain only finite values")

    visit(values)


def _shape_3(values, outer: int, middle: int, inner: int, name: str) -> None:
    if len(values) != outer:
        raise ValueError(f"{name} must have {outer} environments")
    if any(len(row) != middle for row in values):
        raise ValueError(f"{name} must have {middle} agents per environment")
    if any(len(vector) != inner for row in values for vector in row):
        raise ValueError(f"{name} vectors must have length {inner}")
    _finite(values, name)


def _shape_2(values, outer: int, inner: int, name: str) -> None:
    if len(values) != outer or any(len(row) != inner for row in values):
        raise ValueError(f"{name} must have shape [{outer}, {inner}]")
    _finite(values, name)


def _shape_vector(values, length: int, name: str) -> None:
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}")
    _finite(values, name)


def _shape_actor_recurrent(
    values, environments: int, agents: int, layers: int, hidden: int, name: str
) -> None:
    if len(values) != environments:
        raise ValueError(f"{name} must have {environments} environments")
    if any(len(env) != agents for env in values):
        raise ValueError(f"{name} must have {agents} agents per environment")
    if any(len(agent) != layers for env in values for agent in env):
        raise ValueError(f"{name} must have {layers} recurrent layers")
    if any(len(layer) != hidden for env in values for agent in env for layer in agent):
        raise ValueError(f"{name} recurrent vectors must have length {hidden}")
    _finite(values, name)


def _shape_critic_recurrent(values, environments: int, layers: int, hidden: int, name: str) -> None:
    if len(values) != environments:
        raise ValueError(f"{name} must have {environments} environments")
    if any(len(env) != layers for env in values):
        raise ValueError(f"{name} must have {layers} recurrent layers")
    if any(len(layer) != hidden for env in values for layer in env):
        raise ValueError(f"{name} recurrent vectors must have length {hidden}")
    _finite(values, name)


def _all_zero(values) -> bool:
    if isinstance(values, (tuple, list)):
        return all(_all_zero(value) for value in values)
    return float(values) == 0.0


@dataclass(frozen=True)
class RolloutConfig:
    """Fixed dimensions and return-estimation settings for one rollout."""

    horizon: int = 128
    num_envs: int = 4
    num_agents: int = 4
    actor_observation_dim: int = 55
    critic_state_dim: int = 80
    action_dim: int = 4
    recurrent_layers: int = 1
    recurrent_hidden_size: int = 256
    chunk_length: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95

    def __post_init__(self) -> None:
        integer_fields = (
            "horizon",
            "num_envs",
            "num_agents",
            "actor_observation_dim",
            "critic_state_dim",
            "action_dim",
            "recurrent_layers",
            "recurrent_hidden_size",
            "chunk_length",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_length > self.horizon:
            raise ValueError("chunk_length cannot exceed horizon")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.gae_lambda) or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be finite and in [0, 1]")

    @classmethod
    def from_dict(cls, values: dict) -> RolloutConfig:
        expected = set(cls.__dataclass_fields__)
        unknown = set(values) - expected
        missing = expected - set(values)
        if unknown or missing:
            raise ValueError(
                "rollout configuration keys mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**values)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def validate_dimensions(
        self,
        *,
        num_envs: int,
        num_agents: int,
        actor_observation_dim: int,
        critic_state_dim: int,
        action_dim: int,
    ) -> None:
        expected = {
            "num_envs": num_envs,
            "num_agents": num_agents,
            "actor_observation_dim": actor_observation_dim,
            "critic_state_dim": critic_state_dim,
            "action_dim": action_dim,
        }
        mismatches = {
            name: (getattr(self, name), value)
            for name, value in expected.items()
            if getattr(self, name) != value
        }
        if mismatches:
            raise ValueError(f"rollout dimensions do not match task contract: {mismatches}")


@dataclass(frozen=True)
class RecurrentFrame:
    """Inputs and recurrent states immediately before one environment action."""

    actor_observations: AgentTensor
    critic_states: Matrix
    actor_hidden: ActorRecurrentTensor
    actor_cell: ActorRecurrentTensor
    critic_hidden: CriticRecurrentTensor
    critic_cell: CriticRecurrentTensor


@dataclass(frozen=True)
class RolloutTransition:
    """Executed actions plus cooperative rewards and pre-reset bootstrap values."""

    actions: AgentTensor
    old_log_probs: Matrix
    team_rewards: Vector
    values: Vector
    bootstrap_values: Vector
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]


@dataclass(frozen=True)
class ActorSequenceChunk:
    """One padded, episode-contained sequence for one shared-policy actor lane."""

    environment_id: int
    agent_id: int
    start_step: int
    valid_length: int
    valid_mask: Vector
    actor_observations: Matrix
    actions: Matrix
    old_log_probs: Vector
    values: Vector
    advantages: Vector
    returns: Vector
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    initial_hidden: Matrix
    initial_cell: Matrix


@dataclass(frozen=True)
class CriticSequenceChunk:
    """One padded, episode-contained centralized critic sequence."""

    environment_id: int
    start_step: int
    valid_length: int
    valid_mask: Vector
    critic_states: Matrix
    team_rewards: Vector
    values: Vector
    advantages: Vector
    returns: Vector
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    initial_hidden: Matrix
    initial_cell: Matrix


@dataclass(frozen=True)
class SequenceChunks:
    actor: tuple[ActorSequenceChunk, ...]
    critic: tuple[CriticSequenceChunk, ...]


class RecurrentRollout:
    """Validated time-major storage with separate actor and team-critic lanes."""

    def __init__(self, config: RolloutConfig, initial_frame: RecurrentFrame):
        self.config = config
        self._validate_frame(initial_frame, "initial_frame")
        self.frames = [initial_frame]
        self.transitions: list[RolloutTransition] = []
        self._advantages: tuple[Vector, ...] | None = None
        self._returns: tuple[Vector, ...] | None = None

    @property
    def full(self) -> bool:
        return len(self.transitions) == self.config.horizon

    def append(self, transition: RolloutTransition, next_frame: RecurrentFrame) -> None:
        if self.full:
            raise RuntimeError("rollout is already full")
        self._validate_transition(transition)
        self._validate_frame(next_frame, "next_frame")
        for env_id, (terminated, truncated) in enumerate(
            zip(transition.terminated, transition.truncated, strict=True)
        ):
            if terminated and truncated:
                raise ValueError("terminated and truncated must be mutually exclusive")
            if terminated and transition.bootstrap_values[env_id] != 0.0:
                raise ValueError("true terminations require zero bootstrap values")
            if (terminated or truncated) and not self._environment_memory_is_zero(
                next_frame, env_id
            ):
                raise ValueError("actor and critic memory must reset after every episode boundary")
        self.transitions.append(transition)
        self.frames.append(next_frame)
        self._advantages = None
        self._returns = None

    def compute_gae(self) -> tuple[Vector, ...]:
        """Compute cooperative GAE with separate bootstrap and trace masks."""
        if not self.full:
            raise RuntimeError("rollout must be full before computing advantages")
        cfg = self.config
        advantages = [[0.0 for _ in range(cfg.num_envs)] for _ in range(cfg.horizon)]
        running = [0.0 for _ in range(cfg.num_envs)]
        for step in range(cfg.horizon - 1, -1, -1):
            item = self.transitions[step]
            for env_id in range(cfg.num_envs):
                bootstrap_mask = 0.0 if item.terminated[env_id] else 1.0
                trace_mask = 0.0 if (item.terminated[env_id] or item.truncated[env_id]) else 1.0
                delta = (
                    item.team_rewards[env_id]
                    + cfg.gamma * bootstrap_mask * item.bootstrap_values[env_id]
                    - item.values[env_id]
                )
                running[env_id] = delta + cfg.gamma * cfg.gae_lambda * trace_mask * running[env_id]
                advantages[step][env_id] = running[env_id]
        self._advantages = tuple(tuple(envs) for envs in advantages)
        self._returns = tuple(
            tuple(
                self._advantages[step][env] + self.transitions[step].values[env]
                for env in range(cfg.num_envs)
            )
            for step in range(cfg.horizon)
        )
        return self._advantages

    @property
    def returns(self) -> tuple[Vector, ...]:
        if self._returns is None:
            raise RuntimeError("compute_gae must run before reading returns")
        return self._returns

    def sequence_chunks(self) -> SequenceChunks:
        """Split actor and critic lanes at boundaries and pad only trailing rows."""
        if self._advantages is None or self._returns is None:
            raise RuntimeError("compute_gae must run before creating sequence chunks")
        actor_chunks = []
        critic_chunks = []
        for env_id in range(self.config.num_envs):
            segment_start = 0
            for step, transition in enumerate(self.transitions):
                if transition.terminated[env_id] or transition.truncated[env_id]:
                    self._append_segment_chunks(
                        actor_chunks, critic_chunks, env_id, segment_start, step + 1
                    )
                    segment_start = step + 1
            if segment_start < self.config.horizon:
                self._append_segment_chunks(
                    actor_chunks,
                    critic_chunks,
                    env_id,
                    segment_start,
                    self.config.horizon,
                )
        return SequenceChunks(tuple(actor_chunks), tuple(critic_chunks))

    def shuffled_actor_minibatches(
        self, chunks_per_batch: int, seed: int
    ) -> tuple[tuple[ActorSequenceChunk, ...], ...]:
        return self._shuffled(self.sequence_chunks().actor, chunks_per_batch, seed)

    def shuffled_critic_minibatches(
        self, chunks_per_batch: int, seed: int
    ) -> tuple[tuple[CriticSequenceChunk, ...], ...]:
        return self._shuffled(self.sequence_chunks().critic, chunks_per_batch, seed)

    @staticmethod
    def _shuffled(chunks, chunks_per_batch, seed):
        if chunks_per_batch <= 0:
            raise ValueError("chunks_per_batch must be positive")
        values = list(chunks)
        random.Random(seed).shuffle(values)
        return tuple(
            tuple(values[start : start + chunks_per_batch])
            for start in range(0, len(values), chunks_per_batch)
        )

    def _append_segment_chunks(self, actor, critic, env_id: int, start: int, end: int) -> None:
        for chunk_start in range(start, end, self.config.chunk_length):
            chunk_end = min(chunk_start + self.config.chunk_length, end)
            critic.append(self._make_critic_chunk(env_id, chunk_start, chunk_end))
            for agent_id in range(self.config.num_agents):
                actor.append(self._make_actor_chunk(env_id, agent_id, chunk_start, chunk_end))

    def _padded(self, values, zero, valid_length):
        return tuple(values) + tuple(zero for _ in range(self.config.chunk_length - valid_length))

    def _common_chunk_values(self, env_id: int, start: int, end: int) -> dict:
        valid_length = end - start
        steps = range(start, end)
        return {
            "start_step": start,
            "valid_length": valid_length,
            "valid_mask": self._padded((1.0 for _ in steps), 0.0, valid_length),
            "values": self._padded(
                (self.transitions[t].values[env_id] for t in steps), 0.0, valid_length
            ),
            "advantages": self._padded(
                (self._advantages[t][env_id] for t in steps), 0.0, valid_length
            ),
            "returns": self._padded((self._returns[t][env_id] for t in steps), 0.0, valid_length),
            "terminated": self._padded(
                (self.transitions[t].terminated[env_id] for t in steps), False, valid_length
            ),
            "truncated": self._padded(
                (self.transitions[t].truncated[env_id] for t in steps), False, valid_length
            ),
        }

    def _make_actor_chunk(
        self, env_id: int, agent_id: int, start: int, end: int
    ) -> ActorSequenceChunk:
        valid_length = end - start
        steps = range(start, end)
        return ActorSequenceChunk(
            environment_id=env_id,
            agent_id=agent_id,
            actor_observations=self._padded(
                (self.frames[t].actor_observations[env_id][agent_id] for t in steps),
                (0.0,) * self.config.actor_observation_dim,
                valid_length,
            ),
            actions=self._padded(
                (self.transitions[t].actions[env_id][agent_id] for t in steps),
                (0.0,) * self.config.action_dim,
                valid_length,
            ),
            old_log_probs=self._padded(
                (self.transitions[t].old_log_probs[env_id][agent_id] for t in steps),
                0.0,
                valid_length,
            ),
            initial_hidden=self.frames[start].actor_hidden[env_id][agent_id],
            initial_cell=self.frames[start].actor_cell[env_id][agent_id],
            **self._common_chunk_values(env_id, start, end),
        )

    def _make_critic_chunk(self, env_id: int, start: int, end: int) -> CriticSequenceChunk:
        valid_length = end - start
        steps = range(start, end)
        return CriticSequenceChunk(
            environment_id=env_id,
            critic_states=self._padded(
                (self.frames[t].critic_states[env_id] for t in steps),
                (0.0,) * self.config.critic_state_dim,
                valid_length,
            ),
            team_rewards=self._padded(
                (self.transitions[t].team_rewards[env_id] for t in steps),
                0.0,
                valid_length,
            ),
            initial_hidden=self.frames[start].critic_hidden[env_id],
            initial_cell=self.frames[start].critic_cell[env_id],
            **self._common_chunk_values(env_id, start, end),
        )

    def _environment_memory_is_zero(self, frame: RecurrentFrame, env_id: int) -> bool:
        return all(
            _all_zero(memory[env_id])
            for memory in (
                frame.actor_hidden,
                frame.actor_cell,
                frame.critic_hidden,
                frame.critic_cell,
            )
        )

    def _validate_frame(self, frame: RecurrentFrame, name: str) -> None:
        cfg = self.config
        _shape_3(
            frame.actor_observations,
            cfg.num_envs,
            cfg.num_agents,
            cfg.actor_observation_dim,
            f"{name}.actor_observations",
        )
        _shape_2(frame.critic_states, cfg.num_envs, cfg.critic_state_dim, f"{name}.critic_states")
        for field in ("actor_hidden", "actor_cell"):
            _shape_actor_recurrent(
                getattr(frame, field),
                cfg.num_envs,
                cfg.num_agents,
                cfg.recurrent_layers,
                cfg.recurrent_hidden_size,
                f"{name}.{field}",
            )
        for field in ("critic_hidden", "critic_cell"):
            _shape_critic_recurrent(
                getattr(frame, field),
                cfg.num_envs,
                cfg.recurrent_layers,
                cfg.recurrent_hidden_size,
                f"{name}.{field}",
            )

    def _validate_transition(self, transition: RolloutTransition) -> None:
        cfg = self.config
        _shape_3(
            transition.actions,
            cfg.num_envs,
            cfg.num_agents,
            cfg.action_dim,
            "transition.actions",
        )
        if any(
            value < -1.0 or value > 1.0
            for env in transition.actions
            for agent in env
            for value in agent
        ):
            raise ValueError("transition.actions must be bounded to [-1, 1]")
        _shape_2(
            transition.old_log_probs,
            cfg.num_envs,
            cfg.num_agents,
            "transition.old_log_probs",
        )
        for field in ("team_rewards", "values", "bootstrap_values"):
            _shape_vector(getattr(transition, field), cfg.num_envs, f"transition.{field}")
        for field in ("terminated", "truncated"):
            values = getattr(transition, field)
            if len(values) != cfg.num_envs or any(type(value) is not bool for value in values):
                raise ValueError(f"transition.{field} must contain one bool per environment")
