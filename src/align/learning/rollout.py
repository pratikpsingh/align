"""CPU reference contract for temporally correct recurrent MAPPO rollouts."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TypeAlias

Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[Vector, ...]
AgentTensor: TypeAlias = tuple[Matrix, ...]
RecurrentTensor: TypeAlias = tuple[tuple[Matrix, ...], ...]


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


def _shape_recurrent(
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


def _all_zero(values) -> bool:
    return all(
        float(value) == 0.0 for env in values for agent in env for layer in agent for value in layer
    )


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
    recurrent_hidden_size: int = 128
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
    """Inputs and recurrent states before one environment action."""

    actor_observations: AgentTensor
    critic_states: Matrix
    actor_hidden: RecurrentTensor
    actor_cell: RecurrentTensor
    critic_hidden: RecurrentTensor
    critic_cell: RecurrentTensor


@dataclass(frozen=True)
class RolloutTransition:
    """Outputs of one action plus pre-reset value of its final observation."""

    actions: AgentTensor
    old_log_probs: Matrix
    rewards: Matrix
    values: Matrix
    bootstrap_values: Matrix
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]


@dataclass(frozen=True)
class SequenceChunk:
    """One padded, episode-contained sequence for a single shared-policy agent."""

    environment_id: int
    agent_id: int
    start_step: int
    valid_length: int
    valid_mask: Vector
    actor_observations: Matrix
    critic_states: Matrix
    actions: Matrix
    old_log_probs: Vector
    values: Vector
    advantages: Vector
    returns: Vector
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    initial_actor_hidden: Matrix
    initial_actor_cell: Matrix
    initial_critic_hidden: Matrix
    initial_critic_cell: Matrix


class RecurrentRollout:
    """Validated time-major storage with episode-safe recurrent chunks."""

    def __init__(self, config: RolloutConfig, initial_frame: RecurrentFrame):
        self.config = config
        self._validate_frame(initial_frame, "initial_frame")
        self.frames = [initial_frame]
        self.transitions: list[RolloutTransition] = []
        self._advantages: tuple[tuple[tuple[float, ...], ...], ...] | None = None
        self._returns: tuple[tuple[tuple[float, ...], ...], ...] | None = None

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
            if terminated and any(
                transition.bootstrap_values[env_id][agent] != 0.0
                for agent in range(self.config.num_agents)
            ):
                raise ValueError("true terminations require zero bootstrap values")
            if (terminated or truncated) and not self._environment_memory_is_zero(
                next_frame, env_id
            ):
                raise ValueError("recurrent state must reset after every episode boundary")
        self.transitions.append(transition)
        self.frames.append(next_frame)
        self._advantages = None
        self._returns = None

    def compute_gae(self) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Compute GAE using separate bootstrap and cross-episode trace masks."""
        if not self.full:
            raise RuntimeError("rollout must be full before computing advantages")
        cfg = self.config
        advantages = [
            [[0.0 for _ in range(cfg.num_agents)] for _ in range(cfg.num_envs)]
            for _ in range(cfg.horizon)
        ]
        running = [[0.0 for _ in range(cfg.num_agents)] for _ in range(cfg.num_envs)]
        for step in range(cfg.horizon - 1, -1, -1):
            item = self.transitions[step]
            for env_id in range(cfg.num_envs):
                bootstrap_mask = 0.0 if item.terminated[env_id] else 1.0
                trace_mask = 0.0 if (item.terminated[env_id] or item.truncated[env_id]) else 1.0
                for agent_id in range(cfg.num_agents):
                    delta = (
                        item.rewards[env_id][agent_id]
                        + cfg.gamma * bootstrap_mask * item.bootstrap_values[env_id][agent_id]
                        - item.values[env_id][agent_id]
                    )
                    running[env_id][agent_id] = (
                        delta + cfg.gamma * cfg.gae_lambda * trace_mask * running[env_id][agent_id]
                    )
                    advantages[step][env_id][agent_id] = running[env_id][agent_id]
        self._advantages = tuple(
            tuple(tuple(agent for agent in env) for env in step) for step in advantages
        )
        self._returns = tuple(
            tuple(
                tuple(
                    self._advantages[step][env][agent] + self.transitions[step].values[env][agent]
                    for agent in range(cfg.num_agents)
                )
                for env in range(cfg.num_envs)
            )
            for step in range(cfg.horizon)
        )
        return self._advantages

    @property
    def returns(self) -> tuple[tuple[tuple[float, ...], ...], ...]:
        if self._returns is None:
            raise RuntimeError("compute_gae must run before reading returns")
        return self._returns

    def sequence_chunks(self) -> tuple[SequenceChunk, ...]:
        """Split each agent trajectory into padded chunks that never cross episodes."""
        if self._advantages is None or self._returns is None:
            raise RuntimeError("compute_gae must run before creating sequence chunks")
        cfg = self.config
        chunks = []
        for env_id in range(cfg.num_envs):
            segment_start = 0
            for step, transition in enumerate(self.transitions):
                boundary = transition.terminated[env_id] or transition.truncated[env_id]
                if boundary:
                    chunks.extend(self._segment_chunks(env_id, segment_start, step + 1))
                    segment_start = step + 1
            if segment_start < cfg.horizon:
                chunks.extend(self._segment_chunks(env_id, segment_start, cfg.horizon))
        return tuple(chunks)

    def shuffled_minibatches(
        self, chunks_per_batch: int, seed: int
    ) -> tuple[tuple[SequenceChunk, ...], ...]:
        if chunks_per_batch <= 0:
            raise ValueError("chunks_per_batch must be positive")
        chunks = list(self.sequence_chunks())
        random.Random(seed).shuffle(chunks)
        return tuple(
            tuple(chunks[start : start + chunks_per_batch])
            for start in range(0, len(chunks), chunks_per_batch)
        )

    def _segment_chunks(self, env_id: int, start: int, end: int) -> list[SequenceChunk]:
        chunks = []
        for chunk_start in range(start, end, self.config.chunk_length):
            chunk_end = min(chunk_start + self.config.chunk_length, end)
            for agent_id in range(self.config.num_agents):
                chunks.append(self._make_chunk(env_id, agent_id, chunk_start, chunk_end))
        return chunks

    def _make_chunk(self, env_id: int, agent_id: int, start: int, end: int) -> SequenceChunk:
        cfg = self.config
        valid_length = end - start

        def padded(values, zero):
            return tuple(values) + tuple(zero for _ in range(cfg.chunk_length - valid_length))

        steps = range(start, end)
        return SequenceChunk(
            environment_id=env_id,
            agent_id=agent_id,
            start_step=start,
            valid_length=valid_length,
            valid_mask=padded((1.0 for _ in steps), 0.0),
            actor_observations=padded(
                (self.frames[t].actor_observations[env_id][agent_id] for t in steps),
                (0.0,) * cfg.actor_observation_dim,
            ),
            critic_states=padded(
                (self.frames[t].critic_states[env_id] for t in steps),
                (0.0,) * cfg.critic_state_dim,
            ),
            actions=padded(
                (self.transitions[t].actions[env_id][agent_id] for t in steps),
                (0.0,) * cfg.action_dim,
            ),
            old_log_probs=padded(
                (self.transitions[t].old_log_probs[env_id][agent_id] for t in steps), 0.0
            ),
            values=padded((self.transitions[t].values[env_id][agent_id] for t in steps), 0.0),
            advantages=padded((self._advantages[t][env_id][agent_id] for t in steps), 0.0),
            returns=padded((self._returns[t][env_id][agent_id] for t in steps), 0.0),
            terminated=padded((self.transitions[t].terminated[env_id] for t in steps), False),
            truncated=padded((self.transitions[t].truncated[env_id] for t in steps), False),
            initial_actor_hidden=self.frames[start].actor_hidden[env_id][agent_id],
            initial_actor_cell=self.frames[start].actor_cell[env_id][agent_id],
            initial_critic_hidden=self.frames[start].critic_hidden[env_id][agent_id],
            initial_critic_cell=self.frames[start].critic_cell[env_id][agent_id],
        )

    def _environment_memory_is_zero(self, frame: RecurrentFrame, env_id: int) -> bool:
        return all(
            _all_zero((memory[env_id],))
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
        for field in ("actor_hidden", "actor_cell", "critic_hidden", "critic_cell"):
            _shape_recurrent(
                getattr(frame, field),
                cfg.num_envs,
                cfg.num_agents,
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
        for field in ("old_log_probs", "rewards", "values", "bootstrap_values"):
            _shape_2(
                getattr(transition, field),
                cfg.num_envs,
                cfg.num_agents,
                f"transition.{field}",
            )
        for field in ("terminated", "truncated"):
            values = getattr(transition, field)
            if len(values) != cfg.num_envs or any(type(value) is not bool for value in values):
                raise ValueError(f"transition.{field} must contain one bool per environment")
