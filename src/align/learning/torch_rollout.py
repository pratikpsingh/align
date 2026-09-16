"""Device-resident recurrent rollout storage; requires vendor PyTorch."""

from __future__ import annotations

from typing import NamedTuple

import torch

from align.learning.rollout import (
    RecurrentFrame,
    RecurrentRollout,
    RolloutConfig,
    RolloutTransition,
)
from align.policies.torch_recurrent import RecurrentState


class TorchRecurrentFrame(NamedTuple):
    actor_observations: torch.Tensor
    critic_states: torch.Tensor
    actor_memory: RecurrentState
    critic_memory: RecurrentState


class TorchRolloutTransition(NamedTuple):
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    team_rewards: torch.Tensor
    values: torch.Tensor
    bootstrap_values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class TorchActorChunks(NamedTuple):
    environment_ids: torch.Tensor
    agent_ids: torch.Tensor
    start_steps: torch.Tensor
    valid_lengths: torch.Tensor
    valid_mask: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    initial_memory: RecurrentState


class TorchCriticChunks(NamedTuple):
    environment_ids: torch.Tensor
    start_steps: torch.Tensor
    valid_lengths: torch.Tensor
    valid_mask: torch.Tensor
    states: torch.Tensor
    team_rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    initial_memory: RecurrentState


class TorchSequenceChunks(NamedTuple):
    actor: TorchActorChunks
    critic: TorchCriticChunks


def _require_shape(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(tensor.shape)}")


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a finite floating-point tensor")


class TorchRecurrentRollout:
    """Preallocated time-major tensors for one cooperative recurrent rollout."""

    def __init__(self, config: RolloutConfig, initial_frame: TorchRecurrentFrame):
        self.config = config
        self._validate_frame(initial_frame, "initial_frame")
        cfg = config
        device = initial_frame.actor_observations.device
        dtype = initial_frame.actor_observations.dtype
        if initial_frame.critic_states.device != device:
            raise ValueError("actor and critic inputs must use the same device")
        self.device = device
        self.dtype = dtype
        self.cursor = 0
        self.actor_observations = torch.empty(
            cfg.horizon + 1,
            cfg.num_envs,
            cfg.num_agents,
            cfg.actor_observation_dim,
            device=device,
            dtype=dtype,
        )
        self.critic_states = torch.empty(
            cfg.horizon + 1,
            cfg.num_envs,
            cfg.critic_state_dim,
            device=device,
            dtype=dtype,
        )
        self.actor_hidden = torch.empty(
            cfg.horizon + 1,
            cfg.recurrent_layers,
            cfg.num_envs,
            cfg.num_agents,
            cfg.recurrent_hidden_size,
            device=device,
            dtype=dtype,
        )
        self.actor_cell = torch.empty_like(self.actor_hidden)
        self.critic_hidden = torch.empty(
            cfg.horizon + 1,
            cfg.recurrent_layers,
            cfg.num_envs,
            cfg.recurrent_hidden_size,
            device=device,
            dtype=dtype,
        )
        self.critic_cell = torch.empty_like(self.critic_hidden)
        self.actions = torch.empty(
            cfg.horizon,
            cfg.num_envs,
            cfg.num_agents,
            cfg.action_dim,
            device=device,
            dtype=dtype,
        )
        self.old_log_probs = torch.empty(
            cfg.horizon, cfg.num_envs, cfg.num_agents, device=device, dtype=dtype
        )
        self.team_rewards = torch.empty(cfg.horizon, cfg.num_envs, device=device, dtype=dtype)
        self.values = torch.empty_like(self.team_rewards)
        self.bootstrap_values = torch.empty_like(self.team_rewards)
        self.terminated = torch.empty(cfg.horizon, cfg.num_envs, device=device, dtype=torch.bool)
        self.truncated = torch.empty_like(self.terminated)
        self._advantages: torch.Tensor | None = None
        self._returns: torch.Tensor | None = None
        self._store_frame(0, initial_frame)

    @property
    def full(self) -> bool:
        return self.cursor == self.config.horizon

    @property
    def bytes_allocated(self) -> int:
        tensors = (
            self.actor_observations,
            self.critic_states,
            self.actor_hidden,
            self.actor_cell,
            self.critic_hidden,
            self.critic_cell,
            self.actions,
            self.old_log_probs,
            self.team_rewards,
            self.values,
            self.bootstrap_values,
            self.terminated,
            self.truncated,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def append(self, transition: TorchRolloutTransition, next_frame: TorchRecurrentFrame) -> None:
        if self.full:
            raise RuntimeError("rollout is already full")
        self._validate_transition(transition)
        self._validate_frame(next_frame, "next_frame")
        boundary = transition.terminated | transition.truncated
        if bool((transition.terminated & transition.truncated).any()):
            raise ValueError("terminated and truncated must be mutually exclusive")
        if bool((transition.bootstrap_values[transition.terminated] != 0).any()):
            raise ValueError("true terminations require zero bootstrap values")
        if bool(boundary.any()):
            actor_hidden = next_frame.actor_memory.hidden.view(
                self.config.recurrent_layers,
                self.config.num_envs,
                self.config.num_agents,
                self.config.recurrent_hidden_size,
            )
            actor_cell = next_frame.actor_memory.cell.view_as(actor_hidden)
            if bool((actor_hidden[:, boundary] != 0).any()) or bool(
                (actor_cell[:, boundary] != 0).any()
            ):
                raise ValueError("actor memory must reset after every episode boundary")
            if bool((next_frame.critic_memory.hidden[:, boundary] != 0).any()) or bool(
                (next_frame.critic_memory.cell[:, boundary] != 0).any()
            ):
                raise ValueError("critic memory must reset after every episode boundary")
        step = self.cursor
        with torch.no_grad():
            self.actions[step].copy_(transition.actions)
            self.old_log_probs[step].copy_(transition.old_log_probs)
            self.team_rewards[step].copy_(transition.team_rewards)
            self.values[step].copy_(transition.values)
            self.bootstrap_values[step].copy_(transition.bootstrap_values)
            self.terminated[step].copy_(transition.terminated)
            self.truncated[step].copy_(transition.truncated)
            self._store_frame(step + 1, next_frame)
        self.cursor += 1
        self._advantages = None
        self._returns = None

    def compute_gae(self) -> torch.Tensor:
        if not self.full:
            raise RuntimeError("rollout must be full before computing advantages")
        cfg = self.config
        advantages = torch.empty_like(self.team_rewards)
        running = torch.zeros(cfg.num_envs, device=self.device, dtype=self.dtype)
        for step in range(cfg.horizon - 1, -1, -1):
            bootstrap_mask = (~self.terminated[step]).to(self.dtype)
            trace_mask = (~(self.terminated[step] | self.truncated[step])).to(self.dtype)
            delta = (
                self.team_rewards[step]
                + cfg.gamma * bootstrap_mask * self.bootstrap_values[step]
                - self.values[step]
            )
            running = delta + cfg.gamma * cfg.gae_lambda * trace_mask * running
            advantages[step] = running
        self._advantages = advantages
        self._returns = advantages + self.values
        return advantages

    @property
    def returns(self) -> torch.Tensor:
        if self._returns is None:
            raise RuntimeError("compute_gae must run before reading returns")
        return self._returns

    def sequence_chunks(self) -> TorchSequenceChunks:
        if self._advantages is None or self._returns is None:
            raise RuntimeError("compute_gae must run before creating sequence chunks")
        actor_rows: list[dict] = []
        critic_rows: list[dict] = []
        boundaries = (self.terminated | self.truncated).detach().cpu()
        for env_id in range(self.config.num_envs):
            segment_start = 0
            for step in range(self.config.horizon):
                if bool(boundaries[step, env_id]):
                    self._append_tensor_segment(
                        actor_rows, critic_rows, env_id, segment_start, step + 1
                    )
                    segment_start = step + 1
            if segment_start < self.config.horizon:
                self._append_tensor_segment(
                    actor_rows,
                    critic_rows,
                    env_id,
                    segment_start,
                    self.config.horizon,
                )
        return TorchSequenceChunks(
            actor=self._stack_actor_chunks(actor_rows),
            critic=self._stack_critic_chunks(critic_rows),
        )

    def to_cpu_reference(self) -> RecurrentRollout:
        """Copy a completed rollout into the simulator-independent audit contract."""
        if not self.full:
            raise RuntimeError("rollout must be full before conversion")

        def frame_at(step: int) -> RecurrentFrame:
            actor_hidden = self.actor_hidden[step].permute(1, 2, 0, 3).cpu().tolist()
            actor_cell = self.actor_cell[step].permute(1, 2, 0, 3).cpu().tolist()
            critic_hidden = self.critic_hidden[step].permute(1, 0, 2).cpu().tolist()
            critic_cell = self.critic_cell[step].permute(1, 0, 2).cpu().tolist()
            return RecurrentFrame(
                actor_observations=self._tuples(self.actor_observations[step].cpu().tolist()),
                critic_states=self._tuples(self.critic_states[step].cpu().tolist()),
                actor_hidden=self._tuples(actor_hidden),
                actor_cell=self._tuples(actor_cell),
                critic_hidden=self._tuples(critic_hidden),
                critic_cell=self._tuples(critic_cell),
            )

        reference = RecurrentRollout(self.config, frame_at(0))
        for step in range(self.config.horizon):
            reference.append(
                RolloutTransition(
                    actions=self._tuples(self.actions[step].cpu().tolist()),
                    old_log_probs=self._tuples(self.old_log_probs[step].cpu().tolist()),
                    team_rewards=tuple(self.team_rewards[step].cpu().tolist()),
                    values=tuple(self.values[step].cpu().tolist()),
                    bootstrap_values=tuple(self.bootstrap_values[step].cpu().tolist()),
                    terminated=tuple(self.terminated[step].cpu().tolist()),
                    truncated=tuple(self.truncated[step].cpu().tolist()),
                ),
                frame_at(step + 1),
            )
        return reference

    @staticmethod
    def _tuples(value):
        if isinstance(value, list):
            return tuple(TorchRecurrentRollout._tuples(item) for item in value)
        return value

    def _append_tensor_segment(self, actor, critic, env_id: int, start: int, end: int) -> None:
        for chunk_start in range(start, end, self.config.chunk_length):
            chunk_end = min(chunk_start + self.config.chunk_length, end)
            critic.append(self._critic_chunk_row(env_id, chunk_start, chunk_end))
            for agent_id in range(self.config.num_agents):
                actor.append(self._actor_chunk_row(env_id, agent_id, chunk_start, chunk_end))

    def _pad(self, value: torch.Tensor, valid_length: int) -> torch.Tensor:
        result = value.new_zeros((self.config.chunk_length, *value.shape[1:]))
        result[:valid_length] = value
        return result

    def _common_tensor_row(self, env_id: int, start: int, end: int) -> dict:
        valid_length = end - start
        mask = torch.zeros(self.config.chunk_length, device=self.device, dtype=self.dtype)
        mask[:valid_length] = 1.0
        return {
            "environment_id": env_id,
            "start_step": start,
            "valid_length": valid_length,
            "valid_mask": mask,
            "values": self._pad(self.values[start:end, env_id], valid_length),
            "advantages": self._pad(self._advantages[start:end, env_id], valid_length),
            "returns": self._pad(self._returns[start:end, env_id], valid_length),
        }

    def _actor_chunk_row(self, env_id: int, agent_id: int, start: int, end: int) -> dict:
        valid_length = end - start
        return {
            "agent_id": agent_id,
            "observations": self._pad(
                self.actor_observations[start:end, env_id, agent_id], valid_length
            ),
            "actions": self._pad(self.actions[start:end, env_id, agent_id], valid_length),
            "old_log_probs": self._pad(
                self.old_log_probs[start:end, env_id, agent_id], valid_length
            ),
            "initial_hidden": self.actor_hidden[start, :, env_id, agent_id],
            "initial_cell": self.actor_cell[start, :, env_id, agent_id],
            **self._common_tensor_row(env_id, start, end),
        }

    def _critic_chunk_row(self, env_id: int, start: int, end: int) -> dict:
        valid_length = end - start
        return {
            "states": self._pad(self.critic_states[start:end, env_id], valid_length),
            "team_rewards": self._pad(self.team_rewards[start:end, env_id], valid_length),
            "initial_hidden": self.critic_hidden[start, :, env_id],
            "initial_cell": self.critic_cell[start, :, env_id],
            **self._common_tensor_row(env_id, start, end),
        }

    def _stack_actor_chunks(self, rows: list[dict]) -> TorchActorChunks:
        return TorchActorChunks(
            environment_ids=torch.tensor(
                [row["environment_id"] for row in rows], device=self.device
            ),
            agent_ids=torch.tensor([row["agent_id"] for row in rows], device=self.device),
            start_steps=torch.tensor([row["start_step"] for row in rows], device=self.device),
            valid_lengths=torch.tensor([row["valid_length"] for row in rows], device=self.device),
            valid_mask=torch.stack([row["valid_mask"] for row in rows]),
            observations=torch.stack([row["observations"] for row in rows]),
            actions=torch.stack([row["actions"] for row in rows]),
            old_log_probs=torch.stack([row["old_log_probs"] for row in rows]),
            values=torch.stack([row["values"] for row in rows]),
            advantages=torch.stack([row["advantages"] for row in rows]),
            returns=torch.stack([row["returns"] for row in rows]),
            initial_memory=RecurrentState(
                torch.stack([row["initial_hidden"] for row in rows], dim=1),
                torch.stack([row["initial_cell"] for row in rows], dim=1),
            ),
        )

    def _stack_critic_chunks(self, rows: list[dict]) -> TorchCriticChunks:
        return TorchCriticChunks(
            environment_ids=torch.tensor(
                [row["environment_id"] for row in rows], device=self.device
            ),
            start_steps=torch.tensor([row["start_step"] for row in rows], device=self.device),
            valid_lengths=torch.tensor([row["valid_length"] for row in rows], device=self.device),
            valid_mask=torch.stack([row["valid_mask"] for row in rows]),
            states=torch.stack([row["states"] for row in rows]),
            team_rewards=torch.stack([row["team_rewards"] for row in rows]),
            values=torch.stack([row["values"] for row in rows]),
            advantages=torch.stack([row["advantages"] for row in rows]),
            returns=torch.stack([row["returns"] for row in rows]),
            initial_memory=RecurrentState(
                torch.stack([row["initial_hidden"] for row in rows], dim=1),
                torch.stack([row["initial_cell"] for row in rows], dim=1),
            ),
        )

    def _store_frame(self, step: int, frame: TorchRecurrentFrame) -> None:
        cfg = self.config
        self.actor_observations[step].copy_(frame.actor_observations.detach())
        self.critic_states[step].copy_(frame.critic_states.detach())
        self.actor_hidden[step].copy_(
            frame.actor_memory.hidden.detach().view(
                cfg.recurrent_layers,
                cfg.num_envs,
                cfg.num_agents,
                cfg.recurrent_hidden_size,
            )
        )
        self.actor_cell[step].copy_(
            frame.actor_memory.cell.detach().view(
                cfg.recurrent_layers,
                cfg.num_envs,
                cfg.num_agents,
                cfg.recurrent_hidden_size,
            )
        )
        self.critic_hidden[step].copy_(frame.critic_memory.hidden.detach())
        self.critic_cell[step].copy_(frame.critic_memory.cell.detach())

    def _validate_frame(self, frame: TorchRecurrentFrame, name: str) -> None:
        cfg = self.config
        _require_shape(
            frame.actor_observations,
            (cfg.num_envs, cfg.num_agents, cfg.actor_observation_dim),
            f"{name}.actor_observations",
        )
        _require_shape(
            frame.critic_states,
            (cfg.num_envs, cfg.critic_state_dim),
            f"{name}.critic_states",
        )
        for field in ("actor_observations", "critic_states"):
            _require_finite(getattr(frame, field), f"{name}.{field}")
        actor_shape = (
            cfg.recurrent_layers,
            cfg.num_envs * cfg.num_agents,
            cfg.recurrent_hidden_size,
        )
        critic_shape = (cfg.recurrent_layers, cfg.num_envs, cfg.recurrent_hidden_size)
        for state_name, state, shape in (
            ("actor_memory", frame.actor_memory, actor_shape),
            ("critic_memory", frame.critic_memory, critic_shape),
        ):
            _require_shape(state.hidden, shape, f"{name}.{state_name}.hidden")
            _require_shape(state.cell, shape, f"{name}.{state_name}.cell")
            _require_finite(state.hidden, f"{name}.{state_name}.hidden")
            _require_finite(state.cell, f"{name}.{state_name}.cell")

    def _validate_transition(self, transition: TorchRolloutTransition) -> None:
        cfg = self.config
        expected = {
            "actions": (cfg.num_envs, cfg.num_agents, cfg.action_dim),
            "old_log_probs": (cfg.num_envs, cfg.num_agents),
            "team_rewards": (cfg.num_envs,),
            "values": (cfg.num_envs,),
            "bootstrap_values": (cfg.num_envs,),
            "terminated": (cfg.num_envs,),
            "truncated": (cfg.num_envs,),
        }
        for name, shape in expected.items():
            tensor = getattr(transition, name)
            _require_shape(tensor, shape, f"transition.{name}")
            if name in ("terminated", "truncated"):
                if tensor.dtype is not torch.bool:
                    raise ValueError(f"transition.{name} must be bool")
            else:
                _require_finite(tensor, f"transition.{name}")
        if bool(((transition.actions < -1.0) | (transition.actions > 1.0)).any()):
            raise ValueError("transition.actions must be bounded to [-1, 1]")
