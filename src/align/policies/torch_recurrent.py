"""PyTorch recurrent actor, centralized critic, and bounded action distribution.

This module is imported only inside the pinned simulator image, whose vendor
Python supplies PyTorch 2.2.2+cu118. Import ``align.policies`` for host-safe
configuration without importing PyTorch.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from align.policies.config import RecurrentPolicyConfig


class RecurrentState(NamedTuple):
    """LSTM memory in PyTorch's [layers, batch, hidden] convention."""

    hidden: torch.Tensor
    cell: torch.Tensor


class ActorOutput(NamedTuple):
    """Unsquashed Gaussian parameters and memory after a sequence."""

    latent_mean: torch.Tensor
    log_std: torch.Tensor
    state: RecurrentState


class PolicyAction(NamedTuple):
    """Executed bounded action, its transformed log probability, and memory."""

    action: torch.Tensor
    log_prob: torch.Tensor
    state: RecurrentState


class CriticOutput(NamedTuple):
    """Centralized state-value sequence and critic memory."""

    value: torch.Tensor
    state: RecurrentState


def _require_sequence(value: torch.Tensor, feature_dim: int, name: str) -> tuple[int, int]:
    if value.ndim != 3 or value.shape[-1] != feature_dim:
        raise ValueError(f"{name} must have shape [batch, time, {feature_dim}]")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.shape[0], value.shape[1]


def _checked_mask(
    mask: torch.Tensor | None,
    *,
    batch: int,
    time: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if mask is None:
        return reference.new_ones((batch, time))
    if mask.shape != (batch, time):
        raise ValueError(f"{name} must have shape [{batch}, {time}]")
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} must contain only zero or one")
    return mask


class MaskedLSTM(nn.Module):
    """An LSTM unrolled over real time with explicit reset and padding masks."""

    def __init__(self, input_size: int, hidden_size: int, layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(layers)
            ]
        )
        for cell in self.cells:
            nn.init.orthogonal_(cell.weight_ih)
            nn.init.orthogonal_(cell.weight_hh)
            nn.init.zeros_(cell.bias_ih)
            nn.init.zeros_(cell.bias_hh)

    def initial_state(self, reference: torch.Tensor, batch: int) -> RecurrentState:
        shape = (self.layers, batch, self.hidden_size)
        return RecurrentState(reference.new_zeros(shape), reference.new_zeros(shape))

    def _checked_state(
        self, state: RecurrentState | None, reference: torch.Tensor, batch: int
    ) -> RecurrentState:
        if state is None:
            return self.initial_state(reference, batch)
        expected = (self.layers, batch, self.hidden_size)
        if state.hidden.shape != expected or state.cell.shape != expected:
            raise ValueError(f"recurrent state tensors must have shape {list(expected)}")
        hidden = state.hidden.to(device=reference.device, dtype=reference.dtype)
        cell = state.cell.to(device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(cell).all()):
            raise ValueError("recurrent state must contain only finite values")
        return RecurrentState(hidden, cell)

    def forward(
        self,
        inputs: torch.Tensor,
        state: RecurrentState | None = None,
        memory_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        batch, time, _ = inputs.shape
        memory = _checked_mask(
            memory_mask,
            batch=batch,
            time=time,
            reference=inputs,
            name="memory_mask",
        )
        valid = _checked_mask(
            valid_mask,
            batch=batch,
            time=time,
            reference=inputs,
            name="valid_mask",
        )
        if time > 1 and bool(((valid[:, 1:] - valid[:, :-1]) > 0).any()):
            raise ValueError("valid_mask padding must be trailing")
        checked = self._checked_state(state, inputs, batch)
        hidden = list(checked.hidden.unbind(0))
        cell = list(checked.cell.unbind(0))
        outputs = []
        for step in range(time):
            carry = memory[:, step].unsqueeze(-1)
            is_valid = valid[:, step].unsqueeze(-1)
            layer_input = inputs[:, step]
            for layer, recurrent_cell in enumerate(self.cells):
                previous_h = hidden[layer]
                previous_c = cell[layer]
                candidate_h, candidate_c = recurrent_cell(
                    layer_input, (previous_h * carry, previous_c * carry)
                )
                hidden[layer] = is_valid * candidate_h + (1.0 - is_valid) * previous_h
                cell[layer] = is_valid * candidate_c + (1.0 - is_valid) * previous_c
                layer_input = hidden[layer]
            outputs.append(layer_input * is_valid)
        return torch.stack(outputs, dim=1), RecurrentState(
            torch.stack(hidden, dim=0), torch.stack(cell, dim=0)
        )


class BoundedTanhNormal:
    """Diagonal Normal followed by tanh and an affine action-bounds map."""

    def __init__(
        self,
        latent_mean: torch.Tensor,
        log_std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        epsilon: float,
    ):
        if latent_mean.shape != log_std.shape:
            raise ValueError("latent_mean and log_std must have identical shapes")
        if low.shape != (latent_mean.shape[-1],) or high.shape != low.shape:
            raise ValueError("action bounds must match the final action dimension")
        self.latent_mean = latent_mean
        self.log_std = log_std
        self.low = low.to(device=latent_mean.device, dtype=latent_mean.dtype)
        self.high = high.to(device=latent_mean.device, dtype=latent_mean.dtype)
        self.scale = (self.high - self.low) / 2.0
        self.bias = (self.high + self.low) / 2.0
        self.epsilon = epsilon
        self.base = torch.distributions.Normal(latent_mean, log_std.exp())

    @staticmethod
    def _log_one_minus_tanh_squared(latent: torch.Tensor) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))

    def _transform(self, latent: torch.Tensor) -> torch.Tensor:
        return self.bias + self.scale * torch.tanh(latent)

    def _log_prob_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        correction = self.scale.log() + self._log_one_minus_tanh_squared(latent)
        return (self.base.log_prob(latent) - correction).sum(dim=-1)

    def rsample_with_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.base.rsample()
        return self._transform(latent), self._log_prob_from_latent(latent)

    def deterministic_with_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._transform(self.latent_mean), self._log_prob_from_latent(self.latent_mean)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != self.latent_mean.shape:
            raise ValueError("action shape must match the policy distribution")
        normalized = (action - self.bias) / self.scale
        if bool(((normalized < -1.0 - self.epsilon) | (normalized > 1.0 + self.epsilon)).any()):
            raise ValueError("action lies outside the configured bounds")
        normalized = normalized.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        latent = torch.atanh(normalized)
        return self._log_prob_from_latent(latent)


class _RecurrentBackbone(nn.Module):
    def __init__(self, input_dim: int, config: RecurrentPolicyConfig):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, config.encoder_hidden_size)
        self.feature_norm = nn.LayerNorm(config.encoder_hidden_size)
        self.recurrent = MaskedLSTM(
            config.encoder_hidden_size,
            config.recurrent_hidden_size,
            config.recurrent_layers,
        )
        self.recurrent_norm = nn.LayerNorm(config.recurrent_hidden_size)
        nn.init.orthogonal_(self.input_projection.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.input_projection.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        state: RecurrentState | None,
        memory_mask: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        features = self.feature_norm(torch.tanh(self.input_projection(self.input_norm(inputs))))
        sequence, next_state = self.recurrent(features, state, memory_mask, valid_mask)
        return self.recurrent_norm(sequence), next_state


class SharedRecurrentActor(nn.Module):
    """One actor reused for every drone; its only data input is local observation."""

    def __init__(self, config: RecurrentPolicyConfig):
        super().__init__()
        self.config = config
        self.backbone = _RecurrentBackbone(config.actor_observation_dim, config)
        self.action_mean = nn.Linear(config.recurrent_hidden_size, config.action_dim)
        self.log_std_parameter = nn.Parameter(
            torch.full((config.action_dim,), config.log_std_initial)
        )
        self.register_buffer("action_low", torch.tensor(config.action_low))
        self.register_buffer("action_high", torch.tensor(config.action_high))
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

    def forward(
        self,
        observations: torch.Tensor,
        state: RecurrentState | None = None,
        memory_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> ActorOutput:
        _require_sequence(observations, self.config.actor_observation_dim, "observations")
        features, next_state = self.backbone(observations, state, memory_mask, valid_mask)
        latent_mean = self.action_mean(features)
        bounded_log_std = self.log_std_parameter.clamp(
            self.config.log_std_min, self.config.log_std_max
        )
        return ActorOutput(latent_mean, bounded_log_std.expand_as(latent_mean), next_state)

    def distribution(self, output: ActorOutput) -> BoundedTanhNormal:
        return BoundedTanhNormal(
            output.latent_mean,
            output.log_std,
            self.action_low,
            self.action_high,
            self.config.squash_epsilon,
        )

    def act(
        self,
        observations: torch.Tensor,
        state: RecurrentState | None = None,
        memory_mask: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyAction:
        output = self(observations, state, memory_mask)
        distribution = self.distribution(output)
        if deterministic:
            action, log_prob = distribution.deterministic_with_log_prob()
        else:
            action, log_prob = distribution.rsample_with_log_prob()
        return PolicyAction(action, log_prob, output.state)


class CentralizedRecurrentCritic(nn.Module):
    """Training-only recurrent state-value model over the declared global state."""

    def __init__(self, config: RecurrentPolicyConfig):
        super().__init__()
        self.config = config
        self.backbone = _RecurrentBackbone(config.critic_state_dim, config)
        self.value_head = nn.Linear(config.recurrent_hidden_size, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        states: torch.Tensor,
        recurrent_state: RecurrentState | None = None,
        memory_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> CriticOutput:
        _require_sequence(states, self.config.critic_state_dim, "states")
        features, next_state = self.backbone(states, recurrent_state, memory_mask, valid_mask)
        return CriticOutput(self.value_head(features).squeeze(-1), next_state)
