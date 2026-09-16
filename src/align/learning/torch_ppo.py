"""Masked recurrent MAPPO losses and full-batch optimizer updates.

This module requires the pinned simulator image's vendor PyTorch. It does not
import or initialize Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.torch_rollout import TorchSequenceChunks
from align.policies.torch_recurrent import (
    ActorOutput,
    CentralizedRecurrentCritic,
    SharedRecurrentActor,
)


@dataclass(frozen=True)
class PPOLosses:
    actor: torch.Tensor
    critic: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    value_clip_fraction: torch.Tensor
    explained_variance: torch.Tensor


def _valid(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.ndim != 2 or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} must be a two-dimensional binary mask")
    selected = mask.bool()
    if not bool(selected.any()):
        raise ValueError(f"{name} must contain at least one valid timestep")
    return selected


def compute_ppo_losses(
    actor: SharedRecurrentActor,
    critic: CentralizedRecurrentCritic,
    chunks: TorchSequenceChunks,
    config: RecurrentPPOConfig,
) -> PPOLosses:
    """Evaluate whole temporal chunks and reduce only valid timesteps."""
    actor_rows = chunks.actor
    critic_rows = chunks.critic
    actor_valid = _valid(actor_rows.valid_mask, "actor valid_mask")
    critic_valid = _valid(critic_rows.valid_mask, "critic valid_mask")
    actor_output = actor(
        actor_rows.observations,
        actor_rows.initial_memory,
        valid_mask=actor_rows.valid_mask,
    )
    selected_output = ActorOutput(
        actor_output.latent_mean[actor_valid],
        actor_output.log_std[actor_valid],
        actor_output.state,
    )
    distribution = actor.distribution(selected_output)
    new_log_probs = distribution.log_prob(actor_rows.actions[actor_valid])
    old_log_probs = actor_rows.old_log_probs[actor_valid].detach()
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("nonfinite PPO importance ratio")

    advantages = actor_rows.advantages[actor_valid].detach()
    if config.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + config.advantage_epsilon
        )
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - config.policy_clip_ratio, 1 + config.policy_clip_ratio) * (advantages)
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    _, sampled_log_probs = distribution.rsample_with_log_prob()
    entropy = -sampled_log_probs.mean()
    actor_loss = policy_loss - config.entropy_coefficient * entropy
    approximate_kl = (ratio - 1 - log_ratio).mean()
    clip_fraction = ((ratio - 1).abs() > config.policy_clip_ratio).float().mean()

    predicted_values = critic(
        critic_rows.states,
        critic_rows.initial_memory,
        valid_mask=critic_rows.valid_mask,
    ).value[critic_valid]
    old_values = critic_rows.values[critic_valid].detach()
    returns = critic_rows.returns[critic_valid].detach()
    value_delta = predicted_values - old_values
    clipped_values = old_values + value_delta.clamp(
        -config.value_clip_range, config.value_clip_range
    )
    value_loss = (
        0.5
        * torch.maximum(
            (predicted_values - returns).square(), (clipped_values - returns).square()
        ).mean()
    )
    critic_loss = config.value_loss_coefficient * value_loss
    value_clip_fraction = (value_delta.abs() > config.value_clip_range).float().mean()
    return_variance = returns.var(unbiased=False)
    explained_variance = torch.where(
        return_variance > config.advantage_epsilon,
        1.0 - (returns - predicted_values).var(unbiased=False) / return_variance,
        return_variance.new_zeros(()),
    )
    losses = PPOLosses(
        actor_loss,
        critic_loss,
        policy_loss,
        value_loss,
        entropy,
        approximate_kl,
        clip_fraction,
        value_clip_fraction,
        explained_variance,
    )
    if not all(bool(torch.isfinite(value)) for value in vars(losses).values()):
        raise ValueError("nonfinite recurrent PPO loss or diagnostic")
    return losses


def _finite_gradients(module: nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    return bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
    )


def update_recurrent_ppo(
    actor: SharedRecurrentActor,
    critic: CentralizedRecurrentCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    chunks: TorchSequenceChunks,
    config: RecurrentPPOConfig,
) -> list[dict]:
    """Run configured full-batch epochs; return measured per-epoch diagnostics."""
    results = []
    for epoch in range(config.update_epochs):
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        losses = compute_ppo_losses(actor, critic, chunks, config)
        losses.actor.backward()
        losses.critic.backward()
        if not _finite_gradients(actor) or not _finite_gradients(critic):
            raise ValueError("missing or nonfinite recurrent PPO gradient")
        actor_norm = nn.utils.clip_grad_norm_(actor.parameters(), config.max_gradient_norm)
        critic_norm = nn.utils.clip_grad_norm_(critic.parameters(), config.max_gradient_norm)
        if not bool(torch.isfinite(actor_norm)) or not bool(torch.isfinite(critic_norm)):
            raise ValueError("nonfinite recurrent PPO gradient norm")
        actor_optimizer.step()
        critic_optimizer.step()
        results.append(
            {
                "epoch": epoch + 1,
                "actor_loss": float(losses.actor.detach().cpu()),
                "critic_loss": float(losses.critic.detach().cpu()),
                "policy_loss": float(losses.policy.detach().cpu()),
                "value_loss": float(losses.value.detach().cpu()),
                "sampled_action_entropy": float(losses.entropy.detach().cpu()),
                "approximate_kl": float(losses.approximate_kl.detach().cpu()),
                "policy_clip_fraction": float(losses.clip_fraction.detach().cpu()),
                "value_clip_fraction": float(losses.value_clip_fraction.detach().cpu()),
                "explained_variance": float(losses.explained_variance.detach().cpu()),
                "actor_gradient_norm_before_clip": float(actor_norm.detach().cpu()),
                "critic_gradient_norm_before_clip": float(critic_norm.detach().cpu()),
            }
        )
    return results
