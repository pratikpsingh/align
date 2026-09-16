"""Vendor-PyTorch acceptance probe for one recurrent MAPPO update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import torch

from align.artifacts import write_json_atomic
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.torch_ppo import compute_ppo_losses, update_recurrent_ppo
from align.learning.torch_rollout import (
    TorchActorChunks,
    TorchCriticChunks,
    TorchSequenceChunks,
)
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import (
    CentralizedRecurrentCritic,
    RecurrentState,
    SharedRecurrentActor,
)

SEED = 20260917
TOLERANCE = 3e-6


def _optimizer(module, learning_rate: float, config: RecurrentPPOConfig):
    return torch.optim.Adam(module.parameters(), lr=learning_rate, eps=config.adam_epsilon)


def _maximum_parameter_change(before: dict, module: torch.nn.Module) -> float:
    return max(
        float((value.detach() - before[name]).abs().max().cpu())
        for name, value in module.state_dict().items()
        if torch.is_floating_point(value)
    )


def _maximum_model_difference(left: torch.nn.Module, right: torch.nn.Module) -> float:
    return max(
        float((a.detach() - b.detach()).abs().max().cpu())
        for a, b in zip(left.state_dict().values(), right.state_dict().values(), strict=True)
        if torch.is_floating_point(a)
    )


def _artifact(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(
    actor: SharedRecurrentActor,
    critic: CentralizedRecurrentCritic,
    policy: RecurrentPolicyConfig,
    device: torch.device,
) -> TorchSequenceChunks:
    chunk_count, critic_count, length, agents = 8, 2, 4, 4
    generator = torch.Generator().manual_seed(SEED + 1)
    observations = torch.randn(
        chunk_count, length, policy.actor_observation_dim, generator=generator
    ).to(device)
    states = torch.randn(critic_count, length, policy.critic_state_dim, generator=generator).to(
        device
    )
    critic_valid = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], device=device, dtype=torch.float32)
    actor_valid = critic_valid.repeat_interleave(agents, dim=0)
    actor_memory = RecurrentState(
        torch.zeros(
            policy.recurrent_layers, chunk_count, policy.recurrent_hidden_size, device=device
        ),
        torch.zeros(
            policy.recurrent_layers, chunk_count, policy.recurrent_hidden_size, device=device
        ),
    )
    critic_memory = RecurrentState(
        torch.zeros(
            policy.recurrent_layers, critic_count, policy.recurrent_hidden_size, device=device
        ),
        torch.zeros(
            policy.recurrent_layers, critic_count, policy.recurrent_hidden_size, device=device
        ),
    )
    with torch.no_grad():
        actor_output = actor(observations, actor_memory, valid_mask=actor_valid)
        torch.manual_seed(SEED + 2)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED + 2)
        actions, old_log_probs = actor.distribution(actor_output).rsample_with_log_prob()
        old_values = critic(states, critic_memory, valid_mask=critic_valid).value
    team_advantages = torch.tensor([[-0.8, -0.3, 0.4, 1.0], [0.7, -0.5, 0.2, 0.0]], device=device)
    actor_advantages = team_advantages.repeat_interleave(agents, dim=0)
    actor_values = old_values.repeat_interleave(agents, dim=0)
    actor_returns = (old_values + team_advantages).repeat_interleave(agents, dim=0)
    return TorchSequenceChunks(
        actor=TorchActorChunks(
            environment_ids=torch.arange(critic_count, device=device).repeat_interleave(agents),
            agent_ids=torch.arange(agents, device=device).repeat(critic_count),
            start_steps=torch.zeros(chunk_count, device=device, dtype=torch.long),
            valid_lengths=actor_valid.sum(1).long(),
            valid_mask=actor_valid,
            observations=observations,
            actions=actions.detach(),
            old_log_probs=old_log_probs.detach(),
            values=actor_values.detach(),
            advantages=actor_advantages,
            returns=actor_returns.detach(),
            initial_memory=actor_memory,
        ),
        critic=TorchCriticChunks(
            environment_ids=torch.arange(critic_count, device=device),
            start_steps=torch.zeros(critic_count, device=device, dtype=torch.long),
            valid_lengths=critic_valid.sum(1).long(),
            valid_mask=critic_valid,
            states=states,
            team_rewards=torch.zeros_like(old_values),
            values=old_values.detach(),
            advantages=team_advantages,
            returns=(old_values + team_advantages).detach(),
            initial_memory=critic_memory,
        ),
    )


def _corrupt_padding(chunks: TorchSequenceChunks) -> TorchSequenceChunks:
    actor_invalid = ~chunks.actor.valid_mask.bool()
    critic_invalid = ~chunks.critic.valid_mask.bool()
    observations = chunks.actor.observations.clone()
    actions = chunks.actor.actions.clone()
    old_log_probs = chunks.actor.old_log_probs.clone()
    advantages = chunks.actor.advantages.clone()
    returns = chunks.actor.returns.clone()
    states = chunks.critic.states.clone()
    critic_values = chunks.critic.values.clone()
    critic_returns = chunks.critic.returns.clone()
    observations[actor_invalid] = 7.0
    actions[actor_invalid] = -7.0
    old_log_probs[actor_invalid] = 13.0
    advantages[actor_invalid] = -17.0
    returns[actor_invalid] = 19.0
    states[critic_invalid] = -5.0
    critic_values[critic_invalid] = 11.0
    critic_returns[critic_invalid] = -23.0
    return TorchSequenceChunks(
        actor=chunks.actor._replace(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
        ),
        critic=chunks.critic._replace(states=states, values=critic_values, returns=critic_returns),
    )


def evaluate(
    policy_config: RecurrentPolicyConfig,
    ppo_config: RecurrentPPOConfig,
    device: torch.device,
    output: Path,
) -> dict:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    actor = SharedRecurrentActor(policy_config).to(device)
    critic = CentralizedRecurrentCritic(policy_config).to(device)
    chunks = _fixture(actor, critic, policy_config, device)
    padded_chunks = _corrupt_padding(chunks)
    actor_clone = SharedRecurrentActor(policy_config).to(device)
    critic_clone = CentralizedRecurrentCritic(policy_config).to(device)
    actor_clone.load_state_dict(actor.state_dict())
    critic_clone.load_state_dict(critic.state_dict())
    actor_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    initial_losses = compute_ppo_losses(actor, critic, chunks, ppo_config)

    torch.manual_seed(SEED + 3)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 3)
    updates = update_recurrent_ppo(
        actor,
        critic,
        _optimizer(actor, ppo_config.actor_learning_rate, ppo_config),
        _optimizer(critic, ppo_config.critic_learning_rate, ppo_config),
        chunks,
        ppo_config,
    )
    torch.manual_seed(SEED + 3)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 3)
    padded_updates = update_recurrent_ppo(
        actor_clone,
        critic_clone,
        _optimizer(actor_clone, ppo_config.actor_learning_rate, ppo_config),
        _optimizer(critic_clone, ppo_config.critic_learning_rate, ppo_config),
        padded_chunks,
        ppo_config,
    )
    actor_change = _maximum_parameter_change(actor_before, actor)
    critic_change = _maximum_parameter_change(critic_before, critic)
    actor_padding_error = _maximum_model_difference(actor, actor_clone)
    critic_padding_error = _maximum_model_difference(critic, critic_clone)
    diagnostics = [value for row in updates for value in row.values() if not isinstance(value, int)]
    valid_actor_samples = int(chunks.actor.valid_mask.sum().item())
    valid_critic_samples = int(chunks.critic.valid_mask.sum().item())
    advantage_values = chunks.actor.advantages[chunks.actor.valid_mask.bool()]
    checkpoint_path = output / "updated-policy.pt"
    torch.save(
        {
            "policy_config": policy_config.to_dict(),
            "ppo_config": ppo_config.to_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "updates": updates,
            "seed": SEED,
        },
        checkpoint_path,
    )
    os.chmod(checkpoint_path, 0o644)
    checks = {
        "update_ran_on_requested_device": next(actor.parameters()).device == device,
        "actor_and_critic_losses_are_finite": all(
            bool(torch.isfinite(value)) for value in vars(initial_losses).values()
        ),
        "all_update_diagnostics_are_finite": all(math.isfinite(value) for value in diagnostics),
        "actor_parameters_changed": actor_change > 0.0,
        "critic_parameters_changed": critic_change > 0.0,
        "padded_actor_values_do_not_change_update": actor_padding_error <= TOLERANCE,
        "padded_critic_values_do_not_change_update": critic_padding_error <= TOLERANCE,
        "team_advantages_are_broadcast_to_all_agents": all(
            bool(torch.equal(chunks.actor.advantages[agent], chunks.actor.advantages[0]))
            for agent in range(4)
        )
        and all(
            bool(torch.equal(chunks.actor.advantages[4 + agent], chunks.actor.advantages[4]))
            for agent in range(4)
        ),
        "advantage_normalization_is_well_defined": float(advantage_values.std(unbiased=False)) > 0,
        "policy_diagnostics_are_bounded": all(
            0.0 <= row["policy_clip_fraction"] <= 1.0
            and 0.0 <= row["value_clip_fraction"] <= 1.0
            and row["approximate_kl"] >= -TOLERANCE
            for row in updates
        ),
        "gradient_norms_are_positive_and_finite": all(
            row["actor_gradient_norm_before_clip"] > 0
            and row["critic_gradient_norm_before_clip"] > 0
            for row in updates
        ),
        "valid_sample_counts_exclude_padding": valid_actor_samples == 28
        and valid_critic_samples == 7,
        "configured_epoch_count_was_applied": len(updates) == ppo_config.update_epochs,
        "padding_comparison_used_identical_diagnostics": updates == padded_updates,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "valid_actor_samples": valid_actor_samples,
        "valid_critic_samples": valid_critic_samples,
        "updates": updates,
        "measurements": {
            "initial_actor_loss": float(initial_losses.actor.detach().cpu()),
            "initial_critic_loss": float(initial_losses.critic.detach().cpu()),
            "actor_parameter_max_abs_change": actor_change,
            "critic_parameter_max_abs_change": critic_change,
            "actor_padding_max_abs_error": actor_padding_error,
            "critic_padding_max_abs_error": critic_padding_error,
        },
        "checkpoint": _artifact(checkpoint_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--ppo-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    policy = RecurrentPolicyConfig.from_dict(json.loads(args.policy_config.read_text()))
    ppo = RecurrentPPOConfig.from_dict(json.loads(args.ppo_config.read_text()))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but vendor PyTorch cannot use it")
    args.output.mkdir(parents=True, exist_ok=True)
    result = evaluate(policy, ppo, device, args.output)
    result.update(
        schema_version=1,
        python_version=platform.python_version(),
        duration_seconds=time.perf_counter() - started,
        policy_config=policy.to_dict(),
        ppo_config=ppo.to_dict(),
    )
    write_json_atomic(args.output / "metrics.json", result, mode=0o644)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
