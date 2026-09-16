"""Tensor acceptance probe for the recurrent actor, critic, and action distribution."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from align.artifacts import write_json_atomic
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import CentralizedRecurrentCritic, SharedRecurrentActor

SEED = 20260916
TOLERANCE = 3e-5


def maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach().cpu())


def all_finite_gradients(module: torch.nn.Module) -> tuple[bool, int, int]:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    finite = all(item is not None and bool(torch.isfinite(item).all()) for item in gradients)
    nonzero = sum(item is not None and bool((item != 0).any()) for item in gradients)
    return finite, nonzero, len(gradients)


def evaluate(config: RecurrentPolicyConfig, device: torch.device) -> dict:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    actor_cpu = SharedRecurrentActor(config)
    critic_cpu = CentralizedRecurrentCritic(config)
    actor = SharedRecurrentActor(config).to(device)
    critic = CentralizedRecurrentCritic(config).to(device)
    actor.load_state_dict(actor_cpu.state_dict())
    critic.load_state_dict(critic_cpu.state_dict())

    batch, environments, steps = 8, 2, 6
    generator = torch.Generator().manual_seed(SEED + 1)
    observations_cpu = torch.randn(batch, steps, config.actor_observation_dim, generator=generator)
    states_cpu = torch.randn(environments, steps, config.critic_state_dim, generator=generator)
    memory_cpu = torch.ones(batch, steps)
    memory_cpu[:, 0] = 0
    memory_cpu[:, 3] = 0
    critic_memory_cpu = memory_cpu[:environments].clone()

    observations = observations_cpu.to(device)
    states = states_cpu.to(device)
    memory = memory_cpu.to(device)
    critic_memory = critic_memory_cpu.to(device)

    actor_full = actor(observations, memory_mask=memory)
    actor_first = actor(observations[:, :3], memory_mask=memory[:, :3])
    actor_second = actor(observations[:, 3:], actor_first.state, memory_mask=memory[:, 3:])
    actor_chunked = torch.cat((actor_first.latent_mean, actor_second.latent_mean), dim=1)
    actor_chunk_error = maximum_error(actor_full.latent_mean, actor_chunked)
    actor_state_error = max(
        maximum_error(actor_full.state.hidden, actor_second.state.hidden),
        maximum_error(actor_full.state.cell, actor_second.state.cell),
    )

    critic_full = critic(states, memory_mask=critic_memory)
    critic_first = critic(states[:, :3], memory_mask=critic_memory[:, :3])
    critic_second = critic(states[:, 3:], critic_first.state, memory_mask=critic_memory[:, 3:])
    critic_chunked = torch.cat((critic_first.value, critic_second.value), dim=1)
    critic_chunk_error = maximum_error(critic_full.value, critic_chunked)
    critic_state_error = max(
        maximum_error(critic_full.state.hidden, critic_second.state.hidden),
        maximum_error(critic_full.state.cell, critic_second.state.cell),
    )

    reset_observations = observations[:2].clone()
    reset_observations[1, :3] = -reset_observations[0, :3]
    reset_observations[1, 3:] = reset_observations[0, 3:]
    reset_mask = torch.ones(2, steps, device=device)
    reset_mask[:, 0] = 0
    reset_mask[:, 3] = 0
    reset_output = actor(reset_observations, memory_mask=reset_mask).latent_mean
    reset_isolation_error = maximum_error(reset_output[0, 3:], reset_output[1, 3:])

    history_observations = reset_observations.clone()
    history_mask = torch.ones(2, steps, device=device)
    history_mask[:, 0] = 0
    history_output = actor(history_observations, memory_mask=history_mask).latent_mean
    history_effect = maximum_error(history_output[0, -1], history_output[1, -1])

    permutation = torch.tensor([7, 2, 5, 0, 6, 1, 4, 3], device=device)
    permuted = actor(observations[permutation], memory_mask=memory[permutation]).latent_mean
    sharing_error = maximum_error(permuted, actor_full.latent_mean[permutation])

    padding_observations = torch.cat(
        (observations, torch.zeros(batch, 2, config.actor_observation_dim, device=device)), dim=1
    )
    padding_memory = torch.cat((memory, torch.ones(batch, 2, device=device)), dim=1)
    valid = torch.cat(
        (torch.ones(batch, steps, device=device), torch.zeros(batch, 2, device=device)), dim=1
    )
    padded = actor(padding_observations, memory_mask=padding_memory, valid_mask=valid)
    padding_state_error = max(
        maximum_error(padded.state.hidden, actor_full.state.hidden),
        maximum_error(padded.state.cell, actor_full.state.cell),
    )

    distribution = actor.distribution(actor_full)
    samples = []
    sample_log_probs = []
    roundtrip_errors = []
    for _ in range(32):
        action, log_prob = distribution.rsample_with_log_prob()
        samples.append(action.detach())
        sample_log_probs.append(log_prob.detach())
        roundtrip_errors.append(maximum_error(log_prob, distribution.log_prob(action)))
    sampled = torch.stack(samples)
    sampled_log_prob = torch.stack(sample_log_probs)
    action_min = sampled.amin(dim=(0, 1, 2)).cpu().tolist()
    action_max = sampled.amax(dim=(0, 1, 2)).cpu().tolist()
    strict_bounds = bool((sampled > actor.action_low).all() and (sampled < actor.action_high).all())
    log_prob_roundtrip_error = max(roundtrip_errors)

    actor.zero_grad(set_to_none=True)
    fixed_actions = torch.zeros_like(actor_full.latent_mean)
    actor_loss = (
        -distribution.log_prob(fixed_actions).mean() + 0.01 * actor_full.latent_mean.square().mean()
    )
    actor_loss.backward()
    actor_gradients_finite, actor_nonzero_gradients, actor_gradient_tensors = all_finite_gradients(
        actor
    )

    critic.zero_grad(set_to_none=True)
    critic_loss = (critic_full.value - 0.5).square().mean()
    critic_loss.backward()
    critic_gradients_finite, critic_nonzero_gradients, critic_gradient_tensors = (
        all_finite_gradients(critic)
    )

    with torch.no_grad():
        cpu_actor_output = actor_cpu(observations_cpu, memory_mask=memory_cpu).latent_mean
        cpu_critic_output = critic_cpu(states_cpu, memory_mask=critic_memory_cpu).value
    cpu_cuda_actor_error = maximum_error(cpu_actor_output, actor_full.latent_mean.cpu())
    cpu_cuda_critic_error = maximum_error(cpu_critic_output, critic_full.value.cpu())

    actor_parameter_ids = {id(parameter) for parameter in actor.parameters()}
    critic_parameter_ids = {id(parameter) for parameter in critic.parameters()}
    checks = {
        "actor_full_sequence_matches_chunks": actor_chunk_error <= TOLERANCE,
        "actor_final_memory_matches_chunks": actor_state_error <= TOLERANCE,
        "critic_full_sequence_matches_chunks": critic_chunk_error <= TOLERANCE,
        "critic_final_memory_matches_chunks": critic_state_error <= TOLERANCE,
        "reset_mask_prevents_episode_memory_leak": reset_isolation_error <= TOLERANCE,
        "earlier_observations_affect_later_action": history_effect > 1e-8,
        "shared_actor_is_batch_permutation_equivariant": sharing_error <= TOLERANCE,
        "padding_does_not_change_memory": padding_state_error <= TOLERANCE,
        "sampled_actions_are_strictly_bounded": strict_bounds,
        "sampled_log_probabilities_are_finite": bool(torch.isfinite(sampled_log_prob).all()),
        "sample_and_evaluation_log_probabilities_match": log_prob_roundtrip_error <= TOLERANCE,
        "actor_gradients_are_finite": actor_gradients_finite,
        "actor_recurrent_path_has_gradients": actor_nonzero_gradients == actor_gradient_tensors,
        "critic_gradients_are_finite": critic_gradients_finite,
        "critic_recurrent_path_has_gradients": critic_nonzero_gradients == critic_gradient_tensors,
        "actor_and_critic_parameters_are_disjoint": not (
            actor_parameter_ids & critic_parameter_ids
        ),
        "cpu_cuda_actor_outputs_match": cpu_cuda_actor_error <= TOLERANCE,
        "cpu_cuda_critic_outputs_match": cpu_cuda_critic_error <= TOLERANCE,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": SEED,
        "tolerance": TOLERANCE,
        "shapes": {
            "actor_input": list(observations.shape),
            "critic_input": list(states.shape),
            "action": list(actor_full.latent_mean.shape),
            "actor_memory": list(actor_full.state.hidden.shape),
            "critic_memory": list(critic_full.state.hidden.shape),
        },
        "parameters": {
            "actor": sum(parameter.numel() for parameter in actor.parameters()),
            "critic": sum(parameter.numel() for parameter in critic.parameters()),
            "actor_fp32_bytes": sum(parameter.numel() * 4 for parameter in actor.parameters()),
            "critic_fp32_bytes": sum(parameter.numel() * 4 for parameter in critic.parameters()),
        },
        "measurements": {
            "actor_chunk_max_abs_error": actor_chunk_error,
            "actor_state_max_abs_error": actor_state_error,
            "critic_chunk_max_abs_error": critic_chunk_error,
            "critic_state_max_abs_error": critic_state_error,
            "reset_isolation_max_abs_error": reset_isolation_error,
            "history_effect_max_abs_difference": history_effect,
            "sharing_max_abs_error": sharing_error,
            "padding_state_max_abs_error": padding_state_error,
            "action_min_by_dimension": action_min,
            "action_max_by_dimension": action_max,
            "log_prob_roundtrip_max_abs_error": log_prob_roundtrip_error,
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_nonzero_gradient_tensors": actor_nonzero_gradients,
            "actor_gradient_tensors": actor_gradient_tensors,
            "critic_nonzero_gradient_tensors": critic_nonzero_gradients,
            "critic_gradient_tensors": critic_gradient_tensors,
            "cpu_cuda_actor_max_abs_error": cpu_cuda_actor_error,
            "cpu_cuda_critic_max_abs_error": cpu_cuda_critic_error,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = RecurrentPolicyConfig.from_dict(json.loads(args.config.read_text()))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but vendor PyTorch cannot use it")
    result = evaluate(config, device)
    result.update(
        schema_version=1,
        python_version=platform.python_version(),
        duration_seconds=time.perf_counter() - started,
        config=config.to_dict(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output / "metrics.json", result, mode=0o644)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
