"""Live vector-task collection using the recurrent policy; imported inside Isaac Sim."""

from __future__ import annotations

import csv
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from align.learning.collector_config import (
    CollectorProbeConfig,
    select_episode_boundary_memory,
)
from align.learning.rollout import RolloutConfig
from align.learning.torch_rollout import (
    TorchRecurrentFrame,
    TorchRecurrentRollout,
    TorchRolloutTransition,
)
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import (
    CentralizedRecurrentCritic,
    RecurrentState,
    SharedRecurrentActor,
)

COLLECTOR_COLUMNS = (
    "global_step",
    "env_id",
    "agent_id",
    "episode_step",
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "old_log_prob",
    "team_reward",
    "value",
    "bootstrap_value",
    "terminated",
    "truncated",
    "reason_code",
)


def _zero_done_actor_memory(
    state: RecurrentState,
    done: torch.Tensor,
    rollout: RolloutConfig,
) -> RecurrentState:
    hidden = state.hidden.clone().view(
        rollout.recurrent_layers,
        rollout.num_envs,
        rollout.num_agents,
        rollout.recurrent_hidden_size,
    )
    cell = state.cell.clone().view_as(hidden)
    hidden[:, done] = 0
    cell[:, done] = 0
    return RecurrentState(
        hidden.view(
            rollout.recurrent_layers,
            rollout.num_envs * rollout.num_agents,
            rollout.recurrent_hidden_size,
        ),
        cell.view(
            rollout.recurrent_layers,
            rollout.num_envs * rollout.num_agents,
            rollout.recurrent_hidden_size,
        ),
    )


def _zero_done_critic_memory(state: RecurrentState, done: torch.Tensor) -> RecurrentState:
    hidden = state.hidden.clone()
    cell = state.cell.clone()
    hidden[:, done] = 0
    cell[:, done] = 0
    return RecurrentState(hidden, cell)


def _force_safety_termination(env, environment: int) -> None:
    world_pos, world_rot = env.drone.get_world_poses(clone=True)
    world_pos = world_pos.reshape(env.task_cfg.num_envs, env.construction_cfg.num_agents, 3)
    world_rot = world_rot.reshape(env.task_cfg.num_envs, env.construction_cfg.num_agents, 4)
    world_pos[environment, :, 0] = (
        env.envs_positions[environment, 0] + env.task_cfg.safety_xy_limit_m + 1.0
    )
    env_id = torch.tensor([environment], device=env.device)
    env.drone.set_world_poses(
        world_pos[environment : environment + 1],
        world_rot[environment : environment + 1],
        env_id,
    )
    env.drone.set_velocities(
        torch.zeros(
            1,
            env.construction_cfg.num_agents,
            6,
            device=env.device,
        ),
        env_id,
    )
    env.state = env.drone.get_state().clone()


def _maximum_valid_error(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((left[mask] - right[mask]).abs().max().detach().cpu())


def _save_raw_rollout(output: Path, buffer: TorchRecurrentRollout) -> dict:
    path = output / "rollout.npz"
    np.savez_compressed(
        path,
        actor_observations=buffer.actor_observations.cpu().numpy(),
        critic_states=buffer.critic_states.cpu().numpy(),
        actions=buffer.actions.cpu().numpy(),
        old_log_probs=buffer.old_log_probs.cpu().numpy(),
        team_rewards=buffer.team_rewards.cpu().numpy(),
        values=buffer.values.cpu().numpy(),
        bootstrap_values=buffer.bootstrap_values.cpu().numpy(),
        advantages=buffer.compute_gae().cpu().numpy(),
        returns=buffer.returns.cpu().numpy(),
        terminated=buffer.terminated.cpu().numpy(),
        truncated=buffer.truncated.cpu().numpy(),
        actor_hidden=buffer.actor_hidden.cpu().numpy(),
        actor_cell=buffer.actor_cell.cpu().numpy(),
        critic_hidden=buffer.critic_hidden.cpu().numpy(),
        critic_cell=buffer.critic_cell.cpu().numpy(),
    )
    os.chmod(path, 0o644)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def collect_live_rollout(
    *,
    env,
    initial,
    output: Path,
    policy_config: RecurrentPolicyConfig,
    rollout_config: RolloutConfig,
    collector_config: CollectorProbeConfig,
    event,
) -> dict:
    """Collect one fixed-horizon rollout and validate it before any optimization."""
    started = time.perf_counter()
    cfg = rollout_config
    collector_config.validate(horizon=cfg.horizon, num_envs=cfg.num_envs)
    policy_config.validate_task_dimensions(
        actor_observation_dim=cfg.actor_observation_dim,
        critic_state_dim=cfg.critic_state_dim,
        action_dim=cfg.action_dim,
    )
    if (policy_config.recurrent_layers, policy_config.recurrent_hidden_size) != (
        cfg.recurrent_layers,
        cfg.recurrent_hidden_size,
    ):
        raise ValueError("policy and rollout recurrent dimensions differ")
    if (env.num_envs, env.construction_cfg.num_agents) != (cfg.num_envs, cfg.num_agents):
        raise ValueError("live environment and rollout batch dimensions differ")

    torch.manual_seed(collector_config.policy_seed)
    torch.cuda.manual_seed_all(collector_config.policy_seed)
    actor = SharedRecurrentActor(policy_config).to(env.device).eval()
    critic = CentralizedRecurrentCritic(policy_config).to(env.device).eval()
    actor_memory = actor.backbone.recurrent.initial_state(
        initial[("agents", "observation")], cfg.num_envs * cfg.num_agents
    )
    critic_memory = critic.backbone.recurrent.initial_state(
        initial[("agents", "state")], cfg.num_envs
    )
    current_observation = initial[("agents", "observation")].clone()
    current_state = initial[("agents", "state")].clone()
    buffer = TorchRecurrentRollout(
        cfg,
        TorchRecurrentFrame(
            current_observation,
            current_state,
            actor_memory,
            critic_memory,
        ),
    )

    checkpoint_path = output / "initial-policy.pt"
    torch.save(
        {
            "policy_config": policy_config.to_dict(),
            "policy_seed": collector_config.policy_seed,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        checkpoint_path,
    )
    os.chmod(checkpoint_path, 0o644)
    checkpoint = {
        "path": checkpoint_path.name,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }

    done_events = []
    reset_events = []
    forced_observed = False
    truncation_observed = False
    action_saturation_count = 0
    reward_mean_max_error = 0.0
    unaffected_observation_max_error = 0.0
    unaffected_critic_state_max_error = 0.0
    unaffected_actor_memory_max_error = 0.0
    unaffected_critic_memory_max_error = 0.0
    peak_allocated_before = torch.cuda.max_memory_allocated()

    with (output / "collector.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLLECTOR_COLUMNS)
        writer.writeheader()
        with torch.no_grad():
            for step in range(cfg.horizon):
                if step == collector_config.forced_termination_step:
                    _force_safety_termination(env, collector_config.forced_termination_environment)
                    event(
                        "collector_forced_safety_state",
                        global_step=step,
                        env_id=collector_config.forced_termination_environment,
                    )

                flat_observation = current_observation.reshape(
                    cfg.num_envs * cfg.num_agents, 1, cfg.actor_observation_dim
                )
                actor_result = actor.act(
                    flat_observation,
                    actor_memory,
                    torch.ones(
                        cfg.num_envs * cfg.num_agents,
                        1,
                        device=env.device,
                    ),
                    deterministic=not collector_config.stochastic_actions,
                )
                actions = actor_result.action[:, 0].reshape(
                    cfg.num_envs, cfg.num_agents, cfg.action_dim
                )
                old_log_probs = actor_result.log_prob[:, 0].reshape(cfg.num_envs, cfg.num_agents)
                critic_result = critic(
                    current_state.unsqueeze(1),
                    critic_memory,
                    torch.ones(cfg.num_envs, 1, device=env.device),
                )
                values = critic_result.value[:, 0]

                output_td = env.step_actions(actions)
                action_saturation_count += env.action_saturation_count
                final_observation = output_td[("agents", "observation")].clone()
                final_state = output_td[("agents", "state")].clone()
                terminated = output_td["terminated"].squeeze(-1).clone()
                truncated = output_td["truncated"].squeeze(-1).clone()
                done = terminated | truncated
                next_value = critic(
                    final_state.unsqueeze(1),
                    critic_result.state,
                    torch.ones(cfg.num_envs, 1, device=env.device),
                ).value[:, 0]
                bootstrap_values = torch.where(terminated, torch.zeros_like(next_value), next_value)
                team_rewards = env.last_team_reward.clone()
                per_agent_rewards = output_td[("agents", "reward")].squeeze(-1)
                reward_mean_max_error = max(
                    reward_mean_max_error,
                    float((team_rewards - per_agent_rewards.mean(dim=-1)).abs().max().cpu()),
                )

                codes = env.last_reason_code.clone()
                progress_before_reset = env.progress_buf.clone()
                for env_id in done.nonzero().flatten().cpu().tolist():
                    item = {
                        "global_step": step,
                        "env_id": env_id,
                        "episode_step": int(progress_before_reset[env_id].item()),
                        "terminated": bool(terminated[env_id].item()),
                        "truncated": bool(truncated[env_id].item()),
                        "reason_code": int(codes[env_id].item()),
                        "bootstrap_value": float(bootstrap_values[env_id].item()),
                    }
                    done_events.append(item)
                    event("collector_episode_done", **item)
                forced_env = collector_config.forced_termination_environment
                if step == collector_config.forced_termination_step:
                    forced_observed = bool(
                        terminated[forced_env].item() and codes[forced_env].item() == 4
                    )
                truncation_observed |= bool(truncated.any().item())

                next_actor_memory = _zero_done_actor_memory(actor_result.state, done, cfg)
                next_critic_memory = _zero_done_critic_memory(critic_result.state, done)
                if bool(done.any()):
                    actor_before_reset = actor_result.state.hidden.view(
                        cfg.recurrent_layers,
                        cfg.num_envs,
                        cfg.num_agents,
                        cfg.recurrent_hidden_size,
                    )
                    critic_before_reset = critic_result.state.hidden
                    reset_td = env.reset_mask(done)
                    current_observation = reset_td[("agents", "observation")].clone()
                    current_state = reset_td[("agents", "state")].clone()
                    unaffected = ~done
                    if bool(unaffected.any()):
                        unaffected_observation_max_error = max(
                            unaffected_observation_max_error,
                            float(
                                (current_observation[unaffected] - final_observation[unaffected])
                                .abs()
                                .max()
                                .cpu()
                            ),
                        )
                        unaffected_critic_state_max_error = max(
                            unaffected_critic_state_max_error,
                            float(
                                (current_state[unaffected] - final_state[unaffected])
                                .abs()
                                .max()
                                .cpu()
                            ),
                        )
                        actor_after_reset = next_actor_memory.hidden.view_as(actor_before_reset)
                        unaffected_actor_memory_max_error = max(
                            unaffected_actor_memory_max_error,
                            float(
                                (
                                    actor_after_reset[:, unaffected]
                                    - actor_before_reset[:, unaffected]
                                )
                                .abs()
                                .max()
                                .cpu()
                            ),
                        )
                        unaffected_critic_memory_max_error = max(
                            unaffected_critic_memory_max_error,
                            float(
                                (
                                    next_critic_memory.hidden[:, unaffected]
                                    - critic_before_reset[:, unaffected]
                                )
                                .abs()
                                .max()
                                .cpu()
                            ),
                        )
                    reset_events.append(
                        {
                            "global_step": step,
                            "env_ids": done.nonzero().flatten().cpu().tolist(),
                            "post_reset_progress": env.progress_buf.cpu().tolist(),
                        }
                    )
                else:
                    current_observation = final_observation
                    current_state = final_state

                buffer.append(
                    TorchRolloutTransition(
                        actions,
                        old_log_probs,
                        team_rewards,
                        values,
                        bootstrap_values,
                        terminated,
                        truncated,
                    ),
                    TorchRecurrentFrame(
                        current_observation,
                        current_state,
                        next_actor_memory,
                        next_critic_memory,
                    ),
                )
                actor_memory = next_actor_memory
                critic_memory = next_critic_memory

                actions_cpu = actions.cpu().tolist()
                log_probs_cpu = old_log_probs.cpu().tolist()
                for env_id in range(cfg.num_envs):
                    for agent_id in range(cfg.num_agents):
                        writer.writerow(
                            {
                                "global_step": step,
                                "env_id": env_id,
                                "agent_id": agent_id,
                                "episode_step": int(progress_before_reset[env_id].item()),
                                **{
                                    f"action_{index}": actions_cpu[env_id][agent_id][index]
                                    for index in range(cfg.action_dim)
                                },
                                "old_log_prob": log_probs_cpu[env_id][agent_id],
                                "team_reward": float(team_rewards[env_id].item()),
                                "value": float(values[env_id].item()),
                                "bootstrap_value": float(bootstrap_values[env_id].item()),
                                "terminated": bool(terminated[env_id].item()),
                                "truncated": bool(truncated[env_id].item()),
                                "reason_code": int(codes[env_id].item()),
                            }
                        )
                stream.flush()

    advantages = buffer.compute_gae()
    chunks = buffer.sequence_chunks()
    reference = buffer.to_cpu_reference()
    reference_advantages = torch.tensor(
        reference.compute_gae(), device=env.device, dtype=buffer.dtype
    )
    reference_chunks = reference.sequence_chunks()
    gae_error = float((advantages - reference_advantages).abs().max().cpu())
    returns_error = float(
        (buffer.returns - torch.tensor(reference.returns, device=env.device)).abs().max().cpu()
    )

    with torch.no_grad():
        actor_chunk_output = actor(
            chunks.actor.observations,
            chunks.actor.initial_memory,
            valid_mask=chunks.actor.valid_mask,
        )
        evaluated_log_probs = actor.distribution(actor_chunk_output).log_prob(chunks.actor.actions)
        actor_log_prob_error = _maximum_valid_error(
            evaluated_log_probs,
            chunks.actor.old_log_probs,
            chunks.actor.valid_mask.bool(),
        )
        critic_chunk_output = critic(
            chunks.critic.states,
            chunks.critic.initial_memory,
            valid_mask=chunks.critic.valid_mask,
        )
        critic_value_error = _maximum_valid_error(
            critic_chunk_output.value,
            chunks.critic.values,
            chunks.critic.valid_mask.bool(),
        )

    actor_metadata = [
        (item.environment_id, item.agent_id, item.start_step, item.valid_length)
        for item in reference_chunks.actor
    ]
    critic_metadata = [
        (item.environment_id, item.start_step, item.valid_length)
        for item in reference_chunks.critic
    ]
    device_actor_metadata = list(
        zip(
            chunks.actor.environment_ids.cpu().tolist(),
            chunks.actor.agent_ids.cpu().tolist(),
            chunks.actor.start_steps.cpu().tolist(),
            chunks.actor.valid_lengths.cpu().tolist(),
            strict=True,
        )
    )
    device_critic_metadata = list(
        zip(
            chunks.critic.environment_ids.cpu().tolist(),
            chunks.critic.start_steps.cpu().tolist(),
            chunks.critic.valid_lengths.cpu().tolist(),
            strict=True,
        )
    )
    boundary = buffer.terminated | buffer.truncated
    actor_boundary_memory = select_episode_boundary_memory(buffer.actor_hidden, boundary)
    critic_boundary_memory = select_episode_boundary_memory(buffer.critic_hidden, boundary)
    all_tensors = (
        buffer.actor_observations,
        buffer.critic_states,
        buffer.actions,
        buffer.old_log_probs,
        buffer.team_rewards,
        buffer.values,
        buffer.bootstrap_values,
        advantages,
        buffer.returns,
    )
    tolerance = 3e-5
    checks = {
        "full_horizon_collected": buffer.full,
        "rollout_tensors_remained_on_cuda": all(
            tensor.device.type == "cuda" for tensor in all_tensors
        ),
        "rollout_tensors_are_finite": all(
            bool(torch.isfinite(tensor).all()) for tensor in all_tensors
        ),
        "executed_actions_are_bounded": bool(
            ((buffer.actions >= -1.0) & (buffer.actions <= 1.0)).all()
        ),
        "environment_never_clipped_policy_actions": action_saturation_count == 0,
        "stored_log_probabilities_match_chunk_evaluation": actor_log_prob_error <= tolerance,
        "stored_values_match_chunk_evaluation": critic_value_error <= tolerance,
        "device_gae_matches_cpu_reference": gae_error <= 1e-5,
        "device_returns_match_cpu_reference": returns_error <= 1e-5,
        "actor_chunk_metadata_matches_cpu_reference": device_actor_metadata == actor_metadata,
        "critic_chunk_metadata_matches_cpu_reference": device_critic_metadata == critic_metadata,
        "one_actor_chunk_per_agent_and_critic_chunk": (
            len(reference_chunks.actor) == len(reference_chunks.critic) * cfg.num_agents
        ),
        "true_termination_observed": (
            forced_observed if collector_config.require_true_termination else True
        ),
        "time_limit_truncation_observed": (
            truncation_observed if collector_config.require_truncation else True
        ),
        "true_termination_bootstrap_is_zero": bool(
            (buffer.bootstrap_values[buffer.terminated] == 0).all()
        ),
        "truncation_bootstrap_is_finite": bool(
            torch.isfinite(buffer.bootstrap_values[buffer.truncated]).all()
            and buffer.truncated.any()
        ),
        "boundary_actor_memory_is_zero": bool((actor_boundary_memory == 0).all()),
        "boundary_critic_memory_is_zero": bool((critic_boundary_memory == 0).all()),
        "partial_reset_preserves_other_observations": unaffected_observation_max_error <= 1e-7,
        "partial_reset_preserves_other_critic_states": unaffected_critic_state_max_error <= 1e-7,
        "partial_reset_preserves_other_actor_memory": unaffected_actor_memory_max_error == 0.0,
        "partial_reset_preserves_other_critic_memory": unaffected_critic_memory_max_error == 0.0,
        "team_reward_is_agent_mean": reward_mean_max_error <= 1e-7,
    }
    raw_rollout = _save_raw_rollout(output, buffer)
    torch.cuda.synchronize()
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "policy_seed": collector_config.policy_seed,
        "stochastic_actions": collector_config.stochastic_actions,
        "steps": cfg.horizon,
        "agent_transitions": cfg.horizon * cfg.num_envs * cfg.num_agents,
        "environment_transitions": cfg.horizon * cfg.num_envs,
        "done_events": done_events,
        "reset_events": reset_events,
        "actor_chunk_count": len(reference_chunks.actor),
        "critic_chunk_count": len(reference_chunks.critic),
        "rollout_tensor_bytes": buffer.bytes_allocated,
        "peak_cuda_bytes_increment": torch.cuda.max_memory_allocated() - peak_allocated_before,
        "measurements": {
            "action_min": float(buffer.actions.min().cpu()),
            "action_max": float(buffer.actions.max().cpu()),
            "old_log_prob_min": float(buffer.old_log_probs.min().cpu()),
            "old_log_prob_max": float(buffer.old_log_probs.max().cpu()),
            "team_reward_min": float(buffer.team_rewards.min().cpu()),
            "team_reward_max": float(buffer.team_rewards.max().cpu()),
            "advantage_min": float(advantages.min().cpu()),
            "advantage_max": float(advantages.max().cpu()),
            "actor_log_prob_max_abs_error": actor_log_prob_error,
            "critic_value_max_abs_error": critic_value_error,
            "gae_cpu_max_abs_error": gae_error,
            "returns_cpu_max_abs_error": returns_error,
            "team_reward_mean_max_abs_error": reward_mean_max_error,
            "unaffected_observation_max_abs_error": unaffected_observation_max_error,
            "unaffected_critic_state_max_abs_error": unaffected_critic_state_max_error,
            "unaffected_actor_memory_max_abs_error": unaffected_actor_memory_max_error,
            "unaffected_critic_memory_max_abs_error": unaffected_critic_memory_max_error,
        },
        "policy_checkpoint": checkpoint,
        "raw_rollout": raw_rollout,
        "duration_seconds": time.perf_counter() - started,
    }
