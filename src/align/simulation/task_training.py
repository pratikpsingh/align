"""One task-connected recurrent MAPPO update; imported only inside Isaac Sim."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch

from align.learning.checkpoint_store import CheckpointStore
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.recovery_config import RecoveryConfig
from align.learning.rollout import RolloutConfig
from align.learning.torch_ppo import update_recurrent_ppo
from align.learning.torch_recovery import (
    capture_learner_state,
    restore_learner_state,
    validate_learner_payload,
)
from align.learning.torch_rollout import (
    TorchRecurrentFrame,
    TorchRecurrentRollout,
    TorchRolloutTransition,
)
from align.learning.training_config import TaskTrainingConfig
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import (
    CentralizedRecurrentCritic,
    SharedRecurrentActor,
)
from align.simulation.recurrent_collector import (
    zero_done_actor_memory,
    zero_done_critic_memory,
)

ROLLOUT_COLUMNS = (
    "attempt_id",
    "rollout_step",
    "env_id",
    "episode_step",
    "team_reward",
    "value",
    "bootstrap_value",
    "action_min",
    "action_max",
    "assigned_rmse_m",
    "pairwise_rmse_m",
    "minimum_separation_m",
    "terminated",
    "truncated",
    "reason_code",
)


def _optimizer(module, learning_rate: float, config: RecurrentPPOConfig):
    return torch.optim.Adam(module.parameters(), lr=learning_rate, eps=config.adam_epsilon)


def _maximum_parameter_change(before: dict, module: torch.nn.Module) -> float:
    return max(
        float((value.detach() - before[name]).abs().max().cpu())
        for name, value in module.state_dict().items()
        if torch.is_floating_point(value)
    )


def _save_checkpoint(
    store,
    state,
    resolved_config,
    *,
    attempt_id,
    counters,
    source_identity,
    runtime_identity,
    parent_sha256,
):
    return store.save(
        lambda stream: torch.save(state, stream),
        validator=lambda path: validate_learner_payload(path, resolved_config),
        attempt_id=attempt_id,
        completed_updates=counters["completed_updates"],
        environment_transitions=counters["environment_transitions"],
        agent_transitions=counters["agent_transitions"],
        active_training_seconds=counters["active_training_seconds"],
        source_identity=source_identity,
        runtime_identity=runtime_identity,
        parent_checkpoint_sha256=parent_sha256,
    )


def _initial_state(actor, critic, actor_optimizer, critic_optimizer, resolved_config, ppo):
    counters = {
        "completed_updates": 0,
        "environment_transitions": 0,
        "agent_transitions": 0,
        "active_training_seconds": 0.0,
    }
    normalization = {"enabled": False, "contract": "declared_feature_scaling"}
    schedules = {
        "learning_rate_fraction": 1.0,
        "entropy_coefficient": ppo.entropy_coefficient,
    }
    task_sampler = {
        "episodes_completed": 0,
        "unfinished_environment_count": 0,
    }
    return (
        capture_learner_state(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            normalization=normalization,
            schedules=schedules,
            counters=counters,
            resolved_config=resolved_config,
            task_sampler_state=task_sampler,
        ),
        counters,
        normalization,
        schedules,
        task_sampler,
    )


def run_training_attempt(
    *,
    env,
    initial,
    output: Path,
    checkpoint_directory: Path,
    logical_run_id: str,
    attempt_id: str,
    resume: bool,
    resolved_config: dict,
    config_sha256: str,
    source_identity: str,
    runtime_identity: str,
    policy_config: RecurrentPolicyConfig,
    rollout_config: RolloutConfig,
    ppo_config: RecurrentPPOConfig,
    recovery_config: RecoveryConfig,
    training_config: TaskTrainingConfig,
    event,
) -> dict:
    """Collect one real-task rollout, update once, and commit the boundary."""
    del recovery_config
    started = time.perf_counter()
    cfg = rollout_config
    if (env.num_envs, env.construction_cfg.num_agents) != (cfg.num_envs, cfg.num_agents):
        raise ValueError("training environment and rollout batch dimensions differ")
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
    torch.manual_seed(training_config.policy_seed)
    torch.cuda.manual_seed_all(training_config.policy_seed)
    actor = SharedRecurrentActor(policy_config).to(env.device).train()
    critic = CentralizedRecurrentCritic(policy_config).to(env.device).train()
    actor_optimizer = _optimizer(actor, ppo_config.actor_learning_rate, ppo_config)
    critic_optimizer = _optimizer(critic, ppo_config.critic_learning_rate, ppo_config)
    store = CheckpointStore(
        checkpoint_directory,
        run_id=logical_run_id,
        config_sha256=config_sha256,
        file_mode=0o644,
    )

    restored = None
    if resume:
        start_manifest, payload = store.latest_valid()
        restored = restore_learner_state(
            payload,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            expected_config=resolved_config,
        )
        counters = dict(restored["counters"])
        normalization = restored["normalization"]
        schedules = restored["schedules"]
        task_sampler = dict(restored["task_sampler_state"])
        abandoned = int(task_sampler["unfinished_environment_count"])
        event(
            "training_resumed",
            checkpoint_id=start_manifest["checkpoint_id"],
            completed_updates=counters["completed_updates"],
            abandoned_environment_episodes=abandoned,
        )
    else:
        if list(checkpoint_directory.glob("checkpoint-*.json")):
            raise RuntimeError("a new logical run requires an empty checkpoint directory")
        state, counters, normalization, schedules, task_sampler = _initial_state(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            resolved_config,
            ppo_config,
        )
        start_manifest = _save_checkpoint(
            store,
            state,
            resolved_config,
            attempt_id=attempt_id,
            counters=counters,
            source_identity=source_identity,
            runtime_identity=runtime_identity,
            parent_sha256=None,
        )
        abandoned = 0
        event("training_initial_checkpoint", checkpoint_id=start_manifest["checkpoint_id"])

    if training_config.normalization_enabled:
        raise NotImplementedError("empirical observation normalization is not implemented")
    start_counters = dict(counters)
    actor_memory = actor.backbone.recurrent.initial_state(
        initial[("agents", "observation")], cfg.num_envs * cfg.num_agents
    )
    critic_memory = critic.backbone.recurrent.initial_state(
        initial[("agents", "state")], cfg.num_envs
    )
    recurrent_memory_zero = bool(
        (actor_memory.hidden == 0).all()
        and (actor_memory.cell == 0).all()
        and (critic_memory.hidden == 0).all()
        and (critic_memory.cell == 0).all()
    )
    current_observation = initial[("agents", "observation")].clone()
    current_state = initial[("agents", "state")].clone()
    buffer = TorchRecurrentRollout(
        cfg,
        TorchRecurrentFrame(current_observation, current_state, actor_memory, critic_memory),
    )
    reward_sum = 0.0
    reward_min = math.inf
    reward_max = -math.inf
    action_min = math.inf
    action_max = -math.inf
    action_saturation_count = 0
    episode_count = int(task_sampler["episodes_completed"])
    raw_rows = 0
    collection_started = time.perf_counter()
    with (output / "rollout.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROLLOUT_COLUMNS)
        writer.writeheader()
        for step in range(cfg.horizon):
            with torch.no_grad():
                actor_result = actor.act(
                    current_observation.reshape(
                        cfg.num_envs * cfg.num_agents, 1, cfg.actor_observation_dim
                    ),
                    actor_memory,
                    torch.ones(cfg.num_envs * cfg.num_agents, 1, device=env.device),
                    deterministic=not training_config.stochastic_actions,
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
                progress_before_reset = env.progress_buf.clone()
                reasons = env.last_reason_code.clone()
                next_actor_memory = zero_done_actor_memory(actor_result.state, done, cfg)
                next_critic_memory = zero_done_critic_memory(critic_result.state, done)
                actions_cpu = actions.cpu()
                for env_id in range(cfg.num_envs):
                    writer.writerow(
                        {
                            "attempt_id": attempt_id,
                            "rollout_step": step,
                            "env_id": env_id,
                            "episode_step": int(progress_before_reset[env_id].item()),
                            "team_reward": float(team_rewards[env_id].item()),
                            "value": float(values[env_id].item()),
                            "bootstrap_value": float(bootstrap_values[env_id].item()),
                            "action_min": float(actions_cpu[env_id].min().item()),
                            "action_max": float(actions_cpu[env_id].max().item()),
                            "assigned_rmse_m": float(env.last_assigned_rmse[env_id].item()),
                            "pairwise_rmse_m": float(env.last_pairwise_rmse[env_id].item()),
                            "minimum_separation_m": float(
                                env.last_minimum_separation[env_id].item()
                            ),
                            "terminated": bool(terminated[env_id].item()),
                            "truncated": bool(truncated[env_id].item()),
                            "reason_code": int(reasons[env_id].item()),
                        }
                    )
                    raw_rows += 1
                stream.flush()
                reward_sum += float(team_rewards.sum().item())
                reward_min = min(reward_min, float(team_rewards.min().item()))
                reward_max = max(reward_max, float(team_rewards.max().item()))
                action_min = min(action_min, float(actions.min().item()))
                action_max = max(action_max, float(actions.max().item()))
                action_saturation_count += env.action_saturation_count
                episode_count += int(done.sum().item())
                if bool(done.any()):
                    reset_td = env.reset_mask(done)
                    current_observation = reset_td[("agents", "observation")].clone()
                    current_state = reset_td[("agents", "state")].clone()
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
    collection_seconds = time.perf_counter() - collection_started

    advantages = buffer.compute_gae()
    chunks = buffer.sequence_chunks()
    actor_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    update_started = time.perf_counter()
    update_diagnostics = update_recurrent_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        chunks,
        ppo_config,
    )
    update_seconds = time.perf_counter() - update_started
    actor_change = _maximum_parameter_change(actor_before, actor)
    critic_change = _maximum_parameter_change(critic_before, critic)
    completed_updates = start_counters["completed_updates"] + 1
    counters = {
        "completed_updates": completed_updates,
        "environment_transitions": start_counters["environment_transitions"]
        + cfg.horizon * cfg.num_envs,
        "agent_transitions": start_counters["agent_transitions"]
        + cfg.horizon * cfg.num_envs * cfg.num_agents,
        "active_training_seconds": float(start_counters["active_training_seconds"])
        + collection_seconds
        + update_seconds,
    }
    task_sampler = {
        "episodes_completed": episode_count,
        "unfinished_environment_count": int((env.progress_buf > 0).sum().item()),
    }
    state = capture_learner_state(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        normalization=normalization,
        schedules=schedules,
        counters=counters,
        resolved_config=resolved_config,
        task_sampler_state=task_sampler,
    )
    committed = _save_checkpoint(
        store,
        state,
        resolved_config,
        attempt_id=attempt_id,
        counters=counters,
        source_identity=source_identity,
        runtime_identity=runtime_identity,
        parent_sha256=start_manifest["payload"]["sha256"],
    )
    verified, _ = store.verify(
        checkpoint_directory / f"checkpoint-{committed['checkpoint_id']}.json"
    )
    with (output / "updates.csv").open("x", newline="", encoding="utf-8") as stream:
        fieldnames = ("attempt_id", "completed_update", *update_diagnostics[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in update_diagnostics:
            writer.writerow(
                {"attempt_id": attempt_id, "completed_update": completed_updates, **row}
            )

    diagnostic_values = [
        value for row in update_diagnostics for key, value in row.items() if key != "epoch"
    ]
    tensors = (
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
    expected_environment_increment = cfg.horizon * cfg.num_envs
    expected_agent_increment = expected_environment_increment * cfg.num_agents
    checks = {
        "real_task_rollout_is_full": buffer.full,
        "rollout_tensors_are_finite": all(bool(torch.isfinite(value).all()) for value in tensors),
        "rollout_tensors_remained_on_cuda": all(value.device.type == "cuda" for value in tensors),
        "sampled_actions_are_bounded": action_min >= -1.0 and action_max <= 1.0,
        "environment_did_not_clip_actions": action_saturation_count == 0,
        "ppo_diagnostics_are_finite": all(math.isfinite(value) for value in diagnostic_values),
        "actor_parameters_changed": (actor_change > 0.0)
        if training_config.require_parameter_change
        else True,
        "critic_parameters_changed": (critic_change > 0.0)
        if training_config.require_parameter_change
        else True,
        "completed_update_advanced_once": counters["completed_updates"]
        == start_counters["completed_updates"] + 1,
        "environment_counter_advanced_once": counters["environment_transitions"]
        == start_counters["environment_transitions"] + expected_environment_increment,
        "agent_counter_advanced_once": counters["agent_transitions"]
        == start_counters["agent_transitions"] + expected_agent_increment,
        "recurrent_memory_started_zero": recurrent_memory_zero,
        "resume_reset_contract_applied": (not resume)
        or (
            restored["discard_partial_rollout"]
            and restored["reset_environment_and_recurrent_memory"]
        ),
        "checkpoint_was_verified": verified["checkpoint_id"] == committed["checkpoint_id"],
        "checkpoint_parent_matches_start": committed["parent_checkpoint_sha256"]
        == start_manifest["payload"]["sha256"],
        "raw_metric_row_count_is_exact": raw_rows == cfg.horizon * cfg.num_envs,
    }
    torch.cuda.synchronize()
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "attempt_id": attempt_id,
        "resumed": resume,
        "recovery_mode": "training_resume_with_environment_reset",
        "loaded_checkpoint_id": start_manifest["checkpoint_id"] if resume else None,
        "start_checkpoint_id": start_manifest["checkpoint_id"],
        "committed_checkpoint_id": committed["checkpoint_id"],
        "committed_checkpoint_sha256": committed["payload"]["sha256"],
        "checkpoint_parent_sha256": committed["parent_checkpoint_sha256"],
        "start_counters": start_counters,
        "end_counters": counters,
        "abandoned_environment_episodes": abandoned,
        "unfinished_environment_count": task_sampler["unfinished_environment_count"],
        "episodes_completed_total": episode_count,
        "rollout_rows": raw_rows,
        "measurements": {
            "team_reward_mean": reward_sum / (cfg.horizon * cfg.num_envs),
            "team_reward_min": reward_min,
            "team_reward_max": reward_max,
            "action_min": action_min,
            "action_max": action_max,
            "actor_parameter_max_abs_change": actor_change,
            "critic_parameter_max_abs_change": critic_change,
            "collection_seconds": collection_seconds,
            "update_seconds": update_seconds,
            "attempt_seconds": time.perf_counter() - started,
        },
        "updates": update_diagnostics,
    }
