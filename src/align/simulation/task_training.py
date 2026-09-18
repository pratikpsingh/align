"""One task-connected recurrent MAPPO update; imported only inside Isaac Sim."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import torch

from align.artifacts import write_json_atomic
from align.learning.checkpoint_store import CheckpointStore
from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.recovery_config import RecoveryConfig
from align.learning.rollout import RolloutConfig
from align.learning.torch_normalization import (
    CriticGroupAccumulator,
    FrozenCriticGroupNormalizer,
    disabled_normalization_state,
    normalization_matches_config,
    summarize_critic_rollout_distribution,
)
from align.learning.torch_ppo import evaluate_ppo_diagnostics, update_recurrent_ppo
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
    "phase",
    "formation_kind",
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


class InjectedTrainingInterruption(RuntimeError):
    """Expected acceptance-only interruption after a partial live rollout."""


NORMALIZATION_WARMUP_COLUMNS = (
    "warmup_step",
    "position_count",
    "position_sum",
    "position_sum_squares",
    "velocity_count",
    "velocity_sum",
    "velocity_sum_squares",
    "target_count",
    "target_sum",
    "target_sum_squares",
    "action_min",
    "action_max",
    "terminated_count",
    "truncated_count",
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


def _initial_state(
    actor, critic, actor_optimizer, critic_optimizer, resolved_config, ppo, normalization
):
    counters = {
        "completed_updates": 0,
        "environment_transitions": 0,
        "agent_transitions": 0,
        "active_training_seconds": 0.0,
    }
    schedules = {
        "learning_rate_fraction": 1.0,
        "entropy_coefficient": ppo.entropy_coefficient,
    }
    task_sampler = {
        "episodes_completed": 0,
        "unfinished_environment_count": 0,
        "next_template_batch_index": 0,
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


def _collect_normalization_warmup(
    *,
    env,
    initial,
    actor,
    output: Path,
    rollout_config: RolloutConfig,
    training_config: TaskTrainingConfig,
    normalization_config: CriticNormalizationConfig,
    event,
):
    """Collect no-gradient active-slot moments, freeze them, then reset the task."""
    if not normalization_config.enabled:
        return (
            initial,
            disabled_normalization_state(),
            {
                "performed": False,
                "reused_from_checkpoint": False,
                "warmup_steps": 0,
                "environment_transitions": 0,
                "agent_transitions": 0,
                "episodes_completed": 0,
                "seconds": 0.0,
            },
        )
    cfg = rollout_config
    accumulator = CriticGroupAccumulator(
        env.observation_cfg, env.construction_cfg.num_agents, env.device
    )
    current_observation = initial[("agents", "observation")].clone()
    current_state = initial[("agents", "state")].clone()
    actor_memory = actor.backbone.recurrent.initial_state(
        current_observation, cfg.num_envs * cfg.num_agents
    )
    episode_count = 0
    started = time.perf_counter()
    event(
        "critic_normalization_warmup_started",
        warmup_steps=normalization_config.warmup_steps,
    )
    with (output / "normalization-warmup.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=NORMALIZATION_WARMUP_COLUMNS)
        writer.writeheader()
        with torch.no_grad():
            for step in range(normalization_config.warmup_steps):
                reductions = accumulator.update(current_state)
                result = actor.act(
                    current_observation.reshape(
                        cfg.num_envs * cfg.num_agents, 1, cfg.actor_observation_dim
                    ),
                    actor_memory,
                    torch.ones(cfg.num_envs * cfg.num_agents, 1, device=env.device),
                    deterministic=not training_config.stochastic_actions,
                )
                actions = result.action[:, 0].reshape(cfg.num_envs, cfg.num_agents, cfg.action_dim)
                output_td = env.step_actions(actions)
                terminated = output_td["terminated"].squeeze(-1).clone()
                truncated = output_td["truncated"].squeeze(-1).clone()
                done = terminated | truncated
                row = {
                    "warmup_step": step,
                    "action_min": float(actions.min().item()),
                    "action_max": float(actions.max().item()),
                    "terminated_count": int(terminated.sum().item()),
                    "truncated_count": int(truncated.sum().item()),
                }
                for name, values in reductions.items():
                    row[f"{name}_count"] = values["count"]
                    row[f"{name}_sum"] = values["sum"]
                    row[f"{name}_sum_squares"] = values["sum_squares"]
                writer.writerow(row)
                stream.flush()
                episode_count += int(done.sum().item())
                actor_memory = zero_done_actor_memory(result.state, done, cfg)
                if bool(done.any()):
                    reset = env.reset_mask(done)
                    current_observation = reset[("agents", "observation")].clone()
                    current_state = reset[("agents", "state")].clone()
                else:
                    current_observation = output_td[("agents", "observation")].clone()
                    current_state = output_td[("agents", "state")].clone()
    normalization = accumulator.freeze(normalization_config, normalization_config.warmup_steps)
    all_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    fresh_initial = env.reset_mask(all_mask)
    seconds = time.perf_counter() - started
    metrics = {
        "performed": True,
        "reused_from_checkpoint": False,
        "warmup_steps": normalization_config.warmup_steps,
        "environment_transitions": normalization_config.warmup_steps * cfg.num_envs,
        "agent_transitions": (normalization_config.warmup_steps * cfg.num_envs * cfg.num_agents),
        "episodes_completed": episode_count,
        "seconds": seconds,
    }
    event(
        "critic_normalization_warmup_finished",
        environment_transitions=metrics["environment_transitions"],
        groups=normalization["groups"],
    )
    return fresh_initial, normalization, metrics


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
    critic_normalization_config: CriticNormalizationConfig,
    event,
    critic_calibration_config=None,
    fault_after_rollout_step: int | None = None,
) -> dict:
    """Collect one real-task rollout, update once, and commit the boundary."""
    del recovery_config
    started = time.perf_counter()
    cfg = rollout_config
    if fault_after_rollout_step is not None and (
        type(fault_after_rollout_step) is not int or not 1 <= fault_after_rollout_step < cfg.horizon
    ):
        raise ValueError("fault_after_rollout_step must be inside the rollout horizon")
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

    if training_config.normalization_enabled:
        raise ValueError(
            "training.normalization_enabled is a retired actor-normalization flag; "
            "use critic_normalization instead"
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
        if not normalization_matches_config(normalization, critic_normalization_config):
            raise ValueError("checkpoint critic normalization differs from the resolved config")
        schedules = restored["schedules"]
        task_sampler = dict(restored["task_sampler_state"])
        abandoned = int(task_sampler["unfinished_environment_count"])
        warmup_metrics = {
            "performed": False,
            "reused_from_checkpoint": normalization["enabled"],
            "warmup_steps": 0,
            "environment_transitions": 0,
            "agent_transitions": 0,
            "episodes_completed": 0,
            "seconds": 0.0,
        }
        event(
            "training_resumed",
            checkpoint_id=start_manifest["checkpoint_id"],
            completed_updates=counters["completed_updates"],
            abandoned_environment_episodes=abandoned,
            critic_normalization_reused=normalization["enabled"],
        )
    else:
        if list(checkpoint_directory.glob("checkpoint-*.json")):
            raise RuntimeError("a new logical run requires an empty checkpoint directory")
        initial, normalization, warmup_metrics = _collect_normalization_warmup(
            env=env,
            initial=initial,
            actor=actor,
            output=output,
            rollout_config=rollout_config,
            training_config=training_config,
            normalization_config=critic_normalization_config,
            event=event,
        )
        state, counters, normalization, schedules, task_sampler = _initial_state(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            resolved_config,
            ppo_config,
            normalization,
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
        event(
            "training_initial_checkpoint",
            checkpoint_id=start_manifest["checkpoint_id"],
            critic_normalization_enabled=normalization["enabled"],
        )

    normalizer = FrozenCriticGroupNormalizer(normalization, env.observation_cfg)
    start_counters = dict(counters)
    template_batch_index = start_counters["completed_updates"]
    restored_template_batch = int(
        task_sampler.get("next_template_batch_index", template_batch_index)
    )
    if restored_template_batch != template_batch_index:
        raise ValueError("checkpoint template schedule differs from the completed-update counter")
    env.set_template_batch(template_batch_index)
    all_mask = torch.ones(cfg.num_envs, dtype=torch.bool, device=env.device)
    initial = env.reset_mask(all_mask)
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
    current_raw_state = initial[("agents", "state")].clone()
    current_state = normalizer.apply(current_raw_state)
    raw_critic_states = torch.empty(
        cfg.horizon,
        cfg.num_envs,
        cfg.critic_state_dim,
        device=env.device,
        dtype=current_raw_state.dtype,
    )
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
    template_metrics = {
        kind: {
            "rows": 0,
            "formation_rows": 0,
            "team_reward_sum": 0.0,
            "assigned_rmse_sum_m": 0.0,
            "pairwise_rmse_sum_m": 0.0,
            "minimum_separation_m": math.inf,
            "outcomes": 0,
        }
        for kind in env.formation_schedule.kinds
    }
    collection_started = time.perf_counter()
    with (output / "rollout.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROLLOUT_COLUMNS)
        writer.writeheader()
        for step in range(cfg.horizon):
            with torch.no_grad():
                raw_critic_states[step].copy_(current_raw_state)
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
                final_raw_state = output_td[("agents", "state")].clone()
                final_state = normalizer.apply(final_raw_state)
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
                phases = env.last_reward_phase.clone()
                next_actor_memory = zero_done_actor_memory(actor_result.state, done, cfg)
                next_critic_memory = zero_done_critic_memory(critic_result.state, done)
                actions_cpu = actions.cpu()
                template_names = env.template_names()
                for env_id in range(cfg.num_envs):
                    kind = template_names[env_id]
                    template_row = template_metrics[kind]
                    phase = ("ground", "takeoff", "formation")[int(phases[env_id].item())]
                    template_row["rows"] += 1
                    template_row["formation_rows"] += int(phase == "formation")
                    template_row["team_reward_sum"] += float(team_rewards[env_id].item())
                    template_row["assigned_rmse_sum_m"] += float(
                        env.last_assigned_rmse[env_id].item()
                    )
                    template_row["pairwise_rmse_sum_m"] += float(
                        env.last_pairwise_rmse[env_id].item()
                    )
                    template_row["minimum_separation_m"] = min(
                        template_row["minimum_separation_m"],
                        float(env.last_minimum_separation[env_id].item()),
                    )
                    template_row["outcomes"] += int(done[env_id].item())
                    writer.writerow(
                        {
                            "attempt_id": attempt_id,
                            "rollout_step": step,
                            "env_id": env_id,
                            "episode_step": int(progress_before_reset[env_id].item()),
                            "phase": phase,
                            "formation_kind": kind,
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
                    current_raw_state = reset_td[("agents", "state")].clone()
                    current_state = normalizer.apply(current_raw_state)
                else:
                    current_observation = final_observation
                    current_raw_state = final_raw_state
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
                completed_rollout_steps = step + 1
                if fault_after_rollout_step == completed_rollout_steps:
                    stream.flush()
                    os.fsync(stream.fileno())
                    interruption = {
                        "schema_version": 1,
                        "kind": "injected_mid_rollout_interruption",
                        "attempt_id": attempt_id,
                        "start_checkpoint_id": start_manifest["checkpoint_id"],
                        "start_checkpoint_sha256": start_manifest["payload"]["sha256"],
                        "completed_updates_before": start_counters["completed_updates"],
                        "partial_rollout_steps": completed_rollout_steps,
                        "discarded_environment_transitions": (
                            completed_rollout_steps * cfg.num_envs
                        ),
                        "discarded_agent_transitions": (
                            completed_rollout_steps * cfg.num_envs * cfg.num_agents
                        ),
                        "ppo_update_started": False,
                        "checkpoint_committed": False,
                    }
                    write_json_atomic(output / "interruption.json", interruption, mode=0o644)
                    event("training_interruption_injected", **interruption)
                    raise InjectedTrainingInterruption(
                        f"injected interruption after {completed_rollout_steps} rollout steps"
                    )
    collection_seconds = time.perf_counter() - collection_started
    critic_distribution = summarize_critic_rollout_distribution(
        raw_critic_states,
        buffer.critic_states[:-1],
        normalization,
        env.observation_cfg,
        cfg.num_agents,
    )
    with (output / "critic-distribution.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("completed_update", *critic_distribution[0]))
        writer.writeheader()
        for row in critic_distribution:
            writer.writerow({"completed_update": start_counters["completed_updates"] + 1, **row})

    advantages = buffer.compute_gae()
    chunks = buffer.sequence_chunks()
    critic_calibration = None
    if critic_calibration_config is not None:
        from align.simulation.critic_calibration import calibrate_critic_steps

        critic_before_calibration = {
            name: value.detach().clone() for name, value in critic.state_dict().items()
        }
        cpu_rng_before_calibration = torch.get_rng_state()
        cuda_rng_before_calibration = torch.cuda.get_rng_state_all()
        try:
            critic_calibration = calibrate_critic_steps(
                critic=critic,
                chunks=chunks,
                rollout=buffer,
                output=output,
                policy_config=policy_config,
                observation_config=env.observation_cfg,
                ppo_config=ppo_config,
                calibration_config=critic_calibration_config,
            )
        finally:
            torch.set_rng_state(cpu_rng_before_calibration)
            torch.cuda.set_rng_state_all(cuda_rng_before_calibration)
        primary_critic_unchanged = all(
            torch.equal(critic_before_calibration[name], value)
            for name, value in critic.state_dict().items()
        )
        critic_calibration["checks"]["primary_critic_parameters_unchanged"] = (
            primary_critic_unchanged
        )
        critic_calibration["primary_learner_rng_restored"] = True
        critic_calibration["status"] = (
            "passed" if all(critic_calibration["checks"].values()) else "failed"
        )
        if critic_calibration["status"] != "passed":
            raise RuntimeError("matched-rollout critic calibration checks failed")
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
    post_update_diagnostics = evaluate_ppo_diagnostics(actor, critic, chunks, ppo_config)
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
        "next_template_batch_index": completed_updates,
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
    ] + list(post_update_diagnostics.values())
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
    template_measurements = {
        kind: {
            "rows": values["rows"],
            "formation_rows": values["formation_rows"],
            "team_reward_mean": values["team_reward_sum"] / values["rows"],
            "assigned_rmse_mean_m": values["assigned_rmse_sum_m"] / values["rows"],
            "pairwise_rmse_mean_m": values["pairwise_rmse_sum_m"] / values["rows"],
            "minimum_separation_m": values["minimum_separation_m"],
            "outcomes": values["outcomes"],
        }
        for kind, values in template_metrics.items()
        if values["rows"]
    }
    checks = {
        "real_task_rollout_is_full": buffer.full,
        "all_declared_formation_templates_observed": set(template_measurements)
        == set(env.formation_schedule.kinds),
        "all_declared_templates_reached_formation_phase": all(
            values["formation_rows"] > 0 for values in template_measurements.values()
        ),
        "formation_target_sets_are_distinct": len(
            {tuple(layout.assigned_target_positions_m) for layout in env.formation_layouts}
        )
        == len(env.formation_schedule.kinds),
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
        "critic_normalization_matches_config": normalization_matches_config(
            normalization, critic_normalization_config
        ),
        "critic_normalization_is_frozen": normalization["frozen"],
        "critic_distribution_has_exact_active_counts": (
            len(critic_distribution) == 3
            and all(
                row["count"] == cfg.horizon * cfg.num_envs * cfg.num_agents * 3
                for row in critic_distribution
            )
        ),
        "critic_distribution_is_finite": all(
            math.isfinite(value)
            for row in critic_distribution
            for key, value in row.items()
            if key != "group"
        ),
        "critic_normalization_warmup_contract_applied": (
            (
                resume
                and not warmup_metrics["performed"]
                and warmup_metrics["reused_from_checkpoint"] == critic_normalization_config.enabled
            )
            or (
                not resume
                and warmup_metrics["performed"] == critic_normalization_config.enabled
                and warmup_metrics["environment_transitions"]
                == critic_normalization_config.warmup_steps * cfg.num_envs
            )
        ),
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
    metrics = {
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
        "formation_schedule": env.formation_schedule.to_dict(),
        "template_batch_index": template_batch_index,
        "template_measurements": template_measurements,
        "rollout_rows": raw_rows,
        "critic_normalization": normalization,
        "critic_distribution": critic_distribution,
        "critic_normalization_warmup": warmup_metrics,
        "critic_normalization_warmup_environment_transitions": warmup_metrics[
            "environment_transitions"
        ],
        "critic_normalization_warmup_agent_transitions": warmup_metrics["agent_transitions"],
        "measurements": {
            "template_measurements": template_measurements,
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
        "post_update": post_update_diagnostics,
    }
    if critic_calibration is not None:
        metrics["critic_calibration"] = critic_calibration
    return metrics
