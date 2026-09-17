"""Fresh-process deterministic evaluation of a committed recurrent policy."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import torch

from align.learning.checkpoint_store import CheckpointStore
from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.rollout import RolloutConfig
from align.learning.stability_config import StabilityConfig
from align.learning.torch_normalization import (
    FrozenCriticGroupNormalizer,
    normalization_matches_config,
)
from align.learning.torch_recovery import restore_learner_state
from align.learning.training_config import TaskTrainingConfig
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import CentralizedRecurrentCritic, SharedRecurrentActor
from align.simulation.recurrent_collector import zero_done_actor_memory

EVALUATION_COLUMNS = (
    "evaluation_step",
    "env_id",
    "episode_step",
    "phase",
    "team_reward",
    "assigned_rmse_m",
    "pairwise_rmse_m",
    "minimum_separation_m",
    "action_min",
    "action_max",
    "terminated",
    "truncated",
    "reason_code",
)


def _optimizer(module, learning_rate: float, config: RecurrentPPOConfig):
    return torch.optim.Adam(module.parameters(), lr=learning_rate, eps=config.adam_epsilon)


def run_policy_evaluation(
    *,
    env,
    initial,
    output: Path,
    checkpoint_directory: Path,
    logical_run_id: str,
    resolved_config: dict,
    config_sha256: str,
    policy_config: RecurrentPolicyConfig,
    ppo_config: RecurrentPPOConfig,
    rollout_config: RolloutConfig,
    training_config: TaskTrainingConfig,
    critic_normalization_config: CriticNormalizationConfig,
    stability_config: StabilityConfig,
    event,
    checkpoint_update: int | None = None,
) -> dict:
    """Load one requested checkpoint and execute deterministic actor means only."""
    actor = SharedRecurrentActor(policy_config).to(env.device)
    critic = CentralizedRecurrentCritic(policy_config).to(env.device)
    actor_optimizer = _optimizer(actor, ppo_config.actor_learning_rate, ppo_config)
    critic_optimizer = _optimizer(critic, ppo_config.critic_learning_rate, ppo_config)
    store = CheckpointStore(
        checkpoint_directory,
        run_id=logical_run_id,
        config_sha256=config_sha256,
        file_mode=0o644,
    )
    if checkpoint_update is None:
        manifest, payload = store.latest_valid()
        expected_update = stability_config.updates_per_seed
    else:
        manifest, payload = store.for_update(checkpoint_update)
        expected_update = checkpoint_update
    restored = restore_learner_state(
        payload,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        expected_config=resolved_config,
    )
    normalization = restored["normalization"]
    normalization_before = copy.deepcopy(normalization)
    normalizer = FrozenCriticGroupNormalizer(normalization, env.observation_cfg)
    normalized_initial_state = normalizer.apply(initial[("agents", "state")])
    if not normalization_matches_config(normalization, critic_normalization_config):
        raise ValueError("checkpoint critic normalization differs from evaluation config")
    actor.eval()
    before = {name: value.detach().clone() for name, value in actor.state_dict().items()}

    num_envs = env.num_envs
    num_agents = env.construction_cfg.num_agents
    observation_dim = policy_config.actor_observation_dim
    actor_memory = actor.backbone.recurrent.initial_state(
        initial[("agents", "observation")], num_envs * num_agents
    )
    current_observation = initial[("agents", "observation")].clone()
    action_min = math.inf
    action_max = -math.inf
    reward_sum = 0.0
    assigned_sum = 0.0
    pairwise_sum = 0.0
    minimum_separation = math.inf
    saturation_count = 0
    nonfinite_rows = 0
    outcome_counts = {str(code): 0 for code in range(1, 7)}
    phase_rows = {"ground": 0, "takeoff": 0, "formation": 0}
    raw_rows = 0
    with (output / "evaluation.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS)
        writer.writeheader()
        with torch.no_grad():
            for step in range(stability_config.evaluation_steps):
                result = actor.act(
                    current_observation.reshape(num_envs * num_agents, 1, observation_dim),
                    actor_memory,
                    torch.ones(num_envs * num_agents, 1, device=env.device),
                    deterministic=True,
                )
                actions = result.action[:, 0].reshape(num_envs, num_agents, -1)
                output_td = env.step_actions(actions)
                terminated = output_td["terminated"].squeeze(-1).clone()
                truncated = output_td["truncated"].squeeze(-1).clone()
                done = terminated | truncated
                progress = env.progress_buf.clone()
                phases = env.last_reward_phase.clone()
                reasons = env.last_reason_code.clone()
                team_rewards = env.last_team_reward.clone()
                values = torch.stack(
                    (
                        team_rewards,
                        env.last_assigned_rmse,
                        env.last_pairwise_rmse,
                        env.last_minimum_separation,
                    ),
                    dim=-1,
                )
                nonfinite_rows += int((~torch.isfinite(values).all(dim=-1)).sum().item())
                actions_cpu = actions.cpu()
                for env_id in range(num_envs):
                    phase = ("ground", "takeoff", "formation")[int(phases[env_id].item())]
                    phase_rows[phase] += 1
                    reason = int(reasons[env_id].item())
                    if reason:
                        outcome_counts[str(reason)] += 1
                    writer.writerow(
                        {
                            "evaluation_step": step,
                            "env_id": env_id,
                            "episode_step": int(progress[env_id].item()),
                            "phase": phase,
                            "team_reward": float(team_rewards[env_id].item()),
                            "assigned_rmse_m": float(env.last_assigned_rmse[env_id].item()),
                            "pairwise_rmse_m": float(env.last_pairwise_rmse[env_id].item()),
                            "minimum_separation_m": float(
                                env.last_minimum_separation[env_id].item()
                            ),
                            "action_min": float(actions_cpu[env_id].min().item()),
                            "action_max": float(actions_cpu[env_id].max().item()),
                            "terminated": bool(terminated[env_id].item()),
                            "truncated": bool(truncated[env_id].item()),
                            "reason_code": reason,
                        }
                    )
                    raw_rows += 1
                stream.flush()
                action_min = min(action_min, float(actions.min().item()))
                action_max = max(action_max, float(actions.max().item()))
                reward_sum += float(team_rewards.sum().item())
                assigned_sum += float(env.last_assigned_rmse.sum().item())
                pairwise_sum += float(env.last_pairwise_rmse.sum().item())
                minimum_separation = min(
                    minimum_separation, float(env.last_minimum_separation.min().item())
                )
                saturation_count += env.action_saturation_count
                actor_memory = zero_done_actor_memory(result.state, done, rollout_config)
                final_observation = output_td[("agents", "observation")].clone()
                if bool(done.any()):
                    reset = env.reset_mask(done)
                    current_observation = reset[("agents", "observation")].clone()
                else:
                    current_observation = final_observation

    parameters_unchanged = all(
        torch.equal(before[name], value) for name, value in actor.state_dict().items()
    )
    denominator = stability_config.evaluation_steps * num_envs
    checks = {
        "loaded_requested_completed_update": manifest["completed_updates"] == expected_update,
        "deterministic_actor_used": True,
        "raw_row_count_is_exact": raw_rows == denominator,
        "observations_and_metrics_are_finite": nonfinite_rows == 0
        and bool(torch.isfinite(current_observation).all()),
        "actions_are_bounded": action_min >= -1.0 and action_max <= 1.0,
        "environment_did_not_clip_actions": saturation_count == 0,
        "actor_parameters_unchanged": parameters_unchanged,
        "critic_normalization_is_frozen": normalization["frozen"],
        "critic_normalization_matches_config": normalization_matches_config(
            normalization, critic_normalization_config
        ),
        "critic_normalization_state_is_finite": bool(
            torch.isfinite(normalized_initial_state).all()
        ),
        "critic_normalization_unchanged": normalization == normalization_before,
        "phase_accounting_is_exact": sum(phase_rows.values()) == raw_rows,
    }
    event(
        "deterministic_evaluation_finished",
        checkpoint_id=manifest["checkpoint_id"],
        checks=checks,
    )
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "policy_seed": training_config.policy_seed,
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_sha256": manifest["payload"]["sha256"],
        "completed_updates": manifest["completed_updates"],
        "requested_completed_update": expected_update,
        "deterministic_actions": True,
        "optimizer_updates": 0,
        "critic_normalization": normalization,
        "critic_normalization_updates": 0,
        "evaluation_steps": stability_config.evaluation_steps,
        "raw_rows": raw_rows,
        "phase_rows": phase_rows,
        "formation_phase_reached": phase_rows["formation"] > 0,
        "outcome_counts": outcome_counts,
        "measurements": {
            "team_reward_mean": reward_sum / denominator,
            "assigned_rmse_mean_m": assigned_sum / denominator,
            "pairwise_rmse_mean_m": pairwise_sum / denominator,
            "minimum_separation_m": minimum_separation,
            "action_min": action_min,
            "action_max": action_max,
        },
    }
