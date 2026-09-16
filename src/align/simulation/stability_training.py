"""Multi-update stability calibration inside one live simulator process."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from align.artifacts import write_json_atomic
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.recovery_config import RecoveryConfig
from align.learning.rollout import RolloutConfig
from align.learning.stability_config import StabilityConfig
from align.learning.training_config import TaskTrainingConfig
from align.policies.config import RecurrentPolicyConfig
from align.simulation.task_training import run_training_attempt


def _assessment(metrics: dict, config: StabilityConfig) -> dict:
    post = metrics["post_update"]
    rows = metrics["updates"]
    actor_gradient = max(row["actor_gradient_norm_before_clip"] for row in rows)
    critic_gradient = max(row["critic_gradient_norm_before_clip"] for row in rows)
    return {
        "post_update_approximate_kl_within_guidance": abs(post["approximate_kl"])
        <= config.max_post_update_approximate_kl,
        "post_update_policy_clip_fraction_within_guidance": post["policy_clip_fraction"]
        <= config.max_post_update_policy_clip_fraction,
        "post_update_value_clip_fraction_within_guidance": post["value_clip_fraction"]
        <= config.max_post_update_value_clip_fraction,
        "actor_gradient_norm_within_guidance": actor_gradient
        <= config.max_actor_gradient_norm_before_clip,
        "critic_gradient_norm_within_guidance": critic_gradient
        <= config.max_critic_gradient_norm_before_clip,
    }


def run_stability_training(
    *,
    env,
    initial,
    output: Path,
    checkpoint_directory: Path,
    logical_run_id: str,
    resolved_config: dict,
    config_sha256: str,
    source_identity: str,
    runtime_identity: str,
    policy_config: RecurrentPolicyConfig,
    rollout_config: RolloutConfig,
    ppo_config: RecurrentPPOConfig,
    recovery_config: RecoveryConfig,
    training_config: TaskTrainingConfig,
    stability_config: StabilityConfig,
    event,
) -> dict:
    """Collect several independent full-horizon batches for one configured seed."""
    if training_config.attempts != stability_config.updates_per_seed:
        raise ValueError("training attempts must equal stability updates_per_seed")
    if training_config.policy_seed not in stability_config.policy_seeds:
        raise ValueError("training policy_seed is not in stability policy_seeds")

    attempts = []
    all_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    next_initial = initial
    for number in range(1, stability_config.updates_per_seed + 1):
        attempt_id = f"update-{number:04d}"
        attempt_output = output / attempt_id
        attempt_output.mkdir()
        if number > 1:
            next_initial = env.reset_mask(all_mask)
        event(
            "stability_update_started",
            update=number,
            policy_seed=training_config.policy_seed,
            resume=number > 1,
        )
        metrics = run_training_attempt(
            env=env,
            initial=next_initial,
            output=attempt_output,
            checkpoint_directory=checkpoint_directory,
            logical_run_id=logical_run_id,
            attempt_id=attempt_id,
            resume=number > 1,
            resolved_config=resolved_config,
            config_sha256=config_sha256,
            source_identity=source_identity,
            runtime_identity=runtime_identity,
            policy_config=policy_config,
            rollout_config=rollout_config,
            ppo_config=ppo_config,
            recovery_config=recovery_config,
            training_config=training_config,
            event=event,
        )
        metrics["stability_guidance"] = _assessment(metrics, stability_config)
        write_json_atomic(attempt_output / "metrics.json", metrics, mode=0o644)
        attempts.append(metrics)
        if metrics["status"] != "passed":
            raise RuntimeError(f"{attempt_id} training checks failed")
        event(
            "stability_update_finished",
            update=number,
            checkpoint_id=metrics["committed_checkpoint_id"],
            guidance=metrics["stability_guidance"],
        )

    final = attempts[-1]
    expected_environment = (
        stability_config.updates_per_seed * rollout_config.horizon * rollout_config.num_envs
    )
    expected_agents = expected_environment * rollout_config.num_agents
    checkpoint_manifests = list(checkpoint_directory.glob("checkpoint-*.json"))
    diagnostic_values = [
        value
        for attempt in attempts
        for value in (
            *attempt["post_update"].values(),
            *(
                metric
                for row in attempt["updates"]
                for key, metric in row.items()
                if key != "epoch"
            ),
        )
    ]
    checks = {
        "all_updates_passed": all(item["status"] == "passed" for item in attempts),
        "update_count_is_exact": len(attempts) == stability_config.updates_per_seed,
        "completed_update_counter_is_exact": final["end_counters"]["completed_updates"]
        == stability_config.updates_per_seed,
        "environment_transition_counter_is_exact": final["end_counters"]["environment_transitions"]
        == expected_environment,
        "agent_transition_counter_is_exact": final["end_counters"]["agent_transitions"]
        == expected_agents,
        "checkpoint_count_is_exact": len(checkpoint_manifests)
        == stability_config.updates_per_seed + 1,
        "diagnostics_are_finite": all(math.isfinite(value) for value in diagnostic_values),
        "rollout_budget_reaches_configured_formation_phase": rollout_config.horizon
        > env.construction_cfg.ground_steps + env.construction_cfg.takeoff_steps,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "policy_seed": training_config.policy_seed,
        "updates": stability_config.updates_per_seed,
        "rollout_horizon": rollout_config.horizon,
        "environment_transitions": expected_environment,
        "agent_transitions": expected_agents,
        "final_counters": final["end_counters"],
        "initial_checkpoint_id": attempts[0]["start_checkpoint_id"],
        "final_checkpoint_id": final["committed_checkpoint_id"],
        "guidance_is_acceptance_gate": False,
        "guidance": [item["stability_guidance"] for item in attempts],
        "measurements": [
            {
                "completed_update": item["end_counters"]["completed_updates"],
                **item["measurements"],
                **{f"post_{key}": value for key, value in item["post_update"].items()},
                "max_actor_gradient_norm_before_clip": max(
                    row["actor_gradient_norm_before_clip"] for row in item["updates"]
                ),
                "max_critic_gradient_norm_before_clip": max(
                    row["critic_gradient_norm_before_clip"] for row in item["updates"]
                ),
            }
            for item in attempts
        ],
    }
