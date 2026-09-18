"""Deterministic target-directed reference control in the real vector task."""

from __future__ import annotations

import csv
import math
from contextlib import ExitStack
from pathlib import Path

import torch

from align.simulation.multi_drone_contract import velocity_actions, velocity_commands
from align.simulation.policy_evaluation import EVALUATION_COLUMNS, write_telemetry_rows
from align.tasks.policy_telemetry import TELEMETRY_COLUMNS

PHASE_NAMES = ("ground", "takeoff", "formation")
WAYPOINT_COLUMNS = (
    "evaluation_step",
    "env_id",
    "episode_step",
    "active",
    "index_before",
    "index_after",
    "settled",
    "dwell_steps",
    "advanced",
    "complete",
    "maximum_agent_error_m",
    "minimum_separation_m",
    "maximum_speed_m_s",
    "success_dwell_steps",
    "reason_code",
)


def reference_actions(env) -> torch.Tensor:
    """Use exact task targets and positions to issue bounded world-frame velocity actions."""
    positions = env.state[..., :3].detach().cpu().tolist()
    targets = env.targets_for_progress().detach().cpu().tolist()
    config = env.construction_cfg
    actions = [
        velocity_actions(
            velocity_commands(
                env_positions,
                env_targets,
                position_gain_s_inv=config.position_gain_s_inv,
                max_speed_m_s=config.max_speed_m_s,
            ),
            config.max_speed_m_s,
        )
        for env_positions, env_targets in zip(positions, targets, strict=True)
    ]
    return torch.tensor(actions, device=env.device, dtype=torch.float32)


def run_reference_evaluation(*, env, output: Path, evaluation_steps: int, event) -> dict:
    """Evaluate a target-following rule without a learned actor or checkpoint."""
    if evaluation_steps < 1:
        raise ValueError("evaluation_steps must be positive")
    num_envs = env.num_envs
    num_agents = env.construction_cfg.num_agents
    env.set_template_batch(0)
    env.reset_mask(torch.ones(num_envs, dtype=torch.bool, device=env.device))
    phase_rows = {phase: 0 for phase in PHASE_NAMES}
    outcome_counts = {str(code): 0 for code in range(1, 7)}
    action_min, action_max = math.inf, -math.inf
    assigned_sum, pairwise_sum = 0.0, 0.0
    minimum_separation = math.inf
    nonfinite_rows = 0
    saturation_count = 0
    environment_rows = 0
    drone_rows = 0
    directed_rows = 0
    active_rows = 0
    with ExitStack() as stack:
        waypoint_stream = (
            stack.enter_context(
                (output / "waypoint-progress.csv").open("x", newline="", encoding="utf-8")
            )
            if env.waypoint_route_config is not None
            else None
        )
        waypoint_writer = (
            csv.DictWriter(waypoint_stream, fieldnames=WAYPOINT_COLUMNS)
            if waypoint_stream is not None
            else None
        )
        if waypoint_writer is not None:
            waypoint_writer.writeheader()
        aggregate_stream = stack.enter_context(
            (output / "evaluation.csv").open("x", newline="", encoding="utf-8")
        )
        telemetry_stream = stack.enter_context(
            (output / "policy-telemetry.csv").open("x", newline="", encoding="utf-8")
        )
        aggregate = csv.DictWriter(aggregate_stream, fieldnames=EVALUATION_COLUMNS)
        telemetry = csv.DictWriter(telemetry_stream, fieldnames=TELEMETRY_COLUMNS)
        aggregate.writeheader()
        telemetry.writeheader()
        with torch.no_grad():
            for step in range(evaluation_steps):
                actions = reference_actions(env)
                pre_state = env.state.clone()
                pre_targets = env.targets_for_progress().clone()
                error = pre_targets - pre_state[..., :3]
                commands = actions[..., :3] * (
                    actions[..., 3:4].abs() * env.construction_cfg.max_speed_m_s
                )
                active = commands.norm(dim=-1) > 1e-7
                active_rows += int(active.sum().item())
                directed_rows += int(
                    (((commands * error).sum(dim=-1) >= -1e-7) & active).sum().item()
                )
                output_td = env.step_actions(actions)
                terminated = output_td["terminated"].squeeze(-1)
                truncated = output_td["truncated"].squeeze(-1)
                done = terminated | truncated
                values = torch.stack(
                    (
                        env.last_team_reward,
                        env.last_assigned_rmse,
                        env.last_pairwise_rmse,
                        env.last_minimum_separation,
                    ),
                    dim=-1,
                )
                nonfinite_rows += int((~torch.isfinite(values).all(dim=-1)).sum().item())
                action_min = min(action_min, float(actions.min().item()))
                action_max = max(action_max, float(actions.max().item()))
                saturation_count += env.action_saturation_count
                assigned_sum += float(env.last_assigned_rmse.sum().item())
                pairwise_sum += float(env.last_pairwise_rmse.sum().item())
                minimum_separation = min(
                    minimum_separation, float(env.last_minimum_separation.min().item())
                )
                names = env.template_names()
                actions_cpu = actions.cpu()
                for env_id in range(num_envs):
                    phase = PHASE_NAMES[int(env.last_reward_phase[env_id].item())]
                    reason = int(env.last_reason_code[env_id].item())
                    progress = int(env.progress_buf[env_id].item())
                    phase_rows[phase] += 1
                    if reason:
                        outcome_counts[str(reason)] += 1
                    aggregate.writerow(
                        {
                            "evaluation_step": step,
                            "env_id": env_id,
                            "episode_step": progress,
                            "phase": phase,
                            "formation_kind": names[env_id],
                            "team_reward": float(env.last_team_reward[env_id].item()),
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
                    if waypoint_writer is not None:
                        route_target_error = (
                            env.state[env_id, :, :3] - env.last_targets[env_id]
                        ).norm(dim=-1)
                        route_speed = env.state[env_id, :, 7:10].norm(dim=-1)
                        waypoint_writer.writerow(
                            {
                                "evaluation_step": step,
                                "env_id": env_id,
                                "episode_step": progress,
                                "active": bool(env.last_waypoint_active[env_id].item()),
                                "index_before": int(env.last_waypoint_index[env_id].item()),
                                "index_after": int(env.waypoint_index[env_id].item()),
                                "settled": bool(env.waypoint_settled_this_step[env_id].item()),
                                "dwell_steps": int(env.waypoint_dwell[env_id].item()),
                                "advanced": bool(env.waypoint_advanced_this_step[env_id].item()),
                                "complete": bool(env.waypoint_complete[env_id].item()),
                                "maximum_agent_error_m": float(route_target_error.max().item()),
                                "minimum_separation_m": float(
                                    env.last_minimum_separation[env_id].item()
                                ),
                                "maximum_speed_m_s": float(route_speed.max().item()),
                                "success_dwell_steps": int(env.success_dwell[env_id].item()),
                                "reason_code": reason,
                            }
                        )
                    environment_rows += 1
                    drone_rows += write_telemetry_rows(
                        telemetry,
                        env=env,
                        output_td=output_td,
                        pre_state=pre_state,
                        actions_cpu=actions_cpu,
                        env_id=env_id,
                        step=step,
                        progress=progress,
                        phase=phase,
                        kind=names[env_id],
                    )
                aggregate_stream.flush()
                telemetry_stream.flush()
                if waypoint_stream is not None:
                    waypoint_stream.flush()
                if bool(done.any()):
                    env.reset_mask(done)
    expected = evaluation_steps * num_envs
    checks = {
        "exact_environment_rows": environment_rows == expected,
        "exact_drone_rows": drone_rows == expected * num_agents,
        "finite_metrics_and_state": nonfinite_rows == 0 and bool(torch.isfinite(env.state).all()),
        "bounded_reference_actions": action_min >= -1.0 and action_max <= 1.0,
        "environment_did_not_clip_actions": saturation_count == 0,
        "commands_point_toward_targets": directed_rows == active_rows,
        "no_policy_or_optimizer_used": True,
        "shape_commands_observed_if_requested": (
            env.shape_transition_config is None or env.transition_total_commands >= num_envs
        ),
        "waypoint_commands_observed_if_requested": (
            env.waypoint_route_config is None or env.waypoint_total_commands >= num_envs
        ),
    }
    event("reference_loop_finished", checks=checks, outcome_counts=outcome_counts)
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "controller": "target_proportional_world_velocity",
        "controller_position_gain_s_inv": env.construction_cfg.position_gain_s_inv,
        "max_speed_m_s": env.construction_cfg.max_speed_m_s,
        "privileged_exact_target_and_pose": True,
        "optimizer_updates": 0,
        "evaluation_steps": evaluation_steps,
        "raw_rows": environment_rows,
        "telemetry_rows": drone_rows,
        "phase_rows": phase_rows,
        "formation_phase_reached": phase_rows["formation"] > 0,
        "outcome_counts": outcome_counts,
        "active_drone_steps": active_rows,
        "target_directed_drone_steps": directed_rows,
        "shape_transition_commands": env.transition_total_commands,
        "waypoint_route_commands": env.waypoint_total_commands,
        "waypoint_route_advances": env.waypoint_total_advances,
        "waypoint_route_completions": env.waypoint_total_completions,
        "waypoint_route_config": (
            env.waypoint_route_config.to_dict() if env.waypoint_route_config is not None else None
        ),
        "shape_transition_config": (
            env.shape_transition_config.to_dict()
            if env.shape_transition_config is not None
            else None
        ),
        "measurements": {
            "assigned_rmse_mean_m": assigned_sum / expected,
            "pairwise_rmse_mean_m": pairwise_sum / expected,
            "minimum_separation_m": minimum_separation,
            "action_min": action_min,
            "action_max": action_max,
        },
    }
