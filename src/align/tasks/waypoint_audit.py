"""Audit raw reference flight rows for a synchronized fixed-ID waypoint route."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from align.tasks.waypoints import WaypointPlan, WaypointRouteConfig


def _read_first_episodes(path: Path) -> dict[int, list[dict]]:
    by_env: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            env_id = int(row["env_id"])
            first = by_env[env_id]
            if first and int(first[-1]["reason_code"]) != 0:
                continue
            first.append(row)
    if not by_env:
        raise ValueError(f"{path.name} contains no episodes")
    for env_id, rows in by_env.items():
        if [int(row["episode_step"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"environment {env_id} first episode steps are not contiguous")
    return by_env


def audit_waypoint_route(
    evaluation_csv: Path,
    telemetry_csv: Path,
    progress_csv: Path,
    plan: WaypointPlan,
    config: WaypointRouteConfig,
    *,
    control_dt_seconds: float,
    formation_kind: str,
) -> dict:
    """Check command, assigned targets, gate progression, and first-episode outcomes."""
    if control_dt_seconds <= 0 or not math.isfinite(control_dt_seconds):
        raise ValueError("control timestep must be finite and positive")
    evaluation = _read_first_episodes(evaluation_csv)
    progress = _read_first_episodes(progress_csv)
    if set(evaluation) != set(progress):
        raise ValueError("evaluation and progress environment sets differ")
    target_steps: dict[tuple[int, int], int] = {}
    summaries = []
    for env_id in sorted(evaluation):
        eval_rows = evaluation[env_id]
        gate_rows = progress[env_id]
        if len(eval_rows) != len(gate_rows):
            raise ValueError(f"environment {env_id} progress row count differs")
        index = 0
        dwell = 0
        complete = False
        command_count = 0
        advances = 0
        completions = 0
        min_sep = math.inf
        pairwise_errors = []
        assigned_errors = []
        leg_completion_steps = []
        for evaluation_row, row in zip(eval_rows, gate_rows, strict=True):
            step = int(row["episode_step"])
            if step != int(evaluation_row["episode_step"]):
                raise ValueError(f"environment {env_id} step mismatch")
            if int(row["reason_code"]) != int(evaluation_row["reason_code"]):
                raise ValueError(f"environment {env_id} reason mismatch")
            if evaluation_row["formation_kind"] != formation_kind:
                raise ValueError("route must preserve the source template label")
            active = row["active"] == "True"
            expected_active = step > config.command_step
            if active != expected_active:
                raise ValueError(f"environment {env_id} command step differs")
            if step == config.command_step + 1:
                command_count += 1
            before = int(row["index_before"])
            after = int(row["index_after"])
            if before != index:
                raise ValueError(f"environment {env_id} waypoint index before differs")
            error = float(row["maximum_agent_error_m"])
            separation = float(row["minimum_separation_m"])
            speed = float(row["maximum_speed_m_s"])
            if not all(math.isfinite(value) for value in (error, separation, speed)):
                raise ValueError("nonfinite waypoint gate measurement")
            min_sep = min(min_sep, separation) if active else min_sep
            if active:
                target_steps[(env_id, step)] = before
                pairwise_errors.append(float(evaluation_row["pairwise_rmse_m"]))
                assigned_errors.append(float(evaluation_row["assigned_rmse_m"]))
            expected_settled = (
                active
                and not complete
                and error <= config.arrival_radius_m + 1e-5
                and separation >= config.minimum_arrival_separation_m - 1e-5
                and speed <= config.maximum_arrival_speed_m_s + 1e-5
            )
            if (row["settled"] == "True") != expected_settled:
                raise ValueError(f"environment {env_id} arrival gate differs at {step}")
            dwell = dwell + 1 if expected_settled else 0
            reached = dwell >= config.arrival_dwell_steps
            expected_advance = reached and index < len(plan.centers_m) - 1
            expected_complete = complete or (reached and not expected_advance)
            if expected_advance:
                index += 1
                advances += 1
                leg_completion_steps.append(step)
            if reached and not expected_advance:
                completions += 1
                leg_completion_steps.append(step)
            if reached:
                dwell = 0
            if after != index or (row["advanced"] == "True") != expected_advance:
                raise ValueError(f"environment {env_id} waypoint advancement differs at {step}")
            if (row["complete"] == "True") != expected_complete:
                raise ValueError(f"environment {env_id} completion flag differs at {step}")
            if int(row["dwell_steps"]) != dwell:
                raise ValueError(f"environment {env_id} dwell counter differs at {step}")
            if int(evaluation_row["reason_code"]) == 1 and not expected_complete:
                raise ValueError("success recorded before route completion")
            complete = expected_complete
        if command_count != 1:
            raise ValueError(f"environment {env_id} lacks one route command")
        post = len(pairwise_errors)
        if post == 0:
            raise ValueError(f"environment {env_id} has no post-command trajectory")
        summaries.append(
            {
                "env_id": env_id,
                "first_episode_steps": len(eval_rows),
                "commanded_steps": post,
                "advanced_legs": advances,
                "completed_legs": advances + completions,
                "route_complete": complete,
                "leg_completion_steps": leg_completion_steps,
                "minimum_post_command_separation_m": min_sep,
                "post_command_pairwise_rmse_mean_m": math.fsum(pairwise_errors) / post,
                "post_command_assigned_rmse_mean_m": math.fsum(assigned_errors) / post,
                "terminal_reason_code": int(eval_rows[-1]["reason_code"]),
                "success": int(eval_rows[-1]["reason_code"]) == 1,
                "time_from_command_to_outcome_s": post * control_dt_seconds,
            }
        )
    found: set[tuple[int, int, int]] = set()
    with telemetry_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["env_id"]), int(row["episode_step"]))
            if key not in target_steps:
                continue
            agent_id = int(row["agent_id"])
            if agent_id < 0 or agent_id >= len(plan.assigned_targets_m[0]):
                raise ValueError("invalid telemetry agent ID")
            item = (*key, agent_id)
            if item in found:
                continue  # Later episode with the same episode_step.
            expected = plan.assigned_targets_m[target_steps[key]][agent_id]
            actual = tuple(float(row[f"target_{axis}_m"]) for axis in "xyz")
            if not all(math.isfinite(value) for value in actual):
                raise ValueError("nonfinite waypoint target")
            if math.dist(actual, expected) > 1e-4:
                raise ValueError(f"waypoint target differs at {item}")
            found.add(item)
    expected_count = len(target_steps) * len(plan.assigned_targets_m[0])
    if len(found) != expected_count:
        raise ValueError(f"missing waypoint target rows: {len(found)}/{expected_count}")
    return {
        "status": "passed",
        "environment_count": len(summaries),
        "agent_count": len(plan.assigned_targets_m[0]),
        "leg_count": len(plan.centers_m),
        "success_count": sum(row["success"] for row in summaries),
        "route_completion_count": sum(row["route_complete"] for row in summaries),
        "safety_termination_count": sum(
            row["terminal_reason_code"] in {2, 3, 4, 5} for row in summaries
        ),
        "timeout_count": sum(row["terminal_reason_code"] == 6 for row in summaries),
        "episodes": summaries,
        "physical_trajectory_observed": True,
    }
