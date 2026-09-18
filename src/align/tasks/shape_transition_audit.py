"""Host audit of a commanded shape switch in saved reference flight rows."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


def audit_shape_transition(evaluation_csv: Path, telemetry_csv: Path, plan: dict) -> dict:
    """Require the first episode of every environment to switch at the declared step."""
    command_step = plan["command_step"]
    source_kind = plan["source_kind"]
    destination_kind = plan["destination_kind"]
    destinations = plan["assignment"]["assigned_destination_m"]
    agent_count = len(destinations)
    by_env: dict[int, list[dict]] = defaultdict(list)
    with evaluation_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_env[int(row["env_id"])].append(row)
    if not by_env:
        raise ValueError("reference evaluation has no environment rows")
    first_episodes = {}
    for env_id, rows in by_env.items():
        first = []
        for row in rows:
            first.append(row)
            if int(row["reason_code"]) != 0:
                break
        steps = [int(row["episode_step"]) for row in first]
        if steps != list(range(1, len(first) + 1)):
            raise ValueError(f"environment {env_id} first episode steps are not contiguous")
        command_rows = [row for row in first if int(row["episode_step"]) == command_step + 1]
        if len(command_rows) != 1:
            raise ValueError(f"environment {env_id} did not issue the in-flight command")
        if any(
            row["formation_kind"]
            != (source_kind if int(row["episode_step"]) <= command_step else destination_kind)
            for row in first
        ):
            raise ValueError(f"environment {env_id} source/destination labels differ")
        first_episodes[env_id] = first
    target_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with telemetry_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["episode_step"]) == command_step + 1:
                target_rows[(int(row["env_id"]), int(row["agent_id"]))].append(row)
    for env_id in first_episodes:
        for agent_id, expected in enumerate(destinations):
            matches = target_rows[(env_id, agent_id)]
            if not matches:
                raise ValueError(f"missing destination target for env {env_id}, drone {agent_id}")
            actual = tuple(float(matches[0][f"target_{axis}_m"]) for axis in "xyz")
            if math.dist(actual, expected) > 1e-4:
                raise ValueError(f"destination target differs for env {env_id}, drone {agent_id}")
    summaries = []
    for env_id, rows in sorted(first_episodes.items()):
        after = [row for row in rows if int(row["episode_step"]) > command_step]
        separations = [float(row["minimum_separation_m"]) for row in after]
        errors = [float(row["assigned_rmse_m"]) for row in after]
        if not all(math.isfinite(value) for value in (*separations, *errors)):
            raise ValueError("nonfinite post-command geometry")
        reason = int(rows[-1]["reason_code"])
        summaries.append(
            {
                "env_id": env_id,
                "first_episode_steps": len(rows),
                "commanded_target_rows": len(after),
                "first_post_command_assigned_rmse_m": errors[0],
                "last_100_assigned_rmse_mean_m": math.fsum(errors[-100:]) / min(100, len(errors)),
                "minimum_post_command_separation_m": min(separations),
                "terminal_reason_code": reason,
                "success": reason == 1,
                "safety_termination": reason in {2, 3, 4, 5},
                "time_from_command_to_outcome_s": (
                    len(after) * plan["control_dt_seconds"] if reason else None
                ),
            }
        )
    return {
        "status": "passed",
        "environment_count": len(summaries),
        "agent_count": agent_count,
        "success_count": sum(row["success"] for row in summaries),
        "safety_termination_count": sum(row["safety_termination"] for row in summaries),
        "timeout_count": sum(row["terminal_reason_code"] == 6 for row in summaries),
        "episodes": summaries,
        "physical_trajectory_observed": True,
        "geometry_plan_clearance_m": plan["assignment"]["minimum_interpolated_separation_m"],
    }
