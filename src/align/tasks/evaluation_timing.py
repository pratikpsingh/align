"""Pure validation of evaluation-only phase durations for a frozen checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

from align.learning.stability_config import StabilityConfig
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig


@dataclass(frozen=True)
class EvaluationTimingConfig:
    schema_version: int
    takeoff_seconds: float
    formation_seconds: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("evaluation timing schema_version must be 1")
        for name in ("takeoff_seconds", "formation_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_dict(cls, value: dict) -> EvaluationTimingConfig:
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ValueError("evaluation timing fields mismatch")
        return cls(**value)

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def apply_evaluation_timing(
    timing: EvaluationTimingConfig,
    construction: MultiDroneConfig,
    task: TaskEnvironmentConfig,
    stability: StabilityConfig,
) -> tuple[MultiDroneConfig, TaskEnvironmentConfig, StabilityConfig, dict]:
    """Change only takeoff deadline and episode/evaluation length; retain dwell."""
    if stability.evaluation_steps != task.max_episode_steps:
        raise ValueError("source evaluation length must equal source episode limit")
    original_formation_steps = (
        task.max_episode_steps - construction.ground_steps - construction.takeoff_steps
    )
    if original_formation_steps <= task.success_dwell_steps:
        raise ValueError("source evaluation has no post-dwell formation interval")
    original_formation_seconds = original_formation_steps * construction.physics_dt
    if (
        timing.takeoff_seconds < construction.takeoff_seconds
        or timing.formation_seconds < original_formation_seconds
    ):
        raise ValueError("timing probe must not shorten either source phase")
    for name in ("takeoff_seconds", "formation_seconds"):
        value = getattr(timing, name)
        steps = value / construction.physics_dt
        if not math.isclose(steps, round(steps), abs_tol=1e-8):
            raise ValueError(f"{name} must be an exact number of control steps")
    formation_steps = round(timing.formation_seconds / construction.physics_dt)
    if formation_steps <= task.success_dwell_steps:
        raise ValueError("formation phase must exceed the success dwell")
    effective_construction = replace(
        construction,
        takeoff_seconds=timing.takeoff_seconds,
        formation_timeout_seconds=timing.formation_seconds,
    )
    total_steps = (
        effective_construction.ground_steps + effective_construction.takeoff_steps + formation_steps
    )
    if total_steps > 4000:
        raise ValueError("bounded timing probe permits at most 4000 evaluation steps")
    effective_task = replace(task, max_episode_steps=total_steps)
    effective_stability = replace(stability, evaluation_steps=total_steps)
    return (
        effective_construction,
        effective_task,
        effective_stability,
        {
            "source_ground_steps": construction.ground_steps,
            "source_takeoff_steps": construction.takeoff_steps,
            "source_formation_steps": original_formation_steps,
            "source_evaluation_steps": stability.evaluation_steps,
            "effective_ground_steps": effective_construction.ground_steps,
            "effective_takeoff_steps": effective_construction.takeoff_steps,
            "effective_formation_steps": formation_steps,
            "effective_evaluation_steps": total_steps,
            "success_dwell_steps": task.success_dwell_steps,
        },
    )


def summarize_evaluation_phases(path, *, final_window: int = 50) -> dict:
    """Compare physical task metrics within each phase, preserving raw row counts."""
    import csv
    from collections import defaultdict

    if final_window < 1:
        raise ValueError("final_window must be positive")
    phases: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["phase"] not in {"ground", "takeoff", "formation"}:
                raise ValueError("unknown evaluation phase")
            phases[row["phase"]].append(row)
    if not phases.get("formation"):
        raise ValueError("formation phase missing")
    result = {}
    for phase, rows in phases.items():
        fields = ("team_reward", "assigned_rmse_m", "pairwise_rmse_m", "minimum_separation_m")
        result[phase] = {
            "environment_rows": len(rows),
            **{
                f"mean_{field}": math.fsum(float(r[field]) for r in rows) / len(rows)
                for field in fields
            },
            "minimum_separation_m": min(float(r["minimum_separation_m"]) for r in rows),
        }
    formation_by_env: dict[int, list[dict]] = defaultdict(list)
    for row in phases["formation"]:
        formation_by_env[int(row["env_id"])].append(row)
    tail = [row for rows in formation_by_env.values() for row in rows[-final_window:]]
    result["formation"]["last_window_environment_rows"] = len(tail)
    result["formation"]["last_window_assigned_rmse_m"] = math.fsum(
        float(row["assigned_rmse_m"]) for row in tail
    ) / len(tail)
    return result


def compare_shared_prefix(
    baseline_path, extended_path, *, divergence_step: int, num_envs: int, num_agents: int
) -> dict:
    """Require identical raw drone rows before the timing schedules diverge."""
    import csv

    if divergence_step < 1 or num_envs < 1 or num_agents < 2:
        raise ValueError("invalid shared-prefix shape")
    expected = divergence_step * num_envs * num_agents
    count = 0
    with (
        baseline_path.open(newline="", encoding="utf-8") as baseline_stream,
        extended_path.open(newline="", encoding="utf-8") as extended_stream,
    ):
        baseline = csv.DictReader(baseline_stream)
        extended = csv.DictReader(extended_stream)
        if baseline.fieldnames != extended.fieldnames:
            raise ValueError("timing arms have different telemetry schemas")
        for left, right in zip(baseline, extended, strict=False):
            if int(left["evaluation_step"]) >= divergence_step:
                break
            if left != right:
                raise ValueError(f"timing arms diverged early at row {count}")
            count += 1
    if count != expected:
        raise ValueError(f"expected {expected} identical prefix rows; found {count}")
    return {
        "status": "passed",
        "identical_drone_rows": count,
        "shared_episode_steps": divergence_step,
        "divergence_at_episode_step": divergence_step + 1,
    }


def formation_switch_altitude(path) -> dict:
    """Record the first formation target altitude and actual height per drone."""
    import csv

    first = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["phase"] == "formation":
                first.setdefault((int(row["env_id"]), int(row["agent_id"])), row)
    if not first:
        raise ValueError("formation phase missing")
    rows = list(first.values())
    heights = [float(row["z_m"]) for row in rows]
    deficits = [float(row["target_z_m"]) - float(row["z_m"]) for row in rows]
    steps = {int(row["episode_step"]) for row in rows}
    if len(steps) != 1:
        raise ValueError("environments have inconsistent formation switch steps")
    return {
        "first_formation_episode_step": steps.pop(),
        "drone_count": len(rows),
        "mean_height_m": math.fsum(heights) / len(heights),
        "minimum_height_m": min(heights),
        "maximum_height_m": max(heights),
        "mean_target_altitude_deficit_m": math.fsum(deficits) / len(deficits),
        "minimum_target_altitude_deficit_m": min(deficits),
        "maximum_target_altitude_deficit_m": max(deficits),
    }
