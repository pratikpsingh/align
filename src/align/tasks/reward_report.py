"""Audit task rewards on an accepted multi-drone trajectory without Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from align.artifacts import (
    artifact_logger,
    as_ist,
    create_run_directory,
    utc_now,
    write_json_atomic,
)
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.simulation.multi_drone_report import assess_saved_run
from align.tasks.reward import (
    COMPONENT_NAMES,
    RewardConfig,
    RewardMemory,
    compute_step_reward,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "task-reward-baseline.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "task-reward"
REWARD_COLUMNS = [
    "repeat",
    "step",
    "t",
    "phase",
    "agent_id",
    "target_distance_m",
    "speed_m_s",
    "minimum_separation_m",
    "airborne_contact",
    *[f"raw_{name}" for name in COMPONENT_NAMES],
    *[f"weighted_{name}" for name in COMPONENT_NAMES],
    "total_reward",
    "team_reward",
]


def load_reward_config(path: Path) -> RewardConfig:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("reward configuration root must be an object")
    return RewardConfig.from_dict(value)


def audit_rows(
    rows: Sequence[dict[str, object]],
    source_config: MultiDroneConfig,
    reward_config: RewardConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Recompute every reward component from raw, post-physics trajectory rows."""
    by_repeat_step: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["repeat"]), int(row["step"]))
        by_repeat_step.setdefault(key, []).append(row)
    if not by_repeat_step:
        raise ValueError("trajectory has no samples")

    output_rows: list[dict[str, object]] = []
    repeat_summaries: dict[str, dict[str, object]] = {}
    exact_totals = True
    exact_team_means = True
    all_finite = True
    for repeat in sorted({key[0] for key in by_repeat_step}):
        memory = RewardMemory()
        previous_phase = None
        team_rewards: list[float] = []
        raw_sums = {name: 0.0 for name in COMPONENT_NAMES}
        weighted_sums = {name: 0.0 for name in COMPONENT_NAMES}
        repeat_rows = 0
        keys = sorted(key for key in by_repeat_step if key[0] == repeat)
        if [step for _, step in keys] != list(range(len(keys))):
            raise ValueError(f"repeat {repeat} has missing or unexpected steps")
        for key in keys:
            samples = sorted(by_repeat_step[key], key=lambda row: int(row["agent_id"]))
            if [int(row["agent_id"]) for row in samples] != list(range(source_config.num_agents)):
                raise ValueError("each reward step must contain every agent exactly once")
            phases = {str(row["phase"]) for row in samples}
            if len(phases) != 1:
                raise ValueError("agents disagree on task phase")
            phase = phases.pop()
            if phase != previous_phase:
                memory = RewardMemory()
                previous_phase = phase

            positions = tuple(_xyz(row, "") for row in samples)
            targets = tuple(_xyz(row, "target_") for row in samples)
            velocities = tuple(_xyz(row, "v") for row in samples)
            actions = tuple(tuple(float(row[f"a{index}"]) for index in range(4)) for row in samples)
            contacts = tuple(float(row["contact_force_n"]) for row in samples)
            airborne = tuple(
                phase != "ground" and float(row["z"]) >= source_config.airborne_height_m
                for row in samples
            )
            reward, memory = compute_step_reward(
                positions,
                targets,
                velocities,
                actions,
                contacts,
                airborne,
                reward_config,
                memory,
            )
            team_rewards.append(reward.team_reward)
            exact_team_means &= math.isclose(
                reward.team_reward,
                math.fsum(reward.total_by_agent) / source_config.num_agents,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for agent, (sample, raw, weighted, total) in enumerate(
                zip(
                    samples,
                    reward.raw_by_agent,
                    reward.weighted_by_agent,
                    reward.total_by_agent,
                    strict=True,
                )
            ):
                exact_totals &= math.isclose(
                    total,
                    weighted.total(),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                all_finite &= all(
                    math.isfinite(value)
                    for value in (
                        total,
                        reward.team_reward,
                        *raw.to_dict().values(),
                        *weighted.to_dict().values(),
                    )
                )
                output = {
                    "repeat": repeat,
                    "step": key[1],
                    "t": float(sample["t"]),
                    "phase": phase,
                    "agent_id": agent,
                    "target_distance_m": reward.target_distances_m[agent],
                    "speed_m_s": reward.speeds_m_s[agent],
                    "minimum_separation_m": reward.minimum_separations_m[agent],
                    "airborne_contact": int(reward.airborne_contact[agent]),
                    **{f"raw_{name}": getattr(raw, name) for name in COMPONENT_NAMES},
                    **{f"weighted_{name}": getattr(weighted, name) for name in COMPONENT_NAMES},
                    "total_reward": total,
                    "team_reward": reward.team_reward,
                }
                output_rows.append(output)
                for name in COMPONENT_NAMES:
                    raw_sums[name] += getattr(raw, name)
                    weighted_sums[name] += getattr(weighted, name)
                repeat_rows += 1

        repeat_summaries[str(repeat)] = {
            "step_count": len(keys),
            "agent_sample_count": repeat_rows,
            "team_reward_sum_over_steps": math.fsum(team_rewards),
            "team_reward_mean_per_step": math.fsum(team_rewards) / len(team_rewards),
            "raw_component_mean_per_agent_step": {
                name: raw_sums[name] / repeat_rows for name in COMPONENT_NAMES
            },
            "weighted_component_mean_per_agent_step": {
                name: weighted_sums[name] / repeat_rows for name in COMPONENT_NAMES
            },
            "minimum_total_reward": min(
                float(row["total_reward"]) for row in output_rows if int(row["repeat"]) == repeat
            ),
            "maximum_total_reward": max(
                float(row["total_reward"]) for row in output_rows if int(row["repeat"]) == repeat
            ),
        }

    formation_active = reward_config.formation_weight > 0.0
    formation_observed = any(abs(float(row["raw_formation"])) > 0.0 for row in output_rows)
    checks = {
        "finite": all_finite,
        "weighted_total_reconstructs": exact_totals,
        "team_reward_is_agent_mean": exact_team_means,
        "formation_weight_active": formation_active,
        "formation_component_observed": formation_observed,
        "safety_thresholds_match_source": math.isclose(
            reward_config.minimum_separation_m,
            source_config.minimum_separation_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            reward_config.contact_force_threshold_n,
            source_config.contact_force_threshold_n,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "control_timestep_matches_source": math.isclose(
            reward_config.control_dt_seconds,
            source_config.physics_dt * source_config.control_decimation,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "aggregation": {
            "formation": "mean all-pair squared distance distortion / target diameter squared",
            "team_reward": "arithmetic mean of per-agent total rewards",
            "state_costs": "formation/tracking/safety/settling/effort multiplied by control dt",
            "transition_terms": "progress and smoothness apply once per policy decision",
            "trajectory_summary": (
                "means use agent-policy-step samples; returns sum team means over steps"
            ),
        },
        "phase_memory": (
            "progress and smoothness memory resets at episode reset and target-phase change"
        ),
        "repeat_summaries": repeat_summaries,
        "row_count": len(output_rows),
    }, output_rows


def run_audit(
    source_run: Path,
    reward_config: RewardConfig,
    output_dir: Path,
    command: str,
) -> tuple[Path, dict[str, object]]:
    source_run = source_run.resolve()
    source_metrics, rows = assess_saved_run(source_run)
    source_config = MultiDroneConfig.from_dict(json.loads((source_run / "config.json").read_text()))
    reward_report, reward_rows = audit_rows(rows, source_config, reward_config)
    reward_report["checks"]["source_run_passed"] = source_metrics["status"] == "passed"
    reward_report["status"] = "passed" if all(reward_report["checks"].values()) else "failed"
    created_at_utc = utc_now()
    reward_report.update(
        {
            "schema_version": 1,
            "created_at_utc": created_at_utc,
            "created_at_ist": as_ist(created_at_utc),
            "command": command,
            "source_run": str(source_run),
            "source_run_id": source_run.name,
            "source_trajectory_sha256": hashlib.sha256(
                (source_run / "trajectory.csv").read_bytes()
            ).hexdigest(),
            "reward_config": reward_config.to_dict(),
        }
    )
    run_dir = create_run_directory(output_dir)
    with (run_dir / "reward.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REWARD_COLUMNS)
        writer.writeheader()
        writer.writerows(reward_rows)
    write_json_atomic(run_dir / "report.json", reward_report)
    return run_dir, reward_report


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    supplied = list(arguments) if arguments is not None else sys.argv[1:]
    command = shlex.join(
        ["uv", "run", "--locked", "python", "scripts/audit_task_rewards.py", *supplied]
    )
    started = time.perf_counter()
    reward_config = load_reward_config(args.config)
    run_dir, report = run_audit(args.source_run, reward_config, args.output_dir, command)
    report["duration_seconds"] = time.perf_counter() - started
    write_json_atomic(run_dir / "report.json", report)
    with artifact_logger(
        run_dir,
        "INFO",
        log_filename="reward.log",
        namespace="align.task_reward",
    ) as logger:
        logger.info(
            "Task reward audit %s: %s",
            report["status"],
            run_dir,
            extra={
                "event": "reward_audit_completed",
                "check_name": "task_reward",
                "check_status": report["status"],
            },
        )
    return 0 if report["status"] == "passed" else 1


def _xyz(row: dict[str, object], prefix: str) -> tuple[float, float, float]:
    keys = ("x", "y", "z") if not prefix else tuple(f"{prefix}{axis}" for axis in "xyz")
    return tuple(float(row[key]) for key in keys)
