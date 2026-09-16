"""Audit bounded local observations on a saved trajectory without Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from align.artifacts import (
    artifact_logger,
    as_ist,
    create_run_directory,
    utc_now,
    write_json_atomic,
)
from align.runtime.diagnostics import probe_source
from align.simulation.multi_drone_report import assess_saved_run
from align.tasks.observation import ObservationConfig, build_observations

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "local-observation-baseline.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "local-observation"


def load_observation_config(path: Path) -> ObservationConfig:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("observation configuration root must be an object")
    return ObservationConfig.from_dict(value)


def audit_rows(
    rows: Sequence[dict[str, object]], config: ObservationConfig
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Reconstruct local actor inputs and one critic state for each task step."""
    by_repeat_step: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["repeat"]), int(row["step"]))
        by_repeat_step.setdefault(key, []).append(row)
    if not by_repeat_step:
        raise ValueError("trajectory has no samples")

    output_rows = []
    finite = True
    bounded = True
    fixed_actor_dimension = True
    fixed_critic_dimension = True
    hard_radius = True
    budget = True
    valid_padding = True
    deterministic = True
    actor_saturations = 0
    critic_saturations = 0
    candidate_edges = 0
    selected_edges = 0
    budget_limited_agent_steps = 0
    counts = Counter()
    repeat_steps = Counter()

    for key in sorted(by_repeat_step):
        samples = sorted(by_repeat_step[key], key=lambda row: int(row["agent_id"]))
        agent_ids = tuple(int(row["agent_id"]) for row in samples)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("each step must contain unique agent identities")
        phases = {str(row["phase"]) for row in samples}
        if len(phases) != 1:
            raise ValueError("agents disagree on task phase")
        batch = build_observations(
            agent_ids,
            tuple(_xyz(row, "") for row in samples),
            tuple(_xyz(row, "v") for row in samples),
            tuple(_xyz(row, "target_") for row in samples),
            config,
        )
        deterministic &= batch == build_observations(
            agent_ids,
            tuple(_xyz(row, "") for row in samples),
            tuple(_xyz(row, "v") for row in samples),
            tuple(_xyz(row, "target_") for row in samples),
            config,
        )
        critic_values = batch.critic.flat()
        finite &= all(math.isfinite(value) for value in critic_values)
        bounded &= all(-1.0 <= value <= 1.0 for value in critic_values)
        fixed_critic_dimension &= len(critic_values) == config.critic_dimension
        critic_saturations += batch.critic.saturation_count
        repeat_steps[key[0]] += 1

        for sample, actor in zip(samples, batch.actors, strict=True):
            values = actor.flat()
            selected_count = sum(actor.neighbor_mask)
            finite &= all(math.isfinite(value) for value in values)
            bounded &= all(-1.0 <= value <= 1.0 for value in values)
            fixed_actor_dimension &= len(values) == config.actor_dimension
            hard_radius &= all(
                distance is None or distance <= config.neighbor_radius_m
                for distance in actor.neighbor_distances_m
            )
            budget &= selected_count <= config.max_neighbors
            for slot, mask, neighbor_id, distance in zip(
                actor.neighbor_features,
                actor.neighbor_mask,
                actor.neighbor_ids,
                actor.neighbor_distances_m,
                strict=True,
            ):
                valid_padding &= (mask and neighbor_id is not None and distance is not None) or (
                    not mask and neighbor_id is None and distance is None and slot == (0.0,) * 6
                )
            counts[selected_count] += 1
            candidate_edges += actor.candidate_count
            selected_edges += selected_count
            budget_limited_agent_steps += actor.candidate_count > config.max_neighbors
            actor_saturations += actor.saturation_count
            output_rows.append(
                {
                    "repeat": key[0],
                    "step": key[1],
                    "t": float(sample["t"]),
                    "phase": str(sample["phase"]),
                    "agent_id": actor.agent_id,
                    "candidate_count": actor.candidate_count,
                    "selected_count": selected_count,
                    "neighbor_ids": "|".join(
                        "" if value is None else str(value) for value in actor.neighbor_ids
                    ),
                    "neighbor_distances_m": "|".join(
                        "" if value is None else format(value, ".17g")
                        for value in actor.neighbor_distances_m
                    ),
                    "saturation_count": actor.saturation_count,
                    **{f"obs_{index}": value for index, value in enumerate(values)},
                }
            )

    checks = {
        "finite": finite,
        "bounded": bounded,
        "fixed_actor_dimension": fixed_actor_dimension,
        "fixed_critic_dimension": fixed_critic_dimension,
        "hard_neighbor_radius": hard_radius,
        "neighbor_budget": budget,
        "mask_and_padding": valid_padding,
        "deterministic": deterministic,
    }
    total_agent_steps = len(output_rows)
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "contract": {
            "actor_frame": "shared gravity-aligned world axes; no absolute position",
            "self_features": "normalized world velocity xyz, relative assigned target xyz",
            "neighbor_features": "normalized relative position xyz and velocity xyz",
            "neighbor_rule": "Euclidean radius filter, then nearest-distance budget, then padding",
            "tie_break": "ascending agent identity for equal distance",
            "critic": "separate padded global position/velocity/target state for training only",
            "observed_edges": (
                "directed observation topology; not measured or modeled radio traffic"
            ),
        },
        "dimensions": {
            "actor": config.actor_dimension,
            "actor_self": config.self_feature_count,
            "neighbor_slots": config.max_neighbors,
            "neighbor_features_per_slot": config.neighbor_feature_count,
            "neighbor_masks": config.max_neighbors,
            "critic": config.critic_dimension,
        },
        "metrics": {
            "agent_steps": total_agent_steps,
            "environment_steps": sum(repeat_steps.values()),
            "repeat_environment_steps": {str(key): value for key, value in repeat_steps.items()},
            "neighbor_count_histogram": {
                str(key): counts[key] for key in range(config.max_neighbors + 1)
            },
            "mean_selected_neighbors": selected_edges / total_agent_steps,
            "maximum_selected_neighbors": max(counts),
            "in_radius_candidate_directed_edges": candidate_edges,
            "selected_directed_edges": selected_edges,
            "budget_limited_agent_steps": budget_limited_agent_steps,
            "actor_saturation_count": actor_saturations,
            "critic_saturation_count": critic_saturations,
        },
    }, output_rows


def run_audit(
    source_run: Path,
    config: ObservationConfig,
    output_dir: Path,
    command: str,
) -> tuple[Path, dict[str, object]]:
    source_run = source_run.resolve()
    source_metrics, rows = assess_saved_run(source_run)
    report, output_rows = audit_rows(rows, config)
    report["checks"]["source_run_passed"] = source_metrics["status"] == "passed"
    report["status"] = "passed" if all(report["checks"].values()) else "failed"
    created_at_utc = utc_now()
    report.update(
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
            "implementation_source": probe_source(30).details,
            "observation_config": config.to_dict(),
        }
    )
    run_dir = create_run_directory(output_dir)
    fieldnames = [
        "repeat",
        "step",
        "t",
        "phase",
        "agent_id",
        "candidate_count",
        "selected_count",
        "neighbor_ids",
        "neighbor_distances_m",
        "saturation_count",
        *[f"obs_{index}" for index in range(config.actor_dimension)],
    ]
    with (run_dir / "observations.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    write_json_atomic(run_dir / "report.json", report)
    return run_dir, report


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
        ["uv", "run", "--locked", "python", "scripts/audit_local_observations.py", *supplied]
    )
    started = time.perf_counter()
    config = load_observation_config(args.config)
    run_dir, report = run_audit(args.source_run, config, args.output_dir, command)
    report["duration_seconds"] = time.perf_counter() - started
    write_json_atomic(run_dir / "report.json", report)
    with artifact_logger(
        run_dir,
        "INFO",
        log_filename="observation.log",
        namespace="align.local_observation",
    ) as logger:
        logger.info(
            "Local observation audit %s: %s",
            report["status"],
            run_dir,
            extra={
                "event": "observation_audit_completed",
                "check_name": "local_observation",
                "check_status": report["status"],
            },
        )
    return 0 if report["status"] == "passed" else 1


def _xyz(row: dict[str, object], prefix: str) -> tuple[float, float, float]:
    keys = ("x", "y", "z") if not prefix else tuple(f"{prefix}{axis}" for axis in "xyz")
    return tuple(float(row[key]) for key in keys)
