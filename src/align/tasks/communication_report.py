"""Sweep local neighbor budgets over an immutable saved construction trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
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
from align.tasks.communication import CommunicationConfig, account_snapshot
from align.tasks.observation import ObservationConfig, build_actor_observations
from align.tasks.observation_report import DEFAULT_CONFIG, load_observation_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMMUNICATION_CONFIG = (
    REPOSITORY_ROOT / "configs" / "communication-accounting-baseline.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "communication-accounting"


def sweep_rows(
    rows: Sequence[dict[str, object]],
    observation: ObservationConfig,
    communication: CommunicationConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Reconstruct radius-filtered topology; do not replay or score a new policy."""
    snapshots: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        snapshots[(int(row["repeat"]), int(row["step"]))].append(row)
    if not snapshots:
        raise ValueError("trajectory has no samples")
    output: list[dict[str, object]] = []
    for repeat, step in sorted(snapshots):
        samples = sorted(snapshots[(repeat, step)], key=lambda row: int(row["agent_id"]))
        agent_ids = tuple(int(row["agent_id"]) for row in samples)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("duplicate agent in trajectory snapshot")
        positions = tuple(_xyz(row, "") for row in samples)
        velocities = tuple(_xyz(row, "v") for row in samples)
        targets = tuple(_xyz(row, "target_") for row in samples)
        for budget in communication.budgets:
            actors = build_actor_observations(
                agent_ids,
                positions,
                velocities,
                targets,
                replace(observation, max_neighbors=budget),
            )
            cost = account_snapshot(actors, communication)
            output.append(
                {
                    "repeat": repeat,
                    "step": step,
                    "t": float(samples[0]["t"]),
                    "phase": str(samples[0]["phase"]),
                    "budget": budget,
                    "radius_m": observation.neighbor_radius_m,
                    "candidate_directed_edges": sum(actor.candidate_count for actor in actors),
                    **{key: value for key, value in cost.items() if key != "directed_edges"},
                    "directed_edges": "|".join(
                        f"{sender}>{receiver}" for sender, receiver in cost["directed_edges"]
                    ),
                }
            )
    summaries = []
    for budget in communication.budgets:
        subset = [row for row in output if row["budget"] == budget]
        summaries.append(
            {
                "budget": budget,
                "snapshots": len(subset),
                "agent_steps": sum(
                    len(snapshots[(int(row["repeat"]), int(row["step"]))]) for row in subset
                ),
                "candidate_directed_edges": sum(
                    int(row["candidate_directed_edges"]) for row in subset
                ),
                "selected_directed_edges": sum(
                    int(row["selected_directed_edges"]) for row in subset
                ),
                "unicast_proxy_bytes": sum(int(row["unicast_proxy_bytes"]) for row in subset),
                "ideal_broadcast_proxy_bytes": sum(
                    int(row["ideal_broadcast_proxy_bytes"]) for row in subset
                ),
            }
        )
    return output, summaries


def run_audit(
    source_run: Path,
    observation: ObservationConfig,
    communication: CommunicationConfig,
    output_dir: Path,
    command: str,
) -> tuple[Path, dict[str, object]]:
    source_run = source_run.resolve()
    source_report, rows = assess_saved_run(source_run)
    topology_rows, summaries = sweep_rows(rows, observation, communication)
    source_hash = hashlib.sha256((source_run / "trajectory.csv").read_bytes()).hexdigest()
    created_at_utc = utc_now()
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if source_report["status"] == "passed" else "failed",
        "created_at_utc": created_at_utc,
        "created_at_ist": as_ist(created_at_utc),
        "command": command,
        "source_run": str(source_run),
        "source_run_id": source_run.name,
        "source_trajectory_sha256": source_hash,
        "implementation_source": probe_source(30).details,
        "observation_config": observation.to_dict(),
        "communication_config": communication.to_dict(),
        "packet_bytes": communication.packet_bytes,
        "cost_label": "counterfactual packet-byte proxy, not measured radio traffic",
        "assumptions": [
            "One position/velocity packet per selected directed edge "
            "per saved snapshot for unicast.",
            "Ideal broadcast sends one packet per selected sender per snapshot.",
            "No discovery, retry, acknowledgement, contention, routing, "
            "or PHY overhead beyond configured packet bytes.",
            "The saved flight trajectory is held fixed; changing the budget here "
            "cannot establish a performance effect.",
        ],
        "budgets": summaries,
    }
    run_dir = create_run_directory(output_dir)
    with (run_dir / "topology.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(topology_rows[0]))
        writer.writeheader()
        writer.writerows(topology_rows)
    with (run_dir / "budget-summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json_atomic(run_dir / "report.json", report)
    return run_dir, report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--observation-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--communication-config", type=Path, default=DEFAULT_COMMUNICATION_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(arguments)
    supplied = list(arguments) if arguments is not None else sys.argv[1:]
    command = shlex.join(
        ["uv", "run", "--locked", "python", "scripts/audit_communication.py", *supplied]
    )
    observation = load_observation_config(args.observation_config)
    communication = CommunicationConfig.from_dict(
        json.loads(args.communication_config.read_text(encoding="utf-8"))
    )
    started = time.perf_counter()
    run_dir, report = run_audit(
        args.source_run, observation, communication, args.output_dir, command
    )
    report["duration_seconds"] = time.perf_counter() - started
    write_json_atomic(run_dir / "report.json", report)
    with artifact_logger(
        run_dir, "INFO", log_filename="communication.log", namespace="align.communication"
    ) as logger:
        logger.info("Communication audit %s: %s", report["status"], run_dir)
    return 0 if report["status"] == "passed" else 1


def _xyz(row: dict[str, object], prefix: str) -> tuple[float, float, float]:
    keys = ("x", "y", "z") if not prefix else tuple(f"{prefix}{axis}" for axis in "xyz")
    return tuple(float(row[key]) for key in keys)
