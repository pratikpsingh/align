"""CPU-only paired summary of frozen-actor evaluation rows."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from align.tasks.policy_telemetry import GUARDED_TELEMETRY_COLUMNS
from align.tasks.wake_guard import guard_focal_command

PAIR_IDENTITY_FIELDS = (
    "source_run_id",
    "source_image_id",
    "image_id",
    "policy_seed",
    "requested_completed_update",
    "checkpoint_id",
    "checkpoint_sha256",
    "source_config_sha256",
)


def validate_pair(baseline: dict, guarded: dict) -> None:
    """Require one checkpoint and simulator image; guard is the declared change."""
    for key in PAIR_IDENTITY_FIELDS:
        if not baseline.get(key) or baseline[key] != guarded.get(key):
            raise ValueError(f"paired replay identity mismatch: {key}")
    if baseline.get("wake_guard_enabled", False) is not False:
        raise ValueError("baseline replay unexpectedly enabled wake guard")
    if guarded.get("wake_guard_enabled") is not True:
        raise ValueError("treatment replay did not enable wake guard")
    for item in (baseline, guarded):
        if item.get("training_performed") is not False or item.get("optimizer_updates") != 0:
            raise ValueError("paired replay must use the frozen actor")


def summarize_evaluation_rows(path: Path) -> dict:
    """Read task outcomes and exposure directly from immutable evaluation CSV."""
    counts = {"ground": 0, "takeoff": 0, "formation": 0}
    outcomes = {str(code): 0 for code in range(1, 7)}
    formation_assigned_sum = 0.0
    formation_pairwise_sum = 0.0
    min_airborne_separation = math.inf
    first_formation_step: int | None = None
    rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["evaluation_step"])
            phase = row["phase"]
            if phase not in counts:
                raise ValueError("unknown evaluation phase")
            counts[phase] += 1
            rows += 1
            assigned = float(row["assigned_rmse_m"])
            pairwise = float(row["pairwise_rmse_m"])
            separation = float(row["minimum_separation_m"])
            if not all(math.isfinite(value) for value in (assigned, pairwise, separation)):
                raise ValueError("nonfinite evaluation measurement")
            if phase != "ground":
                min_airborne_separation = min(min_airborne_separation, separation)
            if phase == "formation":
                formation_assigned_sum += assigned
                formation_pairwise_sum += pairwise
                if first_formation_step is None:
                    first_formation_step = step
            reason = int(row["reason_code"])
            if reason:
                if str(reason) not in outcomes:
                    raise ValueError("unknown outcome code")
                outcomes[str(reason)] += 1
    if rows == 0:
        raise ValueError("empty evaluation CSV")
    formation = counts["formation"]
    return {
        "environment_rows": rows,
        "phase_rows": counts,
        "outcome_counts": outcomes,
        "formation_fraction": formation / rows,
        "first_formation_evaluation_step": first_formation_step,
        "formation_assigned_rmse_mean_m": formation_assigned_sum / formation if formation else None,
        "formation_pairwise_rmse_mean_m": formation_pairwise_sum / formation if formation else None,
        "minimum_nonground_separation_m": (
            min_airborne_separation if math.isfinite(min_airborne_separation) else None
        ),
    }


def summarize_guard_applicability(path: Path) -> dict:
    """Count when the same rule would trigger outside its takeoff-only contract.

    This is a one-step diagnostic on saved observations/requests, not a
    counterfactual trajectory or a claim that extending the rule is safe.
    """
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != GUARDED_TELEMETRY_COLUMNS:
            raise ValueError("guarded telemetry schema mismatch")
        for row in reader:
            groups[int(row["evaluation_step"]), int(row["env_id"])].append(row)
    if not groups:
        raise ValueError("empty guarded telemetry")
    by_phase = {
        phase: {"environment_steps": 0, "candidate_active_steps": 0, "unresolved_steps": 0}
        for phase in ("ground", "takeoff", "formation")
    }
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["agent_id"]))
        if len(rows) != 4 or [int(row["agent_id"]) for row in rows] != list(range(4)):
            raise ValueError(f"missing or duplicate drones at {key}")
        phase = rows[0]["phase"]
        if phase not in by_phase or any(row["phase"] != phase for row in rows):
            raise ValueError(f"invalid phase at {key}")
        positions = tuple(tuple(float(row[f"pre_{axis}_m"]) for axis in "xyz") for row in rows)
        requests = tuple(
            tuple(float(row[f"requested_v{axis}_m_s"]) for axis in "xyz") for row in rows
        )
        _, details = guard_focal_command(positions, requests, 2)
        summary = by_phase[phase]
        summary["environment_steps"] += 1
        summary["candidate_active_steps"] += int(details["active"])
        summary["unresolved_steps"] += int(details["unresolved"])
    return by_phase
