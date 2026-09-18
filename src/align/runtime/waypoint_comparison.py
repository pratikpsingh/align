"""Compare two audited reference flights with identical contracts except route leg length."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import finish, new_report, project_root


def compare_waypoint_reports(waypoint: dict, direct: dict) -> dict:
    """Require matched provenance and report paired first-episode flight metrics."""
    for name, report in (("waypoint", waypoint), ("direct", direct)):
        if report.get("status") != "passed" or report.get("audit", {}).get("status") != "passed":
            raise ValueError(f"{name} run did not pass its raw audit")
        if report.get("training_performed") or report.get("checkpoint_loaded"):
            raise ValueError(f"{name} arm is not a deterministic reference flight")
    if waypoint["image_id"] != direct["image_id"]:
        raise ValueError("comparison arms use different simulator images")
    if waypoint["config_sha256"] != direct["config_sha256"]:
        raise ValueError("resolved task/control bundles differ")
    if waypoint["host_gpu_index"] != direct["host_gpu_index"]:
        raise ValueError("host GPU allocation differs")
    for key in ("construction", "task", "observation", "reward"):
        if waypoint["input_sha256"][key] != direct["input_sha256"][key]:
            raise ValueError(f"{key} configuration differs")
    left_config = waypoint["route_config"]
    right_config = direct["route_config"]
    if set(left_config) != set(right_config):
        raise ValueError("route configuration schemas differ")
    changed = {key for key in left_config if left_config[key] != right_config[key]}
    if changed != {"maximum_leg_length_m"}:
        raise ValueError("arms differ beyond maximum_leg_length_m")
    left = waypoint["plan"]
    right = direct["plan"]
    if (
        left["source_center_m"] != right["source_center_m"]
        or left["goal_center_m"] != right["goal_center_m"]
    ):
        raise ValueError("route endpoints differ")
    if left["path_length_m"] != right["path_length_m"]:
        raise ValueError("route path lengths differ")
    left_route = waypoint["audit"]["leg_count"]
    right_route = direct["audit"]["leg_count"]
    if not left_route > right_route == 1:
        raise ValueError("expected multi-leg waypoint and one-leg direct arms")
    way_rows = {row["env_id"]: row for row in waypoint["audit"]["episodes"]}
    direct_rows = {row["env_id"]: row for row in direct["audit"]["episodes"]}
    if not way_rows or set(way_rows) != set(direct_rows):
        raise ValueError("paired world IDs differ")
    rows = []
    for env_id in sorted(way_rows):
        item = {"env_id": env_id}
        for label, source in (("waypoint", way_rows[env_id]), ("direct", direct_rows[env_id])):
            for key in (
                "route_complete",
                "success",
                "terminal_reason_code",
                "time_from_command_to_outcome_s",
                "minimum_post_command_separation_m",
                "post_command_pairwise_rmse_mean_m",
                "post_command_assigned_rmse_mean_m",
            ):
                item[f"{label}_{key}"] = source[key]
        item["waypoint_minus_direct_outcome_time_s"] = (
            item["waypoint_time_from_command_to_outcome_s"]
            - item["direct_time_from_command_to_outcome_s"]
        )
        if not all(
            math.isfinite(value)
            for key, value in item.items()
            if key.endswith(("_m", "_s")) and isinstance(value, (int, float))
        ):
            raise ValueError("nonfinite paired route metric")
        rows.append(item)
    return {
        "status": "passed",
        "comparison": "matched deterministic 3 m waypoint versus direct reference",
        "image_id": waypoint["image_id"],
        "task_bundle_sha256": waypoint["config_sha256"],
        "path_length_m": left["path_length_m"],
        "waypoint_legs": left_route,
        "direct_legs": right_route,
        "world_count": len(rows),
        "paired_rows": rows,
        "independent_random_seeds": False,
        "learned_policy_evaluated": False,
        "superiority_established": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoint-run", required=True, type=Path)
    parser.add_argument("--direct-run", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.waypoint_run.resolve() == args.direct_run.resolve():
        parser.error("Comparison arms must be distinct runs")
    root = project_root()
    for path in (args.waypoint_run, args.direct_run):
        if path.resolve().parent != (root / "runs/waypoint-reference").resolve():
            parser.error("Both arms must be immutable runs/waypoint-reference directories")
    run = create_run_directory(root / "runs/waypoint-comparison")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        waypoint_source=str(args.waypoint_run.resolve()),
        direct_source=str(args.direct_run.resolve()),
        source=probe_source(30).details,
        system=probe_system().details,
    )
    try:
        comparison = compare_waypoint_reports(
            json.loads((args.waypoint_run / "report.json").read_text()),
            json.loads((args.direct_run / "report.json").read_text()),
        )
        report.update(comparison)
        with (run / "paired-worlds.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison["paired_rows"][0]))
            writer.writeheader()
            writer.writerows(comparison["paired_rows"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
