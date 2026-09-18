"""Save a CPU-only, fixed-identity group waypoint plan and target rows."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.waypoints import WaypointRouteConfig, make_waypoint_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-config", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    args = parser.parse_args(argv)
    root = project_root()
    inputs = (args.construction_config, args.task_config, args.route_config)
    construction = MultiDroneConfig.from_dict(json.loads(inputs[0].read_text()))
    task = TaskEnvironmentConfig.from_dict(json.loads(inputs[1].read_text()))
    route = WaypointRouteConfig.from_dict(json.loads(inputs[2].read_text()))
    run = create_run_directory(root / "runs/waypoint-plan")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        input_sha256={path.name: digest(path) for path in inputs},
        construction_config=construction.to_dict(),
        task_config=task.to_dict(),
        route_config=route.to_dict(),
        source=probe_source(30).details,
        system=probe_system().details,
    )
    try:
        plan = make_waypoint_plan(construction, task, route)
        with (run / "assigned-waypoints.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "waypoint_index",
                    "agent_id",
                    "center_x_m",
                    "center_y_m",
                    "center_z_m",
                    "target_x_m",
                    "target_y_m",
                    "target_z_m",
                )
            )
            for index, (center, targets) in enumerate(
                zip(plan.centers_m, plan.assigned_targets_m, strict=True)
            ):
                for agent_id, target in enumerate(targets):
                    writer.writerow((index, agent_id, *center, *target))
        report.update(status="passed", plan=plan.to_dict(), simulator_travel_tested=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
