"""Host-only report for a fixed-ID, geometry-checked shape command."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.formations import (
    ShapeTransitionConfig,
    assign_transition_slots,
    generate_template,
    linear_path_clearance,
    place_template,
)
from align.formations.downwash import sample_nominal_downwash
from align.formations.geometry import assign_agents_to_slots
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.simulation.multi_drone_contract import MultiDroneConfig, build_group_layout
from align.tasks.environment import TaskEnvironmentConfig


def make_transition_report(
    construction: MultiDroneConfig,
    task: TaskEnvironmentConfig,
    command: ShapeTransitionConfig,
) -> dict:
    """Plan a four-shape command without making a flight-safety claim."""
    if construction.formation_kind != command.source_kind:
        raise ValueError("construction formation kind differs from transition source")
    if construction.num_agents > command.maximum_search_agents:
        raise ValueError("formation exceeds transition search limit")
    formation_start = construction.ground_steps + construction.takeoff_steps
    latest_dwell_start = task.max_episode_steps - task.success_dwell_steps + 1
    if not formation_start < command.command_step < latest_dwell_start:
        raise ValueError("shape command must leave a formation phase and final dwell window")
    source = build_group_layout(construction).assigned_target_positions_m
    destination = place_template(
        generate_template(
            command.destination_kind, construction.num_agents, construction.target_spacing_m
        ),
        command.destination_center_m or construction.target_center_m,
    )
    center = command.destination_center_m or construction.target_center_m
    if command.destination_horizontal_scale != 1.0:
        destination = tuple(
            (
                center[0] + (point[0] - center[0]) * command.destination_horizontal_scale,
                center[1] + (point[1] - center[1]) * command.destination_horizontal_scale,
                point[2],
            )
            for point in destination
        )
    for point in destination:
        if (
            abs(point[0]) >= task.safety_xy_limit_m
            or abs(point[1]) >= task.safety_xy_limit_m
            or not task.crash_height_m < point[2] < task.safety_z_limit_m
        ):
            raise ValueError("destination template is outside the task safety envelope")
    assignment = assign_transition_slots(
        source,
        destination,
        minimum_planned_separation_m=command.minimum_planned_separation_m,
        maximum_search_agents=command.maximum_search_agents,
    )
    ground = build_group_layout(construction).ground_positions_m
    ground_assignment = assign_agents_to_slots(ground, destination)
    direct = linear_path_clearance(
        source, tuple(destination[slot] for slot in ground_assignment.slot_for_agent)
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "source_kind": command.source_kind,
        "destination_kind": command.destination_kind,
        "source_center_m": construction.target_center_m,
        "destination_center_m": center,
        "destination_horizontal_scale": command.destination_horizontal_scale,
        "command_step": command.command_step,
        "command_time_s": command.command_step * construction.physics_dt,
        "control_dt_seconds": construction.physics_dt,
        "formation_start_step": formation_start,
        "latest_final_dwell_start_step": latest_dwell_start,
        "agent_ids": list(range(construction.num_agents)),
        "source_positions_m": source,
        "destination_slots_m": destination,
        "assignment": assignment.to_dict(),
        "nominal_downwash_proxy": sample_nominal_downwash(
            source, assignment.assigned_destination_m
        ),
        "naive_ground_assigned_minimum_interpolated_separation_m": direct[0],
        "speed_limited_transition_lower_bound_s": (
            assignment.maximum_agent_travel_m / construction.max_speed_m_s
        ),
        "post_command_time_available_s": (task.max_episode_steps - command.command_step)
        * construction.physics_dt,
        "synchronous_linear_path_only": True,
        "physical_transition_tested": False,
        "learned_policy_transition_tested": False,
    }
    if not math.isfinite(assignment.minimum_interpolated_separation_m):
        raise ValueError("planned separation is not finite")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-config", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--transition-config", type=Path, required=True)
    args = parser.parse_args(argv)
    root = project_root()
    inputs = (args.construction_config, args.task_config, args.transition_config)
    construction = MultiDroneConfig.from_dict(json.loads(inputs[0].read_text()))
    task = TaskEnvironmentConfig.from_dict(json.loads(inputs[1].read_text()))
    command = ShapeTransitionConfig.from_dict(json.loads(inputs[2].read_text()))
    started = time.perf_counter()
    run = create_run_directory(root / "runs/shape-transition-plan")
    report = new_report()
    report.update(
        run_id=run.name,
        input_sha256={path.name: digest(path) for path in inputs},
        construction_config=construction.to_dict(),
        task_config=task.to_dict(),
        transition_config=command.to_dict(),
        source=probe_source(30).details,
        system=probe_system().details,
    )
    try:
        report["plan"] = make_transition_report(construction, task, command)
        source = report["plan"]["source_positions_m"]
        destination = report["plan"]["assignment"]["assigned_destination_m"]
        with (run / "planned-positions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("alpha", "agent_id", "x_m", "y_m", "z_m"))
            for step in range(101):
                alpha = step / 100
                for agent_id, (before, after) in enumerate(zip(source, destination, strict=True)):
                    writer.writerow(
                        (
                            alpha,
                            agent_id,
                            *(a + alpha * (b - a) for a, b in zip(before, after, strict=True)),
                        )
                    )
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
