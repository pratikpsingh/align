"""Reproducible host-side report for formation geometry contracts."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shlex
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

from align.artifacts import (
    artifact_logger,
    as_ist,
    create_run_directory,
    utc_now,
    write_json_atomic,
)
from align.formations.geometry import (
    SUPPORTED_KINDS,
    assign_agents_to_slots,
    evaluate_formation,
    generate_template,
    place_template,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "formations" / "eight-drone-suite.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "formation-geometry"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the small, explicit geometry-report configuration."""
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be an object")
    expected = {
        "schema_version",
        "num_agents",
        "minimum_spacing_m",
        "center_m",
        "yaw_rad",
        "kinds",
    }
    unknown = set(config) - expected
    missing = expected - set(config)
    if unknown or missing:
        raise ValueError(
            f"configuration keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if isinstance(config["schema_version"], bool) or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    num_agents = config["num_agents"]
    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents < 1:
        raise ValueError("num_agents must be a positive integer")
    minimum_spacing = config["minimum_spacing_m"]
    if (
        isinstance(minimum_spacing, bool)
        or not isinstance(minimum_spacing, (int, float))
        or not math.isfinite(minimum_spacing)
        or minimum_spacing <= 0.0
    ):
        raise ValueError("minimum_spacing_m must be a finite positive number")
    center = config["center_m"]
    if (
        not isinstance(center, list)
        or len(center) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in center
        )
    ):
        raise ValueError("center_m must contain three finite numbers")
    yaw = config["yaw_rad"]
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ValueError("yaw_rad must be a finite number")
    kinds = config["kinds"]
    if not isinstance(kinds, list) or not kinds or not all(isinstance(kind, str) for kind in kinds):
        raise ValueError("kinds must be a non-empty list of strings")
    if len(set(kinds)) != len(kinds):
        raise ValueError("kinds must not contain duplicates")
    unsupported = set(kinds) - set(SUPPORTED_KINDS)
    if unsupported:
        raise ValueError(f"unsupported formation kinds: {sorted(unsupported)}")
    return config


def build_report(config: dict[str, Any], *, config_path: Path, command: str) -> dict[str, Any]:
    """Build a JSON report and exercise one-to-one assignment with reversed slots."""
    formations = []
    for kind in config["kinds"]:
        template = generate_template(
            kind,
            config["num_agents"],
            config["minimum_spacing_m"],
        )
        targets = place_template(template, config["center_m"], config["yaw_rad"])
        reversed_agents = tuple(reversed(targets))
        assignment = assign_agents_to_slots(reversed_agents, targets)
        assigned_targets = tuple(targets[slot] for slot in assignment.slot_for_agent)
        metrics = evaluate_formation(reversed_agents, assigned_targets)
        centroid_m = tuple(
            math.fsum(point[axis] for point in template.points_m) / len(template.points_m)
            for axis in range(3)
        )
        measured_spacing_m = (
            min(math.dist(first, second) for first, second in combinations(template.points_m, 2))
            if len(template.points_m) > 1
            else 0.0
        )
        template_checks = {
            "count": len(template.points_m) == config["num_agents"],
            "centered": max(map(abs, centroid_m)) <= 1e-12,
            "minimum_spacing": len(template.points_m) == 1
            or math.isclose(
                measured_spacing_m,
                config["minimum_spacing_m"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
        }
        formations.append(
            {
                "kind": kind,
                "template": asdict(template),
                "template_measurements": {
                    "centroid_m": centroid_m,
                    "minimum_spacing_m": measured_spacing_m,
                },
                "template_checks": template_checks,
                "placed_targets_m": targets,
                "reversed_slot_assignment_check": {
                    "assignment": asdict(assignment),
                    "metrics": asdict(metrics),
                    "passed": assignment.total_squared_distance_m2 <= 1e-20,
                },
            }
        )

    created_at_utc = utc_now()
    passed = all(
        all(item["template_checks"].values()) and item["reversed_slot_assignment_check"]["passed"]
        for item in formations
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "created_at_utc": created_at_utc,
        "created_at_ist": as_ist(created_at_utc),
        "command": command,
        "config_path": str(config_path.resolve()),
        "config": config,
        "coordinate_contract": {
            "frame": "right-handed world frame",
            "units": "metres",
            "axes": {"+x": "horizontal", "+y": "horizontal", "+z": "up"},
            "yaw": "radians about +z, positive by right-hand rule",
        },
        "aggregation_contract": {
            "assigned": "mean squared Euclidean agent-to-assigned-target error",
            "pairwise": "mean squared all-pair distance distortion with fixed identities",
            "normalization": "divide each mean squared error by target diameter squared",
            "assignment": "centralized deterministic minimum-squared-distance Hungarian matching",
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "formations": formations,
        "passed": passed,
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    config = load_config(args.config)
    run_dir = create_run_directory(args.output_dir)
    supplied_arguments = arguments if arguments is not None else sys.argv[1:]
    command_parts = [
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/inspect_formations.py",
        *supplied_arguments,
    ]
    started = time.perf_counter()
    with artifact_logger(
        run_dir,
        "INFO",
        log_filename="formation.log",
        namespace="align.formations",
    ) as logger:
        report = build_report(
            config,
            config_path=args.config,
            command=shlex.join(command_parts),
        )
        report["duration_seconds"] = time.perf_counter() - started
        report["status"] = "passed" if report["passed"] else "failed"
        write_json_atomic(run_dir / "report.json", report)
        logger.info(
            "Formation geometry report %s: %s",
            report["status"],
            run_dir,
            extra={
                "event": "report_completed",
                "check_name": "formation_geometry",
                "check_status": report["status"],
            },
        )
    return 0 if report["passed"] else 1
