"""Re-audit an immutable matched takeoff run without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.runtime.takeoff_replay_runtime import validate_baseline_report
from align.tasks.takeoff_replay import ARMS, assess_replays, assess_wake_guard, save_replay_plot


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args(argv)
    root = project_root()
    source = args.source_run.resolve()
    if source.parent != (root / "runs/takeoff-replay").resolve():
        raise ValueError("source must be a direct takeoff-replay run")
    original = json.loads((source / "report.json").read_text())
    if not original.get("include_wake_guard") or not original.get("baseline_run"):
        raise ValueError("source did not include a separate wake guard arm")
    arm = original.get("arms", {}).get("wake_guard", {})
    if not (
        arm.get("exit_code") == 0
        and arm.get("probe", {}).get("status") == "passed"
        and arm.get("probe", {}).get("takeoff_replay_tested") is True
        and arm.get("metrics", {}).get("status") == "passed"
    ):
        raise ValueError("source simulator arm is incomplete")
    if arm["probe"] != json.loads((source / "wake_guard/probe-result.json").read_text()) or arm[
        "metrics"
    ] != json.loads((source / "wake_guard/metrics.json").read_text()):
        raise ValueError("source report and simulator artifacts differ")
    if (
        digest(source / "commands.csv") != original["command_trace_sha256"]
        or digest(source / "config.json") != original["config_sha256"]
    ):
        raise ValueError("source trace or config changed")
    baseline = Path(original["baseline_run"]).resolve()
    if baseline.parent != (root / "runs/takeoff-replay").resolve():
        raise ValueError("baseline path is outside takeoff-replay runs")
    previous = json.loads((baseline / "report.json").read_text())
    validate_baseline_report(
        previous,
        source_run_id=original["source_run_id"],
        source_telemetry_sha256=original["source_telemetry_sha256"],
        config_sha256=original["config_sha256"],
        checkpoint_id=original["source_checkpoint_id"],
        trace_sha256=original["command_trace_sha256"],
    )
    if original["source_run_id"] != previous["source_run_id"]:
        raise ValueError("source and baseline replay identities differ")
    configuration = json.loads((source / "config.json").read_text())
    run = create_run_directory(root / "runs/takeoff-replay-analysis")
    started = time.perf_counter()
    report = new_report()
    report.update(
        source_run=str(source),
        source_run_id=source.name,
        source_report_sha256=digest(source / "report.json"),
        source_status=original["status"],
        source_error=original.get("error"),
        baseline_run=str(baseline),
        baseline_run_id=baseline.name,
        baseline_report_sha256=digest(baseline / "report.json"),
        image_id=original["image_id"],
        command_trace_sha256=original["command_trace_sha256"],
        wake_trajectory_sha256=digest(source / "wake_guard/trajectory.csv"),
    )
    try:
        report["comparison"] = assess_replays(
            source / "commands.csv",
            {name: baseline / name / "trajectory.csv" for name in ARMS},
            airborne_height_m=configuration["task"]["airborne_height_m"],
            contact_threshold_n=configuration["task"]["contact_force_threshold_n"],
        )
        report["wake_guard"] = assess_wake_guard(
            source / "commands.csv",
            source / "wake_guard/trajectory.csv",
            airborne_height_m=configuration["task"]["airborne_height_m"],
            contact_threshold_n=configuration["task"]["contact_force_threshold_n"],
            minimum_separation_m=configuration["construction"]["minimum_separation_m"],
        )
        if (
            report["wake_guard"]["guard_active_steps"] != arm["metrics"]["guard_active_steps"]
            or report["wake_guard"]["guard_unresolved_steps"]
            != arm["metrics"]["guard_unresolved_steps"]
        ):
            raise ValueError("simulator and host guard step counts differ")
        save_replay_plot(
            source / "commands.csv",
            {
                **{name: baseline / name / "trajectory.csv" for name in ARMS},
                "wake_guard": source / "wake_guard/trajectory.csv",
            },
            run / "altitude-and-force.svg",
        )
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
