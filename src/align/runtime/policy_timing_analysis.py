"""Host-only shared-prefix and switch-state audit of a completed timing comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.tasks.evaluation_timing import compare_shared_prefix, formation_switch_altitude


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit a matched policy timing run on CPU")
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args(argv)
    root = project_root()
    source = args.source_run.resolve()
    if source.parent != (root / "runs/policy-timing").resolve():
        raise ValueError("source-run must be a direct child of runs/policy-timing")
    source_report = json.loads((source / "report.json").read_text())
    if source_report["status"] != "passed":
        raise ValueError("source timing run must have passed")
    baseline = source / "baseline/policy-telemetry.csv"
    extended = source / "extended/policy-telemetry.csv"
    run = create_run_directory(root / "runs/policy-timing-analysis")
    started = time.perf_counter()
    report = new_report()
    report.update(
        source_run=str(source),
        source_run_id=source.name,
        baseline_telemetry_sha256=digest(baseline),
        extended_telemetry_sha256=digest(extended),
        checkpoint_sha256=source_report["checkpoint_sha256"],
    )
    try:
        config = json.loads(
            (
                Path(source_report["source_run"])
                / f"seed-{source_report['policy_seed']:010d}/config.json"
            ).read_text()
        )
        details = source_report["timing_details"]
        report["shared_prefix"] = compare_shared_prefix(
            baseline,
            extended,
            divergence_step=details["source_ground_steps"] + details["source_takeoff_steps"],
            num_envs=config["task"]["num_envs"],
            num_agents=config["construction"]["num_agents"],
        )
        report["formation_switch"] = {
            "baseline": formation_switch_altitude(baseline),
            "extended": formation_switch_altitude(extended),
        }
        report["comparison"] = source_report["comparison"]
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
