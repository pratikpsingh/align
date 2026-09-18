"""Host-only analysis of an immutable per-drone policy replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.tasks.policy_telemetry import audit_telemetry, summarize_telemetry


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a passed frozen-policy trace on CPU")
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args(argv)
    root = project_root()
    source = args.source_run.resolve()
    if source.parent != (root / "runs/policy-telemetry").resolve():
        raise ValueError("source-run must be a direct child of runs/policy-telemetry")
    source_report = json.loads((source / "report.json").read_text())
    if source_report["status"] != "passed":
        raise ValueError("source telemetry replay must have passed")
    source_training = Path(source_report["source_run"])
    seed = source_report["policy_seed"]
    config_path = source_training / f"seed-{seed:010d}/config.json"
    if digest(config_path) != source_report["source_config_sha256"]:
        raise ValueError("source configuration hash changed")
    config = json.loads(config_path.read_text())
    evaluation = source / "evaluation"
    run = create_run_directory(root / "runs/policy-telemetry-analysis")
    started = time.perf_counter()
    report = new_report()
    report.update(
        source_run=str(source),
        source_run_id=source.name,
        telemetry_sha256=digest(evaluation / "policy-telemetry.csv"),
        evaluation_sha256=digest(evaluation / "evaluation.csv"),
        checkpoint_id=source_report["checkpoint_id"],
        checkpoint_sha256=source_report["checkpoint_sha256"],
        source_config_sha256=digest(config_path),
    )
    try:
        report["audit"] = audit_telemetry(
            evaluation / "policy-telemetry.csv",
            evaluation / "evaluation.csv",
            num_agents=config["construction"]["num_agents"],
            max_speed_m_s=config["construction"]["max_speed_m_s"],
            reward_config=config["reward"],
        )
        report["analysis"] = summarize_telemetry(
            evaluation / "policy-telemetry.csv",
            max_speed_m_s=config["construction"]["max_speed_m_s"],
            control_dt_seconds=config["reward"]["control_dt_seconds"],
            max_episode_steps=config["task"]["max_episode_steps"],
            success_dwell_steps=config["task"]["success_dwell_steps"],
        )
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
