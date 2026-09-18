"""Create a separate immutable CPU audit of one physical shape-transition run."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.tasks.downwash_audit import audit_downwash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args(argv)
    root = project_root()
    source = args.source_run.resolve()
    if source.parent != (root / "runs/shape-transition-reference").resolve():
        raise ValueError("source run must be in runs/shape-transition-reference")
    source_report_path = source / "report.json"
    source_report = json.loads(source_report_path.read_text())
    if (
        source_report.get("status") != "passed"
        or source_report.get("audit", {}).get("status") != "passed"
    ):
        raise ValueError("source physical telemetry audit did not pass")
    telemetry = source / "evaluation/policy-telemetry.csv"
    episode_lengths = {
        int(item["env_id"]): int(item["first_episode_steps"])
        for item in source_report["audit"]["episodes"]
    }
    run = create_run_directory(root / "runs/shape-downwash-audit")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        source_run_id=source.name,
        source_report_sha256=digest(source_report_path),
        source_telemetry_sha256=digest(telemetry),
        source=probe_source(30).details,
        system=probe_system().details,
    )
    try:
        audit, rows = audit_downwash(telemetry, episode_lengths)
        with (run / "downwash-proxy.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report.update(status="passed", audit=audit)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
