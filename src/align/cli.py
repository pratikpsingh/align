"""Command-line entrypoints; diagnostic work belongs to the runtime module."""

import argparse
import logging
import sys
import time
from importlib import metadata
from pathlib import Path

from align.artifacts import (
    as_ist,
    create_run_directory,
    diagnostic_logger,
    utc_now,
    write_json_atomic,
)
from align.config import DoctorConfig
from align.runtime.diagnostics import collect_checks


def run_doctor(config: DoctorConfig) -> int:
    started = time.perf_counter()
    run_dir = create_run_directory(config.output_dir)
    report_path = run_dir / "report.json"
    report = {
        "schema_version": 1,
        "command": "doctor",
        "run_id": run_dir.name,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "duration_seconds": None,
        "status": "running",
        "exit_code": None,
        "configuration": config.to_dict(),
        "simulation_validation": "not_performed",
        "checks": [],
    }
    report["started_at_ist"] = as_ist(report["started_at_utc"])
    report["timezone"] = "Asia/Kolkata"
    write_json_atomic(report_path, report)
    with diagnostic_logger(run_dir, config.log_level) as logger:
        logger.info("Diagnostic started: %s", run_dir.name, extra={"event": "started"})
        exit_code = 0
        try:
            for check in collect_checks(config, run_dir):
                report["checks"].append(check.to_dict())
                level = {"warning": logging.WARNING, "error": logging.ERROR}.get(
                    check.status, logging.INFO
                )
                logger.log(
                    level,
                    "%s: %s",
                    check.name,
                    check.message,
                    extra={
                        "event": "check_completed",
                        "check_name": check.name,
                        "check_status": check.status,
                    },
                )
                # This command has few probes; retain each completed result on interruption.
                write_json_atomic(report_path, report)
            exit_code = int(any(check["status"] == "error" for check in report["checks"]))
            report["status"] = "requirements_unmet" if exit_code else "completed"
        except KeyboardInterrupt:
            exit_code = 130
            report["status"] = "interrupted"
            logger.warning("Diagnostic interrupted.", extra={"event": "interrupted"})
        except Exception as exc:
            # CLI boundary: expose unexpected errors and retain the completed probes.
            exit_code = 1
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Diagnostic failed.", extra={"event": "failed"})
        report["finished_at_utc"] = utc_now()
        report["finished_at_ist"] = as_ist(report["finished_at_utc"])
        report["duration_seconds"] = time.perf_counter() - started
        report["exit_code"] = exit_code
        write_json_atomic(report_path, report)
        logger.info("Diagnostic status: %s", report["status"], extra={"event": "finished"})
    print(f"Report: {report_path}")
    print(f"Logs:   {run_dir / 'doctor.log'} and {run_dir / 'events.jsonl'}")
    print("Simulation: not tested. Installed packages or a visible GPU do not prove readiness.")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="align", description="ALiGn research project tools.")
    parser.add_argument("--version", action="version", version=f"align {metadata.version('align')}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="Inspect this machine and save a diagnostic report."
    )
    doctor.add_argument(
        "--output-dir", type=Path, default=Path("runs/diagnostics"), help="Report parent directory."
    )
    doctor.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout per external command, in seconds (0–30).",
    )
    doctor.add_argument(
        "--require-nvidia",
        action="store_true",
        help="Exit 1 unless the NVIDIA query finds a device; does not test CUDA or simulation.",
    )
    doctor.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args(argv)
    try:
        config = DoctorConfig(
            output_dir=args.output_dir.expanduser().resolve(),
            timeout_seconds=args.timeout,
            require_nvidia=args.require_nvidia,
            log_level=args.log_level,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return run_doctor(config)
    except OSError as exc:
        print(f"align doctor: could not read/write diagnostic artifacts: {exc}", file=sys.stderr)
        return 1
