"""Independently compare one baseline and one wake-guard frozen-policy replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.runtime.policy_telemetry_runtime import valid_early_contact_capture
from align.tasks.policy_guard_comparison import (
    summarize_evaluation_rows,
    summarize_guard_applicability,
    validate_pair,
)
from align.tasks.policy_telemetry import audit_airborne_contacts, audit_telemetry


def _load_capture(path: Path) -> dict:
    root = project_root()
    if path.parent != (root / "runs/policy-telemetry").resolve():
        raise ValueError("replay must be a direct policy-telemetry run")
    report = json.loads((path / "report.json").read_text())
    early_capture = (
        report.get("status") == "failed"
        and valid_early_contact_capture(
            report.get("exit_code"), report.get("probe"), report.get("metrics")
        )
        and report.get("error")
        == "RuntimeError: simulator evaluation failed; inspect evaluation/console.log"
    )
    if report.get("status") != "passed" and not early_capture:
        raise ValueError("replay is neither passed nor an intact early contact capture")
    evaluation = path / "evaluation"
    if report.get("probe") != json.loads(
        (evaluation / "probe-result.json").read_text()
    ) or report.get("metrics") != json.loads((evaluation / "metrics.json").read_text()):
        raise ValueError("host and simulator reports differ")
    return report


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--guarded-run", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline_path = args.baseline_run.resolve()
    guarded_path = args.guarded_run.resolve()
    if baseline_path == guarded_path:
        raise ValueError("comparison arms must be different immutable runs")
    baseline = _load_capture(baseline_path)
    guarded = _load_capture(guarded_path)
    validate_pair(baseline, guarded)
    source_config = (
        Path(baseline["source_run"]) / f"seed-{baseline['policy_seed']:010d}" / "config.json"
    )
    if digest(source_config) != baseline["source_config_sha256"]:
        raise ValueError("training configuration has changed")
    config = json.loads(source_config.read_text())
    run = create_run_directory(project_root() / "runs/policy-guard-comparison")
    started = time.perf_counter()
    report = new_report()
    report.update(
        baseline_run=str(baseline_path),
        guarded_run=str(guarded_path),
        source_run=baseline["source_run"],
        source_config_sha256=baseline["source_config_sha256"],
        checkpoint_id=baseline["checkpoint_id"],
        checkpoint_sha256=baseline["checkpoint_sha256"],
        image_id=baseline["image_id"],
        policy_seed=baseline["policy_seed"],
        training_performed=False,
        optimizer_updates=0,
    )
    try:
        arms = {}
        for name, path, source in (
            ("baseline", baseline_path, baseline),
            ("guarded", guarded_path, guarded),
        ):
            output = path / "evaluation"
            audit = audit_telemetry(
                output / "policy-telemetry.csv",
                output / "evaluation.csv",
                num_agents=config["construction"]["num_agents"],
                max_speed_m_s=config["construction"]["max_speed_m_s"],
                reward_config=config["reward"],
                wake_guard=name == "guarded",
            )
            if (
                source["metrics"].get("wake_guard_active_steps", 0)
                != audit["wake_guard_active_steps"]
                or source["metrics"].get("wake_guard_unresolved_steps", 0)
                != audit["wake_guard_unresolved_steps"]
            ):
                raise ValueError(f"{name} simulator and raw intervention counts differ")
            contacts = audit_airborne_contacts(
                output / "policy-telemetry.csv",
                output / "evaluation.csv",
                airborne_height_m=config["task"]["airborne_height_m"],
                contact_force_threshold_n=config["task"]["contact_force_threshold_n"],
            )
            summary = summarize_evaluation_rows(output / "evaluation.csv")
            if (
                source["metrics"]["raw_rows"] != summary["environment_rows"]
                or source["metrics"]["phase_rows"] != summary["phase_rows"]
                or source["metrics"]["outcome_counts"] != summary["outcome_counts"]
            ):
                raise ValueError(f"{name} simulator and raw evaluation counts differ")
            arms[name] = {
                "source_status": source["status"],
                "capture_mode": source.get("capture_mode", "early_safety_termination"),
                "telemetry_sha256": digest(output / "policy-telemetry.csv"),
                "evaluation_sha256": digest(output / "evaluation.csv"),
                "telemetry_audit": audit,
                "contact_count": contacts["first_post_airborne_contact_count"],
                "first_post_airborne_contacts": contacts["first_post_airborne_contacts"],
                "evaluation": summary,
            }
        report["arms"] = arms
        report["guard_applicability_on_saved_trace"] = summarize_guard_applicability(
            guarded_path / "evaluation/policy-telemetry.csv"
        )
        if (
            report["guard_applicability_on_saved_trace"]["takeoff"]["candidate_active_steps"]
            != arms["guarded"]["telemetry_audit"]["wake_guard_active_steps"]
        ):
            raise ValueError("guarded takeoff applicability disagrees with applied interventions")
        report["difference_guarded_minus_baseline"] = {
            "post_airborne_contacts": arms["guarded"]["contact_count"]
            - arms["baseline"]["contact_count"],
            "formation_environment_rows": arms["guarded"]["evaluation"]["phase_rows"]["formation"]
            - arms["baseline"]["evaluation"]["phase_rows"]["formation"],
            "successes": arms["guarded"]["evaluation"]["outcome_counts"]["1"]
            - arms["baseline"]["evaluation"]["outcome_counts"]["1"],
            "minimum_nonground_separation_m": arms["guarded"]["evaluation"][
                "minimum_nonground_separation_m"
            ]
            - arms["baseline"]["evaluation"]["minimum_nonground_separation_m"],
        }
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
