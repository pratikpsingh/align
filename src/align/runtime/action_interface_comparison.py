"""Audit a matched speed-coordinate policy-distribution experiment."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from align.artifacts import create_run_directory
from align.policies.action_interface import validate_positive_speed_treatment
from align.policies.config import RecurrentPolicyConfig
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import finish, new_report, project_root
from align.runtime.template_comparison import _load_run, audit_raw_runs

MEASURES = ("assigned_rmse_mean_m", "pairwise_rmse_mean_m", "minimum_separation_m")


def summarize_formation_csv(path: Path) -> dict:
    """Report phase-conditioned quality and exposure from a raw evaluation CSV.

    A controller that crashes at formation entry must not look good merely
    because its aggregate error contains mostly ground and takeoff samples.
    """
    total_rows = 0
    formation_rows = 0
    assigned_sum = 0.0
    pairwise_sum = 0.0
    minimum_separation = math.inf
    terminal_steps = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            total_rows += 1
            if row["phase"] == "formation":
                assigned = float(row["assigned_rmse_m"])
                pairwise = float(row["pairwise_rmse_m"])
                separation = float(row["minimum_separation_m"])
                if not all(math.isfinite(value) for value in (assigned, pairwise, separation)):
                    raise ValueError(f"{path} contains a nonfinite formation metric")
                formation_rows += 1
                assigned_sum += assigned
                pairwise_sum += pairwise
                minimum_separation = min(minimum_separation, separation)
            if int(row["reason_code"]):
                terminal_steps.append(int(row["episode_step"]))
    if total_rows == 0 or formation_rows == 0 or not terminal_steps:
        raise ValueError(f"{path} lacks evaluation or terminal formation evidence")
    return {
        "rows": total_rows,
        "formation_rows": formation_rows,
        "formation_assigned_rmse_mean_m": assigned_sum / formation_rows,
        "formation_pairwise_rmse_mean_m": pairwise_sum / formation_rows,
        "formation_minimum_separation_m": minimum_separation,
        "terminal_episode_steps": terminal_steps,
    }


def compare_action_interface_runs(baseline_run: Path, treatment_run: Path) -> dict:
    """Pair exact-checkpoint plane metrics after all one-variable checks."""
    baseline = _load_run(baseline_run, frozenset(("plane",)))
    treatment = _load_run(treatment_run, frozenset(("plane",)))
    if baseline["path"] == treatment["path"]:
        raise ValueError("comparison arms must be distinct runs")
    if baseline["curve"] != treatment["curve"]:
        raise ValueError("comparison arms differ in seeds, updates, or milestones")
    if baseline["report"]["image_id"] != treatment["report"]["image_id"]:
        raise ValueError("comparison arms use different simulator images")
    if (
        baseline["report"]["source"]["package_sha256"]
        != treatment["report"]["source"]["package_sha256"]
    ):
        raise ValueError("comparison arms use different host package sources")
    before = baseline["config"]
    after = treatment["config"]
    if set(before) != set(after) or {k: v for k, v in before.items() if k != "policy"} != {
        k: v for k, v in after.items() if k != "policy"
    }:
        raise ValueError("resolved task/training configurations differ outside policy")
    max_speed = before["construction"]["max_speed_m_s"]
    intervention = validate_positive_speed_treatment(
        RecurrentPolicyConfig.from_dict(before["policy"]),
        RecurrentPolicyConfig.from_dict(after["policy"]),
        max_speed_m_s=max_speed,
    )
    if baseline["training_exposure"] != treatment["training_exposure"]:
        raise ValueError("training template exposure differs")
    raw_counts = audit_raw_runs(baseline["path"], treatment["path"])
    rows = []
    for left_seed, right_seed in zip(
        baseline["report"]["seeds"], treatment["report"]["seeds"], strict=True
    ):
        if left_seed["policy_seed"] != right_seed["policy_seed"]:
            raise ValueError("paired policy seeds differ")
        for left_eval, right_eval in zip(
            left_seed["evaluations"], right_seed["evaluations"], strict=True
        ):
            if left_eval["completed_update"] != right_eval["completed_update"]:
                raise ValueError("checkpoint milestones differ")
            left = left_eval["metrics"]["template_measurements"]["plane"]
            right = right_eval["metrics"]["template_measurements"]["plane"]
            if left["rows"] != right["rows"]:
                raise ValueError("paired evaluation step budgets differ")
            seed = left_seed["policy_seed"]
            update = left_eval["completed_update"]
            label = f"seed-{seed:010d}/evaluation-update-{update:04d}/evaluation.csv"
            baseline_phase = summarize_formation_csv(baseline["path"] / label)
            treatment_phase = summarize_formation_csv(treatment["path"] / label)
            for reported, raw in ((left, baseline_phase), (right, treatment_phase)):
                if (reported["rows"], reported["formation_rows"]) != (
                    raw["rows"],
                    raw["formation_rows"],
                ):
                    raise ValueError("formation-phase raw counts disagree with the report")
            row = {
                "policy_seed": seed,
                "completed_update": update,
                "baseline_formation_rows": baseline_phase["formation_rows"],
                "treatment_formation_rows": treatment_phase["formation_rows"],
                "formation_phase_exposure_equal": (
                    baseline_phase["formation_rows"] == treatment_phase["formation_rows"]
                ),
                "baseline_successes": left["outcome_counts"]["1"],
                "treatment_successes": right["outcome_counts"]["1"],
                "baseline_timeouts": left["outcome_counts"]["6"],
                "treatment_timeouts": right["outcome_counts"]["6"],
                "baseline_safety_terminations": sum(
                    left["outcome_counts"][str(code)] for code in (2, 3, 4, 5)
                ),
                "treatment_safety_terminations": sum(
                    right["outcome_counts"][str(code)] for code in (2, 3, 4, 5)
                ),
            }
            for metric in MEASURES:
                row[f"baseline_all_phase_{metric}"] = left[metric]
                row[f"treatment_all_phase_{metric}"] = right[metric]
            for metric in (
                "formation_assigned_rmse_mean_m",
                "formation_pairwise_rmse_mean_m",
                "formation_minimum_separation_m",
            ):
                row[f"baseline_{metric}"] = baseline_phase[metric]
                row[f"treatment_{metric}"] = treatment_phase[metric]
            row["baseline_terminal_episode_steps"] = "|".join(
                str(value) for value in baseline_phase["terminal_episode_steps"]
            )
            row["treatment_terminal_episode_steps"] = "|".join(
                str(value) for value in treatment_phase["terminal_episode_steps"]
            )
            rows.append(row)
    return {
        "status": "passed",
        "interpretation": (
            "one-coordinate distribution-bound intervention; initialization, exploration, "
            "and learning all change; all-phase error mixes ground, takeoff, and "
            "formation, while phase-conditioned quality has unequal exposure "
            "when policies terminate at different times"
        ),
        "baseline_run": str(baseline["path"]),
        "treatment_run": str(treatment["path"]),
        "image_id": baseline["report"]["image_id"],
        "source_host_package_sha256": {
            "baseline": baseline["report"]["source"]["package_sha256"],
            "treatment": treatment["report"]["source"]["package_sha256"],
        },
        "learning_curve_config": baseline["curve"],
        "intervention": intervention,
        "raw_audit": {"baseline": raw_counts["plane"], "treatment": raw_counts["generalist"]},
        "paired_rows": rows,
        "update_zero_distributions_are_different": True,
        "superiority_established": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--treatment-run", required=True, type=Path)
    args = parser.parse_args(argv)
    run = create_run_directory(project_root() / "runs/action-interface-comparison")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        baseline_source=str(args.baseline_run.resolve()),
        treatment_source=str(args.treatment_run.resolve()),
        source=probe_source(30).details,
        system=probe_system().details,
    )
    try:
        comparison = compare_action_interface_runs(args.baseline_run, args.treatment_run)
        report.update(comparison)
        with (run / "paired-checkpoints.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison["paired_rows"][0]))
            writer.writeheader()
            writer.writerows(comparison["paired_rows"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
