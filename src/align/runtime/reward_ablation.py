"""Audit a one-variable formation-reward experiment on immutable plane runs."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from align.artifacts import artifact_logger, create_run_directory
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import digest, finish, new_report, project_root
from align.runtime.template_comparison import _load_run, audit_raw_runs

MEASURES = ("assigned_rmse_mean_m", "pairwise_rmse_mean_m", "minimum_separation_m")


def validate_single_weight_change(baseline: dict, treatment: dict) -> tuple[float, float]:
    """Require identical resolved task contracts except the formation reward weight."""
    if set(baseline) != set(treatment):
        raise ValueError("resolved configuration sections differ")
    if set(baseline["reward"]) != set(treatment["reward"]):
        raise ValueError("reward configuration fields differ")
    if {key: value for key, value in baseline.items() if key != "reward"} != {
        key: value for key, value in treatment.items() if key != "reward"
    }:
        raise ValueError("comparison arms differ outside the reward configuration")
    if {key: value for key, value in baseline["reward"].items() if key != "formation_weight"} != {
        key: value for key, value in treatment["reward"].items() if key != "formation_weight"
    }:
        raise ValueError("comparison arms differ beyond formation_weight")
    left = baseline["reward"]["formation_weight"]
    right = treatment["reward"]["formation_weight"]
    if type(left) not in (int, float) or type(right) not in (int, float):
        raise ValueError("formation weights must be numeric")
    if not (math.isfinite(left) and math.isfinite(right) and 0 < left < right):
        raise ValueError("treatment formation weight must exceed positive baseline weight")
    return float(left), float(right)


def _formation_window(path: Path, *, margin_m: float, window: int = 50) -> dict:
    """Summarize separate episodes, including those after an environment reset."""
    if not math.isfinite(margin_m) or margin_m <= 0 or type(window) is not int or window < 1:
        raise ValueError("formation window and safety margin must be positive")
    by_episode: dict[tuple[int, int], list[dict]] = {}
    prior_steps: dict[int, int] = {}
    episode_index: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            env_id = int(row["env_id"])
            step = int(row["episode_step"])
            if env_id in prior_steps and step <= prior_steps[env_id]:
                episode_index[env_id] += 1
            else:
                episode_index.setdefault(env_id, 0)
            prior_steps[env_id] = step
            if row["phase"] == "formation":
                by_episode.setdefault((env_id, episode_index[env_id]), []).append(row)
    if not by_episode:
        raise ValueError("evaluation has no formation-phase rows")
    windowed = [rows for rows in by_episode.values() if len(rows) >= window]
    first = [row for rows in windowed for row in rows[:window]]
    last = [row for rows in windowed for row in rows[-window:]]
    all_rows = [row for rows in by_episode.values() for row in rows]

    def mean(rows: list[dict], key: str) -> float | None:
        if not rows:
            return None
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"nonfinite {key}")
        return math.fsum(values) / len(values)

    separations = [float(row["minimum_separation_m"]) for row in all_rows]
    if not all(math.isfinite(value) for value in separations):
        raise ValueError("nonfinite separation")
    return {
        "formation_rows": len(all_rows),
        "formation_episodes": len(by_episode),
        "windowed_formation_episodes": len(windowed),
        "first_assigned_rmse_m": mean(first, "assigned_rmse_m"),
        "last_assigned_rmse_m": mean(last, "assigned_rmse_m"),
        "first_pairwise_rmse_m": mean(first, "pairwise_rmse_m"),
        "last_pairwise_rmse_m": mean(last, "pairwise_rmse_m"),
        "formation_minimum_separation_m": min(separations),
        "below_margin_fraction": sum(value < margin_m for value in separations) / len(separations),
    }


def compare_runs(baseline_run: Path, treatment_run: Path) -> dict:
    baseline = _load_run(baseline_run, frozenset(("plane",)))
    treatment = _load_run(treatment_run, frozenset(("plane",)))
    if baseline["path"] == treatment["path"]:
        raise ValueError("comparison arms must be distinct immutable runs")
    if baseline["curve"] != treatment["curve"]:
        raise ValueError("comparison arms must use the same seeds, updates, and milestones")
    if baseline["report"]["image_id"] != treatment["report"]["image_id"]:
        raise ValueError("comparison arms must use the same simulator image")
    left_weight, right_weight = validate_single_weight_change(
        baseline["config"], treatment["config"]
    )
    raw_counts = audit_raw_runs(baseline["path"], treatment["path"])
    raw_audit = {"baseline": raw_counts["plane"], "treatment": raw_counts["generalist"]}
    rows = []
    for old_seed, new_seed in zip(
        baseline["report"]["seeds"], treatment["report"]["seeds"], strict=True
    ):
        if old_seed["policy_seed"] != new_seed["policy_seed"]:
            raise ValueError("policy seed pairing differs")
        seed = old_seed["policy_seed"]
        if baseline["training_exposure"][seed] != treatment["training_exposure"][seed]:
            raise ValueError("training template exposure differs")
        for old_eval, new_eval in zip(
            old_seed["evaluations"], new_seed["evaluations"], strict=True
        ):
            update = old_eval["completed_update"]
            if update != new_eval["completed_update"]:
                raise ValueError("evaluation milestones differ")
            old = old_eval["metrics"]["template_measurements"]["plane"]
            new = new_eval["metrics"]["template_measurements"]["plane"]
            if old["rows"] != new["rows"]:
                raise ValueError("evaluation horizon differs")
            old_path = (
                baseline["path"] / f"seed-{seed:010d}/evaluation-update-{update:04d}/evaluation.csv"
            )
            new_path = (
                treatment["path"]
                / f"seed-{seed:010d}/evaluation-update-{update:04d}/evaluation.csv"
            )
            margin = baseline["config"]["reward"]["minimum_separation_m"]
            old_window = _formation_window(old_path, margin_m=margin)
            new_window = _formation_window(new_path, margin_m=margin)
            row = {
                "policy_seed": seed,
                "completed_update": update,
                "baseline_successes": old["outcome_counts"]["1"],
                "treatment_successes": new["outcome_counts"]["1"],
                "baseline_timeouts": old["outcome_counts"]["6"],
                "treatment_timeouts": new["outcome_counts"]["6"],
                "baseline_safety_terminations": sum(
                    old["outcome_counts"][str(code)] for code in (2, 3, 4)
                ),
                "treatment_safety_terminations": sum(
                    new["outcome_counts"][str(code)] for code in (2, 3, 4)
                ),
            }
            for name in MEASURES:
                row[f"baseline_{name}"] = old[name]
                row[f"treatment_{name}"] = new[name]
                row[f"treatment_minus_baseline_{name}"] = new[name] - old[name]
            for name in old_window:
                row[f"baseline_{name}"] = old_window[name]
                row[f"treatment_{name}"] = new_window[name]
            rows.append(row)
    return {
        "status": "passed",
        "interpretation": "matched one-variable formation-weight ablation; no superiority inferred",
        "baseline_run": str(baseline["path"]),
        "treatment_run": str(treatment["path"]),
        "baseline_formation_weight": left_weight,
        "treatment_formation_weight": right_weight,
        "image_id": baseline["report"]["image_id"],
        "learning_curve_config": baseline["curve"],
        "source_package_sha256": {
            "baseline": baseline["report"]["source"]["package_sha256"],
            "treatment": treatment["report"]["source"]["package_sha256"],
        },
        "input_sha256": {
            arm: {name: digest(path) for name, path in item["files"].items()}
            for arm, item in (("baseline", baseline), ("treatment", treatment))
        },
        "raw_audit": raw_audit,
        "paired_rows": rows,
        "team_reward_comparison_valid": False,
        "team_reward_comparison_reason": "reward scales differ by experimental design",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--treatment-run", required=True, type=Path)
    args = parser.parse_args(argv)
    comparison = compare_runs(args.baseline_run, args.treatment_run)
    run = create_run_directory(project_root() / "runs/formation-reward-comparison")
    started = time.perf_counter()
    report = new_report()
    report.update(
        comparison,
        run_id=run.name,
        source=probe_source(30).details,
        system=probe_system().details,
    )
    with (run / "paired-checkpoints.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison["paired_rows"][0]))
        writer.writeheader()
        writer.writerows(comparison["paired_rows"])
    with artifact_logger(
        run, "INFO", log_filename="comparison.log", namespace="align.formation_reward"
    ) as log:
        finish(run, report, started)
        log.info("Matched formation-reward comparison passed: %s", run)
    return 0
