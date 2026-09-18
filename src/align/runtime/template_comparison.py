"""Host-only audit of matched plane and four-template checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from align.artifacts import artifact_logger, create_run_directory
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import digest, finish, new_report, project_root

METRICS = (
    "team_reward_mean",
    "assigned_rmse_mean_m",
    "pairwise_rmse_mean_m",
    "minimum_separation_m",
)
KINDS = frozenset(("cube", "sphere", "pyramid", "plane"))


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _counts(value: dict) -> tuple[int, int]:
    if set(value) != {str(code) for code in range(1, 7)}:
        raise ValueError("terminal outcome codes must be exactly 1 through 6")
    if any(type(count) is not int or count < 0 for count in value.values()):
        raise ValueError("terminal outcome counts must be nonnegative integers")
    total = sum(value.values())
    if total == 0:
        raise ValueError("evaluation has no completed episodes")
    return value["1"], total


def _metric_row(value: dict, kind: str) -> None:
    if value["rows"] <= 0 or value["formation_rows"] <= 0:
        raise ValueError(f"{kind} did not reach the formation phase")
    if value["formation_rows"] > value["rows"]:
        raise ValueError(f"{kind} formation rows exceed total rows")
    _counts(value["outcome_counts"])
    for name in METRICS:
        if not math.isfinite(value[name]):
            raise ValueError(f"nonfinite {kind} {name}")


def _load_run(path: Path, expected_kinds: frozenset[str]) -> dict:
    path = path.resolve(strict=True)
    files = {
        name: path / name
        for name in (
            "report.json",
            "config.json",
            "learning-curve-config.json",
            "build-report.json",
        )
    }
    values = {name: _read(file) for name, file in files.items()}
    report = values["report.json"]
    config = values["config.json"]
    curve = values["learning-curve-config.json"]
    build = values["build-report.json"]
    if report.get("status") != "passed" or not report.get("deterministic_evaluation_performed"):
        raise ValueError(f"{path} has no completed deterministic evaluation")
    if build.get("status") != "built" or report.get("image_id") != build.get("image_id"):
        raise ValueError(f"{path} image identity differs from its build")
    schedule = config["formation_schedule"]
    if frozenset(schedule["kinds"]) != expected_kinds or len(schedule["kinds"]) != len(
        expected_kinds
    ):
        raise ValueError(f"{path} formation schedule differs from expected kinds")
    seeds = curve["policy_seeds"]
    milestones = curve["evaluation_milestones"]
    if seeds != [item["policy_seed"] for item in report["seeds"]]:
        raise ValueError(f"{path} policy seed order differs from curve configuration")
    if report.get("optimizer_updates") != len(seeds) * curve["updates_per_seed"]:
        raise ValueError(f"{path} optimizer update count differs from budget")
    if report.get("evaluation_containers") != len(seeds) * len(milestones):
        raise ValueError(f"{path} evaluation count differs from budget")
    training_exposure = {}
    for item in report["seeds"]:
        measurements = item["train"]["metrics"]["measurements"]
        if [row["completed_update"] for row in measurements] != list(
            range(1, curve["updates_per_seed"] + 1)
        ):
            raise ValueError(f"{path} has missing or repeated training updates")
        exposure = {kind: 0 for kind in expected_kinds}
        for row in measurements:
            templates = row["template_measurements"]
            if frozenset(templates) != expected_kinds:
                raise ValueError(f"{path} training lacks declared template coverage")
            for kind, values in templates.items():
                if (
                    values["rows"] <= 0
                    or values["formation_rows"] <= 0
                    or values["formation_rows"] > values["rows"]
                ):
                    raise ValueError(f"{path} training template has invalid row counts")
                exposure[kind] += values["rows"]
        training_exposure[item["policy_seed"]] = exposure
        if len(item["evaluations"]) != len(milestones):
            raise ValueError(f"{path} has missing evaluations")
        for milestone, evaluation in zip(milestones, item["evaluations"], strict=True):
            metrics = evaluation["metrics"]
            if (
                evaluation["exit_code"] != 0
                or evaluation["completed_update"] != milestone
                or metrics["status"] != "passed"
                or metrics["completed_updates"] != milestone
                or metrics["requested_completed_update"] != milestone
            ):
                raise ValueError(f"{path} did not evaluate the requested checkpoint")
            templates = metrics["template_measurements"]
            if frozenset(templates) != expected_kinds:
                raise ValueError(f"{path} evaluation lacks declared template coverage")
            for kind, row in templates.items():
                _metric_row(row, kind)
            if sum(row["rows"] for row in templates.values()) != metrics["raw_rows"]:
                raise ValueError(f"{path} template rows differ from evaluation rows")
            if {
                str(code): sum(row["outcome_counts"][str(code)] for row in templates.values())
                for code in range(1, 7)
            } != metrics["outcome_counts"]:
                raise ValueError(f"{path} template outcomes differ from evaluation outcomes")
    return {
        "path": path,
        "files": files,
        "report": report,
        "config": config,
        "curve": curve,
        "training_exposure": training_exposure,
    }


def compare_runs(plane_run: Path, generalist_run: Path) -> dict:
    """Return paired plane measurements after strict equal-budget/source validation."""
    plane = _load_run(plane_run, frozenset(("plane",)))
    generalist = _load_run(generalist_run, KINDS)
    if plane["path"] == generalist["path"]:
        raise ValueError("comparison arms must be distinct immutable runs")
    if plane["curve"] != generalist["curve"]:
        raise ValueError("comparison arms use different seeds, budgets, or milestones")
    plane_config = {k: v for k, v in plane["config"].items() if k != "formation_schedule"}
    generalist_config = {k: v for k, v in generalist["config"].items() if k != "formation_schedule"}
    if plane_config != generalist_config:
        raise ValueError("comparison arms differ beyond the formation schedule")
    if plane["report"]["image_id"] != generalist["report"]["image_id"]:
        raise ValueError("comparison arms must use the same derived simulator image")
    if (
        plane["report"]["source"]["package_sha256"]
        != generalist["report"]["source"]["package_sha256"]
    ):
        raise ValueError("comparison arms must use the same host package source")
    exposure = [
        {
            "policy_seed": seed,
            "baseline_plane_training_rows": plane["training_exposure"][seed]["plane"],
            "generalist_plane_training_rows": generalist["training_exposure"][seed]["plane"],
            "generalist_all_training_rows": sum(generalist["training_exposure"][seed].values()),
        }
        for seed in plane["curve"]["policy_seeds"]
    ]
    paired = []
    for plane_seed, generalist_seed in zip(
        plane["report"]["seeds"], generalist["report"]["seeds"], strict=True
    ):
        seed = plane_seed["policy_seed"]
        for plane_eval, generalist_eval in zip(
            plane_seed["evaluations"], generalist_seed["evaluations"], strict=True
        ):
            milestone = plane_eval["completed_update"]
            baseline = plane_eval["metrics"]["template_measurements"]["plane"]
            mixed = generalist_eval["metrics"]["template_measurements"]["plane"]
            baseline_success, baseline_episodes = _counts(baseline["outcome_counts"])
            mixed_success, mixed_episodes = _counts(mixed["outcome_counts"])
            row = {
                "policy_seed": seed,
                "completed_update": milestone,
                "baseline_plane_rows": baseline["rows"],
                "generalist_plane_rows": mixed["rows"],
                "baseline_plane_formation_rows": baseline["formation_rows"],
                "generalist_plane_formation_rows": mixed["formation_rows"],
                "baseline_successes": baseline_success,
                "baseline_completed_episodes": baseline_episodes,
                "generalist_successes": mixed_success,
                "generalist_completed_episodes": mixed_episodes,
            }
            for name in METRICS:
                row[f"baseline_{name}"] = baseline[name]
                row[f"generalist_{name}"] = mixed[name]
                row[f"generalist_minus_baseline_{name}"] = mixed[name] - baseline[name]
            paired.append(row)
    summary = []
    for milestone in plane["curve"]["evaluation_milestones"]:
        rows = [row for row in paired if row["completed_update"] == milestone]
        result = {
            "completed_update": milestone,
            "seed_count": len(rows),
            "baseline_successes": sum(row["baseline_successes"] for row in rows),
            "baseline_completed_episodes": sum(row["baseline_completed_episodes"] for row in rows),
            "generalist_successes": sum(row["generalist_successes"] for row in rows),
            "generalist_completed_episodes": sum(
                row["generalist_completed_episodes"] for row in rows
            ),
        }
        for name in METRICS:
            result[f"mean_paired_generalist_minus_baseline_{name}"] = sum(
                row[f"generalist_minus_baseline_{name}"] for row in rows
            ) / len(rows)
        summary.append(result)
    return {
        "status": "passed",
        "interpretation": "matched integration comparison; no performance superiority inferred",
        "plane_run": str(plane["path"]),
        "generalist_run": str(generalist["path"]),
        "input_sha256": {
            arm: {name: digest(path) for name, path in item["files"].items()}
            for arm, item in (("plane", plane), ("generalist", generalist))
        },
        "image_id": plane["report"]["image_id"],
        "source_package_sha256": plane["report"]["source"]["package_sha256"],
        "learning_curve_config": plane["curve"],
        "comparison_scope": (
            "plane evaluations paired by policy seed and checkpoint; "
            "same total updates, unequal plane exposure"
        ),
        "training_exposure": exposure,
        "paired_plane": paired,
        "plane_summary": summary,
        "generalist_template_evaluation_trends": generalist["report"]["summary"][
            "template_evaluation_trends"
        ],
    }


def audit_raw_csv(path: Path, expected: dict[str, dict], *, evaluation: bool) -> int:
    """Recompute per-template counts and metrics from one immutable simulator CSV."""
    totals = {
        kind: {
            "rows": 0,
            "formation_rows": 0,
            "outcomes": {str(code): 0 for code in range(1, 7)},
            "minimum_separation_m": math.inf,
            **{name: 0.0 for name in METRICS if name != "minimum_separation_m"},
        }
        for kind in expected
    }
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            kind = row["formation_kind"]
            if kind not in totals:
                raise ValueError(f"{path} contains undeclared template {kind}")
            current = totals[kind]
            current["rows"] += 1
            current["formation_rows"] += int(row["phase"] == "formation")
            for name, column in (
                ("team_reward_mean", "team_reward"),
                ("assigned_rmse_mean_m", "assigned_rmse_m"),
                ("pairwise_rmse_mean_m", "pairwise_rmse_m"),
            ):
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError(f"{path} contains nonfinite {column}")
                current[name] += value
            separation = float(row["minimum_separation_m"])
            if not math.isfinite(separation):
                raise ValueError(f"{path} contains nonfinite separation")
            current["minimum_separation_m"] = min(current["minimum_separation_m"], separation)
            reason = row["reason_code"]
            if reason not in {str(code) for code in range(7)}:
                raise ValueError(f"{path} contains invalid outcome code")
            if reason != "0":
                current["outcomes"][reason] += 1
    for kind, current in totals.items():
        declared = expected[kind]
        if (
            current["rows"] != declared["rows"]
            or current["formation_rows"] != declared["formation_rows"]
        ):
            raise ValueError(f"{path} raw {kind} row counts differ from metrics")
        if evaluation and current["outcomes"] != declared["outcome_counts"]:
            raise ValueError(f"{path} raw {kind} outcomes differ from metrics")
        if not evaluation and sum(current["outcomes"].values()) != declared["outcomes"]:
            raise ValueError(f"{path} raw {kind} training outcomes differ from metrics")
        for name in METRICS:
            value = current[name]
            if name != "minimum_separation_m":
                value /= current["rows"]
            if not math.isclose(value, declared[name], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"{path} raw {kind} {name} differs from metrics")
    return sum(row["rows"] for row in totals.values())


def audit_raw_runs(plane_run: Path, generalist_run: Path) -> dict:
    """Audit every source rollout/evaluation CSV before publishing a comparison."""
    counts = {}
    for label, path in (("plane", plane_run), ("generalist", generalist_run)):
        root = path.resolve(strict=True)
        report = _read(root / "report.json")
        curve = _read(root / "learning-curve-config.json")
        arm = {
            "training_csv_count": 0,
            "evaluation_csv_count": 0,
            "training_rows": 0,
            "evaluation_rows": 0,
        }
        for seed in report["seeds"]:
            seed_dir = root / f"seed-{seed['policy_seed']:010d}"
            raw_training_paths = list(seed_dir.glob("train*/update-*/rollout.csv"))
            paths = {
                int(csv_path.parent.name.split("-")[-1]): csv_path
                for csv_path in raw_training_paths
            }
            measurements = seed["train"]["metrics"]["measurements"]
            if len(paths) != len(raw_training_paths):
                raise ValueError(f"{seed_dir} contains duplicate raw training updates")
            if len(paths) != curve["updates_per_seed"]:
                raise ValueError(f"{seed_dir} raw training update count differs from budget")
            for measurement in measurements:
                update = measurement["completed_update"]
                arm["training_rows"] += audit_raw_csv(
                    paths[update], measurement["template_measurements"], evaluation=False
                )
                arm["training_csv_count"] += 1
            for evaluation in seed["evaluations"]:
                update = evaluation["completed_update"]
                csv_path = seed_dir / f"evaluation-update-{update:04d}" / "evaluation.csv"
                arm["evaluation_rows"] += audit_raw_csv(
                    csv_path,
                    evaluation["metrics"]["template_measurements"],
                    evaluation=True,
                )
                arm["evaluation_csv_count"] += 1
        counts[label] = arm
    return counts


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("comparison table cannot be empty")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit matched plane-only and four-template runs")
    parser.add_argument("--plane-run", required=True, type=Path)
    parser.add_argument("--generalist-run", required=True, type=Path)
    args = parser.parse_args(argv)
    comparison = compare_runs(args.plane_run, args.generalist_run)
    comparison["raw_audit"] = audit_raw_runs(args.plane_run, args.generalist_run)
    run = create_run_directory(project_root() / "runs/template-comparison")
    started = time.perf_counter()
    report = new_report()
    report.update(
        comparison,
        run_id=run.name,
        source=probe_source(30).details,
        system=probe_system().details,
    )
    with artifact_logger(
        run, "INFO", log_filename="comparison.log", namespace="align.template"
    ) as log:
        _write_csv(run / "training-exposure.csv", comparison["training_exposure"])
        _write_csv(run / "paired-plane.csv", comparison["paired_plane"])
        _write_csv(run / "plane-summary.csv", comparison["plane_summary"])
        finish(run, report, started)
        log.info("Matched template comparison passed: %s", run)
    return 0
