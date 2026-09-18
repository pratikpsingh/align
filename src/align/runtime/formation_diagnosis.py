"""Host-only phase diagnosis from immutable frozen-policy evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from align.artifacts import artifact_logger, create_run_directory
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import finish, new_report, project_root
from align.runtime.template_comparison import audit_raw_runs, compare_runs

PHASES = ("ground", "takeoff", "formation")


def _number(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key} in evaluation row")
    return value


def analyze_episode(
    rows: list[dict], *, window_steps: int, separation_margin_m: float, action_bound: float
) -> dict:
    """Measure the target-change response of one environment episode."""
    if not rows or window_steps < 1 or not 0 < separation_margin_m or not 0 < action_bound <= 1:
        raise ValueError("episode and diagnostic thresholds must be valid")
    if any(row["formation_kind"] != rows[0]["formation_kind"] for row in rows):
        raise ValueError("formation kind changed inside one episode")
    steps = [int(row["episode_step"]) for row in rows]
    if any(later != earlier + 1 for earlier, later in zip(steps, steps[1:], strict=False)):
        raise ValueError("episode steps are not consecutive")
    if any(row["phase"] not in PHASES for row in rows):
        raise ValueError("unknown episode phase")
    phases = {phase: [row for row in rows if row["phase"] == phase] for phase in PHASES}
    formation = phases["formation"]
    if len(formation) < 2 * window_steps:
        raise ValueError("formation phase is shorter than two diagnostic windows")
    start, end = formation[:window_steps], formation[-window_steps:]
    result = {
        "formation_kind": rows[0]["formation_kind"],
        "episode_rows": len(rows),
        "formation_rows": len(formation),
        "terminal_reason_code": int(rows[-1]["reason_code"]),
        "minimum_separation_m": min(_number(row, "minimum_separation_m") for row in formation),
        "separation_below_margin_rows": sum(
            _number(row, "minimum_separation_m") < separation_margin_m for row in formation
        ),
        "action_near_bound_rows": sum(
            _number(row, "action_min") <= -action_bound
            or _number(row, "action_max") >= action_bound
            for row in formation
        ),
        "mean_row_max_abs_action": sum(
            max(abs(_number(row, "action_min")), abs(_number(row, "action_max")))
            for row in formation
        )
        / len(formation),
        "max_abs_action": max(
            max(abs(_number(row, "action_min")), abs(_number(row, "action_max")))
            for row in formation
        ),
    }
    for phase, values in phases.items():
        result[f"{phase}_rows"] = len(values)
        result[f"{phase}_team_reward_mean"] = (
            sum(_number(row, "team_reward") for row in values) / len(values) if values else None
        )
    for metric, column in (
        ("assigned_rmse_m", "assigned_rmse_m"),
        ("pairwise_rmse_m", "pairwise_rmse_m"),
    ):
        first = sum(_number(row, column) for row in start) / window_steps
        last = sum(_number(row, column) for row in end) / window_steps
        result[f"formation_first_{metric}"] = first
        result[f"formation_last_{metric}"] = last
        result[f"formation_last_minus_first_{metric}"] = last - first
    result["separation_below_margin_fraction"] = result["separation_below_margin_rows"] / len(
        formation
    )
    result["action_near_bound_fraction"] = result["action_near_bound_rows"] / len(formation)
    return result


def _episodes(path: Path) -> list[tuple[int, int, list[dict]]]:
    """Split each environment when episode_step restarts; preserve row order."""
    groups: dict[int, list[list[dict]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            env_id = int(row["env_id"])
            if env_id < 0:
                raise ValueError(f"negative environment id in {path}")
            episodes = groups[env_id]
            if not episodes or int(row["episode_step"]) <= int(episodes[-1][-1]["episode_step"]):
                episodes.append([])
            episodes[-1].append(row)
    return [
        (env_id, index, rows)
        for env_id, episodes in sorted(groups.items())
        for index, rows in enumerate(episodes)
    ]


def diagnose_runs(
    plane_run: Path,
    generalist_run: Path,
    *,
    window_steps: int = 50,
    action_bound: float = 0.98,
) -> dict:
    """Validate both arms, then summarize every frozen-policy episode by phase."""
    comparison = compare_runs(plane_run, generalist_run)
    audit = audit_raw_runs(plane_run, generalist_run)
    rows = []
    for arm, path in (("plane", plane_run), ("generalist", generalist_run)):
        root = path.resolve(strict=True)
        config = json.loads((root / "config.json").read_text())
        margin = config["reward"]["minimum_separation_m"]
        report = json.loads((root / "report.json").read_text())
        for seed in report["seeds"]:
            seed_id = seed["policy_seed"]
            for evaluation in seed["evaluations"]:
                update = evaluation["completed_update"]
                csv_path = (
                    root
                    / f"seed-{seed_id:010d}"
                    / f"evaluation-update-{update:04d}"
                    / "evaluation.csv"
                )
                for env_id, episode_index, episode in _episodes(csv_path):
                    result = analyze_episode(
                        episode,
                        window_steps=window_steps,
                        separation_margin_m=margin,
                        action_bound=action_bound,
                    )
                    rows.append(
                        {
                            "arm": arm,
                            "policy_seed": seed_id,
                            "completed_update": update,
                            "env_id": env_id,
                            "episode_index": episode_index,
                            **result,
                        }
                    )
    summary = []
    keys = sorted({(row["arm"], row["completed_update"], row["formation_kind"]) for row in rows})
    numeric = (
        "formation_first_assigned_rmse_m",
        "formation_last_assigned_rmse_m",
        "formation_last_minus_first_assigned_rmse_m",
        "formation_first_pairwise_rmse_m",
        "formation_last_pairwise_rmse_m",
        "formation_last_minus_first_pairwise_rmse_m",
        "formation_team_reward_mean",
        "minimum_separation_m",
        "separation_below_margin_fraction",
        "action_near_bound_fraction",
        "mean_row_max_abs_action",
        "max_abs_action",
    )
    for arm, update, kind in keys:
        group = [
            row
            for row in rows
            if (row["arm"], row["completed_update"], row["formation_kind"]) == (arm, update, kind)
        ]
        record = {
            "arm": arm,
            "completed_update": update,
            "formation_kind": kind,
            "episode_count": len(group),
            "formation_rows": sum(row["formation_rows"] for row in group),
            "success_count": sum(row["terminal_reason_code"] == 1 for row in group),
            "time_limit_count": sum(row["terminal_reason_code"] == 6 for row in group),
        }
        for name in numeric:
            record[f"mean_{name}"] = sum(row[name] for row in group) / len(group)
        summary.append(record)
    return {
        "status": "passed",
        "plane_run": comparison["plane_run"],
        "generalist_run": comparison["generalist_run"],
        "source_input_sha256": comparison["input_sha256"],
        "image_id": comparison["image_id"],
        "window_steps": window_steps,
        "action_bound": action_bound,
        "separation_margin_m": margin,
        "raw_audit": audit,
        "episode_count": len(rows),
        "episode_diagnostics": rows,
        "phase_summary": summary,
        "unobserved": (
            "per-component reward",
            "per-agent world position and velocity",
            "per-agent assigned target trajectory",
            "commanded velocity and realized controller response",
        ),
    }


def plot_phase_summary(run: Path, rows: list[dict]) -> None:
    """Save an exploratory two-panel scientific plot from the retained summary."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    styles = {
        ("plane", "plane"): ("black", "--", "Plane specialist"),
        ("generalist", "plane"): ("tab:blue", "-", "Generalist: plane"),
        ("generalist", "cube"): ("tab:orange", "-", "Generalist: cube"),
        ("generalist", "pyramid"): ("tab:green", "-", "Generalist: pyramid"),
        ("generalist", "sphere"): ("tab:red", "-", "Generalist: sphere"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for (arm, kind), (color, linestyle, label) in styles.items():
        series = sorted(
            (row for row in rows if row["arm"] == arm and row["formation_kind"] == kind),
            key=lambda row: row["completed_update"],
        )
        if not series:
            continue
        x = [row["completed_update"] for row in series]
        for axis, metric in (
            (axes[0], "mean_formation_last_assigned_rmse_m"),
            (axes[1], "mean_formation_last_pairwise_rmse_m"),
        ):
            axis.plot(
                x,
                [row[metric] for row in series],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
    for axis, title in (
        (axes[0], "Assigned-position error"),
        (axes[1], "Pairwise-distance error"),
    ):
        axis.set(title=title, xlabel="Completed PPO updates", ylabel="Last-50-step RMSE (m)")
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle("Formation-phase error at frozen checkpoints (two policy seeds)")
    figure.tight_layout()
    figure.savefig(run / "phase-trends.pdf")
    figure.savefig(run / "phase-trends.png", dpi=160)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("diagnostic table cannot be empty")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose frozen formation-phase behavior")
    parser.add_argument("--plane-run", required=True, type=Path)
    parser.add_argument("--generalist-run", required=True, type=Path)
    parser.add_argument("--window-steps", type=int, default=50)
    parser.add_argument("--plot", action="store_true", help="requires the pinned reporting extra")
    args = parser.parse_args(argv)
    report = diagnose_runs(args.plane_run, args.generalist_run, window_steps=args.window_steps)
    run = create_run_directory(project_root() / "runs/formation-diagnosis")
    started = time.perf_counter()
    result = new_report()
    result.update(
        report, run_id=run.name, source=probe_source(30).details, system=probe_system().details
    )
    with artifact_logger(
        run, "INFO", log_filename="diagnosis.log", namespace="align.diagnosis"
    ) as log:
        _write_csv(run / "episode-diagnostics.csv", report["episode_diagnostics"])
        _write_csv(run / "phase-summary.csv", report["phase_summary"])
        if args.plot:
            plot_phase_summary(run, report["phase_summary"])
            result["plot_files"] = ["phase-trends.pdf", "phase-trends.png"]
        finish(run, result, started)
        log.info("Formation phase diagnosis passed: %s", run)
    return 0
