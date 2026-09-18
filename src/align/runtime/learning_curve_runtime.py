"""Host launcher for bounded multi-seed checkpoint learning curves."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.critic_distribution_schema import read_valid_critic_distribution
from align.learning.learning_curve_config import LearningCurveConfig
from align.learning.training_config import TaskTrainingConfig
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import (
    SudoKeepalive,
    capture,
    check_gpu_available,
    docker_prefix,
    execute,
    finish,
    new_report,
    project_root,
)
from align.runtime.stability_runtime import (
    valid_normalization_warmup_csv,
    valid_phase,
    validate_stability_resolved_config,
)
from align.runtime.training_runtime import _load, load_resolved_config


def _bounds(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"no rows available for {key}")
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def summarize_learning_curve(items: list[dict], config: LearningCurveConfig) -> dict:
    """Aggregate training diagnostics and checkpoint evaluations by update."""
    if [item["policy_seed"] for item in items] != list(config.policy_seeds):
        raise ValueError("learning curve seed order or count differs")
    training_rows = [row for item in items for row in item["train"]["metrics"]["measurements"]]
    expected_updates = len(config.policy_seeds) * config.updates_per_seed
    if len(training_rows) != expected_updates:
        raise ValueError("learning curve training update count differs")

    training_trends = []
    for update in range(1, config.updates_per_seed + 1):
        rows = [row for row in training_rows if row["completed_update"] == update]
        if len(rows) != len(config.policy_seeds):
            raise ValueError(f"training update {update} does not contain every seed")
        training_trends.append(
            {
                "completed_update": update,
                "post_approximate_kl": _bounds(rows, "post_approximate_kl"),
                "post_policy_clip_fraction": _bounds(rows, "post_policy_clip_fraction"),
                "post_value_clip_fraction": _bounds(rows, "post_value_clip_fraction"),
                "post_explained_variance": _bounds(rows, "post_explained_variance"),
                "post_value_loss": _bounds(rows, "post_value_loss"),
                "team_reward_mean": _bounds(rows, "team_reward_mean"),
            }
        )

    trends = []
    all_checkpoint_ids = []
    for milestone in config.evaluation_milestones:
        evaluations = []
        for item in items:
            matches = [row for row in item["evaluations"] if row["completed_update"] == milestone]
            if len(matches) != 1:
                raise ValueError(
                    f"seed {item['policy_seed']} does not have exactly one "
                    f"evaluation at {milestone}"
                )
            evaluation = matches[0]["metrics"]
            if evaluation["completed_updates"] != milestone:
                raise ValueError("evaluation loaded a different checkpoint update")
            evaluations.append(evaluation)
            all_checkpoint_ids.append(evaluation["checkpoint_id"])
        measurements = [row["measurements"] for row in evaluations]
        outcome_counts = {
            str(code): sum(row["outcome_counts"][str(code)] for row in evaluations)
            for code in range(1, 7)
        }
        trends.append(
            {
                "completed_update": milestone,
                "seed_count": len(evaluations),
                "evaluation_rows": sum(row["raw_rows"] for row in evaluations),
                "formation_phase_reached_all_seeds": all(
                    row["formation_phase_reached"] for row in evaluations
                ),
                "outcome_counts": outcome_counts,
                "team_reward_mean": _bounds(measurements, "team_reward_mean"),
                "assigned_rmse_mean_m": _bounds(measurements, "assigned_rmse_mean_m"),
                "pairwise_rmse_mean_m": _bounds(measurements, "pairwise_rmse_mean_m"),
                "minimum_separation_m": _bounds(measurements, "minimum_separation_m"),
            }
        )
    if len(set(all_checkpoint_ids)) != len(all_checkpoint_ids):
        raise ValueError("checkpoint evaluations must load distinct per-seed milestones")
    evaluation_template_presence = [
        "template_measurements" in evaluation["metrics"]
        for item in items
        for evaluation in item["evaluations"]
    ]
    if any(evaluation_template_presence) and not all(evaluation_template_presence):
        raise ValueError("template evaluation measurements are missing from some milestones")
    evaluation_template_trends = []
    if all(evaluation_template_presence):
        for milestone in config.evaluation_milestones:
            evaluations = [
                next(
                    row["metrics"]
                    for row in item["evaluations"]
                    if row["completed_update"] == milestone
                )
                for item in items
            ]
            declared = tuple(evaluations[0]["template_measurements"])
            if any(tuple(row["template_measurements"]) != declared for row in evaluations):
                raise ValueError("template evaluation coverage differs across seeds")
            for kind in declared:
                metrics = [row["template_measurements"][kind] for row in evaluations]
                if any(row["rows"] <= 0 or row["formation_rows"] <= 0 for row in metrics):
                    raise ValueError("template did not reach formation phase during evaluation")
                evaluation_template_trends.append(
                    {
                        "completed_update": milestone,
                        "formation_kind": kind,
                        "seed_count": len(metrics),
                        "rows": sum(row["rows"] for row in metrics),
                        "formation_rows": sum(row["formation_rows"] for row in metrics),
                        "outcome_counts": {
                            str(code): sum(row["outcome_counts"][str(code)] for row in metrics)
                            for code in range(1, 7)
                        },
                        "team_reward_mean": _bounds(metrics, "team_reward_mean"),
                        "assigned_rmse_mean_m": _bounds(metrics, "assigned_rmse_mean_m"),
                        "pairwise_rmse_mean_m": _bounds(metrics, "pairwise_rmse_mean_m"),
                        "minimum_separation_m": _bounds(metrics, "minimum_separation_m"),
                    }
                )

    training_template_presence = ["template_measurements" in row for row in training_rows]
    if any(training_template_presence) and not all(training_template_presence):
        raise ValueError("template training measurements are missing from some updates")
    training_template_trends = []
    if all(training_template_presence):
        declared = tuple(training_rows[0]["template_measurements"])
        for update in range(1, config.updates_per_seed + 1):
            rows = [row for row in training_rows if row["completed_update"] == update]
            if any(tuple(row["template_measurements"]) != declared for row in rows):
                raise ValueError("template training coverage differs across seeds")
            for kind in declared:
                metrics = [row["template_measurements"][kind] for row in rows]
                if any(row["rows"] <= 0 or row["formation_rows"] <= 0 for row in metrics):
                    raise ValueError("template did not reach formation phase during training")
                training_template_trends.append(
                    {
                        "completed_update": update,
                        "formation_kind": kind,
                        "seed_count": len(metrics),
                        "rows": sum(row["rows"] for row in metrics),
                        "formation_rows": sum(row["formation_rows"] for row in metrics),
                        "team_reward_mean": _bounds(metrics, "team_reward_mean"),
                        "assigned_rmse_mean_m": _bounds(metrics, "assigned_rmse_mean_m"),
                        "pairwise_rmse_mean_m": _bounds(metrics, "pairwise_rmse_mean_m"),
                        "minimum_separation_m": _bounds(metrics, "minimum_separation_m"),
                    }
                )

    distribution_presence = ["critic_distribution" in row for row in training_rows]
    if any(distribution_presence) and not all(distribution_presence):
        raise ValueError("critic distribution measurements are missing from some updates")
    distribution_trends = []
    if all(distribution_presence):
        for update in range(1, config.updates_per_seed + 1):
            rows = [row for row in training_rows if row["completed_update"] == update]
            for group in ("position", "velocity", "target"):
                group_rows = []
                for row in rows:
                    matches = [
                        item for item in row["critic_distribution"] if item["group"] == group
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"update {update} lacks exactly one {group} group")
                    group_rows.append(matches[0])
                effective_presence = [
                    "mean_shift_normalization_standard_deviations" in item
                    and "raw_standard_deviation_normalization_ratio" in item
                    for item in group_rows
                ]
                if any(effective_presence) and not all(effective_presence):
                    raise ValueError("effective critic scale metrics are missing from some seeds")
                trend = {
                    "completed_update": update,
                    "group": group,
                    "clipped_fraction": _bounds(group_rows, "clipped_fraction"),
                    "mean_shift_warmup_standard_deviations": _bounds(
                        group_rows, "mean_shift_warmup_standard_deviations"
                    ),
                    "raw_standard_deviation_ratio": _bounds(
                        group_rows, "raw_standard_deviation_ratio"
                    ),
                }
                if all(effective_presence):
                    trend.update(
                        mean_shift_normalization_standard_deviations=_bounds(
                            group_rows, "mean_shift_normalization_standard_deviations"
                        ),
                        raw_standard_deviation_normalization_ratio=_bounds(
                            group_rows, "raw_standard_deviation_normalization_ratio"
                        ),
                    )
                distribution_trends.append(trend)

    return {
        "seed_count": len(items),
        "seeds": [item["policy_seed"] for item in items],
        "updates_per_seed": config.updates_per_seed,
        "training_update_count": len(training_rows),
        "evaluation_milestones": list(config.evaluation_milestones),
        "evaluation_container_count": len(all_checkpoint_ids),
        "post_update_approximate_kl": _bounds(training_rows, "post_approximate_kl"),
        "post_update_policy_clip_fraction": _bounds(training_rows, "post_policy_clip_fraction"),
        "post_update_value_clip_fraction": _bounds(training_rows, "post_value_clip_fraction"),
        "post_update_explained_variance": _bounds(training_rows, "post_explained_variance"),
        "post_update_value_loss": _bounds(training_rows, "post_value_loss"),
        "critic_gradient_norm_before_clip": _bounds(
            training_rows, "max_critic_gradient_norm_before_clip"
        ),
        "training_trends": training_trends,
        "template_training_trends": training_template_trends,
        "critic_distribution_trends": distribution_trends,
        "evaluation_trends": trends,
        "template_evaluation_trends": evaluation_template_trends,
    }


def combine_segment_metrics(segments: list[dict], config: LearningCurveConfig) -> dict:
    """Join disjoint simulator segments and audit their checkpoint boundary."""
    if not segments:
        raise ValueError("segmented learning requires at least one segment")
    if len(segments) == 1:
        return segments[0]["metrics"]
    actual_ranges = tuple((item["start_update"], item["stop_update"]) for item in segments)
    expected_ranges = config.segment_ranges()
    measurements = [row for item in segments for row in item["metrics"]["measurements"]]
    normalization = segments[0]["metrics"]["critic_normalization"]
    restart_index = config.planned_restart_after_segment
    checks = {
        "segment_ranges_are_exact": actual_ranges == expected_ranges,
        "all_segments_passed": all(item["metrics"]["status"] == "passed" for item in segments),
        "checkpoint_lineage_is_contiguous_across_processes": all(
            current["metrics"]["initial_checkpoint_id"]
            == previous["metrics"]["final_checkpoint_id"]
            and current["metrics"]["initial_checkpoint_sha256"]
            == previous["metrics"]["final_checkpoint_sha256"]
            for previous, current in zip(segments[:-1], segments[1:], strict=True)
        ),
        "completed_updates_are_exact": [row["completed_update"] for row in measurements]
        == list(range(1, config.updates_per_seed + 1)),
        "final_counter_is_exact": segments[-1]["metrics"]["final_counters"]["completed_updates"]
        == config.updates_per_seed,
        "normalization_state_is_identical_across_segments": all(
            item["metrics"]["critic_normalization"] == normalization for item in segments
        ),
        "normalization_warmup_not_repeated": all(
            item["metrics"]["critic_normalization_warmup_environment_transitions"] == 0
            for item in segments[1:]
        ),
        "planned_process_restart_observed": (
            restart_index is not None
            and segments[restart_index - 1]["exit_code"] == 0
            and segments[restart_index]["metrics"]["segment_start_update"]
            == segments[restart_index - 1]["metrics"]["segment_stop_update"]
            and segments[restart_index]["container_name"]
            != segments[restart_index - 1]["container_name"]
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "policy_seed": segments[0]["metrics"]["policy_seed"],
        "segment_count": len(segments),
        "segment_ranges": [list(item) for item in actual_ranges],
        "planned_restart_after_segment": restart_index,
        "updates": config.updates_per_seed,
        "critic_normalization": normalization,
        "critic_normalization_warmup_environment_transitions": sum(
            item["metrics"]["critic_normalization_warmup_environment_transitions"]
            for item in segments
        ),
        "critic_normalization_warmup_agent_transitions": sum(
            item["metrics"]["critic_normalization_warmup_agent_transitions"] for item in segments
        ),
        "critic_normalization_warmup_seconds": sum(
            item["metrics"]["critic_normalization_warmup_seconds"] for item in segments
        ),
        "initial_checkpoint_id": segments[0]["metrics"]["initial_checkpoint_id"],
        "initial_checkpoint_sha256": segments[0]["metrics"]["initial_checkpoint_sha256"],
        "final_checkpoint_id": segments[-1]["metrics"]["final_checkpoint_id"],
        "final_checkpoint_sha256": segments[-1]["metrics"]["final_checkpoint_sha256"],
        "final_counters": segments[-1]["metrics"]["final_counters"],
        "measurements": measurements,
    }


def write_learning_curve_tables(run: Path, summary: dict) -> None:
    """Write compact long-form CSV tables derived from the JSON summary."""
    training_metrics = (
        "post_approximate_kl",
        "post_policy_clip_fraction",
        "post_value_clip_fraction",
        "post_explained_variance",
        "post_value_loss",
        "team_reward_mean",
    )
    evaluation_metrics = (
        "team_reward_mean",
        "assigned_rmse_mean_m",
        "pairwise_rmse_mean_m",
        "minimum_separation_m",
    )
    fields = ("completed_update", "metric", "minimum", "maximum", "mean")
    for filename, trends, metrics in (
        ("training-curve.csv", summary["training_trends"], training_metrics),
        ("evaluation-curve.csv", summary["evaluation_trends"], evaluation_metrics),
    ):
        with (run / filename).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in trends:
                for metric in metrics:
                    writer.writerow(
                        {
                            "completed_update": row["completed_update"],
                            "metric": metric,
                            **row[metric],
                        }
                    )
    template_metrics = (
        "team_reward_mean",
        "assigned_rmse_mean_m",
        "pairwise_rmse_mean_m",
        "minimum_separation_m",
    )
    for filename, trend_key in (
        ("template-training-curve.csv", "template_training_trends"),
        ("template-evaluation-curve.csv", "template_evaluation_trends"),
    ):
        if summary[trend_key]:
            with (run / filename).open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "completed_update",
                        "formation_kind",
                        "seed_count",
                        "rows",
                        "formation_rows",
                        "metric",
                        "minimum",
                        "maximum",
                        "mean",
                    ),
                )
                writer.writeheader()
                for row in summary[trend_key]:
                    for metric in template_metrics:
                        writer.writerow(
                            {
                                "completed_update": row["completed_update"],
                                "formation_kind": row["formation_kind"],
                                "seed_count": row["seed_count"],
                                "rows": row["rows"],
                                "formation_rows": row["formation_rows"],
                                "metric": metric,
                                **row[metric],
                            }
                        )
    if summary["critic_distribution_trends"]:
        with (run / "critic-distribution-curve.csv").open(
            "x", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("completed_update", "group", "metric", "minimum", "maximum", "mean"),
            )
            writer.writeheader()
            for row in summary["critic_distribution_trends"]:
                metrics = [
                    "clipped_fraction",
                    "mean_shift_warmup_standard_deviations",
                    "raw_standard_deviation_ratio",
                ]
                if "mean_shift_normalization_standard_deviations" in row:
                    metrics.extend(
                        (
                            "mean_shift_normalization_standard_deviations",
                            "raw_standard_deviation_normalization_ratio",
                        )
                    )
                for metric in metrics:
                    writer.writerow(
                        {
                            "completed_update": row["completed_update"],
                            "group": row["group"],
                            "metric": metric,
                            **row[metric],
                        }
                    )


def _command(
    docker,
    *,
    run: Path,
    seed_directory: Path,
    output_directory: Path,
    image_id: str,
    gpu: int,
    num_envs: int,
    source_identity: str,
    scenario: str,
    evaluation_update: int | None = None,
    training_start_update: int | None = None,
    training_stop_update: int | None = None,
    fault_at_update: int | None = None,
    fault_after_rollout_step: int | None = None,
    logical_run_name: str | None = None,
) -> list[str]:
    relative_seed = seed_directory.relative_to(run)
    relative_output = output_directory.relative_to(run)
    seed_label = relative_seed.name
    if scenario == "stability" and training_stop_update is not None:
        phase = f"segment-{training_start_update:04d}-{training_stop_update:04d}"
    elif scenario == "stability":
        phase = "train"
    else:
        phase = f"evaluation-{evaluation_update:04d}"
    logical_run_id = f"{logical_run_name or run.name}-seed-{seed_label.split('-')[-1]}"
    command = [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-learning-curve-{run.name.lower()}-{seed_label}-{phase}",
        "--runtime=nvidia",
        "--gpus",
        f"device={gpu}",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "TZ=Asia/Kolkata",
        "--mount",
        f"type=bind,src={run},dst=/output",
        "--mount",
        f"type=bind,src={output_directory / 'kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        f"/output/{relative_seed}/config.json",
        "--output",
        f"/output/{relative_output}",
        "--scenario",
        scenario,
        "--num-envs",
        str(num_envs),
        "--logical-run-id",
        logical_run_id,
        "--attempt-id",
        phase,
        "--checkpoint-directory",
        f"/output/{relative_seed}/checkpoints",
        "--source-identity",
        source_identity,
        "--runtime-identity",
        image_id,
        "--allow-root",
    ]
    if evaluation_update is not None:
        command.extend(("--evaluation-update", str(evaluation_update)))
    if training_stop_update is not None:
        command.extend(
            (
                "--training-start-update",
                str(training_start_update),
                "--training-stop-update",
                str(training_stop_update),
            )
        )
    if fault_at_update is not None:
        command.extend(
            (
                "--fault-at-update",
                str(fault_at_update),
                "--fault-after-rollout-step",
                str(fault_after_rollout_step),
            )
        )
    return command


def run_main(
    argv=None,
    *,
    default_curve_config: str = "learning-curve-baseline.json",
    run_category: str = "learning-curve",
    default_formation_schedule_config: str | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded training with fresh-process checkpoint evaluations."
    )
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--construction-config", type=Path)
    parser.add_argument("--observation-config", type=Path)
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--rollout-config", type=Path)
    parser.add_argument("--ppo-config", type=Path)
    parser.add_argument("--recovery-config", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--critic-normalization-config", type=Path)
    parser.add_argument("--formation-schedule-config", type=Path)
    parser.add_argument("--learning-curve-config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--inject-interruption-seed", type=int)
    parser.add_argument("--inject-interruption-update", type=int)
    parser.add_argument("--inject-interruption-rollout-step", type=int)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60-3600")
    injection_values = (
        args.inject_interruption_seed,
        args.inject_interruption_update,
        args.inject_interruption_rollout_step,
    )
    if any(value is not None for value in injection_values) and not all(
        value is not None for value in injection_values
    ):
        parser.error("Interruption injection requires seed, update, and rollout step together")

    root = project_root()
    args.task_config = args.task_config or root / "configs/recurrent-stability-task.json"
    args.rollout_config = args.rollout_config or root / "configs/recurrent-stability-rollout.json"
    args.ppo_config = args.ppo_config or root / "configs/recurrent-ppo-selected.json"
    if args.formation_schedule_config is None and default_formation_schedule_config is not None:
        args.formation_schedule_config = root / "configs" / default_formation_schedule_config
    curve = _load(
        args.learning_curve_config or root / "configs" / default_curve_config,
        LearningCurveConfig,
    )
    stability = curve.to_stability_config()
    if args.inject_interruption_seed is not None and (
        curve.schema_version != 2
        or args.inject_interruption_seed not in curve.policy_seeds
        or type(args.inject_interruption_update) is not int
        or args.inject_interruption_update
        not in {start + 1 for start, _ in curve.segment_ranges() if start > 0}
    ):
        raise ValueError(
            "interruption injection must target the first update of a noninitial segment"
        )
    resolved = load_resolved_config(root, args)
    if args.inject_interruption_seed is not None and not (
        1 <= args.inject_interruption_rollout_step < resolved["rollout"]["horizon"]
    ):
        raise ValueError("injected rollout step must be inside the configured horizon")
    validate_stability_resolved_config(resolved, stability)
    resolved["stability"] = stability.to_dict()

    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs" / run_category)
    write_json_atomic(run / "config.json", resolved)
    write_json_atomic(run / "learning-curve-config.json", curve.to_dict())
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    for seed in curve.policy_seeds:
        seed_directory = run / f"seed-{seed:010d}"
        (seed_directory / "checkpoints").mkdir(parents=True)
        if curve.updates_per_segment is None:
            (seed_directory / "train/kit-logs").mkdir(parents=True)
        else:
            for start_update, stop_update in curve.segment_ranges():
                (
                    seed_directory
                    / f"train-segment-{start_update:04d}-{stop_update:04d}"
                    / "kit-logs"
                ).mkdir(parents=True)
        for milestone in curve.evaluation_milestones:
            (seed_directory / f"evaluation-update-{milestone:04d}/kit-logs").mkdir(parents=True)
        seed_config = copy.deepcopy(resolved)
        seed_config["training"]["attempts"] = curve.updates_per_seed
        seed_config["training"]["policy_seed"] = seed
        TaskTrainingConfig.from_dict(seed_config["training"])
        write_json_atomic(seed_directory / "config.json", seed_config)

    started = time.perf_counter()
    report = new_report()
    source = probe_source(30).details
    source_identity = f"git:{source.get('git_revision')}:package:{source['package_sha256']}"
    report.update(
        run_id=run.name,
        host_gpu_index=args.gpu,
        image_id=build["image_id"],
        source=source,
        system=probe_system().details,
        learning_curve_config=curve.to_dict(),
        bounded_training_performed=True,
        segmented_training_performed=curve.updates_per_segment is not None,
        fresh_process_training_resumes=max(0, len(curve.segment_ranges()) - 1)
        * len(curve.policy_seeds),
        sustained_training_performed=False,
        deterministic_evaluation_performed=False,
        scientific_result=False,
        planned_training_interruption=(
            {
                "policy_seed": args.inject_interruption_seed,
                "update": args.inject_interruption_update,
                "rollout_step": args.inject_interruption_rollout_step,
            }
            if args.inject_interruption_seed is not None
            else None
        ),
        rendering_validated=False,
        video_validated=False,
        seeds=[],
    )
    report["gpu_before"] = capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ]
    )
    report["gpu_processes_before"] = capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv",
        ]
    )
    docker = docker_prefix(args.docker)
    write_json_atomic(run / "report.json", report)
    print(
        f"{report['started_at_ist']} IST Run: {run}\n"
        "Follow each seed's train*/console.log and evaluation-update-*/console.log files.",
        flush=True,
    )

    active_container = None
    cleanup = False
    keepalive = SudoKeepalive(args.docker == "sudo")
    try:
        keepalive.start()
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for seed in curve.policy_seeds:
            seed_directory = run / f"seed-{seed:010d}"
            item = {"policy_seed": seed, "evaluations": []}
            report["seeds"].append(item)

            segment_records = []
            update_outputs = {}
            for start_update, stop_update in curve.segment_ranges():
                segmented = curve.updates_per_segment is not None
                train_output = (
                    seed_directory / f"train-segment-{start_update:04d}-{stop_update:04d}"
                    if segmented
                    else seed_directory / "train"
                )
                command = _command(
                    docker,
                    run=run,
                    seed_directory=seed_directory,
                    output_directory=train_output,
                    image_id=build["image_id"],
                    gpu=args.gpu,
                    num_envs=resolved["rollout"]["num_envs"],
                    source_identity=source_identity,
                    scenario="stability",
                    training_start_update=start_update if segmented else None,
                    training_stop_update=stop_update if segmented else None,
                    fault_at_update=(
                        args.inject_interruption_update
                        if seed == args.inject_interruption_seed
                        and start_update < args.inject_interruption_update <= stop_update
                        else None
                    ),
                    fault_after_rollout_step=(
                        args.inject_interruption_rollout_step
                        if seed == args.inject_interruption_seed
                        and start_update < args.inject_interruption_update <= stop_update
                        else None
                    ),
                )
                active_container = command[command.index("--name") + 1]
                phase_started = time.perf_counter()
                exit_code = execute(command, train_output / "console.log", args.timeout)
                probe_path = train_output / "probe-result.json"
                metrics_path = train_output / "metrics.json"
                probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
                metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
                segment = {
                    "start_update": start_update,
                    "stop_update": stop_update,
                    "container_name": active_container,
                    "command": command,
                    "exit_code": exit_code,
                    "duration_seconds": time.perf_counter() - phase_started,
                    "probe": probe,
                    "metrics": metrics,
                }
                segment_records.append(segment)
                item["train"] = {"segments": segment_records}
                write_json_atomic(run / "report.json", report)
                if not valid_phase(exit_code, probe, metrics, "training_stability_tested"):
                    if "--fault-at-update" in command:
                        interruption_path = (
                            train_output
                            / f"update-{args.inject_interruption_update:04d}"
                            / "interruption.json"
                        )
                        if interruption_path.is_file():
                            report["observed_training_interruption"] = json.loads(
                                interruption_path.read_text()
                            )
                            report["status"] = "interrupted"
                            write_json_atomic(run / "report.json", report)
                            return 2
                    raise RuntimeError(
                        f"seed {seed} training segment {start_update}:{stop_update} failed"
                    )
                for update in range(start_update + 1, stop_update + 1):
                    update_outputs[update] = train_output
                active_container = None

            combined_metrics = combine_segment_metrics(segment_records, curve)
            if len(segment_records) == 1:
                item["train"] = {
                    **segment_records[0],
                    "segments": segment_records,
                    "metrics": combined_metrics,
                }
            else:
                item["train"] = {
                    "segments": segment_records,
                    "duration_seconds": sum(
                        segment["duration_seconds"] for segment in segment_records
                    ),
                    "metrics": combined_metrics,
                }
            if combined_metrics["status"] != "passed":
                raise RuntimeError(f"seed {seed} segmented training audit failed")

            normalization = resolved["critic_normalization"]
            warmup_paths = [
                path
                for segment in segment_records
                for path in (
                    seed_directory
                    / (
                        f"train-segment-{segment['start_update']:04d}-{segment['stop_update']:04d}"
                        if curve.updates_per_segment is not None
                        else "train"
                    )
                ).glob("update-*/normalization-warmup.csv")
            ]
            warmup_valid = len(warmup_paths) == int(normalization["enabled"]) and (
                not normalization["enabled"]
                or valid_normalization_warmup_csv(
                    warmup_paths[0],
                    warmup_steps=normalization["warmup_steps"],
                    num_envs=resolved["rollout"]["num_envs"],
                    num_agents=resolved["rollout"]["num_agents"],
                )
            )
            item["train"]["normalization_warmup_csv_valid"] = warmup_valid
            write_json_atomic(run / "report.json", report)
            if not warmup_valid:
                raise RuntimeError(f"seed {seed} normalization warmup audit failed")
            expected_scalars = (
                resolved["rollout"]["horizon"]
                * resolved["rollout"]["num_envs"]
                * resolved["rollout"]["num_agents"]
                * 3
            )
            distribution_valid = all(
                read_valid_critic_distribution(
                    update_outputs[update] / f"update-{update:04d}" / "critic-distribution.csv",
                    update=update,
                    scalar_count=expected_scalars,
                    enabled=normalization["enabled"],
                    clip=normalization["clip"],
                )
                is not None
                for update in range(1, curve.updates_per_seed + 1)
            )
            item["train"]["critic_distribution_csv_valid"] = distribution_valid
            write_json_atomic(run / "report.json", report)
            if not distribution_valid:
                raise RuntimeError(f"seed {seed} critic distribution audit failed")

            for milestone in curve.evaluation_milestones:
                evaluation_output = seed_directory / f"evaluation-update-{milestone:04d}"
                command = _command(
                    docker,
                    run=run,
                    seed_directory=seed_directory,
                    output_directory=evaluation_output,
                    image_id=build["image_id"],
                    gpu=args.gpu,
                    num_envs=resolved["rollout"]["num_envs"],
                    source_identity=source_identity,
                    scenario="evaluation",
                    evaluation_update=milestone,
                )
                active_container = command[command.index("--name") + 1]
                phase_started = time.perf_counter()
                exit_code = execute(command, evaluation_output / "console.log", args.timeout)
                probe_path = evaluation_output / "probe-result.json"
                metrics_path = evaluation_output / "metrics.json"
                probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
                metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
                evaluation = {
                    "completed_update": milestone,
                    "command": command,
                    "exit_code": exit_code,
                    "duration_seconds": time.perf_counter() - phase_started,
                    "probe": probe,
                    "metrics": metrics,
                }
                item["evaluations"].append(evaluation)
                write_json_atomic(run / "report.json", report)
                if not valid_phase(exit_code, probe, metrics, "deterministic_evaluation_tested"):
                    raise RuntimeError(f"seed {seed} evaluation at update {milestone} failed")
                if metrics["completed_updates"] != milestone:
                    raise RuntimeError(
                        f"seed {seed} evaluation loaded update "
                        f"{metrics['completed_updates']} instead of {milestone}"
                    )
                active_container = None
            write_json_atomic(run / "report.json", report)

        summary = summarize_learning_curve(report["seeds"], curve)
        write_json_atomic(run / "learning-curve-summary.json", summary)
        write_learning_curve_tables(run, summary)
        report.update(
            status="passed",
            deterministic_evaluation_performed=True,
            optimizer_updates=summary["training_update_count"],
            evaluation_containers=summary["evaluation_container_count"],
            summary=summary,
        )
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        cleanup = True
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
        cleanup = True
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        cleanup = True
    finally:
        if cleanup and active_container is not None:
            report["cleanup"] = capture([*docker, "rm", "--force", active_container])
        report["sudo_keepalive"] = keepalive.stop()
        report["gpu_after"] = capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv",
            ]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
