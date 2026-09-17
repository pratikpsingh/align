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
from align.runtime.stability_runtime import valid_phase, validate_stability_resolved_config
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
        "evaluation_trends": trends,
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
) -> list[str]:
    relative_seed = seed_directory.relative_to(run)
    relative_output = output_directory.relative_to(run)
    seed_label = relative_seed.name
    phase = "train" if scenario == "stability" else f"evaluation-{evaluation_update:04d}"
    logical_run_id = f"{run.name}-seed-{seed_label.split('-')[-1]}"
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
    return command


def run_main(argv=None) -> int:
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
    parser.add_argument("--learning-curve-config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60-3600")

    root = project_root()
    args.task_config = args.task_config or root / "configs/recurrent-stability-task.json"
    args.rollout_config = args.rollout_config or root / "configs/recurrent-stability-rollout.json"
    args.ppo_config = args.ppo_config or root / "configs/recurrent-ppo-selected.json"
    curve = _load(
        args.learning_curve_config or root / "configs/learning-curve-baseline.json",
        LearningCurveConfig,
    )
    stability = curve.to_stability_config()
    resolved = load_resolved_config(root, args)
    validate_stability_resolved_config(resolved, stability)
    resolved["stability"] = stability.to_dict()

    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs/learning-curve")
    write_json_atomic(run / "config.json", resolved)
    write_json_atomic(run / "learning-curve-config.json", curve.to_dict())
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    for seed in curve.policy_seeds:
        seed_directory = run / f"seed-{seed:010d}"
        (seed_directory / "checkpoints").mkdir(parents=True)
        (seed_directory / "train/kit-logs").mkdir(parents=True)
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
        sustained_training_performed=False,
        deterministic_evaluation_performed=False,
        scientific_result=False,
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
        "Follow each seed's train/console.log and evaluation-update-*/console.log files.",
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

            train_output = seed_directory / "train"
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
            )
            active_container = command[command.index("--name") + 1]
            phase_started = time.perf_counter()
            exit_code = execute(command, train_output / "console.log", args.timeout)
            probe_path = train_output / "probe-result.json"
            metrics_path = train_output / "metrics.json"
            probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
            item["train"] = {
                "command": command,
                "exit_code": exit_code,
                "duration_seconds": time.perf_counter() - phase_started,
                "probe": probe,
                "metrics": metrics,
            }
            write_json_atomic(run / "report.json", report)
            if not valid_phase(exit_code, probe, metrics, "training_stability_tested"):
                raise RuntimeError(f"seed {seed} training phase failed")
            active_container = None

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
