"""Host launcher for matched-rollout critic learning-rate calibration."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.critic_calibration_config import CriticCalibrationConfig
from align.learning.training_config import TaskTrainingConfig
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import (
    capture,
    check_gpu_available,
    docker_prefix,
    execute,
    finish,
    new_report,
    project_root,
)
from align.runtime.training_runtime import _load, load_resolved_config
from align.simulation.multi_drone_contract import MultiDroneConfig


def valid_critic_result(exit_code: int, probe: object, metrics: object) -> bool:
    calibration = metrics.get("critic_calibration") if isinstance(metrics, dict) else None
    return (
        exit_code == 0
        and isinstance(probe, dict)
        and probe.get("status") == "passed"
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and probe.get("vector_task_physics_tested") is True
        and probe.get("critic_calibration_tested") is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and isinstance(calibration, dict)
        and calibration.get("status") == "passed"
        and bool(calibration.get("checks"))
        and all(value is True for value in calibration["checks"].values())
    )


def summarize_critic_results(items: list[dict], config: CriticCalibrationConfig) -> dict:
    by_rate = {rate: [] for rate in config.candidate_critic_learning_rates}
    input_ranges = {}
    critic_group_ranges = {}
    relationship_ranges = {}
    for item in items:
        calibration = item["metrics"]["critic_calibration"]
        for name, values in calibration["input_distributions"].items():
            input_ranges.setdefault(name, []).append(values)
        for name, values in calibration["critic_inputs"]["groups"].items():
            critic_group_ranges.setdefault(name, []).append(values)
        for name, values in calibration["source_relationships"].items():
            relationship_ranges.setdefault(name, []).append(values)
        for candidate in calibration["candidates"]:
            by_rate[candidate["critic_learning_rate"]].append(candidate)

    candidates = []
    passing_all_seeds = []
    for rate in config.candidate_critic_learning_rates:
        rows = by_rate[rate]
        if len(rows) != len(config.policy_seeds):
            raise ValueError(f"missing candidate result for learning rate {rate}")

        def bounds(key, candidate_rows=rows):
            values = [float(row[key]) for row in candidate_rows]
            return {
                "minimum": min(values),
                "maximum": max(values),
                "mean": sum(values) / len(values),
            }

        passes = all(
            row["guidance"]["pre_predictions_reproduce_rollout_values"]
            and row["post_value_clip_fraction"] <= config.max_value_clip_fraction
            for row in rows
        )
        if passes:
            passing_all_seeds.append(rate)
        candidates.append(
            {
                "critic_learning_rate": rate,
                "passes_value_clip_guidance_on_all_seeds": passes,
                "post_value_clip_fraction": bounds("post_value_clip_fraction"),
                "pre_prediction_max_abs_error": bounds("pre_prediction_max_abs_error"),
                "critic_gradient_norm_before_clip": bounds("critic_gradient_norm_before_clip"),
                "critic_parameter_max_abs_change": bounds("critic_parameter_max_abs_change"),
                "absolute_value_delta_p95": {
                    "minimum": min(row["absolute_value_delta"]["p95"] for row in rows),
                    "maximum": max(row["absolute_value_delta"]["p95"] for row in rows),
                    "mean": sum(row["absolute_value_delta"]["p95"] for row in rows) / len(rows),
                },
            }
        )
    return {
        "seed_count": len(items),
        "seeds": [item["policy_seed"] for item in items],
        "matched_rollout_within_each_seed": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "rates_passing_value_clip_guidance_on_all_seeds": passing_all_seeds,
        "largest_rate_passing_value_clip_guidance_on_all_seeds": (
            passing_all_seeds[0] if passing_all_seeds else None
        ),
        "input_distribution_ranges": {
            name: {
                "mean_minimum": min(row["mean"] for row in rows),
                "mean_maximum": max(row["mean"] for row in rows),
                "standard_deviation_minimum": min(row["standard_deviation"] for row in rows),
                "standard_deviation_maximum": max(row["standard_deviation"] for row in rows),
            }
            for name, rows in input_ranges.items()
        },
        "critic_input_group_ranges": {
            name: {
                "mean_minimum": min(row["mean"] for row in rows),
                "mean_maximum": max(row["mean"] for row in rows),
                "standard_deviation_minimum": min(row["standard_deviation"] for row in rows),
                "standard_deviation_maximum": max(row["standard_deviation"] for row in rows),
            }
            for name, rows in critic_group_ranges.items()
        },
        "source_relationship_ranges": {
            name: {
                metric: {
                    "minimum": min(row[metric] for row in rows),
                    "maximum": max(row[metric] for row in rows),
                    "mean": sum(row[metric] for row in rows) / len(rows),
                }
                for metric in (
                    "pearson_correlation",
                    "mean_error",
                    "mean_absolute_error",
                    "root_mean_squared_error",
                )
            }
            for name, rows in relationship_ranges.items()
        },
    }


def _feature_csv_is_valid(path: Path, expected_rows: int) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows or {int(row["index"]) for row in rows} != set(
        range(expected_rows)
    ):
        return False
    distribution = (
        "count",
        "minimum",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "maximum",
        "mean",
        "standard_deviation",
        "zero_fraction",
        "boundary_fraction",
    )
    for row in rows:
        active = int(row["active_sample_count"])
        if (
            active < 0
            or not row["name"]
            or row["group"]
            not in {
                "position",
                "velocity",
                "target",
                "mask",
            }
        ):
            return False
        if active:
            if int(row["count"]) != active or not all(
                math.isfinite(float(row[name])) for name in distribution[1:]
            ):
                return False
            if (
                not 0.0 <= float(row["zero_fraction"]) <= 1.0
                or not 0.0 <= float(row["boundary_fraction"]) <= 1.0
            ):
                return False
        elif any(row[name] for name in distribution):
            return False
    return True


def _raw_csv_is_valid(path: Path, expected_rows: int) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "critic_learning_rate",
        "team_reward",
        "old_value",
        "return",
        "advantage",
        "pre_prediction",
        "post_prediction",
        "value_delta",
        "absolute_value_delta",
    )
    return len(rows) == expected_rows and all(
        all(math.isfinite(float(row[name])) for name in numeric) for row in rows
    )


def _command(
    docker,
    *,
    run: Path,
    seed_directory: Path,
    image_id: str,
    gpu: int,
    num_envs: int,
    source_identity: str,
):
    relative = seed_directory.relative_to(run)
    logical_run_id = f"{run.name}-seed-{relative.name.split('-')[-1]}"
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-critic-calibration-{run.name.lower()}-{relative.name}",
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
        f"type=bind,src={seed_directory / 'probe/kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        f"/output/{relative}/config.json",
        "--output",
        f"/output/{relative}/probe",
        "--scenario",
        "critic-calibration",
        "--num-envs",
        str(num_envs),
        "--logical-run-id",
        logical_run_id,
        "--attempt-id",
        "critic-calibration",
        "--checkpoint-directory",
        f"/output/{relative}/checkpoints",
        "--source-identity",
        source_identity,
        "--runtime-identity",
        image_id,
        "--allow-root",
    ]


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare critic learning rates on matched live-task rollouts."
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
    parser.add_argument("--critic-calibration-config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60–3600")

    root = project_root()
    args.task_config = args.task_config or root / "configs/recurrent-stability-task.json"
    args.rollout_config = args.rollout_config or root / "configs/recurrent-stability-rollout.json"
    args.ppo_config = args.ppo_config or root / "configs/recurrent-ppo-calibration.json"
    calibration = _load(
        args.critic_calibration_config or root / "configs/critic-step-calibration.json",
        CriticCalibrationConfig,
    )
    resolved = load_resolved_config(root, args)
    construction = MultiDroneConfig.from_dict(resolved["construction"])
    if resolved["rollout"]["horizon"] <= construction.ground_steps + construction.takeoff_steps:
        raise ValueError("critic calibration rollout must reach the formation phase")
    if resolved["ppo"]["update_epochs"] != 1:
        raise ValueError("critic calibration requires one PPO epoch")
    if not math.isclose(
        calibration.candidate_critic_learning_rates[0],
        resolved["ppo"]["critic_learning_rate"],
    ):
        raise ValueError("first candidate must match the configured primary critic rate")
    resolved["critic_calibration"] = calibration.to_dict()

    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs/critic-calibration")
    write_json_atomic(run / "config.json", resolved)
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    for seed in calibration.policy_seeds:
        seed_directory = run / f"seed-{seed:010d}"
        (seed_directory / "checkpoints").mkdir(parents=True)
        (seed_directory / "probe/kit-logs").mkdir(parents=True)
        seed_config = copy.deepcopy(resolved)
        seed_config["training"]["attempts"] = 1
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
        critic_calibration_config=calibration.to_dict(),
        bounded_training_performed=True,
        sustained_training_performed=False,
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
        f"{report['started_at_ist']} IST Run: {run}\nFollow each seed's probe/console.log.",
        flush=True,
    )

    active_container = None
    cleanup = False
    try:
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for seed in calibration.policy_seeds:
            seed_directory = run / f"seed-{seed:010d}"
            command = _command(
                docker,
                run=run,
                seed_directory=seed_directory,
                image_id=build["image_id"],
                gpu=args.gpu,
                num_envs=resolved["rollout"]["num_envs"],
                source_identity=source_identity,
            )
            active_container = command[command.index("--name") + 1]
            seed_started = time.perf_counter()
            exit_code = execute(command, seed_directory / "probe/console.log", args.timeout)
            probe_path = seed_directory / "probe/probe-result.json"
            metrics_path = seed_directory / "probe/metrics.json"
            probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
            expected_rows = (
                resolved["rollout"]["horizon"]
                * resolved["rollout"]["num_envs"]
                * len(calibration.candidate_critic_learning_rates)
            )
            raw_valid = (
                _raw_csv_is_valid(seed_directory / "probe/critic-samples.csv", expected_rows)
                if (seed_directory / "probe/critic-samples.csv").exists()
                else False
            )
            feature_csv = seed_directory / "probe/critic-feature-summary.csv"
            features_valid = (
                _feature_csv_is_valid(feature_csv, resolved["policy"]["critic_state_dim"])
                if feature_csv.exists()
                else False
            )
            item = {
                "policy_seed": seed,
                "command": command,
                "exit_code": exit_code,
                "duration_seconds": time.perf_counter() - seed_started,
                "probe": probe,
                "metrics": metrics,
                "host_raw_csv_valid": raw_valid,
                "host_feature_csv_valid": features_valid,
            }
            report["seeds"].append(item)
            write_json_atomic(run / "report.json", report)
            if (
                not valid_critic_result(exit_code, probe, metrics)
                or not raw_valid
                or not features_valid
            ):
                raise RuntimeError(f"seed {seed} critic calibration failed")
            active_container = None

        summary = summarize_critic_results(report["seeds"], calibration)
        write_json_atomic(run / "critic-summary.json", summary)
        report.update(status="passed", summary=summary)
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
        report["gpu_after"] = capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv",
            ]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
