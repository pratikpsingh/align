"""Host launcher for multi-seed PPO stability calibration and evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.stability_config import StabilityConfig
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


def validate_stability_resolved_config(resolved: dict, stability: StabilityConfig) -> int:
    """Validate phase coverage using serialized construction inputs."""
    if stability.evaluation_steps != resolved["task"]["max_episode_steps"]:
        raise ValueError("evaluation_steps must equal the task time limit")
    construction = MultiDroneConfig.from_dict(resolved["construction"])
    formation_start_step = construction.ground_steps + construction.takeoff_steps
    if resolved["rollout"]["horizon"] <= formation_start_step:
        raise ValueError("stability rollout must reach the formation phase")
    return formation_start_step


def valid_phase(exit_code: int, probe: object, metrics: object, flag: str) -> bool:
    return (
        exit_code == 0
        and isinstance(probe, dict)
        and probe.get("status") == "passed"
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and probe.get("vector_task_physics_tested") is True
        and probe.get(flag) is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and bool(metrics.get("checks"))
        and all(value is True for value in metrics["checks"].values())
    )


def summarize_seed_results(items: list[dict]) -> dict:
    evaluations = [item["evaluation"]["metrics"]["measurements"] for item in items]
    updates = [row for item in items for row in item["train"]["metrics"]["measurements"]]

    def bounds(rows, key):
        values = [float(row[key]) for row in rows]
        return {"minimum": min(values), "maximum": max(values), "mean": sum(values) / len(values)}

    return {
        "seed_count": len(items),
        "seeds": [item["policy_seed"] for item in items],
        "update_count": len(updates),
        "post_update_approximate_kl": bounds(updates, "post_approximate_kl"),
        "post_update_policy_clip_fraction": bounds(updates, "post_policy_clip_fraction"),
        "post_update_value_clip_fraction": bounds(updates, "post_value_clip_fraction"),
        "critic_gradient_norm_before_clip": bounds(updates, "max_critic_gradient_norm_before_clip"),
        "evaluation_team_reward_mean": bounds(evaluations, "team_reward_mean"),
        "evaluation_assigned_rmse_mean_m": bounds(evaluations, "assigned_rmse_mean_m"),
        "evaluation_pairwise_rmse_mean_m": bounds(evaluations, "pairwise_rmse_mean_m"),
        "evaluation_minimum_separation_m": bounds(evaluations, "minimum_separation_m"),
    }


def _command(
    docker,
    *,
    run: Path,
    seed_directory: Path,
    image_id: str,
    gpu: int,
    num_envs: int,
    source_identity: str,
    scenario: str,
):
    phase = "train" if scenario == "stability" else "evaluation"
    relative = seed_directory.relative_to(run)
    logical_run_id = f"{run.name}-seed-{relative.name.split('-')[-1]}"
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-stability-{run.name.lower()}-{relative.name}-{phase}",
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
        (f"type=bind,src={seed_directory / phase / 'kit-logs'},dst=/isaac-sim/kit/logs"),
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        f"/output/{relative}/config.json",
        "--output",
        f"/output/{relative}/{phase}",
        "--scenario",
        scenario,
        "--num-envs",
        str(num_envs),
        "--logical-run-id",
        logical_run_id,
        "--attempt-id",
        phase,
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
        description="Run bounded multi-seed PPO calibration and fresh-process evaluation."
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
    parser.add_argument("--stability-config", type=Path)
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
    stability = _load(
        args.stability_config or root / "configs/training-stability.json",
        StabilityConfig,
    )
    resolved = load_resolved_config(root, args)
    validate_stability_resolved_config(resolved, stability)
    resolved["stability"] = stability.to_dict()

    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs/training-stability")
    write_json_atomic(run / "config.json", resolved)
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    for seed in stability.policy_seeds:
        seed_directory = run / f"seed-{seed:010d}"
        (seed_directory / "checkpoints").mkdir(parents=True)
        for phase in ("train", "evaluation"):
            (seed_directory / phase / "kit-logs").mkdir(parents=True)
        seed_config = copy.deepcopy(resolved)
        seed_config["training"]["attempts"] = stability.updates_per_seed
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
        stability_config=stability.to_dict(),
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
        f"Follow per-seed train/console.log and evaluation/console.log files.",
        flush=True,
    )

    active_container = None
    cleanup = False
    try:
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for seed in stability.policy_seeds:
            seed_directory = run / f"seed-{seed:010d}"
            item = {"policy_seed": seed}
            report["seeds"].append(item)
            for scenario, phase, flag in (
                ("stability", "train", "training_stability_tested"),
                ("evaluation", "evaluation", "deterministic_evaluation_tested"),
            ):
                command = _command(
                    docker,
                    run=run,
                    seed_directory=seed_directory,
                    image_id=build["image_id"],
                    gpu=args.gpu,
                    num_envs=resolved["rollout"]["num_envs"],
                    source_identity=source_identity,
                    scenario=scenario,
                )
                active_container = command[command.index("--name") + 1]
                phase_started = time.perf_counter()
                exit_code = execute(command, seed_directory / phase / "console.log", args.timeout)
                probe_path = seed_directory / phase / "probe-result.json"
                metrics_path = seed_directory / phase / "metrics.json"
                probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
                metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
                phase_result = {
                    "command": command,
                    "exit_code": exit_code,
                    "duration_seconds": time.perf_counter() - phase_started,
                    "probe": probe,
                    "metrics": metrics,
                }
                item[phase] = phase_result
                write_json_atomic(run / "report.json", report)
                if not valid_phase(exit_code, probe, metrics, flag):
                    raise RuntimeError(f"seed {seed} {phase} phase failed")
                active_container = None
            write_json_atomic(run / "report.json", report)

        summary = summarize_seed_results(report["seeds"])
        write_json_atomic(run / "stability-summary.json", summary)
        report.update(
            status="passed",
            deterministic_evaluation_performed=True,
            optimizer_updates=summary["update_count"],
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
        report["gpu_after"] = capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv",
            ]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
