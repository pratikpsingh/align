"""Launch a deterministic four-drone in-flight plane-to-pyramid command."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.formations import ShapeTransitionConfig
from align.formations.transition_report import make_transition_report
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import (
    SudoKeepalive,
    capture,
    check_gpu_available,
    digest,
    docker_prefix,
    execute,
    finish,
    new_report,
    project_root,
)
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.formation_schedule import FormationScheduleConfig
from align.tasks.observation import ObservationConfig
from align.tasks.reward import RewardConfig
from align.tasks.shape_transition_audit import audit_shape_transition


def transition_command(
    docker: list[str], *, run: Path, image_id: str, gpu: int, num_envs: int
) -> list[str]:
    if gpu < 0 or num_envs < 1:
        raise ValueError("GPU and environment count must be valid")
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-shape-transition-{run.name.lower()}",
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
        f"type=bind,src={run / 'evaluation' / 'kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        "/output/config.json",
        "--output",
        "/output/evaluation",
        "--scenario",
        "reference",
        "--num-envs",
        str(num_envs),
        "--shape-transition-config",
        "/output/transition-config.json",
        "--allow-root",
    ]


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument(
        "--construction-config", type=Path, default=Path("configs/feasible-plane-construction.json")
    )
    parser.add_argument(
        "--task-config", type=Path, default=Path("configs/shape-transition-task.json")
    )
    parser.add_argument(
        "--observation-config", type=Path, default=Path("configs/local-observation-baseline.json")
    )
    parser.add_argument(
        "--reward-config", type=Path, default=Path("configs/task-reward-baseline.json")
    )
    parser.add_argument(
        "--transition-config",
        type=Path,
        default=Path("configs/shape-transition-plane-pyramid.json"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 120 <= args.timeout <= 3600:
        parser.error("Require EULA, a nonnegative GPU, and timeout 120–3600")
    root = project_root()
    inputs = {
        "construction": args.construction_config,
        "task": args.task_config,
        "observation": args.observation_config,
        "reward": args.reward_config,
        "transition": args.transition_config,
    }
    values = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    construction = MultiDroneConfig.from_dict(values["construction"])
    task = TaskEnvironmentConfig.from_dict(values["task"])
    observation = ObservationConfig.from_dict(values["observation"])
    reward = RewardConfig.from_dict(values["reward"])
    transition = ShapeTransitionConfig.from_dict(values["transition"])
    task.validate_compatibility(construction, observation, reward)
    plan = make_transition_report(construction, task, transition)
    if construction.num_agents != 4 or task.num_envs != 4:
        raise ValueError("first physical shape-transition acceptance requires four drones/worlds")
    build_path = args.build_report.resolve()
    if build_path.parent.parent != (root / "runs/runtime-build").resolve():
        raise ValueError("build report must be in runs/runtime-build")
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise ValueError("derived image is not built")
    run = create_run_directory(root / "runs/shape-transition-reference")
    (run / "evaluation" / "kit-logs").mkdir(parents=True)
    bundle = {name: values[name] for name in ("construction", "observation", "reward", "task")}
    bundle["formation_schedule"] = FormationScheduleConfig(
        kinds=(transition.source_kind,), seed=construction.seed
    ).to_dict()
    write_json_atomic(run / "config.json", bundle)
    write_json_atomic(run / "transition-config.json", transition.to_dict())
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        image_id=build["image_id"],
        input_sha256={name: digest(path) for name, path in inputs.items()},
        config_sha256=digest(run / "config.json"),
        transition_config_sha256=digest(run / "transition-config.json"),
        plan=plan,
        host_gpu_index=args.gpu,
        num_envs=task.num_envs,
        checkpoint_loaded=False,
        training_performed=False,
        privileged_exact_target_and_pose=True,
        source=probe_source(30).details,
        system=probe_system().details,
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
    command = transition_command(
        docker, run=run, image_id=build["image_id"], gpu=args.gpu, num_envs=task.num_envs
    )
    report["command"] = command
    write_json_atomic(run / "report.json", report)
    print(f"{report['started_at_ist']} IST Shape transition reference: {run}", flush=True)
    keepalive = SudoKeepalive(args.docker == "sudo")
    try:
        check_gpu_available(report["gpu_before"], report["gpu_processes_before"], args.gpu)
        keepalive.start()
        if capture([*docker, "image", "inspect", build["image_id"]])["exit_code"]:
            raise RuntimeError("built runtime image is unavailable")
        report["container_exit_code"] = execute(command, run / "console.log", args.timeout)
        metrics_path = run / "evaluation" / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        report["metrics"] = metrics
        if (
            report["container_exit_code"] != 0
            or not isinstance(metrics, dict)
            or metrics.get("status") != "passed"
            or metrics.get("shape_transition_commands", 0) < task.num_envs
        ):
            raise RuntimeError("physical shape-transition reference probe failed")
        report["audit"] = audit_shape_transition(
            run / "evaluation" / "evaluation.csv",
            run / "evaluation" / "policy-telemetry.csv",
            plan,
        )
        if report["audit"]["environment_count"] != task.num_envs:
            raise ValueError("raw audit environment count differs")
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if report["status"] != "passed":
            report["container_cleanup"] = capture(
                [*docker, "rm", "--force", f"align-shape-transition-{run.name.lower()}"],
                timeout=30,
            )
        report["sudo_keepalive"] = keepalive.stop()
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
