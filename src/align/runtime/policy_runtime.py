"""Host launcher for the recurrent-policy tensor acceptance probe."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.policies.config import RecurrentPolicyConfig
from align.runtime.diagnostics import probe_source, probe_system
from align.runtime.drone_runtime import (
    capture,
    check_gpu_available,
    digest,
    docker_prefix,
    execute,
    finish,
    new_report,
    project_root,
)


def valid_policy_result(exit_code: int, metrics: object) -> bool:
    return (
        exit_code == 0
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and isinstance(metrics.get("checks"), dict)
        and bool(metrics["checks"])
        and all(value is True for value in metrics["checks"].values())
    )


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the recurrent policy with vendor PyTorch; no simulation or learning."
    )
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 30 <= args.timeout <= 900:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 30–900")

    root = project_root()
    config_path = args.config or root / "configs/recurrent-policy.json"
    config = RecurrentPolicyConfig.from_dict(json.loads(config_path.read_text()))
    config.validate_task_dimensions(
        actor_observation_dim=55,
        critic_state_dim=80,
        action_dim=4,
    )
    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs/recurrent-policy")
    write_json_atomic(run / "config.json", config.to_dict())
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        host_gpu_index=args.gpu,
        image_id=build["image_id"],
        source=probe_source(30).details,
        system=probe_system().details,
        config_sha256=digest(run / "config.json"),
        simulator_started=False,
        training_performed=False,
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
    container_name = f"align-recurrent-policy-{run.name.lower()}"
    write_json_atomic(run / "report.json", report)
    print(f"{report['started_at_ist']} IST Run: {run}\nFollow {run / 'console.log'}", flush=True)

    cleanup = False
    try:
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        command = [
            *docker,
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--name",
            container_name,
            "--runtime=nvidia",
            "--gpus",
            f"device={args.gpu}",
            "-e",
            "ACCEPT_EULA=Y",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "TZ=Asia/Kolkata",
            "--mount",
            f"type=bind,src={run},dst=/output",
            build["image_id"],
            "-m",
            "align.policies.probe",
            "--config",
            "/output/config.json",
            "--output",
            "/output",
            "--device",
            "cuda:0",
        ]
        report["command"] = command
        exit_code = execute(command, run / "console.log", args.timeout)
        metrics_path = run / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        report.update(
            container_exit_code=exit_code,
            metrics=metrics,
            status="passed" if valid_policy_result(exit_code, metrics) else "failed",
        )
        if report["status"] != "passed":
            raise RuntimeError("recurrent policy acceptance probe failed")
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        cleanup = True
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
        cleanup = True
    except (OSError, ValueError, RuntimeError) as exc:
        report.update(status="failed", error=str(exc))
        cleanup = True
    finally:
        if cleanup:
            report["cleanup"] = capture([*docker, "rm", "--force", container_name])
        report["gpu_after"] = capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv",
            ]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
