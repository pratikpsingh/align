"""Host launcher for single and batched vector-task acceptance scenarios."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
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
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.simulation.vector_task_report import audit_saved_scenario
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.observation import ObservationConfig
from align.tasks.reward import RewardConfig


def valid_scenario(exit_code, probe, metrics, host_audit):
    return (
        exit_code == 0
        and isinstance(probe, dict)
        and probe.get("status") == "passed"
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and probe.get("vector_task_physics_tested") is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and bool(metrics.get("checks"))
        and all(value is True for value in metrics["checks"].values())
        and host_audit.get("status") == "passed"
        and all(value is True for value in host_audit["checks"].values())
    )


def load_resolved_config(root, args):
    construction = MultiDroneConfig.from_dict(
        json.loads(
            (args.construction_config or root / "configs/multi-drone-construction.json").read_text()
        )
    )
    observation = ObservationConfig.from_dict(
        json.loads(
            (
                args.observation_config or root / "configs/local-observation-baseline.json"
            ).read_text()
        )
    )
    reward = RewardConfig.from_dict(
        json.loads((args.reward_config or root / "configs/task-reward-baseline.json").read_text())
    )
    task = TaskEnvironmentConfig.from_dict(
        json.loads((args.task_config or root / "configs/vector-task-probe.json").read_text())
    )
    task.validate_compatibility(construction, observation, reward)
    return {
        "construction": construction.to_dict(),
        "observation": observation.to_dict(),
        "reward": reward.to_dict(),
        "task": task.to_dict(),
    }


def run_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one and four cloned OmniDrones task environments; no learning."
    )
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--construction-config", type=Path)
    parser.add_argument("--observation-config", type=Path)
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60–3600")

    root = project_root()
    resolved = load_resolved_config(root, args)
    batch_size = resolved["task"]["num_envs"]
    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs/vector-task")
    write_json_atomic(run / "config.json", resolved)
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
        rendering_validated=False,
        video_validated=False,
        scenarios={},
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
        f"Follow {run / 'single/console.log'} then {run / 'batch/console.log'}",
        flush=True,
    )

    active_name = None
    cleanup = False
    try:
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for scenario, count in (("single", 1), ("batch", batch_size)):
            scenario_run = run / scenario
            scenario_run.mkdir()
            (scenario_run / "kit-logs").mkdir()
            shutil.copy2(run / "config.json", scenario_run / "config.json")
            active_name = f"align-vector-task-{scenario}-{run.name.lower()}"
            command = [
                *docker,
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--name",
                active_name,
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
                f"type=bind,src={scenario_run},dst=/output",
                "--mount",
                f"type=bind,src={scenario_run / 'kit-logs'},dst=/isaac-sim/kit/logs",
                build["image_id"],
                "-m",
                "align.simulation.vector_task",
                "--config",
                "/output/config.json",
                "--output",
                "/output",
                "--scenario",
                scenario,
                "--num-envs",
                str(count),
                "--allow-root",
            ]
            exit_code = execute(command, scenario_run / "console.log", args.timeout)
            probe_path = scenario_run / "probe-result.json"
            metrics_path = scenario_run / "metrics.json"
            probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
            host_audit = audit_saved_scenario(scenario_run, count) if metrics is not None else None
            if host_audit is not None:
                write_json_atomic(scenario_run / "host-audit.json", host_audit)
            passed = valid_scenario(exit_code, probe, metrics, host_audit or {})
            report["scenarios"][scenario] = {
                "num_envs": count,
                "command": command,
                "container_exit_code": exit_code,
                "probe_result": probe,
                "host_audit": host_audit,
                "status": "passed" if passed else "failed",
            }
            write_json_atomic(run / "report.json", report)
            if not passed:
                raise RuntimeError(f"{scenario} vector-task scenario failed")
            active_name = None
        report["status"] = "passed"
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
        if cleanup and active_name:
            report["cleanup"] = capture([*docker, "rm", "--force", active_name])
        report["gpu_after"] = capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv",
            ]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
