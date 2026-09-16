"""Host launcher for the live recurrent rollout collector acceptance run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.collector_config import CollectorProbeConfig
from align.learning.collector_report import audit_collector_run
from align.learning.rollout import RolloutConfig
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
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.observation import ObservationConfig
from align.tasks.reward import RewardConfig


def valid_collector_result(exit_code: int, probe: object, metrics: object, audit: object) -> bool:
    return (
        exit_code == 0
        and isinstance(probe, dict)
        and probe.get("status") == "passed"
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and probe.get("vector_task_physics_tested") is True
        and probe.get("recurrent_collector_tested") is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and bool(metrics.get("checks"))
        and all(value is True for value in metrics["checks"].values())
        and isinstance(audit, dict)
        and audit.get("status") == "passed"
        and all(value is True for value in audit.get("checks", {}).values())
    )


def _load(path: Path, constructor):
    return constructor.from_dict(json.loads(path.read_text()))


def load_resolved_config(root: Path, args) -> dict:
    construction = _load(
        args.construction_config or root / "configs/multi-drone-construction.json",
        MultiDroneConfig,
    )
    observation = _load(
        args.observation_config or root / "configs/local-observation-baseline.json",
        ObservationConfig,
    )
    reward = _load(
        args.reward_config or root / "configs/task-reward-baseline.json",
        RewardConfig,
    )
    task = _load(
        args.task_config or root / "configs/recurrent-collector-task.json",
        TaskEnvironmentConfig,
    )
    policy = _load(
        args.policy_config or root / "configs/recurrent-policy.json",
        RecurrentPolicyConfig,
    )
    rollout = _load(
        args.rollout_config or root / "configs/recurrent-rollout.json",
        RolloutConfig,
    )
    collector = _load(
        args.collector_config or root / "configs/recurrent-collector-probe.json",
        CollectorProbeConfig,
    )
    task.validate_compatibility(construction, observation, reward)
    rollout.validate_dimensions(
        num_envs=task.num_envs,
        num_agents=construction.num_agents,
        actor_observation_dim=observation.actor_dimension,
        critic_state_dim=observation.critic_dimension,
        action_dim=4,
    )
    policy.validate_task_dimensions(
        actor_observation_dim=rollout.actor_observation_dim,
        critic_state_dim=rollout.critic_state_dim,
        action_dim=rollout.action_dim,
    )
    if (policy.recurrent_layers, policy.recurrent_hidden_size) != (
        rollout.recurrent_layers,
        rollout.recurrent_hidden_size,
    ):
        raise ValueError("policy and rollout recurrent dimensions differ")
    collector.validate(horizon=rollout.horizon, num_envs=rollout.num_envs)
    return {
        "construction": construction.to_dict(),
        "observation": observation.to_dict(),
        "reward": reward.to_dict(),
        "task": task.to_dict(),
        "policy": policy.to_dict(),
        "rollout": rollout.to_dict(),
        "collector": collector.to_dict(),
    }


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect and audit one live recurrent rollout; no PPO update or training."
    )
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--construction-config", type=Path)
    parser.add_argument("--observation-config", type=Path)
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--rollout-config", type=Path)
    parser.add_argument("--collector-config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60–3600")

    root = project_root()
    resolved = load_resolved_config(root, args)
    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs/recurrent-collector")
    (run / "kit-logs").mkdir()
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
        training_performed=False,
        optimizer_updates=0,
        rendering_validated=False,
        video_validated=False,
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
    container_name = f"align-recurrent-collector-{run.name.lower()}"
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
            "--mount",
            f"type=bind,src={run / 'kit-logs'},dst=/isaac-sim/kit/logs",
            build["image_id"],
            "-m",
            "align.simulation.vector_task",
            "--config",
            "/output/config.json",
            "--output",
            "/output",
            "--scenario",
            "collector",
            "--num-envs",
            str(resolved["rollout"]["num_envs"]),
            "--allow-root",
        ]
        report["command"] = command
        exit_code = execute(command, run / "console.log", args.timeout)
        probe_path = run / "probe-result.json"
        metrics_path = run / "metrics.json"
        probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        audit = audit_collector_run(run) if metrics is not None else None
        if audit is not None:
            write_json_atomic(run / "host-audit.json", audit)
        passed = valid_collector_result(exit_code, probe, metrics, audit)
        report.update(
            container_exit_code=exit_code,
            probe_result=probe,
            metrics=metrics,
            host_audit=audit,
            status="passed" if passed else "failed",
        )
        if not passed:
            raise RuntimeError("live recurrent collector acceptance failed")
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
