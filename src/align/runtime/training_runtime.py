"""Host launcher for the two-process task-connected training acceptance run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.recovery_config import RecoveryConfig
from align.learning.rollout import RolloutConfig
from align.learning.training_config import TaskTrainingConfig
from align.learning.training_report import audit_training_run, write_combined_tables
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
        args.task_config or root / "configs/recurrent-training-task.json",
        TaskEnvironmentConfig,
    )
    policy = _load(
        args.policy_config or root / "configs/recurrent-policy.json",
        RecurrentPolicyConfig,
    )
    rollout = _load(
        args.rollout_config or root / "configs/recurrent-training-rollout.json",
        RolloutConfig,
    )
    ppo = _load(args.ppo_config or root / "configs/recurrent-ppo.json", RecurrentPPOConfig)
    recovery = _load(
        args.recovery_config or root / "configs/training-recovery.json", RecoveryConfig
    )
    training = _load(
        args.training_config or root / "configs/task-training.json", TaskTrainingConfig
    )
    critic_normalization = _load(
        getattr(args, "critic_normalization_config", None)
        or root / "configs/critic-normalization-disabled.json",
        CriticNormalizationConfig,
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
    if rollout.horizon >= task.max_episode_steps:
        raise ValueError("training rollout must end before the task time limit for reset evidence")
    return {
        "construction": construction.to_dict(),
        "observation": observation.to_dict(),
        "reward": reward.to_dict(),
        "task": task.to_dict(),
        "policy": policy.to_dict(),
        "rollout": rollout.to_dict(),
        "ppo": ppo.to_dict(),
        "recovery": recovery.to_dict(),
        "training": training.to_dict(),
        "critic_normalization": critic_normalization.to_dict(),
    }


def valid_attempt(exit_code: int, probe: object, metrics: object) -> bool:
    return (
        exit_code == 0
        and isinstance(probe, dict)
        and probe.get("status") == "passed"
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and probe.get("vector_task_physics_tested") is True
        and probe.get("task_training_tested") is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and bool(metrics.get("checks"))
        and all(value is True for value in metrics["checks"].values())
    )


def valid_training_result(attempts: list[dict], audit: object) -> bool:
    return (
        len(attempts) == 2
        and all(
            valid_attempt(item["exit_code"], item.get("probe"), item.get("metrics"))
            for item in attempts
        )
        and isinstance(audit, dict)
        and audit.get("status") == "passed"
        and bool(audit.get("checks"))
        and all(value is True for value in audit["checks"].values())
    )


def _attempt_command(
    docker,
    *,
    run,
    attempt,
    image_id,
    gpu,
    num_envs,
    source_identity,
    resume,
):
    attempt_id = f"attempt-{attempt:04d}"
    command = [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-task-training-{run.name.lower()}-{attempt:04d}",
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
        f"type=bind,src={run / attempt_id / 'kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        "/output/config.json",
        "--output",
        f"/output/{attempt_id}",
        "--scenario",
        "training",
        "--num-envs",
        str(num_envs),
        "--logical-run-id",
        run.name,
        "--attempt-id",
        attempt_id,
        "--checkpoint-directory",
        "/output/checkpoints",
        "--source-identity",
        source_identity,
        "--runtime-identity",
        image_id,
        "--allow-root",
    ]
    if resume:
        command.append("--resume")
    return command


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run two bounded task-connected PPO attempts with reset-mode recovery."
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
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60–3600")

    root = project_root()
    resolved = load_resolved_config(root, args)
    if resolved["training"]["attempts"] != 2:
        raise ValueError("task-training acceptance requires exactly two attempts")
    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")
    run = create_run_directory(root / "runs/task-training")
    (run / "checkpoints").mkdir()
    for number in (1, 2):
        attempt = run / f"attempt-{number:04d}"
        attempt.mkdir()
        (attempt / "kit-logs").mkdir()
    write_json_atomic(run / "config.json", resolved)
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
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
        config_sha256=digest(run / "config.json"),
        recovery_mode=resolved["recovery"]["recovery_mode"],
        bounded_training_performed=True,
        sustained_training_performed=False,
        scientific_result=False,
        planned_attempts=2,
        planned_optimizer_updates=2,
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
    report["attempts"] = []
    write_json_atomic(run / "report.json", report)
    print(
        f"{report['started_at_ist']} IST Run: {run}\n"
        f"Follow {run / 'attempt-0001/console.log'} then "
        f"{run / 'attempt-0002/console.log'}",
        flush=True,
    )

    active_container = None
    cleanup = False
    try:
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for number in (1, 2):
            attempt_id = f"attempt-{number:04d}"
            active_container = f"align-task-training-{run.name.lower()}-{number:04d}"
            command = _attempt_command(
                docker,
                run=run,
                attempt=number,
                image_id=build["image_id"],
                gpu=args.gpu,
                num_envs=resolved["rollout"]["num_envs"],
                source_identity=source_identity,
                resume=number == 2,
            )
            attempt_started = time.perf_counter()
            exit_code = execute(command, run / attempt_id / "console.log", args.timeout)
            probe_path = run / attempt_id / "probe-result.json"
            metrics_path = run / attempt_id / "metrics.json"
            probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
            item = {
                "attempt_id": attempt_id,
                "resume": number == 2,
                "command": command,
                "exit_code": exit_code,
                "duration_seconds": time.perf_counter() - attempt_started,
                "probe": probe,
                "metrics": metrics,
            }
            report["attempts"].append(item)
            write_json_atomic(run / "report.json", report)
            if not valid_attempt(exit_code, probe, metrics):
                raise RuntimeError(f"{attempt_id} task-connected training failed")
            active_container = None
        audit = audit_training_run(run)
        write_json_atomic(run / "host-audit.json", audit)
        write_combined_tables(run)
        passed = valid_training_result(report["attempts"], audit)
        report.update(
            host_audit=audit,
            optimizer_updates=audit["final_counters"]["completed_updates"],
            environment_transitions=audit["final_counters"]["environment_transitions"],
            agent_transitions=audit["final_counters"]["agent_transitions"],
            status="passed" if passed else "failed",
        )
        if not passed:
            raise RuntimeError("task-connected training host audit failed")
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
