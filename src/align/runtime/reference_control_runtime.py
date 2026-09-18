"""Run the exact-target reference controller under matched task schedules."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.stability_config import StabilityConfig
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
from align.runtime.stability_runtime import valid_phase
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.evaluation_timing import (
    EvaluationTimingConfig,
    apply_evaluation_timing,
    compare_shared_prefix,
    formation_switch_altitude,
    summarize_evaluation_phases,
)
from align.tasks.policy_telemetry import audit_telemetry, summarize_telemetry


def reference_command(
    docker: list[str],
    *,
    source: Path,
    run: Path,
    image_id: str,
    gpu: int,
    seed: int,
    num_envs: int,
    arm: str,
) -> list[str]:
    if arm not in {"baseline", "extended"}:
        raise ValueError("unknown reference arm")
    label = f"seed-{seed:010d}"
    command = [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-reference-{run.name.lower()}-{arm}",
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
        f"type=bind,src={source},dst=/source,readonly",
        "--mount",
        f"type=bind,src={run},dst=/output",
        "--mount",
        f"type=bind,src={run / arm / 'kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        f"/source/{label}/config.json",
        "--output",
        f"/output/{arm}",
        "--scenario",
        "reference",
        "--num-envs",
        str(num_envs),
        "--allow-root",
    ]
    if arm == "extended":
        command.extend(("--evaluation-timing-config", "/output/timing-config.json"))
    return command


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate exact-target proportional control under two task schedules"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--timing-config", type=Path)
    parser.add_argument("--policy-seed", type=int, default=41)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if (
        not args.accept_eula
        or args.policy_seed < 0
        or args.gpu < 0
        or not 60 <= args.timeout <= 3600
    ):
        parser.error("Require EULA, nonnegative seed/GPU, and timeout 60–3600")
    root = project_root()
    source = args.source_run.resolve()
    if source.parent not in {
        (root / "runs/plane-baseline-training").resolve(),
        (root / "runs/multi-template-training").resolve(),
    }:
        raise ValueError("source-run must be a direct child of a passed training arm")
    source_report = json.loads((source / "report.json").read_text())
    if source_report["status"] != "passed":
        raise ValueError("source training run must have passed")
    if args.policy_seed not in {item["policy_seed"] for item in source_report["seeds"]}:
        raise ValueError("seed is absent from source run")
    source_config_path = source / f"seed-{args.policy_seed:010d}/config.json"
    resolved = json.loads(source_config_path.read_text())
    construction = MultiDroneConfig.from_dict(resolved["construction"])
    task = TaskEnvironmentConfig.from_dict(resolved["task"])
    stability = StabilityConfig.from_dict(resolved["stability"])
    timing_path = (args.timing_config or root / "configs/evaluation-timing-extended.json").resolve()
    timing = EvaluationTimingConfig.from_dict(json.loads(timing_path.read_text()))
    _, effective_task, _, timing_details = apply_evaluation_timing(
        timing, construction, task, stability
    )
    build_path = args.build_report.resolve()
    if build_path.parent.parent != (root / "runs/runtime-build").resolve():
        raise ValueError("build-report must come from runs/runtime-build")
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise ValueError("derived runtime image must be built")
    run = create_run_directory(root / "runs/reference-control")
    for arm in ("baseline", "extended"):
        (run / arm / "kit-logs").mkdir(parents=True)
    write_json_atomic(run / "timing-config.json", timing.to_dict())
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        source_run=str(source),
        source_run_id=source.name,
        source_config_sha256=digest(source_config_path),
        image_id=build["image_id"],
        timing_config=timing.to_dict(),
        timing_config_sha256=digest(run / "timing-config.json"),
        timing_details=timing_details,
        baseline_evaluation_steps=stability.evaluation_steps,
        extended_evaluation_steps=effective_task.max_episode_steps,
        reference_controller="target_proportional_world_velocity",
        reference_position_gain_s_inv=construction.position_gain_s_inv,
        max_speed_m_s=construction.max_speed_m_s,
        policy_seed=args.policy_seed,
        checkpoint_loaded=False,
        training_performed=False,
        optimizer_updates=0,
        privileged_exact_target_and_pose=True,
        scientific_result=False,
        source=probe_source(30).details,
        system=probe_system().details,
        host_gpu_index=args.gpu,
        arms={},
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
    print(f"{report['started_at_ist']} IST Reference control: {run}", flush=True)
    keepalive = SudoKeepalive(args.docker == "sudo")
    active_container = None
    try:
        keepalive.start()
        if capture([*docker, "image", "inspect", build["image_id"]])["exit_code"]:
            raise RuntimeError("built runtime image is unavailable locally")
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for arm in ("baseline", "extended"):
            output = run / arm
            command = reference_command(
                docker,
                source=source,
                run=run,
                image_id=build["image_id"],
                gpu=args.gpu,
                seed=args.policy_seed,
                num_envs=task.num_envs,
                arm=arm,
            )
            arm_report = {"command": command, "status": "running"}
            report["arms"][arm] = arm_report
            write_json_atomic(run / "report.json", report)
            active_container = command[command.index("--name") + 1]
            arm_started = time.perf_counter()
            arm_report["exit_code"] = execute(command, output / "console.log", args.timeout)
            active_container = None
            arm_report["duration_seconds"] = time.perf_counter() - arm_started
            probe_path, metrics_path = output / "probe-result.json", output / "metrics.json"
            arm_report["probe"] = (
                json.loads(probe_path.read_text()) if probe_path.exists() else None
            )
            arm_report["metrics"] = (
                json.loads(metrics_path.read_text()) if metrics_path.exists() else None
            )
            if not valid_phase(
                arm_report["exit_code"],
                arm_report["probe"],
                arm_report["metrics"],
                "reference_evaluation_tested",
            ):
                raise RuntimeError(f"{arm} reference run failed; inspect {output / 'console.log'}")
            metrics = arm_report["metrics"]
            expected_steps = (
                stability.evaluation_steps
                if arm == "baseline"
                else effective_task.max_episode_steps
            )
            if (
                metrics["evaluation_steps"] != expected_steps
                or metrics["optimizer_updates"] != 0
                or arm_report["probe"].get("evaluation_timing")
                != (None if arm == "baseline" else timing_details)
            ):
                raise RuntimeError(f"{arm} reference timing or optimizer contract changed")
            arm_report["audit"] = audit_telemetry(
                output / "policy-telemetry.csv",
                output / "evaluation.csv",
                num_agents=construction.num_agents,
                max_speed_m_s=construction.max_speed_m_s,
                reward_config=resolved["reward"],
            )
            arm_report["trace_summary"] = (
                summarize_telemetry(
                    output / "policy-telemetry.csv",
                    max_speed_m_s=construction.max_speed_m_s,
                    control_dt_seconds=construction.physics_dt,
                    max_episode_steps=expected_steps,
                    success_dwell_steps=task.success_dwell_steps,
                )
                if metrics["formation_phase_reached"]
                else None
            )
            arm_report["phase_metrics"] = (
                summarize_evaluation_phases(output / "evaluation.csv")
                if metrics["formation_phase_reached"]
                else None
            )
            arm_report["formation_switch"] = (
                formation_switch_altitude(output / "policy-telemetry.csv")
                if metrics["formation_phase_reached"]
                else None
            )
            arm_report["status"] = "passed"
            write_json_atomic(run / "report.json", report)
        report["shared_prefix"] = compare_shared_prefix(
            run / "baseline/policy-telemetry.csv",
            run / "extended/policy-telemetry.csv",
            divergence_step=construction.ground_steps + construction.takeoff_steps,
            num_envs=task.num_envs,
            num_agents=construction.num_agents,
        )
        report["comparison"] = {
            "baseline_successes": report["arms"]["baseline"]["metrics"]["outcome_counts"]["1"],
            "extended_successes": report["arms"]["extended"]["metrics"]["outcome_counts"]["1"],
            "baseline_safety_terminations": sum(
                report["arms"]["baseline"]["metrics"]["outcome_counts"][str(code)]
                for code in (2, 3, 4)
            ),
            "extended_safety_terminations": sum(
                report["arms"]["extended"]["metrics"]["outcome_counts"][str(code)]
                for code in (2, 3, 4)
            ),
        }
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if active_container is not None:
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
