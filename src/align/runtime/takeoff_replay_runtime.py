"""Host launcher for matched command and downwash takeoff interventions."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
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
from align.runtime.policy_telemetry_runtime import valid_early_contact_capture
from align.tasks.takeoff_replay import (
    ARMS,
    REPLAY_ARM_CHOICES,
    assess_replays,
    assess_wake_guard,
    prepare_trace,
    save_replay_plot,
)


def replay_command(docker: list[str], *, run: Path, arm: str, image_id: str, gpu: int) -> list[str]:
    if arm not in REPLAY_ARM_CHOICES:
        raise ValueError("unknown replay arm")
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-takeoff-{run.name.lower()}-{arm}",
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
        f"type=bind,src={run / arm / 'kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        "/output/config.json",
        "--output",
        f"/output/{arm}",
        "--scenario",
        "takeoff-replay",
        "--num-envs",
        "1",
        "--replay-trace",
        "/output/commands.csv",
        "--replay-arm",
        arm,
        "--allow-root",
    ]


def validate_baseline_report(
    baseline_report: dict,
    *,
    source_run_id: str,
    source_telemetry_sha256: str,
    config_sha256: str,
    checkpoint_id: str,
    trace_sha256: str,
) -> None:
    """Require an accepted baseline for the same immutable command experiment."""
    if (
        baseline_report.get("status") != "passed"
        or baseline_report.get("source_run_id") != source_run_id
        or baseline_report.get("source_telemetry_sha256") != source_telemetry_sha256
        or baseline_report.get("config_sha256") != config_sha256
        or baseline_report.get("source_checkpoint_id") != checkpoint_id
        or baseline_report.get("command_trace_sha256") != trace_sha256
        or not baseline_report.get("comparison", {}).get(
            "baseline_reproduced_with_declared_tolerance"
        )
    ):
        raise ValueError("baseline run does not match the accepted source replay")


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-per-arm", type=int, default=900)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--include-wake-guard", action="store_true")
    parser.add_argument("--baseline-run", type=Path)
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout_per_arm <= 3600:
        parser.error("Require EULA, a nonnegative GPU, and timeout 60–3600 per arm")
    if args.baseline_run is not None and not args.include_wake_guard:
        parser.error("--baseline-run requires --include-wake-guard")
    root = project_root()
    source = args.source_run.resolve()
    if source.parent != (root / "runs/policy-telemetry").resolve():
        raise ValueError("source must be an immutable policy-telemetry run")
    source_report = json.loads((source / "report.json").read_text())
    valid_capture = source_report.get("status") == "passed" or (
        source_report.get("status") == "failed"
        and valid_early_contact_capture(
            source_report.get("exit_code"),
            source_report.get("probe"),
            source_report.get("metrics"),
        )
    )
    if not valid_capture:
        raise ValueError("source run did not produce an intact telemetry capture")
    telemetry = source / "evaluation/policy-telemetry.csv"
    if (
        not telemetry.is_file()
        or source_report.get("probe")
        != json.loads((source / "evaluation/probe-result.json").read_text())
        or source_report.get("metrics")
        != json.loads((source / "evaluation/metrics.json").read_text())
    ):
        raise ValueError("source telemetry or simulator result is missing/mismatched")
    training = Path(source_report["source_run"])
    config_path = training / f"seed-{source_report['policy_seed']:010d}/config.json"
    if digest(config_path) != source_report["source_config_sha256"]:
        raise ValueError("source training configuration changed")
    config = json.loads(config_path.read_text())
    source_runtime = json.loads((source / "evaluation/runtime.json").read_text())
    if (
        config["construction"]["num_agents"] != 4
        or config["construction"]["physics_dt"] != 0.01
        or config["construction"]["max_speed_m_s"] != 0.5
    ):
        raise ValueError("the matched probe is calibrated for four drones at 100 Hz")
    build_path = args.build_report.resolve()
    if build_path.parent.parent != (root / "runs/runtime-build").resolve():
        raise ValueError("build-report must belong to a runtime-build run")
    build = json.loads(build_path.read_text())
    if build.get("status") != "built":
        raise ValueError("runtime image has not been built")

    baseline = args.baseline_run.resolve() if args.baseline_run is not None else None
    baseline_report = None
    if baseline is not None:
        if baseline.parent != (root / "runs/takeoff-replay").resolve():
            raise ValueError("baseline-run must be a direct child of runs/takeoff-replay")
        baseline_report = json.loads((baseline / "report.json").read_text())
    run = create_run_directory(root / "runs/takeoff-replay")
    active_arms = (
        ("wake_guard",)
        if baseline is not None
        else (REPLAY_ARM_CHOICES if args.include_wake_guard else ARMS)
    )
    for arm in active_arms:
        (run / arm / "kit-logs").mkdir(parents=True)
    shutil.copy2(config_path, run / "config.json")
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    trace_metadata = prepare_trace(telemetry, run / "commands.csv")
    if baseline is not None:
        validate_baseline_report(
            baseline_report,
            source_run_id=source.name,
            source_telemetry_sha256=digest(telemetry),
            config_sha256=digest(config_path),
            checkpoint_id=source_report["checkpoint_id"],
            trace_sha256=digest(run / "commands.csv"),
        )
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        source_run=str(source),
        source_run_id=source.name,
        source_telemetry_sha256=digest(telemetry),
        command_trace_sha256=digest(run / "commands.csv"),
        config_sha256=digest(run / "config.json"),
        source_checkpoint_id=source_report["checkpoint_id"],
        source_checkpoint_sha256=source_report["checkpoint_sha256"],
        source_image_id=source_report["image_id"],
        image_id=build["image_id"],
        host_gpu_index=args.gpu,
        trace=trace_metadata,
        source=probe_source(30).details,
        system=probe_system().details,
        training_performed=False,
        include_wake_guard=args.include_wake_guard,
        baseline_run=str(baseline) if baseline is not None else None,
        baseline_run_id=baseline.name if baseline is not None else None,
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
    write_json_atomic(run / "report.json", report)
    print(f"{report['started_at_ist']} IST Takeoff replay: {run}", flush=True)
    docker = docker_prefix(args.docker)
    keepalive = SudoKeepalive(args.docker == "sudo")
    active_name = None
    try:
        keepalive.start()
        if capture([*docker, "image", "inspect", build["image_id"]])["exit_code"]:
            raise RuntimeError("built image is unavailable")
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for arm in active_arms:
            command = replay_command(
                docker, run=run, arm=arm, image_id=build["image_id"], gpu=args.gpu
            )
            active_name = command[command.index("--name") + 1]
            report["arms"][arm] = {"command": command}
            write_json_atomic(run / "report.json", report)
            print(f"{arm}: follow {run / arm / 'console.log'}", flush=True)
            exit_code = execute(command, run / arm / "console.log", args.timeout_per_arm)
            active_name = None
            probe_path = run / arm / "probe-result.json"
            metrics_path = run / arm / "metrics.json"
            probe = json.loads(probe_path.read_text()) if probe_path.is_file() else None
            metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
            report["arms"][arm].update(exit_code=exit_code, probe=probe, metrics=metrics)
            runtime_path = run / arm / "runtime.json"
            runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
            report["arms"][arm]["source_model_and_controller_match"] = (
                isinstance(runtime, dict)
                and runtime.get("controller") == source_runtime.get("controller")
                and runtime.get("model", {}).get("parameters")
                == source_runtime.get("model", {}).get("parameters")
            )
            if not (
                exit_code == 0
                and isinstance(probe, dict)
                and probe.get("status") == "passed"
                and probe.get("takeoff_replay_tested") is True
                and isinstance(metrics, dict)
                and metrics.get("status") == "passed"
                and metrics.get("arm") == arm
                and metrics.get("downwash_override") == (arm == "no_downwash")
                and report["arms"][arm]["source_model_and_controller_match"]
            ):
                raise RuntimeError(f"{arm} simulator arm failed; inspect its console.log")
        report["comparison"] = assess_replays(
            run / "commands.csv",
            {arm: (baseline or run) / arm / "trajectory.csv" for arm in ARMS},
            airborne_height_m=config["task"]["airborne_height_m"],
            contact_threshold_n=config["task"]["contact_force_threshold_n"],
        )
        if args.include_wake_guard:
            report["wake_guard"] = assess_wake_guard(
                run / "commands.csv",
                run / "wake_guard/trajectory.csv",
                airborne_height_m=config["task"]["airborne_height_m"],
                contact_threshold_n=config["task"]["contact_force_threshold_n"],
                minimum_separation_m=config["construction"]["minimum_separation_m"],
            )
        save_replay_plot(
            run / "commands.csv",
            {
                **{arm: (baseline or run) / arm / "trajectory.csv" for arm in ARMS},
                **(
                    {"wake_guard": run / "wake_guard/trajectory.csv"}
                    if args.include_wake_guard
                    else {}
                ),
            },
            run / "altitude-and-force.svg",
        )
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if active_name is not None:
            report["cleanup"] = capture([*docker, "rm", "--force", active_name])
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
