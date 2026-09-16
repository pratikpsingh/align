"""Host-only launcher for the pinned multi-drone construction probe."""

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
from align.simulation.multi_drone_report import assess_saved_run


def valid_result(exit_code, probe, metrics):
    return (
        exit_code == 0
        and isinstance(probe, dict)
        and probe.get("status") == "passed"
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and probe.get("multi_drone_physics_tested") is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and bool(metrics.get("checks"))
        and all(value is True for value in metrics["checks"].values())
    )


def metrics_match(saved, recomputed):
    """Compare the in-memory result with its JSON-native persisted representation."""
    return json.loads(json.dumps(recomputed, allow_nan=False)) == saved


def run_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate four-drone ground-to-plane construction on one allocated GPU."
    )
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60–3600")

    root = project_root()
    config_path = args.config or root / "configs/multi-drone-construction.json"
    config = MultiDroneConfig.from_dict(json.loads(config_path.read_text()))
    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")

    run = create_run_directory(root / "runs/multi-drone")
    (run / "kit-logs").mkdir()
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
    name = "align-multi-drone-" + run.name.lower()
    command = [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        name,
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
        "align.simulation.multi_drone",
        "--config",
        "/output/config.json",
        "--output",
        "/output",
        "--allow-root",
    ]
    report["command"] = command
    write_json_atomic(run / "report.json", report)
    print(f"{report['started_at_ist']} IST Run: {run}\nFollow {run / 'console.log'}", flush=True)

    cleanup = False
    try:
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        report["container_exit_code"] = execute(command, run / "console.log", args.timeout)
        probe_path, metrics_path = run / "probe-result.json", run / "metrics.json"
        probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        report["probe_result"] = probe
        if metrics is not None:
            assessed, _ = assess_saved_run(run)
            if not metrics_match(metrics, assessed):
                raise ValueError("Container metrics disagree with host evaluation of raw data")
            report["host_recomputed_metrics"] = True
        report["status"] = (
            "passed" if valid_result(report["container_exit_code"], probe, metrics) else "failed"
        )
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
            report["cleanup"] = capture([*docker, "rm", "--force", name])
        report["gpu_after"] = capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv",
            ]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
