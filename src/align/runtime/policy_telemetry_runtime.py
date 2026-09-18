"""Replay an immutable checkpoint in a new image with per-drone telemetry."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.checkpoint_store import CheckpointStore
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
from align.tasks.policy_telemetry import audit_airborne_contacts, audit_telemetry


def replay_command(
    docker: list[str],
    *,
    source: Path,
    run: Path,
    image_id: str,
    gpu: int,
    seed: int,
    update: int,
    num_envs: int,
    source_identity: str,
    output_label: str = "evaluation",
    timing_config: str | None = None,
) -> list[str]:
    if output_label not in {"evaluation", "baseline", "extended"}:
        raise ValueError("unsupported replay output label")
    label = f"seed-{seed:010d}"
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-policy-telemetry-{run.name.lower()}-{output_label}",
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
        f"type=bind,src={run / output_label / 'kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        f"/source/{label}/config.json",
        "--output",
        f"/output/{output_label}",
        "--scenario",
        "evaluation",
        "--num-envs",
        str(num_envs),
        "--logical-run-id",
        f"{source.name}-seed-{seed:010d}",
        "--attempt-id",
        "policy-telemetry",
        "--checkpoint-directory",
        f"/source/{label}/checkpoints",
        "--source-identity",
        source_identity,
        "--runtime-identity",
        image_id,
        "--allow-root",
        "--evaluation-update",
        str(update),
        "--policy-telemetry",
        *(["--evaluation-timing-config", timing_config] if timing_config else []),
    ]


def valid_early_contact_capture(exit_code: int, probe: object, metrics: object) -> bool:
    """Accept an intact telemetry capture whose only failed gate is formation entry."""
    if not (
        exit_code in (0, 1)
        and isinstance(probe, dict)
        and probe.get("status") == "failed"
        and probe.get("error") is None
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and probe.get("vector_task_physics_tested") is True
        and probe.get("deterministic_evaluation_tested") is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "failed"
        and metrics.get("formation_phase_reached") is False
        and int(metrics.get("outcome_counts", {}).get("3", 0)) > 0
    ):
        return False
    checks = metrics.get("checks")
    return (
        isinstance(checks, dict)
        and len(checks) > 1
        and checks.get("all_declared_templates_reached_formation_phase") is False
        and all(
            value is True
            for name, value in checks.items()
            if name != "all_declared_templates_reached_formation_phase"
        )
    )


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay one frozen checkpoint with drone telemetry"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=41)
    parser.add_argument("--evaluation-update", type=int, default=12)
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if (
        not args.accept_eula
        or args.gpu < 0
        or args.policy_seed < 0
        or args.evaluation_update < 0
        or not 60 <= args.timeout <= 3600
    ):
        parser.error("Require EULA, nonnegative GPU/seed/update, and timeout 60–3600")
    root = project_root()
    source = args.source_run.resolve()
    if source.parent not in {
        root / "runs/plane-baseline-training",
        root / "runs/multi-template-training",
    }:
        raise ValueError("source-run must be a direct child of a matched training arm")
    source_report = json.loads((source / "report.json").read_text())
    if source_report["status"] != "passed":
        raise ValueError("source training run has not passed")
    seed_item = next(
        (item for item in source_report["seeds"] if item["policy_seed"] == args.policy_seed), None
    )
    if seed_item is None:
        raise ValueError("policy seed is absent from source run")
    seed_directory = source / f"seed-{args.policy_seed:010d}"
    config_path = seed_directory / "config.json"
    resolved = json.loads(config_path.read_text())
    store = CheckpointStore(
        seed_directory / "checkpoints",
        run_id=f"{source.name}-seed-{args.policy_seed:010d}",
        config_sha256=digest(config_path),
        file_mode=0o644,
    )
    manifest, _ = store.for_update(args.evaluation_update)
    build_path = args.build_report.resolve()
    if build_path.parent.parent != (root / "runs/runtime-build").resolve():
        raise ValueError("build-report must be a runtime-build run report")
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise ValueError("new runtime image must be built")
    run = create_run_directory(root / "runs/policy-telemetry")
    output = run / "evaluation"
    (output / "kit-logs").mkdir(parents=True)
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    started = time.perf_counter()
    report = new_report()
    source_meta = source_report["source"]
    source_identity = (
        f"git:{source_meta.get('git_revision')}:package:{source_meta['package_sha256']}"
    )
    report.update(
        run_id=run.name,
        source_run=str(source),
        source_run_id=source.name,
        source_image_id=source_report["image_id"],
        image_id=build["image_id"],
        policy_seed=args.policy_seed,
        requested_completed_update=args.evaluation_update,
        checkpoint_id=manifest["checkpoint_id"],
        checkpoint_sha256=manifest["payload"]["sha256"],
        source_config_sha256=digest(config_path),
        source=probe_source(30).details,
        system=probe_system().details,
        host_gpu_index=args.gpu,
        training_performed=False,
        optimizer_updates=0,
        scientific_result=False,
    )
    report["gpu_before"] = capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ]
    )
    report["gpu_processes_before"] = capture(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"]
    )
    docker = docker_prefix(args.docker)
    command = replay_command(
        docker,
        source=source,
        run=run,
        image_id=build["image_id"],
        gpu=args.gpu,
        seed=args.policy_seed,
        update=args.evaluation_update,
        num_envs=resolved["rollout"]["num_envs"],
        source_identity=source_identity,
    )
    report["command"] = command
    write_json_atomic(run / "report.json", report)
    print(f"{report['started_at_ist']} IST Policy telemetry: {run}", flush=True)
    keepalive = SudoKeepalive(args.docker == "sudo")
    active = False
    try:
        keepalive.start()
        if capture([*docker, "image", "inspect", build["image_id"]])["exit_code"]:
            raise RuntimeError("built image is unavailable locally")
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        active = True
        report["exit_code"] = execute(command, output / "console.log", args.timeout)
        active = False
        probe_path, metrics_path = output / "probe-result.json", output / "metrics.json"
        report["probe"] = json.loads(probe_path.read_text()) if probe_path.exists() else None
        report["metrics"] = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        standard_capture = valid_phase(
            report["exit_code"],
            report["probe"],
            report["metrics"],
            "deterministic_evaluation_tested",
        )
        early_contact_capture = valid_early_contact_capture(
            report["exit_code"], report["probe"], report["metrics"]
        )
        if not (standard_capture or early_contact_capture):
            raise RuntimeError("simulator evaluation failed; inspect evaluation/console.log")
        report["capture_mode"] = "early_safety_termination" if early_contact_capture else "standard"
        metrics = report["metrics"]
        if (
            metrics["checkpoint_id"] != manifest["checkpoint_id"]
            or metrics["optimizer_updates"] != 0
        ):
            raise RuntimeError("checkpoint lineage or frozen-policy contract changed")
        report["telemetry_audit"] = audit_telemetry(
            output / "policy-telemetry.csv",
            output / "evaluation.csv",
            num_agents=resolved["construction"]["num_agents"],
            max_speed_m_s=resolved["construction"]["max_speed_m_s"],
            reward_config=resolved["reward"],
        )
        report["airborne_contact_audit"] = audit_airborne_contacts(
            output / "policy-telemetry.csv",
            output / "evaluation.csv",
            airborne_height_m=resolved["task"]["airborne_height_m"],
            contact_force_threshold_n=resolved["task"]["contact_force_threshold_n"],
        )
        if (
            early_contact_capture
            and not report["airborne_contact_audit"]["first_post_airborne_contact_count"]
        ):
            raise RuntimeError("early safety capture has no post-airborne contact in raw telemetry")
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
        if active:
            report["cleanup"] = capture(
                [*docker, "rm", "--force", command[command.index("--name") + 1]]
            )
        report["sudo_keepalive"] = keepalive.stop()
        report["gpu_after"] = capture(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv"]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
