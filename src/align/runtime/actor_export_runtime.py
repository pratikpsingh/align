"""Host launcher for a verified, actor-only export in the pinned vendor runtime."""

from __future__ import annotations

import argparse
import json
import os
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
    digest,
    docker_prefix,
    execute,
    finish,
    new_report,
    project_root,
)


def export_command(
    docker: list[str],
    *,
    source: Path,
    output: Path,
    image_id: str,
    seed: int,
    manifest: dict,
    repeats: int,
    container_name: str,
) -> list[str]:
    """Read the verified learner checkpoint and export its actor using vendor CPU PyTorch."""
    if seed < 0 or repeats < 1:
        raise ValueError("seed and benchmark repeats must be valid")
    seed_label = f"seed-{seed:010d}"
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--network=none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "HOME=/tmp",
        "-e",
        "TZ=Asia/Kolkata",
        "--mount",
        f"type=bind,src={source},dst=/source,readonly",
        "--mount",
        f"type=bind,src={output},dst=/output",
        image_id,
        "-m",
        "align.policies.actor_export",
        "--checkpoint",
        f"/source/{seed_label}/checkpoints/{manifest['payload']['file']}",
        "--config",
        f"/source/{seed_label}/config.json",
        "--output",
        "/output/artifact",
        "--checkpoint-sha256",
        manifest["payload"]["sha256"],
        "--checkpoint-id",
        manifest["checkpoint_id"],
        "--completed-update",
        str(manifest["completed_updates"]),
        "--repeats",
        str(repeats),
    ]


def verify_command(
    docker: list[str], *, output: Path, image_id: str, container_name: str
) -> list[str]:
    """Launch a fresh process that can see only the actor artifact, never the checkpoint."""
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--network=none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "HOME=/tmp",
        "-e",
        "TZ=Asia/Kolkata",
        "--mount",
        f"type=bind,src={output / 'artifact'},dst=/artifact,readonly",
        "--mount",
        f"type=bind,src={output},dst=/output",
        image_id,
        "-m",
        "align.policies.actor_inference",
        "--artifact",
        "/artifact",
        "--output",
        "/output/verification.json",
    ]


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--build-report", required=True, type=Path)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--evaluation-update", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if (
        not args.accept_eula
        or args.policy_seed < 0
        or args.evaluation_update < 0
        or not 20 <= args.repeats <= 2000
        or not 60 <= args.timeout <= 900
    ):
        parser.error("Require EULA, nonnegative seed/update, repeats 20–2000, timeout 60–900")
    root = project_root()
    source = args.source_run.resolve()
    if source.parent not in {
        (root / "runs/plane-baseline-training").resolve(),
        (root / "runs/multi-template-training").resolve(),
    }:
        raise ValueError("source-run must be a direct child of a training category")
    source_report = json.loads((source / "report.json").read_text())
    if source_report["status"] != "passed":
        raise ValueError("source training run must have passed")
    if args.policy_seed not in {item["policy_seed"] for item in source_report["seeds"]}:
        raise ValueError("requested policy seed is absent")
    seed_label = f"seed-{args.policy_seed:010d}"
    config_path = source / seed_label / "config.json"
    store = CheckpointStore(
        source / seed_label / "checkpoints",
        run_id=f"{source.name}-seed-{args.policy_seed:010d}",
        config_sha256=digest(config_path),
        file_mode=0o644,
    )
    manifest, _ = store.for_update(args.evaluation_update)
    build_path = args.build_report.resolve()
    if build_path.parent.parent != (root / "runs/runtime-build").resolve():
        raise ValueError("build-report must be in runs/runtime-build")
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise ValueError("derived image must be built")
    run = create_run_directory(root / "runs/actor-export")
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        source_run=str(source),
        source_run_id=source.name,
        source_image_id=source_report["image_id"],
        image_id=build["image_id"],
        policy_seed=args.policy_seed,
        completed_update=args.evaluation_update,
        checkpoint_id=manifest["checkpoint_id"],
        checkpoint_sha256=manifest["payload"]["sha256"],
        source_config_sha256=digest(config_path),
        source=probe_source(30).details,
        system=probe_system().details,
        simulator_started=False,
        training_performed=False,
        critic_exported=False,
        physical_deployment_tested=False,
    )
    docker = docker_prefix(args.docker)
    export_name = f"align-actor-export-{run.name.lower()}"
    verify_name = f"align-actor-verify-{run.name.lower()}"
    command = export_command(
        docker,
        source=source,
        output=run,
        image_id=build["image_id"],
        seed=args.policy_seed,
        manifest=manifest,
        repeats=args.repeats,
        container_name=export_name,
    )
    verification_command = verify_command(
        docker, output=run, image_id=build["image_id"], container_name=verify_name
    )
    report["command"] = command
    report["verification_command"] = verification_command
    write_json_atomic(run / "report.json", report)
    print(f"{report['started_at_ist']} IST Actor export: {run}", flush=True)
    keepalive = SudoKeepalive(args.docker == "sudo")
    try:
        keepalive.start()
        if capture([*docker, "image", "inspect", build["image_id"]])["exit_code"]:
            raise RuntimeError("built runtime image is unavailable locally")
        report["container_exit_code"] = execute(command, run / "console.log", args.timeout)
        artifact_path = run / "artifact"
        metrics_path = artifact_path / "report.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        report["metrics"] = metrics
        if (
            report["container_exit_code"] != 0
            or not isinstance(metrics, dict)
            or metrics.get("status") != "passed"
            or metrics.get("checkpoint_sha256") != manifest["payload"]["sha256"]
            or metrics.get("checkpoint_id") != manifest["checkpoint_id"]
            or not metrics.get("checks")
            or not all(metrics["checks"].values())
        ):
            raise RuntimeError("actor export or checkpoint validation failed")
        for name, item in metrics["artifacts"].items():
            path = artifact_path / name
            if path.name != name or not path.is_file():
                raise ValueError("exported artifact missing or has unsafe name")
            if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("exported artifact hash or size differs")
        report["verification_container_exit_code"] = execute(
            verification_command, run / "verification.log", args.timeout
        )
        verification_path = run / "verification.json"
        verification = (
            json.loads(verification_path.read_text()) if verification_path.exists() else None
        )
        report["verification"] = verification
        if (
            report["verification_container_exit_code"] != 0
            or not isinstance(verification, dict)
            or verification.get("status") != "passed"
            or verification.get("artifact_sha256") != metrics["artifacts"]["actor.pt"]["sha256"]
            or not verification.get("checks")
            or not all(verification["checks"].values())
        ):
            raise RuntimeError("fresh-process actor inference failed")
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if report["status"] != "passed":
            report["container_cleanup"] = [
                capture([*docker, "rm", "--force", name], timeout=30)
                for name in (export_name, verify_name)
            ]
        report["sudo_keepalive"] = keepalive.stop()
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
