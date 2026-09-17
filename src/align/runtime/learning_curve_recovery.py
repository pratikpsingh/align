"""Recover missing evaluations from an immutable timed-out learning-curve run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.checkpoint_store import CheckpointStore
from align.learning.learning_curve_config import LearningCurveConfig
from align.runtime.diagnostics import probe_system
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
from align.runtime.learning_curve_runtime import (
    summarize_learning_curve,
    write_learning_curve_tables,
)
from align.runtime.stability_runtime import valid_phase


def _valid_source_evaluation(item: dict, milestone: int) -> bool:
    return (
        item.get("completed_update") == milestone
        and valid_phase(
            item.get("exit_code", -1),
            item.get("probe"),
            item.get("metrics"),
            "deterministic_evaluation_tested",
        )
        and item["metrics"].get("completed_updates") == milestone
    )


def _command(
    docker,
    *,
    source_run: Path,
    recovery_run: Path,
    source_seed: Path,
    output_directory: Path,
    image_id: str,
    gpu: int,
    num_envs: int,
    source_identity: str,
    milestone: int,
) -> list[str]:
    seed_label = source_seed.name
    relative_output = output_directory.relative_to(recovery_run)
    logical_run_id = f"{source_run.name}-seed-{seed_label.split('-')[-1]}"
    phase = f"evaluation-{milestone:04d}"
    return [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        f"align-learning-recovery-{recovery_run.name.lower()}-{seed_label}-{phase}",
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
        f"type=bind,src={source_run},dst=/source,readonly",
        "--mount",
        f"type=bind,src={recovery_run},dst=/output",
        "--mount",
        f"type=bind,src={output_directory / 'kit-logs'},dst=/isaac-sim/kit/logs",
        image_id,
        "-m",
        "align.simulation.vector_task",
        "--config",
        f"/source/{seed_label}/config.json",
        "--output",
        f"/output/{relative_output}",
        "--scenario",
        "evaluation",
        "--num-envs",
        str(num_envs),
        "--logical-run-id",
        logical_run_id,
        "--attempt-id",
        f"recovery-{phase}",
        "--checkpoint-directory",
        f"/source/{seed_label}/checkpoints",
        "--source-identity",
        source_identity,
        "--runtime-identity",
        image_id,
        "--allow-root",
        "--evaluation-update",
        str(milestone),
    ]


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover only missing checkpoint evaluations from a failed curve run."
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60-3600")

    root = project_root()
    source_run = args.source_run.resolve()
    if source_run.parent != (root / "runs/learning-curve").resolve():
        raise ValueError("source-run must be one direct child of runs/learning-curve")
    source_report = json.loads((source_run / "report.json").read_text())
    if source_report.get("status") not in {"failed", "timed_out", "interrupted"}:
        raise ValueError("source run must be failed, timed_out, or interrupted")
    curve = LearningCurveConfig.from_dict(
        json.loads((source_run / "learning-curve-config.json").read_text())
    )
    resolved = json.loads((source_run / "config.json").read_text())
    if [item.get("policy_seed") for item in source_report.get("seeds", [])] != list(
        curve.policy_seeds
    ):
        raise ValueError("source report does not contain the declared seeds")

    run = create_run_directory(root / "runs/learning-curve-recovery")
    write_json_atomic(run / "config.json", resolved)
    write_json_atomic(run / "learning-curve-config.json", curve.to_dict())
    shutil.copy2(source_run / "build-report.json", run / "build-report.json")
    shutil.copy2(source_run / "context-manifest.json", run / "context-manifest.json")
    for seed in curve.policy_seeds:
        for milestone in curve.evaluation_milestones:
            (run / f"seed-{seed:010d}/evaluation-update-{milestone:04d}/kit-logs").mkdir(
                parents=True
            )

    started = time.perf_counter()
    report = new_report()
    image_id = source_report["image_id"]
    source = source_report["source"]
    source_identity = f"git:{source.get('git_revision')}:package:{source['package_sha256']}"
    report.update(
        source_run=str(source_run),
        source_run_id=source_run.name,
        source_run_status=source_report["status"],
        image_id=image_id,
        host_gpu_index=args.gpu,
        system=probe_system().details,
        learning_curve_config=curve.to_dict(),
        training_reused=True,
        training_performed=False,
        deterministic_evaluation_performed=False,
        scientific_result=False,
        seeds=[],
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
        f"{report['started_at_ist']} IST Recovery run: {run}\nSource: {source_run}",
        flush=True,
    )

    active_container = None
    cleanup = False
    keepalive = SudoKeepalive(args.docker == "sudo")
    try:
        keepalive.start()
        inspection = capture([*docker, "image", "inspect", image_id])
        if inspection["exit_code"]:
            raise RuntimeError("source runtime image is not available locally")
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        for source_item in source_report["seeds"]:
            seed = source_item["policy_seed"]
            if not valid_phase(
                source_item["train"]["exit_code"],
                source_item["train"].get("probe"),
                source_item["train"].get("metrics"),
                "training_stability_tested",
            ):
                raise RuntimeError(f"source seed {seed} training phase is not valid")
            seed_label = f"seed-{seed:010d}"
            source_seed = source_run / seed_label
            logical_run_id = f"{source_run.name}-seed-{seed_label.split('-')[-1]}"
            store = CheckpointStore(
                source_seed / "checkpoints",
                run_id=logical_run_id,
                config_sha256=digest(source_seed / "config.json"),
                file_mode=0o644,
            )
            checkpoint_audit = {}
            for milestone in curve.evaluation_milestones:
                manifest, _ = store.for_update(milestone)
                checkpoint_audit[str(milestone)] = {
                    "checkpoint_id": manifest["checkpoint_id"],
                    "payload_sha256": manifest["payload"]["sha256"],
                }

            item = {
                "policy_seed": seed,
                "train": {
                    **source_item["train"],
                    "artifact_origin": "source_run",
                },
                "checkpoint_audit": checkpoint_audit,
                "evaluations": [],
            }
            report["seeds"].append(item)
            existing = {row["completed_update"]: row for row in source_item.get("evaluations", [])}
            for milestone in curve.evaluation_milestones:
                source_evaluation = existing.get(milestone)
                if source_evaluation is not None and _valid_source_evaluation(
                    source_evaluation, milestone
                ):
                    item["evaluations"].append(
                        {**source_evaluation, "artifact_origin": "source_run"}
                    )
                    continue

                output = run / seed_label / f"evaluation-update-{milestone:04d}"
                command = _command(
                    docker,
                    source_run=source_run,
                    recovery_run=run,
                    source_seed=source_seed,
                    output_directory=output,
                    image_id=image_id,
                    gpu=args.gpu,
                    num_envs=resolved["rollout"]["num_envs"],
                    source_identity=source_identity,
                    milestone=milestone,
                )
                active_container = command[command.index("--name") + 1]
                phase_started = time.perf_counter()
                exit_code = execute(command, output / "console.log", args.timeout)
                probe_path = output / "probe-result.json"
                metrics_path = output / "metrics.json"
                probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
                metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
                evaluation = {
                    "completed_update": milestone,
                    "artifact_origin": "recovery_run",
                    "command": command,
                    "exit_code": exit_code,
                    "duration_seconds": time.perf_counter() - phase_started,
                    "probe": probe,
                    "metrics": metrics,
                }
                item["evaluations"].append(evaluation)
                write_json_atomic(run / "report.json", report)
                if not _valid_source_evaluation(evaluation, milestone):
                    raise RuntimeError(f"seed {seed} evaluation at update {milestone} failed")
                active_container = None
            write_json_atomic(run / "report.json", report)

        summary = summarize_learning_curve(report["seeds"], curve)
        write_json_atomic(run / "learning-curve-summary.json", summary)
        write_learning_curve_tables(run, summary)
        reused = sum(
            row["artifact_origin"] == "source_run"
            for item in report["seeds"]
            for row in item["evaluations"]
        )
        report.update(
            status="passed",
            deterministic_evaluation_performed=True,
            optimizer_updates=summary["training_update_count"],
            evaluation_containers_reused=reused,
            evaluation_containers_run=summary["evaluation_container_count"] - reused,
            summary=summary,
        )
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        cleanup = True
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
        cleanup = True
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
        cleanup = True
    finally:
        if cleanup and active_container is not None:
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
