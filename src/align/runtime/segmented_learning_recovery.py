"""Resume an interrupted segmented-learning run into a separate immutable run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.checkpoint_store import CheckpointStore
from align.learning.critic_distribution_schema import read_valid_critic_distribution
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
    _command,
    combine_segment_metrics,
    summarize_learning_curve,
    write_learning_curve_tables,
)
from align.runtime.stability_runtime import valid_normalization_warmup_csv, valid_phase


def source_inventory(directory: Path) -> dict:
    """Hash every regular source artifact so recovery can prove it stayed unchanged."""
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("source run cannot contain symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": rows,
    }


def inventory_summary(inventory: dict) -> dict:
    """Keep compact inventory identity in the report; retain rows separately."""
    return {
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "tree_sha256": inventory["tree_sha256"],
    }


def completed_source_segments(source_item: dict | None, curve: LearningCurveConfig) -> list[dict]:
    """Return the valid leading source segments; reject gaps and later valid work."""
    if source_item is None:
        return []
    records = source_item.get("train", {}).get("segments", [])
    completed = []
    for expected, record in zip(curve.segment_ranges(), records, strict=False):
        if (record.get("start_update"), record.get("stop_update")) != expected:
            break
        if not valid_phase(
            record.get("exit_code", -1),
            record.get("probe"),
            record.get("metrics"),
            "training_stability_tested",
        ):
            break
        completed.append({**record, "artifact_origin": "source_run"})
    for record in records[len(completed) :]:
        if valid_phase(
            record.get("exit_code", -1),
            record.get("probe"),
            record.get("metrics"),
            "training_stability_tested",
        ):
            raise ValueError("source contains valid training after an incomplete segment")
    return completed


def copy_checkpoint_prefix(
    source: Path,
    destination: Path,
    *,
    logical_run_id: str,
    config_sha256: str,
    through_update: int,
) -> dict:
    """Copy and reverify exactly the committed prefix needed for recovery."""
    source_store = CheckpointStore(
        source,
        run_id=logical_run_id,
        config_sha256=config_sha256,
        file_mode=0o644,
    )
    latest, _ = source_store.latest_valid()
    if latest["completed_updates"] != through_update:
        raise ValueError("latest valid source checkpoint differs from completed segment boundary")
    destination.mkdir(parents=True)
    copied = []
    for update in range(through_update + 1):
        manifest, payload = source_store.for_update(update)
        manifest_path = source / f"checkpoint-{manifest['checkpoint_id']}.json"
        shutil.copy2(payload, destination / payload.name)
        shutil.copy2(manifest_path, destination / manifest_path.name)
        copied.append(
            {
                "completed_update": update,
                "checkpoint_id": manifest["checkpoint_id"],
                "payload_sha256": manifest["payload"]["sha256"],
            }
        )
    write_json_atomic(
        destination / "latest.json",
        {
            "schema_version": 1,
            "manifest": f"checkpoint-{latest['checkpoint_id']}.json",
        },
        mode=0o644,
    )
    destination_store = CheckpointStore(
        destination,
        run_id=logical_run_id,
        config_sha256=config_sha256,
        file_mode=0o644,
    )
    copied_latest, _ = destination_store.latest_valid()
    if copied_latest["payload"]["sha256"] != latest["payload"]["sha256"]:
        raise RuntimeError("copied recovery checkpoint differs from source")
    return {
        "through_update": through_update,
        "checkpoint_count": len(copied),
        "latest_checkpoint_id": latest["checkpoint_id"],
        "latest_payload_sha256": latest["payload"]["sha256"],
        "checkpoints": copied,
    }


def _interruption_contract(source_report: dict, curve: LearningCurveConfig) -> dict:
    planned = source_report.get("planned_training_interruption")
    observed = source_report.get("observed_training_interruption")
    if source_report.get("status") != "interrupted" or not isinstance(planned, dict):
        raise ValueError("source must be a deliberately interrupted segmented run")
    if (
        not isinstance(observed, dict)
        or observed.get("kind") != "injected_mid_rollout_interruption"
    ):
        raise ValueError("source lacks a valid live partial-rollout interruption record")
    if (
        planned.get("update") != observed.get("completed_updates_before", -1) + 1
        or planned.get("rollout_step") != observed.get("partial_rollout_steps")
        or observed.get("ppo_update_started") is not False
        or observed.get("checkpoint_committed") is not False
    ):
        raise ValueError("planned and observed interruption details differ")
    segment_starts = {start for start, _ in curve.segment_ranges() if start > 0}
    if observed["completed_updates_before"] not in segment_starts:
        raise ValueError("interruption did not begin at a committed segment boundary")
    return {
        **observed,
        "policy_seed": planned["policy_seed"],
        "discard_policy": "discard_partial_rollout_and_reset_environment_and_recurrent_memory",
    }


def validate_interrupted_rollout(
    source_run: Path,
    *,
    interruption: dict,
    curve: LearningCurveConfig,
    resolved: dict,
) -> dict:
    """Audit the flushed partial trajectory that must be discarded."""
    before = interruption["completed_updates_before"]
    segment = next(
        (bounds for bounds in curve.segment_ranges() if bounds[0] == before),
        None,
    )
    if segment is None:
        raise ValueError("interrupted update does not start a declared segment")
    update = before + 1
    directory = (
        source_run
        / f"seed-{interruption['policy_seed']:010d}"
        / f"train-segment-{segment[0]:04d}-{segment[1]:04d}"
        / f"update-{update:04d}"
    )
    saved = json.loads((directory / "interruption.json").read_text())
    if saved != {
        key: value
        for key, value in interruption.items()
        if key not in {"policy_seed", "discard_policy"}
    }:
        raise ValueError("source interruption file differs from the source report")
    with (directory / "rollout.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    steps = interruption["partial_rollout_steps"]
    num_envs = resolved["rollout"]["num_envs"]
    if len(rows) != steps * num_envs:
        raise ValueError("partial rollout row count differs from interruption record")
    expected = [(step, env_id) for step in range(steps) for env_id in range(num_envs)]
    actual = [(int(row["rollout_step"]), int(row["env_id"])) for row in rows]
    if actual != expected:
        raise ValueError("partial rollout step/environment ordering differs")
    if (directory / "updates.csv").exists() or (directory / "metrics.json").exists():
        raise ValueError("interrupted update contains post-update artifacts")
    return {
        "directory": str(directory),
        "row_count": len(rows),
        "partial_rollout_steps": steps,
        "environment_count": num_envs,
        "ppo_update_started": False,
        "checkpoint_committed": False,
    }


def _update_directory(
    *,
    source_run: Path,
    recovery_run: Path,
    seed: int,
    segment: dict,
    update: int,
) -> Path:
    root = source_run if segment["artifact_origin"] == "source_run" else recovery_run
    return (
        root
        / f"seed-{seed:010d}"
        / f"train-segment-{segment['start_update']:04d}-{segment['stop_update']:04d}"
        / f"update-{update:04d}"
    )


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Resume interrupted segmented training in a new immutable run."
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
    if source_run.parent != (root / "runs/segmented-learning").resolve():
        raise ValueError("source-run must be one direct child of runs/segmented-learning")
    source_report = json.loads((source_run / "report.json").read_text())
    curve = LearningCurveConfig.from_dict(
        json.loads((source_run / "learning-curve-config.json").read_text())
    )
    if curve.schema_version != 2:
        raise ValueError("segmented recovery requires learning-curve schema 2")
    resolved = json.loads((source_run / "config.json").read_text())
    interruption = _interruption_contract(source_report, curve)
    interrupted_rollout_audit = validate_interrupted_rollout(
        source_run,
        interruption=interruption,
        curve=curve,
        resolved=resolved,
    )
    source_before = source_inventory(source_run)

    run = create_run_directory(root / "runs/segmented-learning-recovery")
    shutil.copy2(source_run / "config.json", run / "config.json")
    shutil.copy2(source_run / "learning-curve-config.json", run / "learning-curve-config.json")
    shutil.copy2(source_run / "build-report.json", run / "build-report.json")
    shutil.copy2(source_run / "context-manifest.json", run / "context-manifest.json")
    write_json_atomic(run / "source-inventory-before.json", source_before)

    source_items = {item["policy_seed"]: item for item in source_report.get("seeds", [])}
    completed_by_seed = {}
    checkpoint_copies = {}
    for seed in curve.policy_seeds:
        seed_label = f"seed-{seed:010d}"
        source_seed = source_run / seed_label
        recovery_seed = run / seed_label
        recovery_seed.mkdir()
        shutil.copy2(source_seed / "config.json", recovery_seed / "config.json")
        completed = completed_source_segments(source_items.get(seed), curve)
        completed_by_seed[seed] = completed
        completed_updates = completed[-1]["stop_update"] if completed else 0
        if seed == interruption["policy_seed"] and (
            completed_updates != interruption["completed_updates_before"]
        ):
            raise ValueError(
                "completed source segments differ from interrupted checkpoint boundary"
            )
        if completed_updates:
            logical_run_id = f"{source_run.name}-seed-{seed:010d}"
            checkpoint_copies[str(seed)] = copy_checkpoint_prefix(
                source_seed / "checkpoints",
                recovery_seed / "checkpoints",
                logical_run_id=logical_run_id,
                config_sha256=digest(recovery_seed / "config.json"),
                through_update=completed_updates,
            )
        else:
            (recovery_seed / "checkpoints").mkdir()
            checkpoint_copies[str(seed)] = {
                "through_update": 0,
                "checkpoint_count": 0,
                "latest_checkpoint_id": None,
                "latest_payload_sha256": None,
                "checkpoints": [],
            }
        for start_update, stop_update in curve.segment_ranges()[len(completed) :]:
            (
                recovery_seed / f"train-segment-{start_update:04d}-{stop_update:04d}" / "kit-logs"
            ).mkdir(parents=True)
        for milestone in curve.evaluation_milestones:
            (recovery_seed / f"evaluation-update-{milestone:04d}/kit-logs").mkdir(parents=True)

    started = time.perf_counter()
    report = new_report()
    image_id = source_report["image_id"]
    source = source_report["source"]
    source_identity = f"git:{source.get('git_revision')}:package:{source['package_sha256']}"
    report.update(
        source_run=str(source_run),
        source_run_id=source_run.name,
        source_run_status=source_report["status"],
        source_inventory_before=inventory_summary(source_before),
        image_id=image_id,
        host_gpu_index=args.gpu,
        system=probe_system().details,
        learning_curve_config=curve.to_dict(),
        interruption=interruption,
        interrupted_rollout_audit=interrupted_rollout_audit,
        checkpoint_copies=checkpoint_copies,
        training_reused=True,
        training_performed=True,
        deterministic_evaluation_performed=False,
        scientific_result=False,
        rendering_validated=False,
        video_validated=False,
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
        f"{report['started_at_ist']} IST Segmented recovery: {run}\nImmutable source: {source_run}",
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
        for seed in curve.policy_seeds:
            seed_directory = run / f"seed-{seed:010d}"
            segment_records = list(completed_by_seed[seed])
            item = {
                "policy_seed": seed,
                "train": {"segments": segment_records},
                "evaluations": [],
            }
            report["seeds"].append(item)
            for start_update, stop_update in curve.segment_ranges()[len(segment_records) :]:
                output = seed_directory / f"train-segment-{start_update:04d}-{stop_update:04d}"
                command = _command(
                    docker,
                    run=run,
                    seed_directory=seed_directory,
                    output_directory=output,
                    image_id=image_id,
                    gpu=args.gpu,
                    num_envs=resolved["rollout"]["num_envs"],
                    source_identity=source_identity,
                    scenario="stability",
                    training_start_update=start_update,
                    training_stop_update=stop_update,
                    logical_run_name=source_run.name,
                )
                active_container = command[command.index("--name") + 1]
                phase_started = time.perf_counter()
                exit_code = execute(command, output / "console.log", args.timeout)
                probe_path = output / "probe-result.json"
                metrics_path = output / "metrics.json"
                probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
                metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
                segment = {
                    "start_update": start_update,
                    "stop_update": stop_update,
                    "container_name": active_container,
                    "artifact_origin": "recovery_run",
                    "command": command,
                    "exit_code": exit_code,
                    "duration_seconds": time.perf_counter() - phase_started,
                    "probe": probe,
                    "metrics": metrics,
                }
                segment_records.append(segment)
                item["train"] = {"segments": segment_records}
                write_json_atomic(run / "report.json", report)
                if not valid_phase(exit_code, probe, metrics, "training_stability_tested"):
                    raise RuntimeError(
                        f"seed {seed} recovery segment {start_update}:{stop_update} failed"
                    )
                active_container = None

            combined = combine_segment_metrics(segment_records, curve)
            item["train"] = {
                "segments": segment_records,
                "duration_seconds": sum(segment["duration_seconds"] for segment in segment_records),
                "metrics": combined,
            }
            if combined["status"] != "passed":
                raise RuntimeError(f"seed {seed} joined recovery audit failed")

            update_directories = {}
            for segment in segment_records:
                for update in range(segment["start_update"] + 1, segment["stop_update"] + 1):
                    update_directories[update] = _update_directory(
                        source_run=source_run,
                        recovery_run=run,
                        seed=seed,
                        segment=segment,
                        update=update,
                    )
            normalization = resolved["critic_normalization"]
            warmups = [
                update_directories[update] / "normalization-warmup.csv"
                for update in update_directories
                if (update_directories[update] / "normalization-warmup.csv").is_file()
            ]
            warmup_valid = len(warmups) == int(normalization["enabled"]) and (
                not normalization["enabled"]
                or valid_normalization_warmup_csv(
                    warmups[0],
                    warmup_steps=normalization["warmup_steps"],
                    num_envs=resolved["rollout"]["num_envs"],
                    num_agents=resolved["rollout"]["num_agents"],
                )
            )
            item["train"]["normalization_warmup_csv_valid"] = warmup_valid
            if not warmup_valid:
                raise RuntimeError(f"seed {seed} joined normalization warmup audit failed")
            expected_scalars = (
                resolved["rollout"]["horizon"]
                * resolved["rollout"]["num_envs"]
                * resolved["rollout"]["num_agents"]
                * 3
            )
            distribution_valid = all(
                read_valid_critic_distribution(
                    update_directories[update] / "critic-distribution.csv",
                    update=update,
                    scalar_count=expected_scalars,
                    enabled=normalization["enabled"],
                    clip=normalization["clip"],
                )
                is not None
                for update in range(1, curve.updates_per_seed + 1)
            )
            item["train"]["critic_distribution_csv_valid"] = distribution_valid
            if not distribution_valid:
                raise RuntimeError(f"seed {seed} joined critic distribution audit failed")

            for milestone in curve.evaluation_milestones:
                output = seed_directory / f"evaluation-update-{milestone:04d}"
                command = _command(
                    docker,
                    run=run,
                    seed_directory=seed_directory,
                    output_directory=output,
                    image_id=image_id,
                    gpu=args.gpu,
                    num_envs=resolved["rollout"]["num_envs"],
                    source_identity=source_identity,
                    scenario="evaluation",
                    evaluation_update=milestone,
                    logical_run_name=source_run.name,
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
                if not valid_phase(exit_code, probe, metrics, "deterministic_evaluation_tested"):
                    raise RuntimeError(
                        f"seed {seed} recovery evaluation at update {milestone} failed"
                    )
                if metrics["completed_updates"] != milestone:
                    raise RuntimeError("recovery evaluation loaded the wrong update")
                active_container = None
            write_json_atomic(run / "report.json", report)

        summary = summarize_learning_curve(report["seeds"], curve)
        source_after = source_inventory(source_run)
        write_json_atomic(run / "source-inventory-after.json", source_after)
        source_unchanged = source_after == source_before
        if not source_unchanged:
            raise RuntimeError("immutable source run changed during recovery")
        completed_updates_by_seed = {
            item["policy_seed"]: [
                row["completed_update"] for row in item["train"]["metrics"]["measurements"]
            ]
            for item in report["seeds"]
        }
        no_duplicate_updates = all(
            updates == list(range(1, curve.updates_per_seed + 1))
            for updates in completed_updates_by_seed.values()
        )
        write_json_atomic(run / "learning-curve-summary.json", summary)
        write_learning_curve_tables(run, summary)
        report.update(
            status="passed",
            source_inventory_after=inventory_summary(source_after),
            source_run_unchanged=source_unchanged,
            partial_rollout_discarded=True,
            discarded_environment_transitions=interruption["discarded_environment_transitions"],
            discarded_agent_transitions=interruption["discarded_agent_transitions"],
            optimizer_updates=summary["training_update_count"],
            recovery_optimizer_updates=sum(
                segment["metrics"]["updates"]
                for item in report["seeds"]
                for segment in item["train"]["segments"]
                if segment["artifact_origin"] == "recovery_run"
            ),
            evaluation_containers=summary["evaluation_container_count"],
            no_duplicate_training_updates=no_duplicate_updates,
            completed_updates_by_seed=completed_updates_by_seed,
            deterministic_evaluation_performed=True,
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
        csv.Error,
    ) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        cleanup = True
    finally:
        if "source_inventory_after" not in report:
            try:
                source_after = source_inventory(source_run)
                write_json_atomic(run / "source-inventory-after.json", source_after)
                report["source_inventory_after"] = inventory_summary(source_after)
                report["source_run_unchanged"] = source_after == source_before
            except (OSError, ValueError, TypeError) as exc:
                report["source_inventory_error"] = f"{type(exc).__name__}: {exc}"
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
