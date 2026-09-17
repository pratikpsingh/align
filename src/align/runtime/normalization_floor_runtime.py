"""Host-only calibration of a frozen critic normalization denominator floor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.learning.critic_distribution_schema import read_valid_critic_distribution
from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.learning.critic_normalization_state import validate_normalization_state
from align.learning.normalization_floor_calibration import (
    NormalizationFloorCalibrationConfig,
    summarize_floor_candidates,
)
from align.runtime.drone_runtime import finish, new_report, project_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(
    source: Path, config: NormalizationFloorCalibrationConfig
) -> tuple[list[dict], list[dict]]:
    report_path = source / "report.json"
    resolved_path = source / "config.json"
    if not report_path.is_file() or not resolved_path.is_file():
        raise ValueError("source learning curve is missing report.json or config.json")
    report = json.loads(report_path.read_text())
    resolved = json.loads(resolved_path.read_text())
    if report.get("status") != "passed":
        raise ValueError("source learning curve did not pass")
    if (
        report.get("optimizer_updates")
        != len(config.policy_seeds) * config.expected_updates_per_seed
    ):
        raise ValueError("source learning curve optimizer-update count is incompatible")
    if resolved["rollout"]["horizon"] * resolved["rollout"]["num_envs"] <= 0:
        raise ValueError("source rollout dimensions are invalid")
    scalar_count = (
        resolved["rollout"]["horizon"]
        * resolved["rollout"]["num_envs"]
        * resolved["rollout"]["num_agents"]
        * 3
    )
    artifacts = [
        {"path": "report.json", "sha256": _sha256(report_path)},
        {"path": "config.json", "sha256": _sha256(resolved_path)},
    ]
    seed_data = []
    reference_clip = None
    for seed in config.policy_seeds:
        train = source / f"seed-{seed:010d}" / "train"
        first_metrics_path = train / "update-0001" / "metrics.json"
        if not first_metrics_path.is_file():
            raise ValueError(f"source seed {seed} is missing update-0001 metrics")
        first_metrics = json.loads(first_metrics_path.read_text())
        normalization = validate_normalization_state(first_metrics["critic_normalization"])
        if (
            not normalization["enabled"]
            or normalization.get("minimum_standard_deviation", 0.0) != 0.0
        ):
            raise ValueError("source normalization must be enabled and unfloored")
        if reference_clip is None:
            reference_clip = normalization["clip"]
        elif normalization["clip"] != reference_clip:
            raise ValueError("source seeds use different normalization clips")
        velocity_updates = []
        for update in range(1, config.expected_updates_per_seed + 1):
            directory = train / f"update-{update:04d}"
            metrics_path = directory / "metrics.json"
            distribution_path = directory / "critic-distribution.csv"
            if not metrics_path.is_file():
                raise ValueError(f"source seed {seed} update {update} is missing metrics")
            metrics = json.loads(metrics_path.read_text())
            if metrics["critic_normalization"] != normalization:
                raise ValueError("source normalization changed across updates")
            rows = read_valid_critic_distribution(
                distribution_path,
                update=update,
                scalar_count=scalar_count,
                enabled=True,
                clip=normalization["clip"],
            )
            if rows is None:
                raise ValueError(f"source seed {seed} update {update} distribution is invalid")
            velocity = next(row for row in rows if row["group"] == "velocity")
            velocity_updates.append(
                {
                    "completed_update": update,
                    "raw_mean": float(velocity["raw_mean"]),
                    "raw_standard_deviation": float(velocity["raw_standard_deviation"]),
                    "raw_minimum": float(velocity["raw_minimum"]),
                    "raw_maximum": float(velocity["raw_maximum"]),
                }
            )
            artifacts.extend(
                (
                    {
                        "path": str(metrics_path.relative_to(source)),
                        "sha256": _sha256(metrics_path),
                    },
                    {
                        "path": str(distribution_path.relative_to(source)),
                        "sha256": _sha256(distribution_path),
                    },
                )
            )
        seed_data.append(
            {
                "policy_seed": seed,
                "normalization": normalization,
                "velocity_updates": velocity_updates,
            }
        )
    return seed_data, artifacts


def _write_candidate_table(path: Path, summary: dict) -> None:
    fields = (
        "minimum_standard_deviation",
        "zero_velocity_clipping_guaranteed_by_observed_extrema",
        "position_and_target_denominators_unchanged",
        "maximum_absolute_velocity_preclamp",
        "maximum_absolute_effective_velocity_mean_shift",
        "maximum_effective_velocity_standard_deviation_ratio",
        "passes_all_gates",
    )
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["candidates"])


def run_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a critic normalization floor from an immutable drift run."
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    root = project_root()
    source = args.source_run.resolve()
    config_path = args.config or root / "configs/critic-normalization-floor-calibration.json"
    config = NormalizationFloorCalibrationConfig.from_dict(json.loads(config_path.read_text()))
    run = create_run_directory(root / "runs/normalization-floor-calibration")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        source_run=str(source),
        calibration_config=config.to_dict(),
        scientific_result=False,
        simulator_used=False,
        gpu_used=False,
    )
    write_json_atomic(run / "report.json", report)
    try:
        seed_data, artifacts = _load_source(source, config)
        summary = summarize_floor_candidates(seed_data, config)
        selected = summary["selected_minimum_standard_deviation"]
        if selected is None:
            raise RuntimeError("no candidate normalization floor passed all gates")
        source_normalization = seed_data[0]["normalization"]
        selected_config = CriticNormalizationConfig(
            schema_version=2,
            enabled=True,
            warmup_steps=source_normalization["warmup_steps"],
            epsilon=source_normalization["epsilon"],
            clip=source_normalization["clip"],
            minimum_standard_deviation=selected,
        ).to_dict()
        tracked = CriticNormalizationConfig.from_dict(
            json.loads((root / "configs/critic-normalization-active-groups-floor.json").read_text())
        ).to_dict()
        if selected_config != tracked:
            raise RuntimeError("selected floor does not match the tracked experiment config")
        _write_candidate_table(run / "candidate-summary.csv", summary)
        write_json_atomic(run / "calibration-summary.json", summary)
        write_json_atomic(run / "selected-critic-normalization.json", selected_config)
        report.update(
            status="passed",
            source_artifacts=artifacts,
            summary=summary,
            selected_critic_normalization=selected_config,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
