"""Host-side audit and table aggregation for bounded task-connected training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from align.learning.checkpoint_store import CheckpointStore


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit_training_run(run: Path) -> dict:
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    rollout = config["rollout"]
    ppo = config["ppo"]
    attempt_ids = ("attempt-0001", "attempt-0002")
    metrics = [json.loads((run / attempt / "metrics.json").read_text()) for attempt in attempt_ids]
    rollouts = [_read_rows(run / attempt / "rollout.csv") for attempt in attempt_ids]
    updates = [_read_rows(run / attempt / "updates.csv") for attempt in attempt_ids]
    expected_rollout_rows = rollout["horizon"] * rollout["num_envs"]
    numeric_rollout = (
        "team_reward",
        "value",
        "bootstrap_value",
        "action_min",
        "action_max",
        "assigned_rmse_m",
        "pairwise_rmse_m",
        "minimum_separation_m",
    )
    finite_rollouts = all(
        math.isfinite(float(row[name]))
        for rows in rollouts
        for row in rows
        for name in numeric_rollout
    )
    bounded_actions = all(
        -1.0 <= float(row["action_min"]) <= float(row["action_max"]) <= 1.0
        for rows in rollouts
        for row in rows
    )
    unique_rollouts = all(
        len({(row["rollout_step"], row["env_id"]) for row in rows}) == len(rows)
        for rows in rollouts
    )
    finite_updates = all(
        math.isfinite(float(value))
        for rows in updates
        for row in rows
        for key, value in row.items()
        if key not in {"attempt_id"}
    )

    config_sha256 = _digest(config_path)
    store = CheckpointStore(
        run / "checkpoints",
        run_id=run.name,
        config_sha256=config_sha256,
        file_mode=0o644,
    )
    verified = []
    for path in sorted((run / "checkpoints").glob("checkpoint-*.json")):
        try:
            manifest, _ = store.verify(path)
            verified.append(manifest)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    by_id = {manifest["checkpoint_id"]: manifest for manifest in verified}
    latest, _ = store.latest_valid()
    first, second = metrics
    first_start = by_id.get(first["start_checkpoint_id"], {})
    first_end = by_id.get(first["committed_checkpoint_id"], {})
    second_end = by_id.get(second["committed_checkpoint_id"], {})
    expected_environment_increment = rollout["horizon"] * rollout["num_envs"]
    expected_agent_increment = expected_environment_increment * rollout["num_agents"]
    checks = {
        "resolved_config_has_exact_sections": set(config)
        == {
            "construction",
            "observation",
            "reward",
            "task",
            "policy",
            "rollout",
            "ppo",
            "recovery",
            "training",
            "critic_normalization",
        },
        "both_attempts_passed_container_checks": all(
            item.get("status") == "passed"
            and bool(item.get("checks"))
            and all(value is True for value in item["checks"].values())
            for item in metrics
        ),
        "attempt_roles_are_explicit": first["resumed"] is False and second["resumed"] is True,
        "second_attempt_loaded_first_update": second["loaded_checkpoint_id"]
        == first["committed_checkpoint_id"],
        "counters_continue_without_duplication": first["start_counters"]["completed_updates"] == 0
        and first["end_counters"]["completed_updates"] == 1
        and second["start_counters"] == first["end_counters"]
        and second["end_counters"]["completed_updates"] == 2,
        "transition_counters_are_exact": second["end_counters"]["environment_transitions"]
        == 2 * expected_environment_increment
        and second["end_counters"]["agent_transitions"] == 2 * expected_agent_increment,
        "three_valid_checkpoints_exist": len(verified) >= 3,
        "first_update_descends_from_initial_checkpoint": first_end.get("parent_checkpoint_sha256")
        == first_start.get("payload", {}).get("sha256"),
        "second_update_descends_from_loaded_checkpoint": second_end.get("parent_checkpoint_sha256")
        == first_end.get("payload", {}).get("sha256"),
        "latest_checkpoint_is_second_update": latest["checkpoint_id"]
        == second["committed_checkpoint_id"],
        "abandoned_episode_accounting_matches_checkpoint": second["abandoned_environment_episodes"]
        == first["unfinished_environment_count"],
        "rollout_row_counts_are_exact": all(
            len(rows) == expected_rollout_rows for rows in rollouts
        ),
        "rollout_keys_are_unique": unique_rollouts,
        "rollout_values_are_finite": finite_rollouts,
        "rollout_actions_are_bounded": bounded_actions,
        "update_epoch_row_counts_are_exact": all(
            len(rows) == ppo["update_epochs"] for rows in updates
        ),
        "update_values_are_finite": finite_updates,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "attempts": list(attempt_ids),
        "valid_checkpoint_count": len(verified),
        "latest_checkpoint_id": latest["checkpoint_id"],
        "rollout_rows_per_attempt": [len(rows) for rows in rollouts],
        "update_rows_per_attempt": [len(rows) for rows in updates],
        "final_counters": second["end_counters"],
    }


def write_combined_tables(run: Path) -> None:
    """Preserve attempt tables and also provide convenient run-level raw tables."""
    for name in ("rollout.csv", "updates.csv"):
        rows = []
        fieldnames = None
        for attempt in ("attempt-0001", "attempt-0002"):
            with (run / attempt / name).open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames
                rows.extend(reader)
        with (run / f"training-{name}").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
