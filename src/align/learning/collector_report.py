"""Host-side audit of saved live recurrent-collector artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_collector_run(run: Path) -> dict:
    config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    rollout = config["rollout"]
    expected_rows = rollout["horizon"] * rollout["num_envs"] * rollout["num_agents"]
    with (run / "collector.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric_columns = (
        "action_0",
        "action_1",
        "action_2",
        "action_3",
        "old_log_prob",
        "team_reward",
        "value",
        "bootstrap_value",
    )
    finite = all(math.isfinite(float(row[column])) for row in rows for column in numeric_columns)
    bounded = all(
        -1.0 <= float(row[f"action_{index}"]) <= 1.0
        for row in rows
        for index in range(rollout["action_dim"])
    )
    keys = [(row["global_step"], row["env_id"], row["agent_id"]) for row in rows]
    mutually_exclusive = all(
        not (row["terminated"] == "True" and row["truncated"] == "True") for row in rows
    )
    step_groups = {}
    for row in rows:
        key = (row["global_step"], row["env_id"])
        values = (
            row["team_reward"],
            row["value"],
            row["bootstrap_value"],
            row["terminated"],
            row["truncated"],
            row["reason_code"],
        )
        step_groups.setdefault(key, set()).add(values)
    raw = metrics["raw_rollout"]
    checkpoint = metrics["policy_checkpoint"]
    raw_path = run / raw["path"]
    checkpoint_path = run / checkpoint["path"]
    checks = {
        "config_has_all_collector_sections": set(config)
        == {"construction", "observation", "reward", "task", "policy", "rollout", "collector"},
        "container_report_passed": metrics.get("status") == "passed"
        and bool(metrics.get("checks"))
        and all(value is True for value in metrics["checks"].values()),
        "complete_csv_row_count": len(rows) == expected_rows,
        "unique_agent_step_rows": len(set(keys)) == len(keys),
        "finite_csv_values": finite,
        "bounded_csv_actions": bounded,
        "terminal_flags_mutually_exclusive": mutually_exclusive,
        "team_fields_consistent_within_environment_step": all(
            len(values) == 1 for values in step_groups.values()
        ),
        "true_termination_rows_present": any(row["terminated"] == "True" for row in rows),
        "truncation_rows_present": any(row["truncated"] == "True" for row in rows),
        "raw_rollout_hash_matches": raw_path.is_file()
        and raw_path.stat().st_size == raw["bytes"]
        and _digest(raw_path) == raw["sha256"],
        "policy_checkpoint_hash_matches": checkpoint_path.is_file()
        and checkpoint_path.stat().st_size == checkpoint["bytes"]
        and _digest(checkpoint_path) == checkpoint["sha256"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "csv_rows": len(rows),
        "expected_csv_rows": expected_rows,
        "unique_environment_steps": len(step_groups),
        "raw_rollout": raw,
        "policy_checkpoint": checkpoint,
    }
