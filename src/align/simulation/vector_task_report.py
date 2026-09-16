"""Host-side structural audit for saved vector-task trajectories."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from align.simulation.multi_drone_contract import MultiDroneConfig


def audit_saved_scenario(run: Path, expected_envs: int):
    config = json.loads((run / "config.json").read_text())
    construction = MultiDroneConfig.from_dict(config["construction"])
    metrics = json.loads((run / "metrics.json").read_text())
    with (run / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("vector task trajectory is empty")

    numeric = (
        "global_step",
        "env_id",
        "episode_step",
        "agent_id",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "target_x",
        "target_y",
        "target_z",
        "a0",
        "a1",
        "a2",
        "a3",
        "contact_force_n",
        "reward",
    )
    finite = all(math.isfinite(float(row[key])) for row in rows for key in numeric)
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        key = (int(row["global_step"]), int(row["env_id"]))
        groups[key].append(row)
        identities.add((key, int(row["agent_id"])))
    complete_groups = all(
        len(group) == construction.num_agents
        and {int(row["agent_id"]) for row in group} == set(range(construction.num_agents))
        for group in groups.values()
    )
    env_ids = {int(row["env_id"]) for row in rows}
    bounded_actions = all(abs(float(row[f"a{index}"])) <= 1.0 for row in rows for index in range(4))
    exclusive_masks = all(
        not (row["terminated"] == "True" and row["truncated"] == "True") for row in rows
    )
    reasons_consistent = all(
        bool(row["reason"]) == (row["terminated"] == "True" or row["truncated"] == "True")
        for row in rows
    )
    checks = {
        "finite_raw_trajectory": finite,
        "complete_agent_groups": complete_groups,
        "unique_agent_rows": len(identities) == len(rows),
        "expected_environments": env_ids == set(range(expected_envs)),
        "bounded_saved_actions": bounded_actions,
        "exclusive_terminal_masks": exclusive_masks,
        "done_reason_consistency": reasons_consistent,
        "reported_sample_count": metrics.get("agent_steps") == len(rows),
        "container_checks_passed": bool(metrics.get("checks"))
        and all(value is True for value in metrics["checks"].values()),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "metrics": {
            "trajectory_rows": len(rows),
            "environment_steps": len(groups),
            "environments": len(env_ids),
            "agents_per_environment": construction.num_agents,
        },
    }


__all__ = ["audit_saved_scenario"]
