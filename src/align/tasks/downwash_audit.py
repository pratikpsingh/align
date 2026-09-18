"""Recompute a vertical-thrust downwash proxy from saved drone positions."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from align.formations.downwash import vertical_downwash_exposure


def audit_downwash(
    telemetry_csv: Path, first_episode_lengths: dict[int, int]
) -> tuple[dict, list[dict]]:
    """Return per-drone exposure rows from complete first-episode snapshots.

    The proxy uses the positions that actually occurred, but assumes each
    upper drone's thrust is vertical. It is not a measured aerodynamic force.
    """
    by_step: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)
    with telemetry_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            env_id = int(row["env_id"])
            evaluation_step = int(row["evaluation_step"])
            if (
                env_id not in first_episode_lengths
                or evaluation_step >= first_episode_lengths[env_id]
            ):
                continue
            episode_step = int(row["episode_step"])
            if episode_step != evaluation_step + 1:
                raise ValueError("first episode telemetry step differs from evaluation step")
            agent_id = int(row["agent_id"])
            group = by_step[(env_id, episode_step)]
            if agent_id in group:
                raise ValueError("duplicate first-episode drone row")
            group[agent_id] = row
    if not by_step:
        raise ValueError("no first-episode drone rows")
    output = []
    maxima = {env_id: (0.0, None, None) for env_id in first_episode_lengths}
    for (env_id, step), group in sorted(by_step.items()):
        if len(group) < 2 or sorted(group) != list(range(len(group))):
            raise ValueError("snapshot has missing or noncontiguous drone IDs")
        positions = tuple(
            tuple(float(group[agent][f"{axis}_m"]) for axis in "xyz") for agent in sorted(group)
        )
        exposure = vertical_downwash_exposure(positions)
        for agent_id, value in enumerate(exposure):
            row = group[agent_id]
            if not math.isfinite(value):
                raise ValueError("nonfinite downwash proxy")
            output.append(
                {
                    "env_id": env_id,
                    "episode_step": step,
                    "agent_id": agent_id,
                    "vertical_thrust_exposure_proxy": value,
                    "z_m": positions[agent_id][2],
                    "target_z_m": float(row["target_z_m"]),
                    "vz_m_s": float(row["vz_m_s"]),
                    "command_vz_m_s": float(row["command_vz_m_s"]),
                    "contact_force_n": float(row["contact_force_n"]),
                }
            )
            if value > maxima[env_id][0]:
                maxima[env_id] = (value, step, agent_id)
    return (
        {
            "status": "passed",
            "model": "pinned_omnidrones_vertical_thrust_proxy_kr2_kz0.3",
            "physical_force_measured": False,
            "raw_drone_rows": len(output),
            "snapshots": len(by_step),
            "environment_maxima": [
                {
                    "env_id": env_id,
                    "maximum_exposure_proxy": maximum,
                    "episode_step": step,
                    "agent_id": agent_id,
                }
                for env_id, (maximum, step, agent_id) in sorted(maxima.items())
            ],
        },
        output,
    )
