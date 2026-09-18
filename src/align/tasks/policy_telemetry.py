"""CPU-only schema and audit for frozen-policy drone telemetry."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from align.tasks.reward import COMPONENT_NAMES, TIME_INTEGRATED_COMPONENTS

VECTOR_FIELDS = {
    "pre_position": ("pre_x_m", "pre_y_m", "pre_z_m"),
    "pre_velocity": ("pre_vx_m_s", "pre_vy_m_s", "pre_vz_m_s"),
    "position": ("x_m", "y_m", "z_m"),
    "velocity": ("vx_m_s", "vy_m_s", "vz_m_s"),
    "target": ("target_x_m", "target_y_m", "target_z_m"),
    "command": ("command_vx_m_s", "command_vy_m_s", "command_vz_m_s"),
}
TELEMETRY_COLUMNS = (
    "evaluation_step",
    "env_id",
    "episode_step",
    "phase",
    "formation_kind",
    "agent_id",
    *(name for fields in VECTOR_FIELDS.values() for name in fields),
    "qw",
    "qx",
    "qy",
    "qz",
    *(f"action_{i}" for i in range(4)),
    *(f"rotor_raw_{i}" for i in range(4)),
    *(f"rotor_applied_{i}" for i in range(4)),
    *(f"raw_{name}" for name in COMPONENT_NAMES),
    *(f"weighted_{name}" for name in COMPONENT_NAMES),
    "agent_reward",
    "team_reward",
    "contact_force_n",
)
GUARDED_TELEMETRY_COLUMNS = TELEMETRY_COLUMNS + (
    "requested_vx_m_s",
    "requested_vy_m_s",
    "requested_vz_m_s",
    "wake_guard_active",
    "wake_guard_unresolved",
)


def audit_telemetry(
    telemetry_path: Path,
    evaluation_path: Path,
    *,
    num_agents: int,
    max_speed_m_s: float,
    tolerance: float = 3e-4,
    reward_config: dict | None = None,
    wake_guard: bool = False,
) -> dict:
    """Cross-check each raw drone transition against aggregate evaluation rows."""
    if num_agents < 2 or max_speed_m_s <= 0 or tolerance <= 0:
        raise ValueError("invalid telemetry audit parameters")
    with evaluation_path.open(newline="", encoding="utf-8") as stream:
        aggregate = {
            (int(row["evaluation_step"]), int(row["env_id"])): row for row in csv.DictReader(stream)
        }
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with telemetry_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = GUARDED_TELEMETRY_COLUMNS if wake_guard else TELEMETRY_COLUMNS
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError("telemetry schema mismatch")
        for row in reader:
            groups[int(row["evaluation_step"]), int(row["env_id"])].append(row)
    if not aggregate or set(groups) != set(aggregate):
        raise ValueError("missing or unexpected environment steps")
    formation_rows = 0
    aligned_commands = 0
    nonzero_commands = 0
    closing_velocity = 0
    guard_active_steps = 0
    guard_unresolved_steps = 0
    for key, rows in groups.items():
        reference = aggregate[key]
        if len(rows) != num_agents or {int(r["agent_id"]) for r in rows} != set(range(num_agents)):
            raise ValueError(f"missing or duplicate agents at {key}")
        positions, targets, rewards = [], [], []
        rows.sort(key=lambda row: int(row["agent_id"]))
        expected_guard_command = None
        guard_details = {"active": False, "unresolved": False}
        if wake_guard:
            from align.tasks.wake_guard import guard_focal_command

            if num_agents != 4:
                raise ValueError("wake guard telemetry requires four agents")
            pre_positions = tuple(
                tuple(float(row[f"pre_{axis}_m"]) for axis in "xyz") for row in rows
            )
            requests = []
            for row in rows:
                action_values = [float(row[f"action_{i}"]) for i in range(4)]
                length = math.sqrt(math.fsum(value * value for value in action_values[:3]))
                requests.append(
                    tuple(
                        value / length * abs(action_values[3]) * max_speed_m_s
                        if length > 1e-8
                        else 0.0
                        for value in action_values[:3]
                    )
                )
            if rows[0]["phase"] == "takeoff":
                expected_guard_command, guard_details = guard_focal_command(
                    pre_positions, tuple(requests), 2
                )
            else:
                expected_guard_command = requests[2]
            guard_active_steps += int(guard_details["active"])
            guard_unresolved_steps += int(guard_details["unresolved"])
        for row in rows:
            if (row["phase"], row["formation_kind"], row["episode_step"]) != (
                reference["phase"],
                reference["formation_kind"],
                reference["episode_step"],
            ):
                raise ValueError(f"phase or template mismatch at {key}")
            values = [
                float(row[name]) for name in columns if name not in {"phase", "formation_kind"}
            ]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"nonfinite telemetry at {key}")
            vectors = {
                name: tuple(float(row[field]) for field in fields)
                for name, fields in VECTOR_FIELDS.items()
            }
            positions.append(vectors["position"])
            targets.append(vectors["target"])
            weights = [float(row[f"weighted_{name}"]) for name in COMPONENT_NAMES]
            if reward_config is not None:
                for name in COMPONENT_NAMES:
                    expected = float(row[f"raw_{name}"]) * float(reward_config[f"{name}_weight"])
                    if name in TIME_INTEGRATED_COMPONENTS:
                        expected *= float(reward_config["control_dt_seconds"])
                    if abs(expected - float(row[f"weighted_{name}"])) > tolerance:
                        raise ValueError(f"reward weighting mismatch at {key}: {name}")
            agent_reward = float(row["agent_reward"])
            if abs(math.fsum(weights) - agent_reward) > tolerance:
                raise ValueError(f"reward components do not close at {key}")
            rewards.append(agent_reward)
            action = [float(row[f"action_{i}"]) for i in range(4)]
            if any(abs(a) > 1.0 + tolerance for a in action):
                raise ValueError(f"unbounded action at {key}")
            norm = math.sqrt(math.fsum(a * a for a in action[:3]))
            expected_command = tuple(
                a / norm * abs(action[3]) * max_speed_m_s if norm > 1e-8 else 0.0
                for a in action[:3]
            )
            if wake_guard:
                requested = tuple(float(row[f"requested_v{axis}_m_s"]) for axis in "xyz")
                if (
                    max(abs(a - b) for a, b in zip(expected_command, requested, strict=True))
                    > tolerance
                ):
                    raise ValueError(f"action-to-request mismatch at {key}")
                agent_id = int(row["agent_id"])
                expected_command = expected_guard_command if agent_id == 2 else expected_command
                if int(float(row["wake_guard_active"])) != (
                    int(guard_details["active"]) if agent_id == 2 else 0
                ) or int(float(row["wake_guard_unresolved"])) != (
                    int(guard_details["unresolved"]) if agent_id == 2 else 0
                ):
                    raise ValueError(f"wake guard flag mismatch at {key}")
            if (
                max(abs(a - b) for a, b in zip(expected_command, vectors["command"], strict=True))
                > tolerance
            ):
                raise ValueError(f"action-to-command mismatch at {key}")
            for i in range(4):
                raw = float(row[f"rotor_raw_{i}"])
                applied = float(row[f"rotor_applied_{i}"])
                if abs(min(1.0, max(-1.0, raw)) - applied) > tolerance:
                    raise ValueError(f"rotor clamp mismatch at {key}")
            if row["phase"] == "formation":
                formation_rows += 1
                command = vectors["command"]
                error = tuple(t - p for t, p in zip(targets[-1], positions[-1], strict=True))
                desired = tuple(
                    c - v for c, v in zip(command, vectors["pre_velocity"], strict=True)
                )
                response = tuple(
                    v - p for v, p in zip(vectors["velocity"], vectors["pre_velocity"], strict=True)
                )
                if math.sqrt(math.fsum(c * c for c in command)) > 1e-6:
                    nonzero_commands += 1
                    aligned_commands += (
                        math.fsum(c * e for c, e in zip(command, error, strict=True)) > 0
                    )
                    closing_velocity += (
                        math.fsum(d * r for d, r in zip(desired, response, strict=True)) > 0
                    )
        if abs(math.fsum(rewards) / num_agents - float(reference["team_reward"])) > tolerance:
            raise ValueError(f"team reward mismatch at {key}")
        assigned = math.sqrt(
            math.fsum(math.dist(p, t) ** 2 for p, t in zip(positions, targets, strict=True))
            / num_agents
        )
        pairs = [(i, j) for i in range(num_agents) for j in range(i + 1, num_agents)]
        pairwise = math.sqrt(
            math.fsum(
                (math.dist(positions[i], positions[j]) - math.dist(targets[i], targets[j])) ** 2
                for i, j in pairs
            )
            / len(pairs)
        )
        if (
            abs(assigned - float(reference["assigned_rmse_m"])) > tolerance
            or abs(pairwise - float(reference["pairwise_rmse_m"])) > tolerance
        ):
            raise ValueError(f"formation geometry mismatch at {key}")
    return {
        "status": "passed",
        "environment_rows": len(groups),
        "drone_rows": sum(map(len, groups.values())),
        "formation_drone_rows": formation_rows,
        "nonzero_formation_commands": nonzero_commands,
        "target_aligned_command_fraction": aligned_commands / nonzero_commands
        if nonzero_commands
        else None,
        "positive_velocity_response_fraction": closing_velocity / nonzero_commands
        if nonzero_commands
        else None,
        "reward_and_geometry_recomputed": True,
        "wake_guard_enabled": wake_guard,
        "wake_guard_active_steps": guard_active_steps,
        "wake_guard_unresolved_steps": guard_unresolved_steps,
    }


def audit_airborne_contacts(
    telemetry_path: Path,
    evaluation_path: Path,
    *,
    airborne_height_m: float,
    contact_force_threshold_n: float,
) -> dict:
    """Find first post-ascent contact per drone and episode in a saved replay."""
    if airborne_height_m <= 0 or contact_force_threshold_n <= 0:
        raise ValueError("contact audit thresholds must be positive")
    with evaluation_path.open(newline="", encoding="utf-8") as stream:
        aggregate = {
            (int(row["evaluation_step"]), int(row["env_id"])): row for row in csv.DictReader(stream)
        }
    if not aggregate:
        raise ValueError("empty evaluation CSV")
    episode_index: dict[int, int] = defaultdict(int)
    last_episode_step: dict[int, int] = {}
    first_airborne: dict[tuple[int, int], int] = {}
    recorded: set[tuple[int, int, int]] = set()
    events = []
    drone_rows = 0
    with telemetry_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) not in (TELEMETRY_COLUMNS, GUARDED_TELEMETRY_COLUMNS):
            raise ValueError("telemetry schema mismatch")
        for row in reader:
            drone_rows += 1
            env_id = int(row["env_id"])
            agent_id = int(row["agent_id"])
            episode_step = int(row["episode_step"])
            key = (int(row["evaluation_step"]), env_id)
            reference = aggregate.get(key)
            if reference is None:
                raise ValueError(f"missing evaluation row at {key}")
            if int(reference["episode_step"]) != episode_step or reference["phase"] != row["phase"]:
                raise ValueError(f"episode or phase mismatch at {key}")
            if env_id in last_episode_step and episode_step < last_episode_step[env_id]:
                episode_index[env_id] += 1
                first_airborne = {
                    agent_key: step
                    for agent_key, step in first_airborne.items()
                    if agent_key[0] != env_id
                }
            last_episode_step[env_id] = episode_step
            height = float(row["z_m"])
            force = float(row["contact_force_n"])
            command_z = float(row["command_vz_m_s"])
            if not all(map(math.isfinite, (height, force, command_z))):
                raise ValueError(f"nonfinite contact telemetry at {key}")
            agent_key = (env_id, agent_id)
            if height > airborne_height_m:
                first_airborne.setdefault(agent_key, episode_step)
            event_key = (env_id, episode_index[env_id], agent_id)
            if (
                agent_key in first_airborne
                and force > contact_force_threshold_n
                and event_key not in recorded
            ):
                recorded.add(event_key)
                events.append(
                    {
                        "env_id": env_id,
                        "episode_index": episode_index[env_id],
                        "agent_id": agent_id,
                        "first_airborne_episode_step": first_airborne[agent_key],
                        "first_contact_episode_step": episode_step,
                        "first_contact_evaluation_step": key[0],
                        "phase": row["phase"],
                        "height_m": height,
                        "contact_force_n": force,
                        "command_vz_m_s": command_z,
                        "reason_code_at_contact": int(reference["reason_code"]),
                        "terminated_at_contact": reference["terminated"] == "True",
                    }
                )
    if drone_rows == 0:
        raise ValueError("empty telemetry CSV")
    return {
        "status": "passed",
        "airborne_height_m": airborne_height_m,
        "contact_force_threshold_n": contact_force_threshold_n,
        "drone_rows": drone_rows,
        "first_post_airborne_contacts": events,
        "first_post_airborne_contact_count": len(events),
    }


def summarize_telemetry(
    telemetry_path: Path,
    *,
    max_speed_m_s: float,
    control_dt_seconds: float,
    max_episode_steps: int,
    success_dwell_steps: int,
) -> dict:
    """Summarize phase behavior and a commanded-speed reachability bound.

    The bound concerns the declared velocity command. It is not a guarantee
    about realized motion when the inner controller overshoots or is disturbed.
    """
    if (
        max_speed_m_s <= 0
        or control_dt_seconds <= 0
        or max_episode_steps <= 0
        or not 0 < success_dwell_steps <= max_episode_steps
    ):
        raise ValueError("invalid reachability contract")
    phases: dict[str, dict] = {}
    first_formation: dict[tuple[int, int], dict] = {}
    with telemetry_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) not in (TELEMETRY_COLUMNS, GUARDED_TELEMETRY_COLUMNS):
            raise ValueError("telemetry schema mismatch")
        for row in reader:
            phase = row["phase"]
            if phase not in ("ground", "takeoff", "formation"):
                raise ValueError("unknown reward phase")
            summary = phases.setdefault(
                phase,
                {
                    "drone_rows": 0,
                    "target_distance_sum_m": 0.0,
                    "command_speed_sum_m_s": 0.0,
                    "realized_speed_sum_m_s": 0.0,
                    "command_toward_target_rows": 0,
                    **{f"weighted_{name}_sum": 0.0 for name in COMPONENT_NAMES},
                },
            )
            summary["drone_rows"] += 1
            position = tuple(float(row[name]) for name in VECTOR_FIELDS["position"])
            target = tuple(float(row[name]) for name in VECTOR_FIELDS["target"])
            command = tuple(float(row[name]) for name in VECTOR_FIELDS["command"])
            velocity = tuple(float(row[name]) for name in VECTOR_FIELDS["velocity"])
            summary["target_distance_sum_m"] += math.dist(position, target)
            summary["command_speed_sum_m_s"] += math.dist(command, (0.0, 0.0, 0.0))
            summary["realized_speed_sum_m_s"] += math.dist(velocity, (0.0, 0.0, 0.0))
            summary["command_toward_target_rows"] += (
                math.fsum(c * (t - p) for c, t, p in zip(command, target, position, strict=True))
                > 0
            )
            for name in COMPONENT_NAMES:
                summary[f"weighted_{name}_sum"] += float(row[f"weighted_{name}"])
            if phase == "formation":
                key = (int(row["env_id"]), int(row["agent_id"]))
                first_formation.setdefault(key, row)
    phase_means = {}
    for phase, summary in phases.items():
        count = summary["drone_rows"]
        phase_means[phase] = {
            "drone_rows": count,
            "mean_target_distance_m": summary["target_distance_sum_m"] / count,
            "mean_command_speed_m_s": summary["command_speed_sum_m_s"] / count,
            "mean_realized_speed_m_s": summary["realized_speed_sum_m_s"] / count,
            "command_toward_target_fraction": summary["command_toward_target_rows"] / count,
            "mean_weighted_reward": {
                name: summary[f"weighted_{name}_sum"] / count for name in COMPONENT_NAMES
            },
        }
    latest_dwell_start = max_episode_steps - success_dwell_steps + 1
    by_environment: dict[int, list[float]] = defaultdict(list)
    available_steps = {}
    for (env_id, _agent_id), row in first_formation.items():
        steps = max(0, latest_dwell_start - int(row["episode_step"]))
        available_steps[env_id] = steps
        z_error = abs(float(row["target_z_m"]) - float(row["z_m"]))
        by_environment[env_id].append(
            max(0.0, z_error - steps * control_dt_seconds * max_speed_m_s)
        )
    return {
        "formation_phase_reached": bool(first_formation),
        "phase_means": phase_means,
        "latest_dwell_start_episode_step": latest_dwell_start,
        "remaining_command_steps_at_first_formation": available_steps,
        "maximum_commanded_travel_before_dwell_m": {
            env_id: steps * control_dt_seconds * max_speed_m_s
            for env_id, steps in available_steps.items()
        },
        "altitude_only_assigned_rmse_lower_bound_at_dwell_m": {
            env_id: math.sqrt(math.fsum(error * error for error in errors) / len(errors))
            for env_id, errors in by_environment.items()
        },
        "bound_assumption": (
            "Velocity command magnitude never exceeds the configured maximum; "
            "actual controller dynamics may differ."
        ),
    }
