"""CPU-only contracts and audits for matched takeoff command replays."""

from __future__ import annotations

import csv
import math
from pathlib import Path

TRACE_COLUMNS = (
    "step",
    "phase",
    "agent_id",
    "command_vx_m_s",
    "command_vy_m_s",
    "command_vz_m_s",
    "source_x_m",
    "source_y_m",
    "source_z_m",
    "source_vz_m_s",
    "source_contact_force_n",
)
ARMS = ("four", "isolated", "no_downwash")
REPLAY_ARM_CHOICES = (*ARMS, "wake_guard")
REPLAY_COLUMNS = (
    "step",
    "phase",
    "agent_id",
    "command_vx_m_s",
    "command_vy_m_s",
    "command_vz_m_s",
    "x_m",
    "y_m",
    "z_m",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "qw",
    "qx",
    "qy",
    "qz",
    "rotor_raw_0",
    "rotor_raw_1",
    "rotor_raw_2",
    "rotor_raw_3",
    "rotor_applied_0",
    "rotor_applied_1",
    "rotor_applied_2",
    "rotor_applied_3",
    "rotor_throttle_0",
    "rotor_throttle_1",
    "rotor_throttle_2",
    "rotor_throttle_3",
    "rotor_thrust_sum_n",
    "model_force_x_n",
    "model_force_y_n",
    "model_force_z_n",
    "contact_force_n",
)
WAKE_COLUMNS = REPLAY_COLUMNS + (
    "requested_vx_m_s",
    "requested_vy_m_s",
    "requested_vz_m_s",
    "pre_x_m",
    "pre_y_m",
    "pre_z_m",
    "guard_active",
    "guard_unresolved",
)


def _finite(row: dict[str, str], names: tuple[str, ...]) -> None:
    for name in names:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite {name}")


def validate_trace_rows(rows: list[dict[str, str]], num_agents: int, focal_agent: int) -> int:
    if num_agents < 2 or not 0 <= focal_agent < num_agents or not rows:
        raise ValueError("invalid drone count, focal agent, or empty trace")
    if len(rows) % num_agents:
        raise ValueError("incomplete command step")
    for step in range(len(rows) // num_agents):
        block = rows[step * num_agents : (step + 1) * num_agents]
        phases = {row["phase"] for row in block}
        if len(phases) != 1 or not phases <= {"ground", "takeoff"}:
            raise ValueError("trace must contain one ground or takeoff phase per step")
        if step and rows[(step - 1) * num_agents]["phase"] == "takeoff" and phases == {"ground"}:
            raise ValueError("phase moved backward")
        for agent_id, row in enumerate(block):
            if int(row["step"]) != step or int(row["agent_id"]) != agent_id:
                raise ValueError("noncontiguous step or agent IDs")
            _finite(
                row, tuple(name for name in TRACE_COLUMNS if name.endswith(("_m", "_m_s", "_n")))
            )
            command = [float(row[f"command_v{axis}_m_s"]) for axis in "xyz"]
            if math.sqrt(sum(value * value for value in command)) > 0.50001:
                raise ValueError("command exceeds source 0.5 m/s limit")
    if not any(row["phase"] == "takeoff" for row in rows):
        raise ValueError("trace contains no takeoff")
    return len(rows) // num_agents


def load_trace(path: Path, num_agents: int = 4, focal_agent: int = 2) -> list[list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
            raise ValueError("command trace schema differs")
        rows = list(reader)
    steps = validate_trace_rows(rows, num_agents, focal_agent)
    return [rows[i * num_agents : (i + 1) * num_agents] for i in range(steps)]


def prepare_trace(
    source: Path, destination: Path, *, num_agents: int = 4, focal_agent: int = 2
) -> dict:
    """Select world zero's first intact episode; preserve its commands and source motion."""
    rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "env_id",
            "episode_step",
            "phase",
            "agent_id",
            "x_m",
            "y_m",
            "z_m",
            "vz_m_s",
            "contact_force_n",
            "command_vx_m_s",
            "command_vy_m_s",
            "command_vz_m_s",
        }
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("source telemetry lacks replay fields")
        for row in reader:
            if int(row["env_id"]) != 0:
                continue
            step = int(row["episode_step"]) - 1
            if step < 0:
                raise ValueError("source episode step is invalid")
            if rows and step < int(rows[-1]["step"]):
                break
            if len(rows) >= 10000 * num_agents:
                raise ValueError("source episode exceeds replay limit")
            rows.append(
                {
                    "step": str(step),
                    "phase": row["phase"],
                    "agent_id": row["agent_id"],
                    **{f"command_v{axis}_m_s": row[f"command_v{axis}_m_s"] for axis in "xyz"},
                    **{f"source_{axis}_m": row[f"{axis}_m"] for axis in "xyz"},
                    "source_vz_m_s": row["vz_m_s"],
                    "source_contact_force_n": row["contact_force_n"],
                }
            )
    steps = validate_trace_rows(rows, num_agents, focal_agent)
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {"steps": steps, "agent_rows": len(rows), "focal_agent": focal_agent, "source_env_id": 0}


def _first_contact(rows: list[dict[str, str]], airborne_height_m: float, threshold_n: float):
    airborne = False
    first_airborne = None
    for row in rows:
        step = int(row["step"])
        z = float(row["z_m"])
        if z >= airborne_height_m and not airborne:
            airborne, first_airborne = True, step
        if airborne and float(row["contact_force_n"]) > threshold_n:
            return first_airborne, step
    return first_airborne, None


def assess_replays(
    trace: Path,
    outputs: dict[str, Path],
    *,
    focal_agent: int = 2,
    airborne_height_m: float = 0.25,
    contact_threshold_n: float = 0.01,
) -> dict:
    commands = load_trace(trace, focal_agent=focal_agent)
    if set(outputs) != set(ARMS):
        raise ValueError("all three declared arms are required")
    summaries = {}
    source_focal = [step[focal_agent] for step in commands]
    for arm in ARMS:
        with outputs[arm].open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REPLAY_COLUMNS:
                raise ValueError(f"{arm}: replay CSV schema differs")
            rows = list(reader)
        if len(rows) != len(commands) * 4:
            raise ValueError(f"{arm}: incomplete replay")
        focal = []
        for step, source_step in enumerate(commands):
            block = rows[step * 4 : (step + 1) * 4]
            for agent_id, row in enumerate(block):
                if int(row["step"]) != step or int(row["agent_id"]) != agent_id:
                    raise ValueError(f"{arm}: noncontiguous step or agent ID")
                if row["phase"] != source_step[agent_id]["phase"]:
                    raise ValueError(f"{arm}: source phase changed")
                _finite(
                    row,
                    tuple(
                        name for name in REPLAY_COLUMNS if name not in {"step", "phase", "agent_id"}
                    ),
                )
                for axis in "xyz":
                    expected = (
                        0.0
                        if arm == "isolated" and agent_id != focal_agent
                        else float(source_step[agent_id][f"command_v{axis}_m_s"])
                    )
                    if abs(float(row[f"command_v{axis}_m_s"]) - expected) > 1e-6:
                        raise ValueError(f"{arm}: command differs from declared replay")
            focal.append(block[focal_agent])
        airborne_step, contact_step = _first_contact(focal, airborne_height_m, contact_threshold_n)
        prefix = min(750, len(focal))
        source_z_rmse = math.sqrt(
            sum(
                (float(focal[i]["z_m"]) - float(source_focal[i]["source_z_m"])) ** 2
                for i in range(prefix)
            )
            / prefix
        )
        summaries[arm] = {
            "focal_first_airborne_step": airborne_step,
            "focal_first_post_airborne_contact_step": contact_step,
            "focal_min_vz_m_s": min(float(row["vz_m_s"]) for row in focal),
            "focal_final_z_m": float(focal[-1]["z_m"]),
            "focal_max_downward_model_force_n": max(
                0.0, -min(float(row["model_force_z_n"]) for row in focal)
            ),
            "focal_max_abs_model_force_n": max(
                math.sqrt(sum(float(row[f"model_force_{axis}_n"]) ** 2 for axis in "xyz"))
                for row in focal
            ),
            "focal_mean_rotor_thrust_n": sum(float(row["rotor_thrust_sum_n"]) for row in focal)
            / len(focal),
            "source_altitude_rmse_first_750_steps_m": source_z_rmse,
        }
    source_contact = next(
        (int(row["step"]) for row in source_focal if float(row["source_z_m"]) >= airborne_height_m),
        None,
    )
    source_first_contact = None
    if source_contact is not None:
        source_first_contact = next(
            (
                int(row["step"])
                for row in source_focal
                if int(row["step"]) >= source_contact
                and float(row["source_contact_force_n"]) > contact_threshold_n
            ),
            None,
        )
    four_contact = summaries["four"]["focal_first_post_airborne_contact_step"]
    baseline_reproduced = (
        source_first_contact is not None
        and four_contact is not None
        and abs(four_contact - source_first_contact) <= 50
        and summaries["four"]["source_altitude_rmse_first_750_steps_m"] <= 0.15
    )
    no_downwash_contact = summaries["no_downwash"]["focal_first_post_airborne_contact_step"]
    isolated_contact = summaries["isolated"]["focal_first_post_airborne_contact_step"]
    downwash_force_removed = summaries["no_downwash"]["focal_max_abs_model_force_n"] <= 1e-5
    return {
        "status": "passed",
        "focal_agent": focal_agent,
        "steps": len(commands),
        "source_first_post_airborne_contact_step": source_first_contact,
        "arms": summaries,
        "baseline_reproduced_with_declared_tolerance": baseline_reproduced,
        "no_downwash_model_force_near_zero": downwash_force_removed,
        "modeled_downwash_causal_support": (
            baseline_reproduced
            and downwash_force_removed
            and no_downwash_contact is None
            and isolated_contact is None
        ),
        "interpretation": (
            "Model downwash is supported as necessary for this replay's contact; "
            "this is a simulator intervention, not a measured aerodynamic force."
            if baseline_reproduced
            and downwash_force_removed
            and no_downwash_contact is None
            and isolated_contact is None
            else "Inconclusive: inspect baseline fidelity, force, contact, and command traces."
        ),
    }


def save_replay_plot(
    trace: Path, outputs: dict[str, Path], destination: Path, focal_agent: int = 2
) -> None:
    """Write a dependency-free SVG; CSV remains the numerical source of truth."""
    commands = load_trace(trace, focal_agent=focal_agent)
    series = {
        "source": [float(step[focal_agent]["source_z_m"]) for step in commands],
    }
    forces = {}
    for arm in ARMS:
        with outputs[arm].open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if int(row["agent_id"]) == focal_agent]
        if len(rows) != len(commands):
            raise ValueError(f"{arm}: incomplete plot source")
        series[arm] = [float(row["z_m"]) for row in rows]
        forces[arm] = [float(row["model_force_z_n"]) for row in rows]
    n = len(commands)
    z_max = max(1.0, *(max(values) for values in series.values()))
    force_min = min(-0.1, *(min(values) for values in forces.values()))
    if "wake_guard" in outputs:
        with outputs["wake_guard"].open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if int(row["agent_id"]) == focal_agent]
        if len(rows) != len(commands):
            raise ValueError("wake_guard: incomplete plot source")
        series["wake_guard"] = [float(row["z_m"]) for row in rows]
        forces["wake_guard"] = [float(row["model_force_z_n"]) for row in rows]
    colors = {
        "source": "#24292f",
        "four": "#d73a49",
        "isolated": "#1f77b4",
        "no_downwash": "#2da44e",
        "wake_guard": "#9467bd",
    }

    def polyline(values, left, top, width, height, minimum, maximum):
        points = []
        for i, value in enumerate(values):
            x = left + width * i / max(n - 1, 1)
            y = top + height * (maximum - value) / (maximum - minimum)
            points.append(f"{x:.2f},{y:.2f}")
        return " ".join(points)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="610" viewBox="0 0 940 610">',
        '<rect width="940" height="610" fill="white"/>',
        '<text x="60" y="26" font-size="18">Focal drone 2: matched takeoff replay</text>',
        '<text x="60" y="53" font-size="13">Altitude (m)</text>',
        '<text x="60" y="349" font-size="13">Vertical model force (N)</text>',
        '<path d="M60 65 V300 H900 M60 360 V555 H900" fill="none" stroke="#666"/>',
    ]
    for label, values in series.items():
        dash = ' stroke-dasharray="7 4"' if label == "source" else ""
        parts.append(
            f'<polyline points="{polyline(values, 60, 65, 840, 235, 0, z_max)}" '
            f'fill="none" stroke="{colors[label]}" stroke-width="1.5"{dash}/>'
        )
    for label, values in forces.items():
        parts.append(
            f'<polyline points="{polyline(values, 60, 360, 840, 195, force_min, 0.1)}" '
            f'fill="none" stroke="{colors[label]}" stroke-width="1.5"/>'
        )
    for index, label in enumerate(series):
        parts.append(
            f'<text x="{80 + index * 185}" y="590" font-size="13" fill="{colors[label]}">'
            f"{label.replace('_', ' ')}</text>"
        )
    parts.append("</svg>")
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def assess_wake_guard(
    trace: Path,
    output: Path,
    *,
    focal_agent: int = 2,
    airborne_height_m: float = 0.25,
    contact_threshold_n: float = 0.01,
    minimum_separation_m: float = 0.55,
) -> dict:
    """Recompute every local guard choice from saved pre-step poses and requests."""
    from align.tasks.wake_guard import guard_focal_command

    commands = load_trace(trace, focal_agent=focal_agent)
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != WAKE_COLUMNS:
            raise ValueError("wake guard CSV schema differs")
        rows = list(reader)
    if len(rows) != len(commands) * 4:
        raise ValueError("wake guard replay is incomplete")
    active_steps = unresolved_steps = 0
    minimum_separation = math.inf
    focal = []
    for step, source_step in enumerate(commands):
        block = rows[step * 4 : (step + 1) * 4]
        positions = tuple(tuple(float(row[f"pre_{axis}_m"]) for axis in "xyz") for row in block)
        requested = tuple(
            tuple(float(source_step[agent][f"command_v{axis}_m_s"]) for axis in "xyz")
            for agent in range(4)
        )
        if source_step[0]["phase"] == "takeoff":
            expected, details = guard_focal_command(positions, requested, focal_agent)
        else:
            expected, details = requested[focal_agent], {"active": False, "unresolved": False}
        active_steps += int(details["active"])
        unresolved_steps += int(details["unresolved"])
        for agent, row in enumerate(block):
            if int(row["step"]) != step or int(row["agent_id"]) != agent:
                raise ValueError("wake guard step or agent ID differs")
            if row["phase"] != source_step[agent]["phase"]:
                raise ValueError("wake guard phase differs")
            _finite(
                row,
                tuple(name for name in WAKE_COLUMNS if name not in {"step", "phase", "agent_id"}),
            )
            for axis in "xyz":
                actual_requested = float(row[f"requested_v{axis}_m_s"])
                if abs(actual_requested - requested[agent]["xyz".index(axis)]) > 1e-6:
                    raise ValueError("wake guard source request differs")
                actual = float(row[f"command_v{axis}_m_s"])
                target = (
                    expected["xyz".index(axis)]
                    if agent == focal_agent
                    else requested[agent]["xyz".index(axis)]
                )
                if abs(actual - target) > 1e-5:
                    raise ValueError("wake guard command differs from CPU rule")
            if agent == focal_agent and (
                int(float(row["guard_active"])) != int(details["active"])
                or int(float(row["guard_unresolved"])) != int(details["unresolved"])
            ):
                raise ValueError("wake guard activity flag differs")
        if source_step[0]["phase"] == "takeoff":
            positions_after = [tuple(float(row[f"{axis}_m"]) for axis in "xyz") for row in block]
            minimum_separation = min(
                minimum_separation,
                *(
                    math.dist(positions_after[a], positions_after[b])
                    for a in range(4)
                    for b in range(a + 1, 4)
                ),
            )
        focal.append(block[focal_agent])
    first_airborne, first_contact = _first_contact(focal, airborne_height_m, contact_threshold_n)
    return {
        "status": "passed",
        "steps": len(commands),
        "agent_rows": len(rows),
        "guard_active_steps": active_steps,
        "guard_unresolved_steps": unresolved_steps,
        "focal_first_airborne_step": first_airborne,
        "focal_first_post_airborne_contact_step": first_contact,
        "focal_final_z_m": float(focal[-1]["z_m"]),
        "focal_max_downward_model_force_n": max(
            0.0, -min(float(row["model_force_z_n"]) for row in focal)
        ),
        "minimum_takeoff_separation_m": minimum_separation,
        "contact_free": first_contact is None,
        "separation_safe": minimum_separation >= minimum_separation_m,
        "candidate_accepted": (
            active_steps > 0
            and unresolved_steps == 0
            and first_contact is None
            and minimum_separation >= minimum_separation_m
        ),
    }
