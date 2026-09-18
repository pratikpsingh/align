"""CPU-only contract for deterministic multi-drone formation construction."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations

from align.formations import (
    SUPPORTED_KINDS,
    Assignment,
    assign_agents_to_slots,
    evaluate_formation,
    generate_template,
    place_template,
)
from align.simulation.contract import direction_speed

_FLOAT32_SERIALIZATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MultiDroneConfig:
    schema_version: int = 2
    seed: int = 23
    num_agents: int = 4
    formation_kind: str = "plane"
    physics_dt: float = 0.01
    control_decimation: int = 1
    ground_height_m: float = 0.06
    ground_spacing_m: float = 1.0
    target_spacing_m: float = 1.0
    target_center_m: tuple[float, float, float] = (0.0, 0.0, 1.5)
    max_speed_m_s: float = 0.5
    position_gain_s_inv: float = 0.8
    ground_settle_seconds: float = 0.5
    takeoff_seconds: float = 5.0
    formation_timeout_seconds: float = 8.0
    dwell_seconds: float = 2.0
    repetitions: int = 2
    airborne_height_m: float = 0.25
    contact_force_threshold_n: float = 0.01
    minimum_separation_m: float = 0.55
    formation_rmse_tolerance_m: float = 0.10
    pairwise_rmse_tolerance_m: float = 0.08
    speed_tolerance_m_s: float = 0.08
    reset_tolerance: float = 1e-6
    repeat_position_tolerance_m: float = 0.0002
    repeat_linear_velocity_tolerance_m_s: float = 0.002
    repeat_attitude_tolerance: float = 0.0001
    repeat_ground_angular_rate_tolerance_rad_s: float = 0.02
    repeat_airborne_angular_rate_tolerance_rad_s: float = 0.01
    repeat_actuator_tolerance: float = 0.002
    repeat_rotor_tolerance: float = 0.001
    safety_xy_limit_m: float = 5.0
    safety_z_limit_m: float = 3.0
    calibration_status: str = "provisional"
    calibration_run_id: str | None = None

    def __post_init__(self):
        if self.schema_version != 2 or self.control_decimation != 1:
            raise ValueError(
                "This contract supports schema 2 and one controller update per physics step"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.num_agents) is not int or self.num_agents < 2:
            raise ValueError("num_agents must be an integer of at least two")
        if type(self.repetitions) is not int or self.repetitions < 2:
            raise ValueError("At least two repetitions are needed to check reset leakage")
        if self.formation_kind not in SUPPORTED_KINDS:
            raise ValueError(f"formation_kind must come from {SUPPORTED_KINDS}")
        if len(self.target_center_m) != 3 or not all(
            type(value) in (int, float) and math.isfinite(value) for value in self.target_center_m
        ):
            raise ValueError("target_center_m must contain three finite numbers")
        positive = (
            "physics_dt",
            "ground_height_m",
            "ground_spacing_m",
            "target_spacing_m",
            "max_speed_m_s",
            "position_gain_s_inv",
            "ground_settle_seconds",
            "takeoff_seconds",
            "formation_timeout_seconds",
            "dwell_seconds",
            "airborne_height_m",
            "contact_force_threshold_n",
            "minimum_separation_m",
            "formation_rmse_tolerance_m",
            "pairwise_rmse_tolerance_m",
            "speed_tolerance_m_s",
            "reset_tolerance",
            "repeat_position_tolerance_m",
            "repeat_linear_velocity_tolerance_m_s",
            "repeat_attitude_tolerance",
            "repeat_ground_angular_rate_tolerance_rad_s",
            "repeat_airborne_angular_rate_tolerance_rad_s",
            "repeat_actuator_tolerance",
            "repeat_rotor_tolerance",
            "safety_xy_limit_m",
            "safety_z_limit_m",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.001 <= self.physics_dt <= 0.02:
            raise ValueError("physics_dt must be between 0.001 and 0.02 seconds")
        for name in (
            "ground_settle_seconds",
            "takeoff_seconds",
            "formation_timeout_seconds",
            "dwell_seconds",
        ):
            steps = getattr(self, name) / self.physics_dt
            if not math.isclose(steps, round(steps)):
                raise ValueError(f"{name} must contain an integer number of physics steps")
        if self.dwell_seconds >= self.formation_timeout_seconds:
            raise ValueError("dwell_seconds must be shorter than formation_timeout_seconds")
        if self.minimum_separation_m >= min(self.ground_spacing_m, self.target_spacing_m):
            raise ValueError("minimum separation must be below ground and target spacing")
        if not self.ground_height_m < self.airborne_height_m < self.target_center_m[2]:
            raise ValueError("airborne height must lie between ground and target altitude")
        if self.target_center_m[2] >= self.safety_z_limit_m:
            raise ValueError("target altitude must be below the safety Z limit")
        if self.calibration_status not in {"provisional", "frozen"}:
            raise ValueError("Unknown calibration status")
        if self.calibration_status == "frozen" and not self.calibration_run_id:
            raise ValueError("Frozen tolerances require calibration run lineage")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.get("schema_version") == 1:
            value.pop("repeat_trajectory_tolerance", None)
            value["schema_version"] = 2
        if "target_center_m" in value:
            value["target_center_m"] = tuple(value["target_center_m"])
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    @property
    def ground_steps(self):
        return round(self.ground_settle_seconds / self.physics_dt)

    @property
    def takeoff_steps(self):
        return round(self.takeoff_seconds / self.physics_dt)

    @property
    def formation_steps(self):
        return round(self.formation_timeout_seconds / self.physics_dt)

    @property
    def dwell_steps(self):
        return round(self.dwell_seconds / self.physics_dt)

    @property
    def maximum_steps(self):
        return self.ground_steps + self.takeoff_steps + self.formation_steps


@dataclass(frozen=True)
class GroupLayout:
    group_id: int
    agent_ids: tuple[int, ...]
    ground_positions_m: tuple[tuple[float, float, float], ...]
    takeoff_positions_m: tuple[tuple[float, float, float], ...]
    assigned_target_positions_m: tuple[tuple[float, float, float], ...]
    assignment: Assignment


def build_group_layout(config: MultiDroneConfig) -> GroupLayout:
    """Build a centred ground line, vertical transit targets, and fixed final slots."""
    midpoint = (config.num_agents - 1) / 2.0
    ground = tuple(
        (
            (index - midpoint) * config.ground_spacing_m,
            0.0,
            config.ground_height_m,
        )
        for index in range(config.num_agents)
    )
    takeoff = tuple((x, y, config.target_center_m[2]) for x, y, _ in ground)
    template = generate_template(
        config.formation_kind,
        config.num_agents,
        config.target_spacing_m,
    )
    targets = place_template(template, config.target_center_m)
    assignment = assign_agents_to_slots(ground, targets)
    assigned_targets = tuple(targets[slot] for slot in assignment.slot_for_agent)
    return GroupLayout(
        group_id=0,
        agent_ids=tuple(range(config.num_agents)),
        ground_positions_m=ground,
        takeoff_positions_m=takeoff,
        assigned_target_positions_m=assigned_targets,
        assignment=assignment,
    )


def phase_at(config: MultiDroneConfig, step: int) -> str:
    if type(step) is not int or step < 0 or step >= config.maximum_steps:
        raise ValueError("step is outside the configured episode")
    if step < config.ground_steps:
        return "ground"
    if step < config.ground_steps + config.takeoff_steps:
        return "takeoff"
    return "formation"


def targets_for_phase(layout: GroupLayout, phase: str) -> tuple[tuple[float, float, float], ...]:
    if phase == "ground":
        return layout.ground_positions_m
    if phase == "takeoff":
        return layout.takeoff_positions_m
    if phase == "formation":
        return layout.assigned_target_positions_m
    raise ValueError(f"Unknown phase {phase!r}")


def velocity_commands(
    positions_m,
    targets_m,
    *,
    position_gain_s_inv: float,
    max_speed_m_s: float,
):
    """Apply a proportional outer loop and norm saturation in the world frame."""
    if len(positions_m) != len(targets_m) or not positions_m:
        raise ValueError("positions and targets must have the same nonzero length")
    commands = []
    for position, target in zip(positions_m, targets_m, strict=True):
        if len(position) != 3 or len(target) != 3:
            raise ValueError("positions and targets must contain three coordinates")
        error = tuple(
            float(right) - float(left) for left, right in zip(position, target, strict=True)
        )
        if not all(math.isfinite(value) for value in error):
            raise ValueError("positions and targets must be finite")
        unconstrained = tuple(position_gain_s_inv * value for value in error)
        norm = math.sqrt(math.fsum(value * value for value in unconstrained))
        scale = min(1.0, max_speed_m_s / norm) if norm > 0.0 else 0.0
        commands.append(tuple(value * scale for value in unconstrained))
    return tuple(commands)


def velocity_actions(velocities_m_s, max_speed_m_s: float):
    """Encode world velocities as bounded [direction_xyz, speed_scale] actions."""
    actions = []
    for velocity in velocities_m_s:
        if len(velocity) != 3 or not all(math.isfinite(value) for value in velocity):
            raise ValueError("velocities must contain three finite coordinates")
        speed = math.sqrt(math.fsum(value * value for value in velocity))
        if speed > max_speed_m_s + 1e-12:
            raise ValueError("velocity exceeds configured maximum")
        if speed == 0.0:
            actions.append((0.0, 0.0, 0.0, 0.0))
        else:
            actions.append(
                (
                    velocity[0] / speed,
                    velocity[1] / speed,
                    velocity[2] / speed,
                    speed / max_speed_m_s,
                )
            )
    return tuple(actions)


def evaluate(rows, resets, episodes, config: MultiDroneConfig):
    """Recompute construction, safety, completion, and reset checks from raw samples."""
    layout = build_group_layout(config)
    numeric_exclusions = {"phase"}
    by_repeat_step = {}
    for row in rows:
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key not in numeric_exclusions
        ):
            raise ValueError("Nonfinite trajectory sample")
        key = (int(row["repeat"]), int(row["step"]))
        by_repeat_step.setdefault(key, []).append(row)

    expected_repeats = set(range(config.repetitions))
    if {int(item["repeat"]) for item in resets} != expected_repeats:
        raise ValueError("Missing or repeated reset identity")
    if {int(item["repeat"]) for item in episodes} != expected_repeats:
        raise ValueError("Missing or repeated episode identity")
    episode_by_repeat = {int(item["repeat"]): item for item in episodes}

    checks = {
        "reset_state": all(
            float(item["max_reset_error"]) <= config.reset_tolerance for item in resets
        )
    }
    metrics = {}
    ordered_by_repeat = {}
    for repeat in range(config.repetitions):
        episode = episode_by_repeat[repeat]
        step_count = int(episode["steps"])
        if not 1 <= step_count <= config.maximum_steps:
            raise ValueError("Invalid episode step count")
        expected_keys = {(repeat, step) for step in range(step_count)}
        actual_keys = {key for key in by_repeat_step if key[0] == repeat}
        if actual_keys != expected_keys:
            raise ValueError("Missing or unexpected trajectory steps")

        ordered_steps = []
        formation_history = []
        minimum_separation = math.inf
        maximum_airborne_contact = 0.0
        maximum_ground_contact = 0.0
        applied_delta = 0.0
        previous_applied = None
        for step in range(step_count):
            samples = sorted(
                by_repeat_step[(repeat, step)],
                key=lambda row: int(row["agent_id"]),
            )
            if [int(row["agent_id"]) for row in samples] != list(range(config.num_agents)):
                raise ValueError("Each step must contain every agent exactly once")
            expected_phase = phase_at(config, step)
            if any(row["phase"] != expected_phase for row in samples):
                raise ValueError("Trajectory phase does not match configured schedule")
            positions = tuple(
                (float(row["x"]), float(row["y"]), float(row["z"])) for row in samples
            )
            targets = tuple(
                (
                    float(row["target_x"]),
                    float(row["target_y"]),
                    float(row["target_z"]),
                )
                for row in samples
            )
            expected_targets = targets_for_phase(layout, expected_phase)
            if any(
                math.dist(target, expected) > _FLOAT32_SERIALIZATION_TOLERANCE
                for target, expected in zip(targets, expected_targets, strict=True)
            ):
                raise ValueError("Saved targets disagree with fixed group layout")
            for row in samples:
                action = tuple(float(row[f"a{index}"]) for index in range(4))
                decoded = direction_speed(action, config.max_speed_m_s)
                saved_velocity = tuple(float(row[f"target_v{axis}"]) for axis in "xyz")
                if math.dist(decoded, saved_velocity) > _FLOAT32_SERIALIZATION_TOLERANCE:
                    raise ValueError("Saved action and target velocity disagree")

            formation = evaluate_formation(positions, targets)
            speed = max(
                math.sqrt(float(row["vx"]) ** 2 + float(row["vy"]) ** 2 + float(row["vz"]) ** 2)
                for row in samples
            )
            contact_by_agent = [float(row["contact_force_n"]) for row in samples]
            airborne_contact = max(
                (
                    contact
                    for contact, position in zip(contact_by_agent, positions, strict=True)
                    if position[2] > config.airborne_height_m
                ),
                default=0.0,
            )
            if expected_phase == "ground":
                maximum_ground_contact = max(maximum_ground_contact, *contact_by_agent)
            maximum_airborne_contact = max(maximum_airborne_contact, airborne_contact)
            minimum_separation = min(
                minimum_separation,
                min(math.dist(first, second) for first, second in combinations(positions, 2)),
            )
            applied = tuple(float(row[f"u{rotor}"]) for row in samples for rotor in range(4))
            if previous_applied is not None:
                applied_delta = max(
                    applied_delta,
                    max(
                        abs(current - previous)
                        for current, previous in zip(applied, previous_applied, strict=True)
                    ),
                )
            previous_applied = applied
            if expected_phase == "formation":
                formation_history.append(
                    {
                        "assigned_rmse_m": formation.assigned_root_mean_squared_error_m,
                        "pairwise_rmse_m": formation.pairwise_root_mean_squared_error_m,
                        "max_speed_m_s": speed,
                    }
                )
            ordered_steps.append(samples)

        label = f"repeat/{repeat}"
        checks[label + "/termination_success"] = episode["termination_reason"] == "success"
        checks[label + "/ground_contact_observed"] = (
            maximum_ground_contact > config.contact_force_threshold_n
        )
        checks[label + "/no_airborne_contact"] = (
            maximum_airborne_contact <= config.contact_force_threshold_n
        )
        checks[label + "/separation"] = minimum_separation >= config.minimum_separation_m
        checks[label + "/actuator_bounds"] = all(
            -1.0 <= float(row[f"u{rotor}"]) <= 1.0
            for samples in ordered_steps
            for row in samples
            for rotor in range(4)
        )
        checks[label + "/quaternion"] = all(
            abs(math.fsum(float(row[key]) ** 2 for key in ("qw", "qx", "qy", "qz")) - 1.0) <= 1e-3
            for samples in ordered_steps
            for row in samples
        )
        checks[label + "/takeoff"] = any(
            min(float(row["z"]) for row in samples)
            >= config.target_center_m[2] - config.formation_rmse_tolerance_m
            for samples in ordered_steps
        )
        if len(formation_history) < config.dwell_steps:
            tail = ()
        else:
            tail = formation_history[-config.dwell_steps :]
        checks[label + "/formation_tracking"] = (
            bool(tail)
            and max(item["assigned_rmse_m"] for item in tail) <= config.formation_rmse_tolerance_m
        )
        checks[label + "/shape_tracking"] = (
            bool(tail)
            and max(item["pairwise_rmse_m"] for item in tail) <= config.pairwise_rmse_tolerance_m
        )
        checks[label + "/settled"] = (
            bool(tail) and max(item["max_speed_m_s"] for item in tail) <= config.speed_tolerance_m_s
        )
        metrics[label] = {
            "samples": step_count * config.num_agents,
            "termination_reason": episode["termination_reason"],
            "completion_time_s": step_count * config.physics_dt,
            "minimum_separation_m": minimum_separation,
            "maximum_ground_contact_force_n": maximum_ground_contact,
            "maximum_airborne_contact_force_n": maximum_airborne_contact,
            "maximum_applied_rotor_step_change": applied_delta,
            "final_dwell_max_assigned_rmse_m": max(
                (item["assigned_rmse_m"] for item in tail), default=None
            ),
            "final_dwell_max_pairwise_rmse_m": max(
                (item["pairwise_rmse_m"] for item in tail), default=None
            ),
            "final_dwell_max_speed_m_s": max(
                (item["max_speed_m_s"] for item in tail), default=None
            ),
        }
        ordered_by_repeat[repeat] = ordered_steps

    reference = ordered_by_repeat[0]
    repeat_maxima = {
        "position_m": math.inf,
        "linear_velocity_m_s": math.inf,
        "attitude_component": math.inf,
        "ground_angular_rate_rad_s": math.inf,
        "airborne_angular_rate_rad_s": math.inf,
        "raw_actuator": math.inf,
        "applied_actuator": math.inf,
        "rotor_state": math.inf,
    }
    if all(len(ordered_by_repeat[repeat]) == len(reference) for repeat in expected_repeats):
        repeat_maxima = {
            "position_m": 0.0,
            "linear_velocity_m_s": 0.0,
            "attitude_component": 0.0,
            "ground_angular_rate_rad_s": 0.0,
            "airborne_angular_rate_rad_s": 0.0,
            "raw_actuator": 0.0,
            "applied_actuator": 0.0,
            "rotor_state": 0.0,
        }
        field_groups = {
            "position_m": ("x", "y", "z"),
            "linear_velocity_m_s": ("vx", "vy", "vz"),
            "attitude_component": ("qw", "qx", "qy", "qz"),
            "raw_actuator": ("raw_u0", "raw_u1", "raw_u2", "raw_u3"),
            "applied_actuator": ("u0", "u1", "u2", "u3"),
            "rotor_state": ("rotor0", "rotor1", "rotor2", "rotor3"),
        }
        for repeat in range(1, config.repetitions):
            for expected_step, candidate_step in zip(
                reference, ordered_by_repeat[repeat], strict=True
            ):
                for expected, candidate in zip(expected_step, candidate_step, strict=True):
                    for group, fields in field_groups.items():
                        repeat_maxima[group] = max(
                            repeat_maxima[group],
                            max(
                                abs(float(candidate[field]) - float(expected[field]))
                                for field in fields
                            ),
                        )
                    angular_group = (
                        "ground_angular_rate_rad_s"
                        if expected["phase"] == "ground"
                        else "airborne_angular_rate_rad_s"
                    )
                    repeat_maxima[angular_group] = max(
                        repeat_maxima[angular_group],
                        max(
                            abs(float(candidate[field]) - float(expected[field]))
                            for field in ("wx", "wy", "wz")
                        ),
                    )

    repeat_checks = {
        "position": repeat_maxima["position_m"] <= config.repeat_position_tolerance_m,
        "linear_velocity": repeat_maxima["linear_velocity_m_s"]
        <= config.repeat_linear_velocity_tolerance_m_s,
        "attitude": repeat_maxima["attitude_component"] <= config.repeat_attitude_tolerance,
        "ground_angular_rate": repeat_maxima["ground_angular_rate_rad_s"]
        <= config.repeat_ground_angular_rate_tolerance_rad_s,
        "airborne_angular_rate": repeat_maxima["airborne_angular_rate_rad_s"]
        <= config.repeat_airborne_angular_rate_tolerance_rad_s,
        "raw_actuator": repeat_maxima["raw_actuator"] <= config.repeat_actuator_tolerance,
        "applied_actuator": repeat_maxima["applied_actuator"] <= config.repeat_actuator_tolerance,
        "rotor_state": repeat_maxima["rotor_state"] <= config.repeat_rotor_tolerance,
    }
    checks.update({f"reset_repeat/{name}": passed for name, passed in repeat_checks.items()})
    checks["reset_repeat"] = all(repeat_checks.values())
    metrics["reset_repeat"] = {
        "maximum_absolute_differences": repeat_maxima,
        "tolerances": {
            "position_m": config.repeat_position_tolerance_m,
            "linear_velocity_m_s": config.repeat_linear_velocity_tolerance_m_s,
            "attitude_component": config.repeat_attitude_tolerance,
            "ground_angular_rate_rad_s": config.repeat_ground_angular_rate_tolerance_rad_s,
            "airborne_angular_rate_rad_s": config.repeat_airborne_angular_rate_tolerance_rad_s,
            "raw_and_applied_actuator": config.repeat_actuator_tolerance,
            "rotor_state": config.repeat_rotor_tolerance,
        },
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "metrics": metrics,
        "layout": asdict(layout),
        "calibration_status": config.calibration_status,
    }
