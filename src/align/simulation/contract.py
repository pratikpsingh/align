"""CPU-only single-drone configuration, command mapping, and acceptance metrics."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class DroneCheckConfig:
    schema_version: int = 1
    seed: int = 17
    physics_dt: float = 0.01
    control_decimation: int = 1
    max_speed_m_s: float = 0.5
    command_speed_m_s: float = 0.25
    initial_height_m: float = 1.5
    hover_seconds: float = 6.0
    settle_seconds: float = 1.0
    command_seconds: float = 3.0
    brake_seconds: float = 2.0
    repetitions: int = 2
    hover_position_tolerance_m: float = 0.08
    hover_speed_tolerance_m_s: float = 0.05
    velocity_tolerance_m_s: float = 0.08
    cross_axis_tolerance_m: float = 0.10
    minimum_axis_displacement_m: float = 0.40
    reset_trajectory_tolerance: float = 0.002
    calibration_status: str = "provisional"
    calibration_run_id: str | None = None

    def __post_init__(self):
        if self.schema_version != 1 or self.control_decimation != 1:
            raise ValueError(
                "This contract supports schema 1 and one controller update per physics step"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.repetitions) is not int or self.repetitions < 2:
            raise ValueError("At least two repetitions are needed to check reset leakage")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is float and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{field.name} must be finite and positive")
        if not 0.001 <= self.physics_dt <= 0.02:
            raise ValueError("physics_dt must be between 0.001 and 0.02 seconds")
        if self.command_speed_m_s > self.max_speed_m_s:
            raise ValueError("Command speed exceeds configured saturation")
        if self.calibration_status not in {"provisional", "frozen"}:
            raise ValueError("Unknown calibration status")
        if self.calibration_status == "frozen" and not self.calibration_run_id:
            raise ValueError("Frozen tolerances require calibration run lineage")
        for value in (
            self.hover_seconds,
            self.settle_seconds,
            self.command_seconds,
            self.brake_seconds,
        ):
            if not math.isclose(value / self.physics_dt, round(value / self.physics_dt)):
                raise ValueError("Durations must contain an integer number of physics steps")
        if min(self.hover_seconds, self.command_seconds, self.brake_seconds) < 1.0:
            raise ValueError("Each measurement segment must last at least one second")

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def direction_speed(action, max_speed):
    """Map bounded [direction_xyz, speed_scale] to world velocity, in m/s.

    Preserves the student's absolute speed scale. Clip all sampled components
    first; a bounded mean alone would not provide this guarantee.
    """
    if len(action) != 4 or not all(math.isfinite(x) for x in action):
        raise ValueError("Expected four finite action values")
    if not math.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("max_speed must be finite and positive")
    clipped = [max(-1.0, min(1.0, x)) for x in action]
    norm = math.sqrt(sum(x * x for x in clipped[:3]))
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return tuple(max_speed * abs(clipped[3]) * x / norm for x in clipped[:3])


def cases():
    """name, world command axis (None for position hold), reset yaw in radians."""
    return (
        ("hover", None, 0.0),
        ("x", 0, 0.0),
        ("y", 1, 0.0),
        ("z", 2, 0.0),
        ("x_yaw90", 0, math.pi / 2),
    )


def command_at(config, axis, step):
    time = step * config.physics_dt
    active = axis is not None and config.settle_seconds <= time < (
        config.settle_seconds + config.command_seconds - config.physics_dt / 2
    )
    action = [0.0] * 4
    if active:
        action[axis] = 1.0
        action[3] = config.command_speed_m_s / config.max_speed_m_s
    return action, direction_speed(action, config.max_speed_m_s), active


def evaluate(rows, resets, config):
    """Evaluate saved physics samples; reject missing, nonfinite, or misordered data.

    Rows contain post-step states. Commands apply over [t-dt, t]. These checks
    are independent of simulator imports and can be rerun from the saved CSV.
    """
    by_case = {}
    for row in rows:
        numeric = [v for k, v in row.items() if k not in {"case"}]
        if not all(math.isfinite(float(v)) for v in numeric):
            raise ValueError("Nonfinite trajectory sample")
        by_case.setdefault((row["case"], int(row["repeat"])), []).append(row)
    metrics, checks = {}, {}
    expected_keys = {
        (name, repeat) for name, _, _ in cases() for repeat in range(config.repetitions)
    }
    if set(by_case) != expected_keys or len(resets) != len(expected_keys):
        raise ValueError("Missing or unexpected episodes/reset records")
    if {(r["case"], r["repeat"]) for r in resets} != expected_keys:
        raise ValueError("Missing or repeated reset identity")
    checks["reset_state"] = all(r["max_reset_error"] <= 1e-6 for r in resets)
    for name, axis, _ in cases():
        duration = (
            config.hover_seconds
            if axis is None
            else (config.settle_seconds + config.command_seconds + config.brake_seconds)
        )
        expected_steps = round(duration / config.physics_dt)
        for repeat in range(config.repetitions):
            sample = by_case[name, repeat]
            if len(sample) != expected_steps or any(
                int(row["step"]) != i
                or not math.isclose(float(row["t"]), (i + 1) * config.physics_dt, abs_tol=1e-8)
                for i, row in enumerate(sample)
            ):
                raise ValueError("Incomplete or misordered physics samples")
            label = f"{name}/{repeat}"
            tail = sample[-round(0.5 / config.physics_dt) :]
            speed = max(math.sqrt(sum(float(r[k]) ** 2 for k in ("vx", "vy", "vz"))) for r in tail)
            error = max(
                math.sqrt(
                    float(r["x"]) ** 2
                    + float(r["y"]) ** 2
                    + (float(r["z"]) - config.initial_height_m) ** 2
                )
                for r in tail
            )
            item = {"samples": len(sample), "final_tail_max_speed_m_s": speed}
            checks[label + "/stopped"] = speed <= config.hover_speed_tolerance_m_s
            if axis is None:
                item["final_tail_max_position_error_m"] = error
                checks[label + "/hover"] = error <= config.hover_position_tolerance_m
            else:
                start = sample[round(config.settle_seconds / config.physics_dt) - 1]
                end_idx = round(
                    (config.settle_seconds + config.command_seconds) / config.physics_dt
                )
                end = sample[end_idx - 1]
                displacement = [float(end[k]) - float(start[k]) for k in ("x", "y", "z")]
                cross = math.sqrt(sum(d * d for i, d in enumerate(displacement) if i != axis))
                vel_tail = sample[end_idx - round(0.5 / config.physics_dt) : end_idx]
                velocity = sum(float(r[("vx", "vy", "vz")[axis]]) for r in vel_tail) / len(vel_tail)
                item.update(
                    axis_displacement_m=displacement[axis],
                    cross_axis_displacement_m=cross,
                    command_tail_mean_velocity_m_s=velocity,
                )
                checks[label + "/axis"] = displacement[axis] >= config.minimum_axis_displacement_m
                checks[label + "/cross_axis"] = cross <= config.cross_axis_tolerance_m
                checks[label + "/velocity"] = (
                    abs(velocity - config.command_speed_m_s) <= config.velocity_tolerance_m_s
                )
            checks[label + "/actuator_bounds"] = all(
                -1 <= float(r[f"u{i}"]) <= 1 for r in sample for i in range(4)
            )
            checks[label + "/quaternion"] = all(
                abs(sum(float(r[k]) ** 2 for k in ("qw", "qx", "qy", "qz")) - 1) < 1e-3
                for r in sample
            )
            metrics[label] = item
        reference = by_case[name, 0]
        keys = (
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "wx",
            "wy",
            "wz",
            "u0",
            "u1",
            "u2",
            "u3",
            "rotor0",
            "rotor1",
            "rotor2",
            "rotor3",
        )
        error = max(
            abs(float(a[k]) - float(b[k]))
            for repeat in range(1, config.repetitions)
            for a, b in zip(reference, by_case[name, repeat], strict=True)
            for k in keys
        )
        metrics[name + "/reset_repeat"] = {"max_absolute_difference_mixed_units": error}
        checks[name + "/reset_repeat"] = error <= config.reset_trajectory_tolerance
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "metrics": metrics,
        "calibration_status": config.calibration_status,
    }
