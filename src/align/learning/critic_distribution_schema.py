"""Host-safe validation of per-update active critic distribution measurements."""

from __future__ import annotations

import csv
import math
from pathlib import Path

GROUPS = ("position", "velocity", "target")
MEASUREMENTS = (
    "raw_mean",
    "raw_standard_deviation",
    "raw_minimum",
    "raw_maximum",
    "warmup_mean",
    "warmup_standard_deviation",
    "mean_shift_warmup_standard_deviations",
    "raw_standard_deviation_ratio",
    "normalized_mean",
    "normalized_standard_deviation",
    "normalized_minimum",
    "normalized_maximum",
    "clipped_fraction",
)
COLUMNS = (
    "completed_update",
    "group",
    "count",
    *MEASUREMENTS[:-1],
    "clipped_count",
    "clipped_fraction",
)


def validate_critic_distribution_rows(
    rows: list[dict], *, update: int, scalar_count: int, enabled: bool, clip: float
) -> bool:
    """Require three complete, finite groups with exact active scalar counts."""
    if len(rows) != 3 or {row.get("group") for row in rows} != set(GROUPS):
        return False
    try:
        for row in rows:
            if set(row) != set(COLUMNS) or int(row["completed_update"]) != update:
                return False
            count = int(row["count"])
            clipped = int(row["clipped_count"])
            values = {name: float(row[name]) for name in MEASUREMENTS}
            if count != scalar_count or not 0 <= clipped <= count:
                return False
            if not all(math.isfinite(value) for value in values.values()):
                return False
            if not math.isclose(values["clipped_fraction"], clipped / count, abs_tol=1e-12):
                return False
            if values["raw_minimum"] > values["raw_maximum"]:
                return False
            if values["normalized_minimum"] > values["normalized_maximum"]:
                return False
            if values["raw_standard_deviation"] < 0 or values["normalized_standard_deviation"] < 0:
                return False
            if (
                values["warmup_standard_deviation"] <= 0
                or values["raw_standard_deviation_ratio"] < 0
            ):
                return False
            if values["raw_minimum"] < -1.000001 or values["raw_maximum"] > 1.000001:
                return False
            limit = clip if enabled else 1.0
            if (
                values["normalized_minimum"] < -limit - 1e-6
                or values["normalized_maximum"] > limit + 1e-6
            ):
                return False
            if not enabled and clipped != 0:
                return False
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return False
    return True


def read_valid_critic_distribution(
    path: Path, *, update: int, scalar_count: int, enabled: bool, clip: float
) -> list[dict] | None:
    """Parse and validate one immutable three-row update table."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            return None
        rows = list(reader)
    if not validate_critic_distribution_rows(
        rows, update=update, scalar_count=scalar_count, enabled=enabled, clip=clip
    ):
        return None
    return rows
