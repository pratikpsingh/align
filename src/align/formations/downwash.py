"""Vertical-thrust downwash proxy for inspecting a geometric flight plan.

OmniDrones' pinned MultirotorBase model uses a direction-dependent downwash
term with kr=2 and kz=0.3. This host-only proxy assumes vertical thrust and
reports dimensionless exposure per lower drone, not measured force.

The coupling expression is adapted from OmniDrones commit
9ce7c2028b71be64d7e748c31f685cd3b54afe27,
omni_drones/robots/drone/multirotor.py. That source is MIT licensed,
copyright (c) 2023 Botian Xu, Tsinghua University; its LICENSE is retained
with the pinned source in the derived image.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def vertical_downwash_exposure(positions_m: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Sum nominal upper-drone coupling factors for each lower drone."""
    points = tuple(tuple(point) for point in positions_m)
    if len(points) < 2 or any(
        len(point) != 3
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
        for point in points
    ):
        raise ValueError("downwash positions must be finite XYZ triples for at least two drones")
    result = []
    for lower in points:
        exposure = 0.0
        for upper in points:
            height = upper[2] - lower[2]
            if height <= 0:
                continue
            horizontal_squared = (upper[0] - lower[0]) ** 2 + (upper[1] - lower[1]) ** 2
            exposure += math.exp(-2.0 * horizontal_squared / height**2) / (1.0 + 0.3 * height) ** 2
        result.append(exposure)
    return tuple(result)


def sample_nominal_downwash(
    source_positions_m: Sequence[Sequence[float]],
    assigned_destinations_m: Sequence[Sequence[float]],
    *,
    samples: int = 101,
) -> dict:
    """Inspect synchronous linear paths; real drone paths may differ."""
    if type(samples) is not int or samples < 2:
        raise ValueError("downwash samples must be an integer >= 2")
    if len(source_positions_m) != len(assigned_destinations_m):
        raise ValueError("source and destination counts differ")
    peak = (-math.inf, None, None)
    final = None
    for step in range(samples):
        alpha = step / (samples - 1)
        points = tuple(
            tuple(a + alpha * (b - a) for a, b in zip(source, target, strict=True))
            for source, target in zip(source_positions_m, assigned_destinations_m, strict=True)
        )
        exposure = vertical_downwash_exposure(points)
        for agent_id, value in enumerate(exposure):
            if value > peak[0]:
                peak = (value, alpha, agent_id)
        final = exposure
    return {
        "model": "pinned_omnidrones_vertical_thrust_proxy_kr2_kz0.3",
        "sample_count": samples,
        "maximum_nominal_exposure_fraction": peak[0],
        "maximum_nominal_exposure_alpha": peak[1],
        "maximum_nominal_exposure_agent_id": peak[2],
        "destination_exposure_by_agent": final,
        "physical_force_measured": False,
        "assumes_synchronous_linear_paths": True,
    }
