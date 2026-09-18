"""Stable input and byte-level comparison for actor export acceptance."""

from __future__ import annotations

import hashlib

import torch


def probe_observation(batch: int, dimensions: int) -> torch.Tensor:
    if batch < 1 or dimensions < 6:
        raise ValueError("actor probe requires a positive batch and six self features")
    sample = torch.zeros((batch, 1, dimensions), dtype=torch.float32)
    sample[:, :, 3] = 0.25
    sample[:, :, 5] = 0.10
    return sample


def tensor_digest(value: torch.Tensor) -> str:
    """Hash exact float32 tensor bytes for a second-process parity check."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
