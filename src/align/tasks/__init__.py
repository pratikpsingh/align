"""Simulator-independent task contracts shared by training and evaluation."""

from align.tasks.reward import (
    COMPONENT_NAMES,
    RewardComponents,
    RewardConfig,
    RewardMemory,
    RewardStep,
    compute_step_reward,
)

__all__ = [
    "COMPONENT_NAMES",
    "RewardComponents",
    "RewardConfig",
    "RewardMemory",
    "RewardStep",
    "compute_step_reward",
]
