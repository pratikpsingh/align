"""Simulator-independent task contracts shared by training and evaluation."""

from align.tasks.environment import BatchedTaskEnvironment, TaskEnvironmentConfig, TaskStep
from align.tasks.observation import (
    ActorObservation,
    CriticObservation,
    ObservationBatch,
    ObservationConfig,
    build_actor_observations,
    build_critic_observation,
    build_observations,
)
from align.tasks.reward import (
    COMPONENT_NAMES,
    RewardComponents,
    RewardConfig,
    RewardMemory,
    RewardStep,
    compute_step_reward,
)

__all__ = [
    "ActorObservation",
    "BatchedTaskEnvironment",
    "COMPONENT_NAMES",
    "CriticObservation",
    "ObservationBatch",
    "ObservationConfig",
    "RewardComponents",
    "RewardConfig",
    "RewardMemory",
    "RewardStep",
    "TaskEnvironmentConfig",
    "TaskStep",
    "build_actor_observations",
    "build_critic_observation",
    "build_observations",
    "compute_step_reward",
]
