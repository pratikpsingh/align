"""Coherent vendor-PyTorch learner state for reset-mode training recovery."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from align.learning.checkpoint_store import RECOVERY_MODE
from align.learning.torch_normalization import validate_normalization_state

STATE_SCHEMA_VERSION = 2
EXPECTED_STATE_KEYS = {
    "state_schema_version",
    "recovery_mode",
    "actor",
    "critic",
    "actor_optimizer",
    "critic_optimizer",
    "normalization",
    "schedules",
    "counters",
    "resolved_config",
    "task_sampler_state",
    "rng",
}


def capture_learner_state(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    normalization: dict,
    schedules: dict,
    counters: dict,
    resolved_config: dict,
    task_sampler_state: dict,
) -> dict:
    """Capture after a completed update, before any next collection starts."""
    required_counts = ("completed_updates", "environment_transitions", "agent_transitions")
    if any(type(counters.get(name)) is not int or counters[name] < 0 for name in required_counts):
        raise ValueError("learner counters must include nonnegative completed progress")
    validate_normalization_state(normalization)
    if not all(
        isinstance(value, dict) for value in (schedules, resolved_config, task_sampler_state)
    ):
        raise ValueError("schedules, configuration, and task sampler state must be dictionaries")
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "recovery_mode": RECOVERY_MODE,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "normalization": normalization,
        "schedules": schedules,
        "counters": counters,
        "resolved_config": resolved_config,
        "task_sampler_state": task_sampler_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def _validate_module_state(module: torch.nn.Module, saved: object, label: str) -> None:
    if not isinstance(saved, Mapping):
        raise ValueError(f"checkpoint {label} state is not a mapping")
    current = module.state_dict()
    if set(saved) != set(current):
        raise ValueError(f"checkpoint {label} parameter names differ")
    for name, value in current.items():
        candidate = saved[name]
        if not torch.is_tensor(candidate) or candidate.shape != value.shape:
            raise ValueError(f"checkpoint {label} parameter shape differs: {name}")


def _load_validated_state(path: Path, expected_config: Mapping, map_location) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or set(state) != EXPECTED_STATE_KEYS:
        raise ValueError("checkpoint learner-state fields are incomplete or unexpected")
    if state["state_schema_version"] != STATE_SCHEMA_VERSION:
        raise ValueError("checkpoint learner-state schema is incompatible")
    if state["recovery_mode"] != RECOVERY_MODE:
        raise ValueError("checkpoint does not use environment-reset recovery")
    if state["resolved_config"] != dict(expected_config):
        raise ValueError("checkpoint resolved configuration differs")
    if not isinstance(state["rng"], Mapping) or set(state["rng"]) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda_all",
    }:
        raise ValueError("checkpoint RNG state is incomplete or unexpected")
    for name in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
        if not isinstance(state[name], Mapping):
            raise ValueError(f"checkpoint {name} state is not a mapping")
    validate_normalization_state(state["normalization"])
    return state


def validate_learner_payload(path: Path, expected_config: Mapping) -> None:
    """Deserialize and validate a newly written temporary payload before publication."""
    _load_validated_state(path, expected_config, "cpu")


def restore_learner_state(
    path: Path,
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    expected_config: Mapping,
) -> dict:
    """Load a previously checksum-verified, trusted checkpoint payload."""
    state = _load_validated_state(path, expected_config, next(actor.parameters()).device)
    _validate_module_state(actor, state["actor"], "actor")
    _validate_module_state(critic, state["critic"], "critic")
    actor.load_state_dict(state["actor"], strict=True)
    critic.load_state_dict(state["critic"], strict=True)
    actor_optimizer.load_state_dict(state["actor_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch_cpu"].cpu())
    if torch.cuda.is_available():
        saved_cuda = state["rng"]["torch_cuda_all"]
        if len(saved_cuda) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all([value.cpu() for value in saved_cuda])
    return {
        "normalization": state["normalization"],
        "schedules": state["schedules"],
        "counters": state["counters"],
        "task_sampler_state": state["task_sampler_state"],
        "recovery_mode": RECOVERY_MODE,
        "discard_partial_rollout": True,
        "reset_environment_and_recurrent_memory": True,
    }
