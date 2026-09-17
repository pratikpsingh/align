"""Vendor-PyTorch acceptance probe for reset-mode training recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from align.artifacts import write_json_atomic
from align.learning.checkpoint_store import CheckpointStore
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.ppo_probe import _fixture
from align.learning.recovery_config import RecoveryConfig
from align.learning.torch_ppo import update_recurrent_ppo
from align.learning.torch_recovery import (
    capture_learner_state,
    restore_learner_state,
    validate_learner_payload,
)
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import CentralizedRecurrentCritic, SharedRecurrentActor

SEED = 20260918


def _optimizer(module, learning_rate: float, config: RecurrentPPOConfig):
    return torch.optim.Adam(module.parameters(), lr=learning_rate, eps=config.adam_epsilon)


def _seed(device: torch.device) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _max_model_difference(left, right) -> float:
    return max(
        float((a.detach() - b.detach()).abs().max().cpu())
        for a, b in zip(left.state_dict().values(), right.state_dict().values(), strict=True)
        if torch.is_floating_point(a)
    )


def _equal_nested(left, right) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _equal_nested(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _save(store, state, resolved, *, update, attempt, parent=None):
    return store.save(
        lambda stream: torch.save(state, stream),
        validator=lambda path: validate_learner_payload(path, resolved),
        attempt_id=attempt,
        completed_updates=update,
        environment_transitions=update * 7,
        agent_transitions=update * 28,
        active_training_seconds=float(update),
        source_identity="acceptance-fixture",
        runtime_identity=f"torch:{torch.__version__}:cuda:{torch.version.cuda}",
        parent_checkpoint_sha256=parent,
    )


def _learner_state(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    *,
    normalization,
    schedules,
    counters,
    resolved,
    task_sampler_state,
):
    return capture_learner_state(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        normalization=normalization,
        schedules=schedules,
        counters=counters,
        resolved_config=resolved,
        task_sampler_state=task_sampler_state,
    )


def evaluate(policy, ppo, resolved, config_path, output, device):
    _seed(device)
    one_epoch = replace(ppo, update_epochs=1)
    actor = SharedRecurrentActor(policy).to(device)
    critic = CentralizedRecurrentCritic(policy).to(device)
    actor_optimizer = _optimizer(actor, ppo.actor_learning_rate, ppo)
    critic_optimizer = _optimizer(critic, ppo.critic_learning_rate, ppo)
    chunks = _fixture(actor, critic, policy, device)
    update_recurrent_ppo(actor, critic, actor_optimizer, critic_optimizer, chunks, one_epoch)
    counters = {
        "completed_updates": 1,
        "environment_transitions": 7,
        "agent_transitions": 28,
        "active_training_seconds": 1.0,
    }
    normalization = {
        "schema_version": 1,
        "enabled": True,
        "contract": "frozen_active_critic_group_standardization",
        "frozen": True,
        "epsilon": 1e-6,
        "clip": 5.0,
        "warmup_steps": 7,
        "environment_samples": 7,
        "active_agent_samples": 28,
        "groups": {
            name: {"count": 84, "mean": mean, "variance": variance}
            for name, mean, variance in (
                ("position", 0.1, 0.2),
                ("velocity", -0.01, 0.03),
                ("target", 0.2, 0.4),
            )
        },
    }
    schedules = {"learning_rate_fraction": 0.9, "entropy_coefficient": 0.01}
    task_sampler_state = {"episode_index": 12, "seed_sequence_position": 4}
    state = _learner_state(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        normalization=normalization,
        schedules=schedules,
        counters=counters,
        resolved=resolved,
        task_sampler_state=task_sampler_state,
    )
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    checkpoints = output / "checkpoints"
    store = CheckpointStore(
        checkpoints,
        run_id="recovery-acceptance",
        config_sha256=config_sha256,
        file_mode=0o644,
    )
    first = _save(store, state, resolved, update=1, attempt="attempt-original")
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    uninterrupted_diagnostics = update_recurrent_ppo(
        actor, critic, actor_optimizer, critic_optimizer, chunks, one_epoch
    )

    resumed_actor = SharedRecurrentActor(policy).to(device)
    resumed_critic = CentralizedRecurrentCritic(policy).to(device)
    resumed_actor_optimizer = _optimizer(resumed_actor, ppo.actor_learning_rate, ppo)
    resumed_critic_optimizer = _optimizer(resumed_critic, ppo.critic_learning_rate, ppo)
    selected, payload = store.latest_valid()
    restored = restore_learner_state(
        payload,
        actor=resumed_actor,
        critic=resumed_critic,
        actor_optimizer=resumed_actor_optimizer,
        critic_optimizer=resumed_critic_optimizer,
        expected_config=resolved,
    )
    restored_python = random.random()
    restored_numpy = float(np.random.random())
    resumed_diagnostics = update_recurrent_ppo(
        resumed_actor,
        resumed_critic,
        resumed_actor_optimizer,
        resumed_critic_optimizer,
        chunks,
        one_epoch,
    )
    resumed_counters = dict(restored["counters"])
    resumed_counters.update(
        completed_updates=resumed_counters["completed_updates"] + 1,
        environment_transitions=resumed_counters["environment_transitions"] + 7,
        agent_transitions=resumed_counters["agent_transitions"] + 28,
        active_training_seconds=resumed_counters["active_training_seconds"] + 1.0,
    )
    second_state = _learner_state(
        resumed_actor,
        resumed_critic,
        resumed_actor_optimizer,
        resumed_critic_optimizer,
        normalization=restored["normalization"],
        schedules=restored["schedules"],
        counters=resumed_counters,
        resolved=resolved,
        task_sampler_state=restored["task_sampler_state"],
    )
    second = _save(
        store,
        second_state,
        resolved,
        update=2,
        attempt="attempt-resumed",
        parent=first["payload"]["sha256"],
    )
    third_state = {
        **second_state,
        "counters": {
            "completed_updates": 3,
            "environment_transitions": 21,
            "agent_transitions": 84,
            "active_training_seconds": 3.0,
        },
    }
    third = _save(
        store,
        third_state,
        resolved,
        update=3,
        attempt="attempt-resumed",
        parent=second["payload"]["sha256"],
    )
    corrupt = _save(
        store,
        {
            **second_state,
            "counters": {
                "completed_updates": 4,
                "environment_transitions": 28,
                "agent_transitions": 112,
                "active_training_seconds": 4.0,
            },
        },
        resolved,
        update=4,
        attempt="attempt-resumed",
        parent=third["payload"]["sha256"],
    )
    (checkpoints / corrupt["payload"]["file"]).write_bytes(b"corrupt")

    def failed_writer(stream):
        stream.write(b"partial")
        raise OSError("simulated disk failure")

    write_failed = False
    try:
        store.save(
            failed_writer,
            attempt_id="attempt-resumed",
            completed_updates=5,
            environment_transitions=35,
            agent_transitions=140,
            active_training_seconds=5.0,
            source_identity="acceptance-fixture",
            runtime_identity="acceptance-fixture",
        )
    except OSError:
        write_failed = True
    fallback, _ = store.latest_valid()
    valid_checkpoint_count = 0
    for manifest_path in checkpoints.glob("checkpoint-*.json"):
        try:
            store.verify(manifest_path)
            valid_checkpoint_count += 1
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    incompatible_config_rejected = False
    try:
        restore_learner_state(
            checkpoints / first["payload"]["file"],
            actor=resumed_actor,
            critic=resumed_critic,
            actor_optimizer=resumed_actor_optimizer,
            critic_optimizer=resumed_critic_optimizer,
            expected_config={**resolved, "incompatible": True},
        )
    except ValueError:
        incompatible_config_rejected = True
    fresh_output = output / "fresh-load.json"
    fresh = subprocess.run(
        [
            sys.executable,
            "-m",
            "align.learning.recovery_fresh_load",
            "--config",
            str(config_path),
            "--checkpoints",
            str(checkpoints),
            "--run-id",
            "recovery-acceptance",
            "--config-sha256",
            config_sha256,
            "--output",
            str(fresh_output),
            "--device",
            str(device),
        ],
        check=False,
    )
    fresh_result = json.loads(fresh_output.read_text()) if fresh_output.exists() else None
    actor_difference = _max_model_difference(actor, resumed_actor)
    critic_difference = _max_model_difference(critic, resumed_critic)
    checks = {
        "latest_manifest_was_verified_before_restore": selected["checkpoint_id"]
        == first["checkpoint_id"],
        "actor_next_update_is_exact": actor_difference == 0.0,
        "critic_next_update_is_exact": critic_difference == 0.0,
        "actor_adam_state_is_exact": _equal_nested(
            actor_optimizer.state_dict(), resumed_actor_optimizer.state_dict()
        ),
        "critic_adam_state_is_exact": _equal_nested(
            critic_optimizer.state_dict(), resumed_critic_optimizer.state_dict()
        ),
        "next_update_diagnostics_are_exact": uninterrupted_diagnostics == resumed_diagnostics,
        "python_rng_is_restored": expected_python == restored_python,
        "numpy_rng_is_restored": expected_numpy == restored_numpy,
        "normalization_and_schedules_are_restored": _equal_nested(
            normalization, restored["normalization"]
        )
        and schedules == restored["schedules"],
        "task_sampler_state_is_restored": task_sampler_state == restored["task_sampler_state"],
        "progress_advances_once_after_resume": resumed_counters["completed_updates"] == 2
        and resumed_counters["environment_transitions"] == 14
        and resumed_counters["agent_transitions"] == 56,
        "resume_requires_environment_and_memory_reset": restored[
            "reset_environment_and_recurrent_memory"
        ]
        and restored["discard_partial_rollout"],
        "lineage_links_to_loaded_checkpoint": second["parent_checkpoint_sha256"]
        == first["payload"]["sha256"],
        "at_least_three_valid_checkpoints_are_retained": valid_checkpoint_count >= 3,
        "incompatible_configuration_is_rejected": incompatible_config_rejected,
        "corrupt_newest_falls_back": fallback["checkpoint_id"] == third["checkpoint_id"],
        "failed_write_preserves_prior_checkpoint": write_failed,
        "fresh_process_can_load_and_evaluate": fresh.returncode == 0
        and isinstance(fresh_result, dict)
        and fresh_result.get("status") == "passed",
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "recovery_mode": restored["recovery_mode"],
        "loaded_checkpoint_id": selected["checkpoint_id"],
        "fallback_checkpoint_id": fallback["checkpoint_id"],
        "fresh_process": fresh_result,
        "measurements": {
            "actor_next_update_max_abs_difference": actor_difference,
            "critic_next_update_max_abs_difference": critic_difference,
            "valid_checkpoint_count": valid_checkpoint_count,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    resolved = json.loads(args.config.read_text())
    policy = RecurrentPolicyConfig.from_dict(resolved["policy"])
    ppo = RecurrentPPOConfig.from_dict(resolved["ppo"])
    RecoveryConfig.from_dict(resolved["recovery"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but vendor PyTorch cannot use it")
    args.output.mkdir(parents=True, exist_ok=True)
    result = evaluate(policy, ppo, resolved, args.config, args.output, device)
    result.update(
        schema_version=1,
        python_version=platform.python_version(),
        duration_seconds=time.perf_counter() - started,
        resolved_config=resolved,
    )
    write_json_atomic(args.output / "metrics.json", result, mode=0o644)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
