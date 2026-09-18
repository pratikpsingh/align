"""Export a verified learner checkpoint as an actor-only PyTorch artifact.

Run this with the pinned vendor Python; no Isaac Sim application is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path

import torch

from align.artifacts import as_ist, utc_now, write_json_atomic
from align.learning.torch_recovery import EXPECTED_STATE_KEYS, STATE_SCHEMA_VERSION
from align.policies.actor_probe import probe_observation, tensor_digest
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import SharedRecurrentActor
from align.tasks.observation import ObservationConfig


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _save_weights(path: Path, state: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".actor-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _benchmark(actor: SharedRecurrentActor, *, batch: int, repeats: int) -> dict:
    if batch < 1 or repeats < 1:
        raise ValueError("benchmark batch and repeats must be positive")
    sample = probe_observation(batch, actor.config.actor_observation_dim)
    memory = actor.backbone.recurrent.initial_state(sample, batch)
    mask = torch.ones((batch, 1), dtype=torch.float32)
    elapsed_us = []
    with torch.inference_mode():
        for index in range(20 + repeats):
            started = time.perf_counter_ns()
            result = actor.act(sample, memory, mask, deterministic=True)
            if index >= 20:
                elapsed_us.append((time.perf_counter_ns() - started) / 1000.0)
            memory = result.state
    ordered = sorted(elapsed_us)
    return {
        "batch": batch,
        "warmup_calls": 20,
        "measured_calls": repeats,
        "median_microseconds_per_call": statistics.median(ordered),
        "p95_microseconds_per_call": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "mean_microseconds_per_agent": math.fsum(elapsed_us) / len(elapsed_us) / batch,
        "scope": (
            "CPU Python/PyTorch actor.act including distribution transform; "
            "excludes sensing and controller"
        ),
    }


def export_actor(
    *,
    checkpoint_path: Path,
    config_path: Path,
    output: Path,
    expected_checkpoint_sha256: str,
    checkpoint_id: str,
    completed_update: int,
    repeats: int = 200,
) -> dict:
    """Validate trusted checkpoint bytes, export only actor tensors, then reload."""
    if len(expected_checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_checkpoint_sha256
    ):
        raise ValueError("expected checkpoint SHA-256 must be lowercase hex")
    if _sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("checkpoint checksum differs before deserialization")
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    policy_config = RecurrentPolicyConfig.from_dict(resolved["policy"])
    observation_config = ObservationConfig.from_dict(resolved["observation"])
    if policy_config.actor_observation_dim != observation_config.actor_dimension:
        raise ValueError("actor observation dimensions differ")
    max_speed_m_s = float(resolved["construction"]["max_speed_m_s"])
    control_dt_seconds = float(resolved["construction"]["physics_dt"])
    if not (math.isfinite(max_speed_m_s) and max_speed_m_s > 0):
        raise ValueError("export action speed must be finite and positive")
    if not (math.isfinite(control_dt_seconds) and control_dt_seconds > 0):
        raise ValueError("export control timestep must be finite and positive")
    if completed_update < 0:
        raise ValueError("completed update must be nonnegative")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != EXPECTED_STATE_KEYS:
        raise ValueError("learner checkpoint fields are incomplete")
    if (
        state["state_schema_version"] != STATE_SCHEMA_VERSION
        or state["resolved_config"] != resolved
        or state["counters"]["completed_updates"] != completed_update
    ):
        raise ValueError("learner checkpoint schema, config, or update differs")
    actor_state = state["actor"]
    if not isinstance(actor_state, dict) or not actor_state:
        raise ValueError("actor state is missing")
    if any(
        not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
        for value in actor_state.values()
    ):
        raise ValueError("actor tensors must be finite")
    torch.set_num_threads(1)
    actor = SharedRecurrentActor(policy_config).cpu().eval()
    actor.load_state_dict(actor_state, strict=True)
    output.mkdir(parents=True, exist_ok=False)
    actor_path = output / "actor.pt"
    config_output = output / "actor-config.json"
    observation_output = output / "observation-config.json"
    interface_output = output / "interface.json"
    write_json_atomic(config_output, policy_config.to_dict())
    write_json_atomic(observation_output, observation_config.to_dict())
    write_json_atomic(
        interface_output,
        {
            "schema_version": 1,
            "observation_shape": ["batch", 1, policy_config.actor_observation_dim],
            "observation_order": (
                "normalized own world velocity xyz; normalized relative target xyz; "
                "nearest-neighbor relative position/velocity slots; neighbor masks"
            ),
            "action_shape": ["batch", 1, policy_config.action_dim],
            "action_meaning": (
                "bounded world direction xyz and speed coordinate; speed magnitude is "
                "abs(action[3]) times max_speed_m_s"
            ),
            "direction_decode": "unit(action[:3]) if norm>1e-8 else zero",
            "max_speed_m_s": max_speed_m_s,
            "control_dt_seconds": control_dt_seconds,
            "inner_controller_not_exported": True,
            "action_low": list(policy_config.action_low),
            "action_high": list(policy_config.action_high),
            "recurrent_hidden_shape": [
                policy_config.recurrent_layers,
                "batch",
                policy_config.recurrent_hidden_size,
            ],
            "recurrent_cell_shape": [
                policy_config.recurrent_layers,
                "batch",
                policy_config.recurrent_hidden_size,
            ],
            "recurrent_reset": (
                "zero hidden and cell for a new episode or memory_mask=0 at episode start; "
                "memory_mask=1 for continuing observations"
            ),
            "deterministic_action": "affine(tanh(latent_mean)); no Gaussian sample",
            "critic_required_for_inference": False,
        },
    )
    _save_weights(actor_path, actor.state_dict())
    exported_state = torch.load(actor_path, map_location="cpu", weights_only=True)
    if set(exported_state) != set(actor.state_dict()):
        raise ValueError("exported actor parameter names differ")
    restored = (
        SharedRecurrentActor(RecurrentPolicyConfig.from_dict(json.loads(config_output.read_text())))
        .cpu()
        .eval()
    )
    restored.load_state_dict(exported_state, strict=True)
    probe = probe_observation(2, policy_config.actor_observation_dim)
    with torch.inference_mode():
        source_action = actor.act(probe, deterministic=True)
        restored_action = restored.act(probe, deterministic=True)
    if not (
        torch.equal(source_action.action, restored_action.action)
        and torch.equal(source_action.state.hidden, restored_action.state.hidden)
        and torch.equal(source_action.state.cell, restored_action.state.cell)
    ):
        raise ValueError("fresh actor differs from checkpoint actor")
    low = torch.tensor(policy_config.action_low, dtype=restored_action.action.dtype)
    high = torch.tensor(policy_config.action_high, dtype=restored_action.action.dtype)
    benchmarks = [_benchmark(restored, batch=batch, repeats=repeats) for batch in (1, 8)]
    parameters = sum(value.numel() for value in actor.parameters())
    parameter_bytes = sum(value.numel() * value.element_size() for value in actor.parameters())
    recurrent_bytes_per_agent = (
        2 * policy_config.recurrent_layers * policy_config.recurrent_hidden_size * 4
    )
    created_at_utc = utc_now()
    report = {
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": created_at_utc,
        "created_at_ist": as_ist(created_at_utc),
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "completed_update": completed_update,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "simulator_started": False,
        "learner_checkpoint_deserialized": True,
        "critic_module_instantiated": False,
        "critic_weights_in_export": False,
        "critic_required_for_inference": False,
        "parameter_count": parameters,
        "parameter_tensor_bytes": parameter_bytes,
        "recurrent_state_bytes_per_agent_float32": recurrent_bytes_per_agent,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "process_peak_rss_scope": (
            "whole exporter process including full learner-checkpoint deserialization, "
            "not isolated actor working memory"
        ),
        "artifact_bytes": actor_path.stat().st_size,
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (actor_path, config_output, observation_output, interface_output)
        },
        "benchmarks": benchmarks,
        "probe": {
            "observation_sha256": tensor_digest(probe),
            "action_sha256": tensor_digest(restored_action.action),
            "hidden_sha256": tensor_digest(restored_action.state.hidden),
            "cell_sha256": tensor_digest(restored_action.state.cell),
        },
        "checks": {
            "checkpoint_checksum_verified": True,
            "actor_state_finite": True,
            "actor_only_artifact_reloaded": True,
            "deterministic_action_and_memory_exact": True,
            "actor_action_bounded": bool(
                ((restored_action.action >= low) & (restored_action.action <= high)).all()
            ),
        },
    }
    if not all(report["checks"].values()):
        raise ValueError("actor export checks failed")
    write_json_atomic(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--completed-update", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)
    report = export_actor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output=args.output,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_id=args.checkpoint_id,
        completed_update=args.completed_update,
        repeats=args.repeats,
    )
    print(
        f"actor export {report['status']}: {report['parameter_count']} parameters, "
        f"{report['artifact_bytes']} bytes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
