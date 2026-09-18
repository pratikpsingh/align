"""Verify an actor artifact in a fresh CPU process without a learner checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from align.artifacts import as_ist, utc_now, write_json_atomic
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


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("process RSS is unavailable")


def verify_artifact(artifact: Path, output: Path) -> dict:
    """Load only exported files and reproduce exact deterministic action/memory."""
    source = json.loads((artifact / "report.json").read_text())
    if source.get("status") != "passed":
        raise ValueError("actor export report did not pass")
    expected_names = {"actor.pt", "actor-config.json", "observation-config.json", "interface.json"}
    if set(source["artifacts"]) != expected_names:
        raise ValueError("actor artifact file inventory differs")
    for name, item in source["artifacts"].items():
        file = artifact / name
        if not file.is_file() or file.is_symlink():
            raise ValueError(f"missing or linked actor artifact: {name}")
        if file.stat().st_size != item["bytes"] or _sha256(file) != item["sha256"]:
            raise ValueError(f"actor artifact hash or size differs: {name}")
    config = RecurrentPolicyConfig.from_dict(
        json.loads((artifact / "actor-config.json").read_text())
    )
    observation = ObservationConfig.from_dict(
        json.loads((artifact / "observation-config.json").read_text())
    )
    interface = json.loads((artifact / "interface.json").read_text())
    if (
        config.actor_observation_dim != observation.actor_dimension
        or interface["observation_shape"] != ["batch", 1, observation.actor_dimension]
        or interface["action_shape"] != ["batch", 1, config.action_dim]
        or interface["schema_version"] != 1
    ):
        raise ValueError("actor input/output contracts differ")
    torch.set_num_threads(1)
    rss_before_model_bytes = _rss_bytes()
    actor = SharedRecurrentActor(config).cpu().eval()
    weights = torch.load(artifact / "actor.pt", map_location="cpu", weights_only=True)
    actor.load_state_dict(weights, strict=True)
    rss_after_load_bytes = _rss_bytes()
    probe = probe_observation(2, config.actor_observation_dim)
    with torch.inference_mode():
        result = actor.act(probe, deterministic=True)
        recurrent_state = actor.backbone.recurrent.initial_state(probe, 2)
        reset_state = type(recurrent_state)(
            torch.ones_like(recurrent_state.hidden), torch.ones_like(recurrent_state.cell)
        )
        after_reset = actor.act(
            probe, reset_state, torch.zeros((2, 1), dtype=torch.float32), deterministic=True
        )
    rss_after_inference_bytes = _rss_bytes()
    probe_hashes = {
        "observation_sha256": tensor_digest(probe),
        "action_sha256": tensor_digest(result.action),
        "hidden_sha256": tensor_digest(result.state.hidden),
        "cell_sha256": tensor_digest(result.state.cell),
    }
    low = torch.tensor(config.action_low, dtype=result.action.dtype)
    high = torch.tensor(config.action_high, dtype=result.action.dtype)
    checks = {
        "exported_hashes_match": True,
        "probe_action_and_memory_exact": probe_hashes == source["probe"],
        "action_finite": bool(torch.isfinite(result.action).all()),
        "action_within_bounds": bool(((result.action >= low) & (result.action <= high)).all()),
        "memory_reset_exact": bool(
            torch.equal(result.action, after_reset.action)
            and torch.equal(result.state.hidden, after_reset.state.hidden)
            and torch.equal(result.state.cell, after_reset.state.cell)
        ),
    }
    now = utc_now()
    report = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "created_at_utc": now,
        "created_at_ist": as_ist(now),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "learner_checkpoint_available": False,
        "simulator_started": False,
        "critic_module_instantiated": False,
        "artifact_sha256": source["artifacts"]["actor.pt"]["sha256"],
        "parameter_count": sum(value.numel() for value in actor.parameters()),
        "rss_before_model_bytes": rss_before_model_bytes,
        "rss_after_load_bytes": rss_after_load_bytes,
        "rss_after_inference_bytes": rss_after_inference_bytes,
        "rss_scope": "whole fresh vendor-Python process, not isolated actor allocation",
        "checks": checks,
        "probe": probe_hashes,
    }
    write_json_atomic(output, report)
    if report["status"] != "passed":
        raise ValueError(f"fresh actor inference checks failed: {checks}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = verify_artifact(args.artifact, args.output)
    print(f"fresh actor inference {report['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
