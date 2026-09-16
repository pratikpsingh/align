"""Fresh-process evaluation of a checksum-verified learner checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from align.artifacts import write_json_atomic
from align.learning.checkpoint_store import CheckpointStore
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.torch_recovery import restore_learner_state
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import (
    CentralizedRecurrentCritic,
    RecurrentState,
    SharedRecurrentActor,
)


def _optimizer(module, learning_rate: float, config: RecurrentPPOConfig):
    return torch.optim.Adam(module.parameters(), lr=learning_rate, eps=config.adam_epsilon)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    resolved = json.loads(args.config.read_text())
    policy = RecurrentPolicyConfig.from_dict(resolved["policy"])
    ppo = RecurrentPPOConfig.from_dict(resolved["ppo"])
    device = torch.device(args.device)
    actor = SharedRecurrentActor(policy).to(device)
    critic = CentralizedRecurrentCritic(policy).to(device)
    actor_optimizer = _optimizer(actor, ppo.actor_learning_rate, ppo)
    critic_optimizer = _optimizer(critic, ppo.critic_learning_rate, ppo)
    store = CheckpointStore(
        args.checkpoints,
        run_id=args.run_id,
        config_sha256=args.config_sha256,
        file_mode=0o644,
    )
    manifest, payload = store.latest_valid()
    restored = restore_learner_state(
        payload,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        expected_config=resolved,
    )
    batch = 3
    actor_memory = RecurrentState(
        torch.zeros(policy.recurrent_layers, batch, policy.recurrent_hidden_size, device=device),
        torch.zeros(policy.recurrent_layers, batch, policy.recurrent_hidden_size, device=device),
    )
    critic_memory = RecurrentState(
        torch.zeros(policy.recurrent_layers, 1, policy.recurrent_hidden_size, device=device),
        torch.zeros(policy.recurrent_layers, 1, policy.recurrent_hidden_size, device=device),
    )
    with torch.no_grad():
        actor_output = actor(
            torch.zeros(batch, 1, policy.actor_observation_dim, device=device), actor_memory
        )
        value = critic(
            torch.zeros(1, 1, policy.critic_state_dim, device=device), critic_memory
        ).value
    passed = bool(torch.isfinite(actor_output.latent_mean).all() and torch.isfinite(value).all())
    result = {
        "status": "passed" if passed else "failed",
        "checkpoint_id": manifest["checkpoint_id"],
        "completed_updates": restored["counters"]["completed_updates"],
        "finite_actor_and_critic_output": passed,
        "reset_environment_and_recurrent_memory": restored[
            "reset_environment_and_recurrent_memory"
        ],
    }
    write_json_atomic(args.output, result, mode=0o644)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
