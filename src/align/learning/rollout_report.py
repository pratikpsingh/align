"""Create a small auditable report for the recurrent rollout contract."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from align.artifacts import (
    artifact_logger,
    as_ist,
    create_run_directory,
    utc_now,
    write_json_atomic,
)
from align.learning.rollout import (
    RecurrentFrame,
    RecurrentRollout,
    RolloutConfig,
    RolloutTransition,
)
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.observation import ObservationConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "recurrent-rollout.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "recurrent-rollout"


def load_config(path: Path) -> RolloutConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("rollout configuration root must be an object")
    return RolloutConfig.from_dict(value)


def _memory(config: RolloutConfig, value: float):
    return tuple(
        tuple(
            tuple(
                tuple(value for _ in range(config.recurrent_hidden_size))
                for _ in range(config.recurrent_layers)
            )
            for _ in range(config.num_agents)
        )
        for _ in range(config.num_envs)
    )


def _frame(config: RolloutConfig, marker: float, *, reset: bool = False) -> RecurrentFrame:
    memory = _memory(config, 0.0 if reset else marker)
    return RecurrentFrame(
        actor_observations=tuple(
            tuple(
                tuple(marker for _ in range(config.actor_observation_dim))
                for _ in range(config.num_agents)
            )
            for _ in range(config.num_envs)
        ),
        critic_states=tuple(
            tuple(100.0 + marker for _ in range(config.critic_state_dim))
            for _ in range(config.num_envs)
        ),
        actor_hidden=memory,
        actor_cell=memory,
        critic_hidden=memory,
        critic_cell=memory,
    )


def _transition(
    config: RolloutConfig,
    reward: float,
    bootstrap: float,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> RolloutTransition:
    agents = tuple(0.0 for _ in range(config.num_agents))
    return RolloutTransition(
        actions=tuple(
            tuple((0.0,) * config.action_dim for _ in range(config.num_agents))
            for _ in range(config.num_envs)
        ),
        old_log_probs=tuple(agents for _ in range(config.num_envs)),
        rewards=tuple(
            tuple(reward for _ in range(config.num_agents)) for _ in range(config.num_envs)
        ),
        values=tuple(agents for _ in range(config.num_envs)),
        bootstrap_values=tuple(
            tuple(bootstrap for _ in range(config.num_agents)) for _ in range(config.num_envs)
        ),
        terminated=tuple(terminated for _ in range(config.num_envs)),
        truncated=tuple(truncated for _ in range(config.num_envs)),
    )


def build_example(production: RolloutConfig) -> dict:
    construction = MultiDroneConfig()
    observation = ObservationConfig()
    task = TaskEnvironmentConfig()
    production.validate_dimensions(
        num_envs=task.num_envs,
        num_agents=construction.num_agents,
        actor_observation_dim=observation.actor_dimension,
        critic_state_dim=observation.critic_dimension,
        action_dim=4,
    )
    demo = RolloutConfig(
        horizon=3,
        num_envs=1,
        num_agents=1,
        actor_observation_dim=2,
        critic_state_dim=3,
        action_dim=1,
        recurrent_layers=1,
        recurrent_hidden_size=2,
        chunk_length=2,
        gamma=0.5,
        gae_lambda=1.0,
    )
    rollout = RecurrentRollout(demo, _frame(demo, 10.0))
    rollout.append(_transition(demo, 1.0, 2.0), _frame(demo, 11.0))
    rollout.append(
        _transition(demo, 3.0, 0.0, terminated=True),
        _frame(demo, 20.0, reset=True),
    )
    rollout.append(
        _transition(demo, 5.0, 4.0, truncated=True),
        _frame(demo, 30.0, reset=True),
    )
    advantages = rollout.compute_gae()
    chunks = rollout.sequence_chunks()
    advantage_values = [advantages[step][0][0] for step in range(demo.horizon)]
    checks = {
        "production_dimensions_match_task": True,
        "expected_advantages": advantage_values == [3.5, 3.0, 7.0],
        "episode_safe_chunk_boundaries": [
            (chunk.start_step, chunk.valid_length) for chunk in chunks
        ]
        == [(0, 2), (2, 1)],
        "padding_mask_present": chunks[1].valid_mask == (1.0, 0.0),
        "post_terminal_memory_reset": chunks[1].initial_actor_hidden == ((0.0, 0.0),),
        "actor_and_critic_inputs_separate": 110.0 not in chunks[0].actor_observations[0],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "production_config": production.to_dict(),
        "worked_example": {
            "advantages": advantage_values,
            "chunk_starts_and_lengths": [
                [chunk.start_step, chunk.valid_length] for chunk in chunks
            ],
            "valid_masks": [list(chunk.valid_mask) for chunk in chunks],
            "boundary_types": ["continuing", "terminated", "truncated"],
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = create_run_directory(args.output_dir.resolve())
    started = utc_now()
    with artifact_logger(
        run_dir,
        args.log_level,
        log_filename="rollout.log",
        namespace="align.recurrent_rollout",
    ) as logger:
        try:
            config = load_config(args.config.resolve())
            shutil.copy2(args.config.resolve(), run_dir / "config.json")
            report = build_example(config)
            report.update(
                {
                    "started_at_utc": started,
                    "started_at_ist": as_ist(started),
                    "finished_at_utc": utc_now(),
                    "command": [sys.executable, *sys.argv],
                }
            )
            report["finished_at_ist"] = as_ist(report["finished_at_utc"])
            write_json_atomic(run_dir / "report.json", report)
            logger.info(
                "Recurrent rollout report %s: %s",
                report["status"],
                run_dir,
                extra={"event": "finished", "check_status": report["status"]},
            )
            return 0 if report["status"] == "passed" else 1
        except Exception:
            logger.exception(
                "Recurrent rollout report failed: %s",
                run_dir,
                extra={"event": "failed", "check_status": "failed"},
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
