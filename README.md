# ALiGn

Tools for reproducible research on UAV formation control with local observations and recurrent multi-agent learning. The target simulator is OmniDrones.

## Current capabilities

The installable package provides machine diagnostics, versioned JSON reports, local logs, and automated checks. The standalone Isaac Sim startup/CUDA smoke test, deterministic one-drone controller suite, and four-drone ground-to-plane construction probe passed on the lab GPU using fast shutdown. Simulator-independent formation geometry, active component-logged rewards, fixed-capacity masked local observations, vector episode semantics, and recurrent rollout storage are implemented. The one-world and four-world cloned vector task passed on the lab GPU. The shared LSTM actor, centralized recurrent critic, transformed bounded-action distribution, live device collector, and masked recurrent PPO optimizer update passed their CUDA contracts. Atomic recovery and a bounded two-process task-connected training run have passed. Multi-seed stability calibration and fresh-process deterministic evaluation have passed on the lab GPU. Matched-rollout calibration selected a stable critic learning rate, and a recovered ten-update two-seed learning curve passed exact milestone evaluation. It showed no formation success and persistently negative explained variance. Semantic critic-input and value-target diagnostics selected critic-only active-group normalization. Its frozen warmup, checkpoint/recovery, host-audited raw reductions, ten-update training, and exact 0/5/10 evaluation contracts passed on two lab seeds. Critic explained variance improved sharply, but velocity statistics drifted into clipping and formation evaluation remained mixed with no successes.

## Run the diagnostic

From this repository root, with uv and Python 3.12 available:

```sh
uv sync --locked
uv run --locked align doctor
```

Each invocation saves report.json, doctor.log, and events.jsonl in a unique folder under runs/diagnostics/. Missing NVIDIA tools are a warning by default. A completed diagnostic does not certify simulation readiness.

To require an NVIDIA device query when checking the lab machine:

```sh
uv run --locked align doctor --require-nvidia
```

## Documentation

Start with the [numbered documentation index](docs/00-README.md). For conceptual background, read the [learning index](learning/00-README.md); for intended research work, read the [planning index](plans/00-README.md). Each folder has its own reading order.

- [Installation](docs/01-installation.md): laptop setup, lab inventory, runtime limitations, troubleshooting.
- [Diagnostic reference](docs/02-diagnostics.md): options, report fields, exit codes, artifact semantics.
- [Development](docs/05-development.md): code layout, tests, formatting, builds, and portability.
- [OmniDrones runtime](docs/07-omnidrones-runtime.md): reproducible derived image, exact dependencies, and Python boundary.
- [Single-drone control](docs/08-single-drone-control.md): commands, frames, reset contract, raw trajectories and lab results.
- [Formation geometry](docs/09-formation-geometry.md): templates, assignment, metrics, configuration, and deterministic reports.
- [Multi-drone construction](docs/10-multi-drone-construction.md): group contract, staged takeoff, safety measurements, and lab execution.
- [Task reward contract](docs/11-task-reward-contract.md): component equations, aggregation, configuration, and offline trajectory audit.
- [Local observation contract](docs/12-local-observation-contract.md): hard radius, neighbor budget, masks, normalization, critic separation, and topology audit.
- [Vectorized task environment](docs/13-vectorized-task-environment.md): cloned worlds, step/reset semantics, terminal masks, GPU command, and evidence.
- [Recurrent rollout contract](docs/14-recurrent-rollout-contract.md): temporal storage, GAE masks, recurrent states, and sequence chunks.
- [Recurrent policy contract](docs/15-recurrent-policy-contract.md): shared actor, centralized critic, bounded actions, and CUDA evidence.
- [Device recurrent collector](docs/16-device-recurrent-collector.md): live rollouts, episode boundaries, CUDA/reference parity, and saved evidence.
- [Recurrent MAPPO optimizer](docs/17-recurrent-ppo-update.md): clipped temporal losses, optimizer diagnostics, padding invariance, and CUDA evidence.
- [Training recovery](docs/18-training-recovery.md): atomic checkpoints, complete learner state, fallback, and reset-mode resume.
- [Task-connected training](docs/19-task-connected-training.md): live task rollouts, PPO updates, process restart, and checkpoint lineage.
- [Training stability and evaluation](docs/20-training-stability-and-evaluation.md): multi-seed calibration, post-update diagnostics, and deterministic checkpoint evaluation.
- [Critic scale calibration](docs/21-critic-scale-calibration.md): matched-rollout critic learning-rate comparison and value distribution evidence.
- [Bounded learning curves](docs/22-bounded-learning-curves.md): ten-update, two-seed training with exact update 0/5/10 checkpoint evaluation.
- [Frozen critic normalization](docs/23-frozen-critic-normalization.md): active-group warmup statistics, checkpoint restoration, evaluation immutability, and lab evidence.
- [Critic distribution drift](docs/24-critic-distribution-drift.md): per-update frozen-normalizer drift metrics and the normalized ten-update curve command.
- [Lab handoff](docs/06-lab-handoff.md): verified runtime, next task, and a continuation prompt.
- [Working agreement](AGENTS.md): implementation, teaching, and research-record requirements.

The current checkout tracks plans/ and learning/ notes. Required usage instructions live in docs/ so this checkout is usable without those notes. Generated runs/ and dist/ are also ignored.

## Verify changes

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Host development stays on Python 3.12. The shared package also passes CPU checks on Python 3.10; the pinned simulator uses bundled Python 3.10.14 and vendor PyTorch 2.2.2+cu118. Its dependency lock is separate from the host lock.
