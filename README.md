# ALiGn

Tools for reproducible research on UAV formation control with local observations and recurrent multi-agent learning. The target simulator is OmniDrones.

## Current capabilities

The installable package provides machine diagnostics, versioned JSON reports, local logs, and automated checks. The standalone Isaac Sim startup/CUDA smoke test has passed on the lab GPU using fast shutdown. A pinned OmniDrones runtime and deterministic single-drone/controller probe are implemented, with lab calibration evidence in the [control guide](docs/08-single-drone-control.md). Policies, training, and checkpoint recovery remain future work.

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
