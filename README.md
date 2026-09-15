# ALiGn

Tools for reproducible research on UAV formation control with local observations and recurrent multi-agent learning. The target simulator is OmniDrones.

## Current capabilities

The installable package provides machine diagnostics, versioned JSON reports, local logs, and automated checks. Simulator integration, policies, training, checkpoint recovery, and flight visualization are not implemented yet.

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
- [Working agreement](AGENTS.md): implementation, teaching, and research-record requirements.

Personal plans/ and learning/ notes are intentionally ignored by Git. Required usage instructions live in docs/ so this checkout is usable without those notes. Generated runs/ and dist/ are also ignored.

## Verify changes

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Validated development interpreter: Python 3.12.3 on Linux x86_64 with uv 0.9.28. The simulator-compatible Python and dependency versions will be selected through lab validation.
