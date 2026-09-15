# Development and validation

## Current code structure

~~~text
src/align/
  __init__.py              # importable package; no simulator initialization
  __main__.py              # python -m align
  cli.py                   # arguments, diagnostic lifecycle, exit codes
  config.py                # immutable validated diagnostic settings
  artifacts.py             # report writes and owned local logging handlers
  runtime/
    __init__.py
    diagnostics.py         # independent host/package/source probes
tests/
  test_artifacts.py
  test_cli.py
  test_diagnostics.py
~~~

The project script declared in pyproject.toml maps align to align.cli:main. The original hello-world main.py has been removed. Run the installed package through uv instead of adding src to sys.path.

uv_build 0.9.28 is the pinned build backend. A build backend turns package source into installable distributions; uv can use its compatible bundled backend. The src layout and native backend behavior are documented in [uv's build backend guide](https://docs.astral.sh/uv/concepts/build-backend/).

The current package uses standard-library code only. Keep future simulator imports behind the runtime boundary so diagnostics and mathematical checks remain usable without GPU software. Add modules for implemented responsibilities, not empty placeholders for the entire roadmap.

## Checks

From the repository root:

~~~sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check .
uv run --locked ruff format --check .
~~~

Tests use Python's unittest runner. The suite checks NVIDIA parsing and failure semantics, strict versus optional checks, finite timeout validation, artifact preservation on failed writes, unique run IDs, log identity/cleanup, partial results on interruption, CLI errors, and installed-package use outside the checkout without simulator imports.

Hardware-free fixtures test what happens when a driver is missing, fails, times out, or reports multiple devices. They do not replace a real GPU/driver test. The ordinary laptop diagnostic provides separate evidence about this actual host.

Ruff checks imports, common errors, and selected maintainability rules; its formatter provides consistent layout. To apply formatting intentionally:

~~~sh
uv run --locked ruff format .
~~~

Do not treat formatting as a scientific validation. Later geometry, recurrence, control, and recovery changes need checks of their actual behavior.

## Building and portability

~~~sh
uv build
~~~

This creates an sdist (.tar.gz source archive) and wheel (.whl installable package) under dist/. The source archive includes public docs, tests, uv.lock, and .python-version; the wheel contains the runtime package. Personal notes and diagnostic runs are not distribution contents.

Validation for this implementation includes a clean copied project with a new virtual environment, an inventory-only installation, offline invocation with cached prerequisites, and a wheel installed into another environment and invoked outside the source checkout. This checks package portability on the tested Linux/Python combination; it does not establish other OS or simulator support.

Build artifacts and diagnostics are ignored by Git. Preserve source and public documentation in version control; copy the required run folder separately when transferring evidence. No commit or external publication is performed automatically by these commands.

## Continuing the implementation

The next runtime work is to collect the lab inventory, select a supported simulator/Python pairing, and run a minimal drone/controller check. Keep Python 3.12 labeled as the current development pin until that pairing is validated. Do not infer readiness from the presence of package metadata.

For every substantive implementation increment, explain its purpose, inputs/outputs, a small worked example, code paths, verification, and limits in the topic-based learning notes. Public docs must remain sufficient when the ignored notes are absent.
