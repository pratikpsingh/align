# Installation and machine inventory

## What works now

ALiGn currently provides a CPU-capable diagnostic command, saved reports, local logs, and automated checks. It has no training command or OmniDrones environment yet. The runtime package uses the Python standard library; Ruff is a development dependency.

Validated locally on Linux x86_64 with Python 3.12.3 and uv 0.9.28. The current lockfile selects Ruff 0.16.7. Python 3.12 is the development interpreter pin, not a validated OmniDrones/Isaac Sim choice. Other operating systems and Python versions have not been validated; detailed RAM probing currently supports Linux.

## Install the development project

Obtain this repository and open a terminal in its root: the directory containing pyproject.toml. For this workspace that directory is align/. Do not run these commands in the parent folder containing all the reference repositories.

If uv is missing, use the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/). The existing machine already had uv installed; its installer was not exercised as part of this change. Ensure a Python 3.12 interpreter is available. uv may download one if needed and permitted; interpreter download was not needed on the tested machine.

Run from the repository root:

~~~sh
uv --version
uv sync --locked
uv run --locked align --help
uv run --locked align doctor
~~~

The sync command installs the project into its local .venv and installs the locked development tools. The --locked option rejects an out-of-date lockfile instead of silently changing dependency resolution. Manual environment activation is unnecessary for uv run. These are [uv project synchronization behaviors](https://docs.astral.sh/uv/concepts/projects/sync/).

Expected result: the command logs its checks, prints a report path under runs/diagnostics/, and prints that simulation has not been tested. Missing NVIDIA tools are a warning for this default invocation. The command should exit successfully if it can complete and save the diagnostic.

For an inventory-only installation, uv sync --locked --no-dev omits Ruff. This is suitable for running doctor; use the normal sync command when working on code. The package has been checked from a fresh copy without personal plans/ or learning/ directories and from a built wheel installed in a separate environment.

After dependencies/interpreter/build support are available locally, offline use is supported:

~~~sh
uv run --locked --offline align doctor
~~~

Offline mode cannot supply packages or a Python interpreter that have never been installed/cached.

## Inspect the result

Each invocation creates a new folder containing report.json, doctor.log, and events.jsonl. Open report.json in the editor. Its status describes the diagnostic, and simulation_validation remains not_performed. The [diagnostic reference](diagnostics.md) explains the fields, exit codes, and limitations.

The laptop check found no nvidia-smi command on PATH and no distribution metadata for the simulator/learning packages in this project environment. This does not independently prove the physical absence of a GPU or an externally installed simulator.

## Collect the lab inventory

Install the same development project in its own environment on the lab machine, then run:

~~~sh
uv run --locked align doctor --require-nvidia
~~~

This additionally requires a successful NVIDIA device query. It still does not initialize CUDA, launch Isaac Sim, or test rendering. The success path for physical NVIDIA hardware must be exercised on the lab machine; parser and failure behavior are checked locally.

Use the per-device name, driver and VRAM fields plus host RAM/OS information to select the simulator stack. System RAM and each GPU's VRAM are separate quantities. The query follows NVIDIA's [nvidia-smi interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Simulator installation remains a separate task

Choose an exact OmniDrones revision, Isaac Sim build, and compatible Python/PyTorch/TorchRL/TensorDict combination before adding those dependencies. The [OmniDrones installation guide](https://omnidrones.readthedocs.io/en/latest/installation.html) documents version-dependent runtime requirements. The current diagnostic cannot certify a pairing.

Keep the project environment separate from simulator-managed Python installations. Do not synchronize the current development lockfile into Isaac Sim's bundled environment: exact synchronization can remove undeclared packages. Record the tested activation, assets, native libraries, and rendering requirements when simulator integration is implemented.

## Common setup problems

| Symptom | Meaning and next action |
|---|---|
| uv: command not found | Install uv or open a terminal where its installation is on PATH |
| No pyproject.toml found | Change into the align repository root |
| Lockfile needs updating | Ensure pyproject.toml and uv.lock are from the same source revision; dependency changes need intentional relocking and verification |
| align command missing | Run uv sync --locked, then invoke through uv run --locked align |
| NVIDIA warning | Read checks[name=nvidia].details.reason; CPU development remains available |
| Strict NVIDIA invocation exits 1 | The requested device query failed, timed out, returned no devices, or could not be parsed |
| Cannot write diagnostic artifacts | Choose a writable --output-dir and check free space; existing output files are not treated as directories |
| Package versions are null | Distribution metadata was not found in this interpreter; inspect external simulator activation separately |

The root README and docs/ are intended for Git. runs/, plans/, and learning/ are ignored. Copy the specific diagnostic folder when transferring evidence; Git will not include it automatically.
