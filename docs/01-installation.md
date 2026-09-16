# Installation and machine inventory

## What works now

ALiGn currently provides a CPU-capable diagnostic command, saved reports, local logs, and automated checks. It also provides a pinned OmniDrones single-drone integration; see [runtime setup](07-omnidrones-runtime.md) and its separately recorded lab acceptance. There is no training command. Host diagnostics use the Python standard library; Ruff is a development dependency. The optional reporting extra adds locked plotting libraries. Simulator dependencies are separately pinned.

Validated locally on Linux x86_64 with Python 3.12.3 and uv 0.9.28. A user-supplied lab report also records successful diagnostic execution with Python 3.12.14 and uv 0.12.12. The current lockfile selects Ruff 0.16.7. Python 3.12 is the host development pin. The shared package supports Python >=3.10; the simulator uses its separate bundled Python 3.10.14. Other operating systems and Python versions have not been validated; detailed RAM probing currently supports Linux.

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

Each invocation creates a new folder containing report.json, doctor.log, and events.jsonl. Open report.json in the editor. Its status describes the diagnostic, and simulation_validation remains not_performed. The [diagnostic reference](02-diagnostics.md) explains the fields, exit codes, and limitations.

The laptop check found no nvidia-smi command on PATH and no distribution metadata for the simulator/learning packages in this project environment. This does not independently prove the physical absence of a GPU or an externally installed simulator.

## Collect the lab inventory

Install the same development project in its own environment on the lab machine, then run:

~~~sh
uv run --locked align doctor --require-nvidia
~~~

This additionally requires a successful NVIDIA device query. It still does not initialize CUDA, launch Isaac Sim, or test rendering. A user-supplied lab report on 2026-09-15 confirms that the NVIDIA query succeeded on five RTX A4000 devices with driver 580.173.02. Simulator execution and GPU-container access remain untested; parser and failure behavior are checked locally.

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

Lab prerequisite update, reviewed 2026-09-16: the user successfully queried one RTX A4000 from an Ubuntu container. The [GPU container setup procedure](03-gpu-container-setup.md) records the result and next image-acquisition commands. Actual Isaac Sim execution remains pending.

## Candidate image and startup validation

The lab user supplied the Isaac Sim 4.1.0 digest sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173. Acquisition is confirmed; simulator execution and OmniDrones compatibility remain unverified. The startup launcher and artifact contract are described in [the smoke-test guide](../docs/04-isaac-sim-smoke.md).
