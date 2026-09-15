# Understanding the runnable project and its diagnostic

## What we built and why

We built an installable Python package and a command that inspects the computer and saves a report. This gives us a repeatable way to run code, see failures, and identify the machine/software used before introducing drone physics or learning.

Imagine a flight behaves differently on your laptop and the lab machine. Before investigating a learning algorithm, we need to know which interpreter, source code, driver, and packages each machine used. The diagnostic collects that evidence. It also establishes the logging and file organization we will reuse and extend as the project grows.

There is no learned policy, drone environment, or training loop in this implementation. The result is a working development tool. Training checkpoints and simulation videos remain future capabilities.

## A package, an environment, and a command

A Python file contains code. A package groups related files so they can be imported consistently. Our package is under src/align/. The project configuration, pyproject.toml, tells tools how to install it and exposes the align command.

A virtual environment is a project-specific place for a Python interpreter and installed packages. Here uv manages .venv. This separates ALiGn's dependencies from other projects. uv.lock records resolved dependency versions; .python-version currently requests Python 3.12.

The existing laptop has Python 3.12.3 and uv 0.9.28. That is enough for this diagnostic. It does not decide the future simulator stack: Isaac Sim and OmniDrones have additional compatibility requirements.

Run these commands in the align repository root:

~~~sh
uv sync --locked
uv run --locked align doctor
~~~

The first command prepares the project's environment. The second runs our installed command in that environment. The --locked flag prevents automatic changes to dependency resolution when the project metadata and lockfile disagree. See the [setup instructions](../docs/installation.md) for operational details and sources.

## Following one invocation through the code

~~~text
uv run --locked align doctor
          |
          v
cli.py: read options and construct DoctorConfig
          |
          v
artifacts.py: create a unique folder, initial report, and log files
          |
          v
runtime/diagnostics.py: collect each independent observation
          |
          v
cli.py: record checks, determine status, finish the report, return an exit code
~~~

The [CLI](../src/align/cli.py) decides which operation was requested. The [configuration](../src/align/config.py) rejects invalid settings such as a negative or infinite timeout. A frozen configuration cannot be changed accidentally halfway through the diagnostic.

The [probes](../src/align/runtime/diagnostics.py) inspect the machine. Each returns a name, status, explanation, and structured details. The [artifact code](../src/align/artifacts.py) handles file writing and logging. Keeping these responsibilities separate lets us test driver failures without changing the real driver, and test report writing without starting a simulator.

## Reading your first result

The command prints a report path. Its parent folder contains three files:

| File | How to use it |
|---|---|
| report.json | Open in the editor to inspect settings, hardware/software details, checks, and final status |
| doctor.log | Read the chronological messages when understanding what happened |
| events.jsonl | Structured versions of events that later tools can process automatically |

JSON is a text format for named values, lists, and nested objects. For example, this simplified result means the check could not query NVIDIA devices:

~~~json
{
  "name": "nvidia",
  "status": "warning",
  "details": {
    "query_status": "missing",
    "devices": [],
    "reason": "nvidia-smi is not available on PATH"
  }
}
~~~

This matches the kind of result observed on this laptop. It is not proof that no GPU physically exists. It says the command used to query NVIDIA devices is unavailable to this process. CPU development can continue.

Likewise, a null simulator-package version means metadata was not found in this interpreter. An externally installed simulator may use a different Python environment.

## Why completed does not mean simulation-ready

The report's status answers whether the diagnostic completed. It does not answer whether a drone can fly in OmniDrones. A successful package lookup does not launch that package; a successful driver query does not execute a CUDA computation.

We therefore record simulation_validation as not_performed. Later, a real simulator check will supply different evidence: constructing an environment, receiving observations, sending commands, advancing physics, and resetting successfully.

For the lab inventory, --require-nvidia makes a failed device query a failed requirement. It still does not certify the runtime. This distinction prevents a reassuring green check from claiming more than we measured.

## RAM, VRAM, and units

RAM is the computer's main memory. VRAM belongs to a GPU. Their capacities are not interchangeable. The laptop report measured about 15.5 GiB of host RAM. The lab's per-device VRAM will be measured when its NVIDIA query runs.

The report stores host RAM in bytes and NVIDIA memory in MiB. One GiB is 1,024 cubed bytes; one MiB is 1,024 squared bytes. If two GPUs each report 16,384 MiB, record two devices with 16 GiB each, not one device with 32 GiB available to an arbitrary simulation.

Host-reported RAM may differ from a container or scheduler's actual allocation. The diagnostic records that limitation rather than guessing an effective training budget.

## Keeping evidence when something fails

Each invocation uses a timestamp and unique suffix, so it does not overwrite an earlier run. The report is refreshed after each completed probe. Before replacing it, the writer serializes and synchronizes a temporary file. If publication fails, the previous complete JSON can remain readable.

This is useful file-writing behavior, but it is not the training recovery system. A future training checkpoint must additionally preserve weights, optimizer state, normalization, randomness, counters, and the declared episode/simulator state.

An unfinished diagnostic may leave status=running. That means no completion record was saved; it is not a successful result. The [report reference](../docs/diagnostics.md) explains all statuses and exit codes.

## How we checked the implementation

Twenty automated tests cover useful failure and boundary cases, including a missing driver, malformed GPU output, unknown memory, timeout, invalid options, failed publication, interruption, and unique logging identities. Tests also check that the installed command can run outside the source folder without importing simulator or learning libraries.

The test command is:

~~~sh
uv run --locked python -m unittest discover -s tests -v
~~~

It should end with OK. Passing these tests supports the diagnostic behavior. It says nothing yet about formation control or reinforcement-learning performance. Clean-copy installation and wheel checks separately verify that the package does not depend on this particular working directory.

## A small exercise

Run the default diagnostic twice and open both report folders. Confirm that their run IDs differ. Find the Python executable, the NVIDIA reason/device list, and simulation_validation. Then try a custom folder:

~~~sh
uv run --locked align doctor --output-dir "runs/my first check"
~~~

The quotes keep the path containing spaces as one argument. The output belongs to this invocation, and earlier diagnostic results remain intact. These commands create small diagnostic files; they do not train a policy or start a simulator.

The next concept to learn is the relationship between a simulator, an environment, and a low-level controller. A known command applied to one drone will give us a concrete observation/action loop before a learned policy enters it.
