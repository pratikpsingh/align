# Setup, reproducibility, and learning documentation

## Current state and runtime decision

The project now provides the installable diagnostic package, uv lockfile, Ruff development tooling, and [public setup instructions](../docs/01-installation.md). It still declares Python >=3.12 and has no simulator or learning dependencies. That declaration is provisional: select Python from the validated OmniDrones/Isaac Sim compatibility requirements, then update the project constraint, interpreter pin, dependency lock, and installation docs together.

The inspected paper-03 repository describes a legacy Isaac Sim 2022.2.0/Python 3.7 stack and pinned dependencies. The [OmniDrones installation guide](https://omnidrones.readthedocs.io/en/latest/installation.html) describes other supported pairings as well. Neither is a reason to install arbitrary latest versions or to assume the current scaffold's Python works. Choose and test an exact compatible stack for ALiGn.

uv manages the project's Python environment and Python dependencies. Isaac Sim also has native libraries, assets, environment settings, and runtime-specific packaging. A lockfile alone does not install or reproduce that entire simulator.

## Supported workflows to establish

| Workflow | Intended capability | Required evidence |
|---|---|---|
| CPU development | Geometry/model/buffer/configuration checks, report generation, supported policy inspection | Tested commands on a clean ordinary Python/uv setup without simulator imports |
| Lab GPU | OmniDrones simulation, training, evaluation, native rendering, GPU integration checks | Recorded compatible GPU/driver/runtime and successful smoke examples |
| Laptop presentation | Read reports, play exported video, make lightweight plots/playback from trajectories | Documented artifact inputs and a tested example |
| Fresh researcher checkout | Install and run the supported workflow from public docs | No dependency on ignored personal notes or local absolute paths |

Do not offer a CPU physics substitute under the OmniDrones name. If a later alternative simulator is added, label it and validate it independently.

## Lab preflight

Lab inventory has now been received: five RTX A4000 devices, each reporting 16,376 MiB, and approximately 503 GiB host RAM. See the [runtime assessment](06-lab-runtime-assessment.md). The standard [RTX A4000 specification](https://www.nvidia.com/en-us/products/workstations/rtx-a4000/) lists 16 GB VRAM; the five devices do not automatically provide one combined memory allocation.

These are information-gathering commands, not an ALiGn installation procedure:

~~~sh
uname -a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -h
df -h .
uv --version
~~~

Record command failures too: a missing nvidia-smi or uv has a different meaning from insufficient VRAM. Also record CPU, free storage on the actual output filesystem, runtime asset locations, and headless-rendering support. Avoid collecting sensitive host/environment details unnecessarily.

After selecting the runtime, measure one- and few-environment smoke examples before scaling parallel environments. Grow the batch using measured memory and throughput, leaving operational headroom. Hardware names alone cannot predict a safe batch size or training duration.

## uv and external runtime integration

Keep project dependencies isolated from a simulator-managed environment. uv's default exact synchronization removes undeclared packages from the target environment; therefore do not blindly run uv sync inside Isaac Sim's bundled Python environment. See [uv synchronization semantics](https://docs.astral.sh/uv/concepts/projects/sync/).

Prefer a supported isolated environment plus a documented runtime activation/bootstrap interface. If the chosen OmniDrones/runtime combination needs supplied activation scripts, audit the required paths/native-library settings and provide a tested integration. Do not assume a conda-based example makes conda mandatory, or that replacing its activation with a virtual environment is automatically sufficient.

Pin Python dependencies and upstream revisions once validated. Use locked synchronization for reproducible installs where applicable. Record external runtime version, build, download/asset requirements, compatibility checks, and licenses separately. Do not modify runtime-managed PyTorch/native libraries through hidden installer side effects.

Training/logging should work offline after required dependencies and assets are available. Document any first-run downloads and cache locations. Keep absolute machine-specific paths in explicit local configuration or documented environment variables, not source files.

## Public documentation to create with implementation

The root README, 01-installation.md, 02-diagnostics.md, and 05-development.md now document implemented behavior. The table below describes the broader documentation to maintain as research functionality is added:

| Document | Required contents |
|---|---|
| Root README.md | Project purpose/status, supported workflows, short tested example, links to detailed docs |
| docs/01-installation.md | Exact tested OS/hardware/runtime versions, prerequisites, uv steps, assets, preflight, expected outputs, troubleshooting |
| docs/configuration.md | Schema, units, defaults, overrides, resolved configuration and compatibility rules |
| docs/training.md | Start, monitor, stop, resume, warm start, checkpoints, recovery limits, resource/storage guidance |
| docs/evaluation.md | Freeze/load policy, suite/seed selection, success/metric definitions, artifact outputs |
| docs/visualization.md | Fresh simulation video versus trajectory playback, inputs, cameras, headless use, laptop playback |
| docs/experiments.md | Reproduce each figure/table, budgets, seed accounting, report commands and data schemas |
| docs/architecture.md | Modules, observation/action/recurrent contracts, simulator boundary, information flow |
| docs/troubleshooting.md | Tested failures and fixes: runtime mismatch, assets, OOM, headless graphics, NaNs, disk/checkpoint failures |

Add concrete commands only when exercised, with platform/runtime and expected output. Clearly label commands awaiting lab validation. A fresh reader should not need to infer setup from source imports or a developer's shell history.

## Learning documents

The initial [learning index](../learning/00-README.md) and conceptual explanations establish vocabulary. For each substantive implementation increment, create or update a document named for the topic, such as formation-geometry.md, recurrent-policies.md, reward-design.md, or checkpoint-recovery.md.

Each explanation should include:

- The problem being solved and why the project needs it.
- Prerequisite concepts, explained in plain language.
- A small worked numerical/data-flow example.
- The actual code/configuration paths once implemented, with dimensions and units where relevant.
- A tested command and expected result when runnable behavior exists.
- What was verified on CPU, what was verified in simulation, and what remains uncertain.
- Common mistakes and a short exercise or check for understanding.

Do not title or describe documents as numbered development installments. Users should be able to read by topic, while implementation proceeds gradually for understanding. Update explanations when code changes, rather than preserving an obsolete tutorial as if it still matches behavior.

## Ignored personal notes and portability

The user intentionally ignores plans/ and learning/. Preserve that decision. Those directories will not normally be included in Git commits or another person's clone. Keep required installation, runtime behavior, metric definitions, and reproducibility instructions in tracked public docs as they become implemented.

Back up the personal notes by copying/archiving them to the user's chosen storage; do not assume Git protects them. The tracked [working agreement](../AGENTS.md) preserves the development/documentation expectations even when personal notes are absent.

## Definition of a documented increment

An increment is ready when its scope is clear, code is maintainable, relevant checks pass, logs/results can be inspected, operational docs match the implementation, and its learning explanation is present. Research claims require additional controlled experimental evidence. Keep those statuses separate in reports and discussion.
