# Lab inventory and simulator runtime assessment

Latest review: 2026-09-16. Lab inventory, the isolated Isaac Sim fast-mode startup/CUDA test, deterministic drone control, and the one/four-world OmniDrones task are confirmed. Full extension cleanup remains unvalidated. Earlier dated findings below are retained as history.

## Evidence received

The supplied diagnostic has run ID 20260915T170626.384884Z-6278da59, status completed, exit code 0, and require_nvidia=true. It reports source revision eca98888797a1043df2ec7c4e49f202852a186fe with a clean worktree at collection time. Its simulation_validation remains not_performed.

| Resource | Reported value | Interpretation |
|---|---|---|
| GPU | Five NVIDIA RTX A4000 devices, indices 0–4 | Start with one explicitly selected, available device |
| VRAM | 16,376 MiB per GPU, approximately 15.99 GiB | Separate device memories; not one approximately 80 GiB allocation |
| Driver | 580.173.02 on every reported GPU | Inventory succeeded; CUDA, Vulkan, and Isaac Sim behavior still need tests |
| CPU | AMD EPYC 9654 96-Core Processor; 192 logical CPUs | Substantial host resources; job allocation and CPU affinity are unknown |
| RAM | 503.39 GiB total, 492.44 GiB available at collection | Host-reported capacity, not a guaranteed scheduler/container allowance |
| Output storage | Approximately 18.89 TiB free | Filesystem capacity; quota, storage type, and I/O throughput are unmeasured |
| OS | Ubuntu 24.04.5 LTS, x86_64; kernel 7.0.0-31-generic | Preserve the exact reported OS/kernel as compatibility-test inputs |
| Python / uv | CPython 3.12.14 / uv 0.12.12 | The diagnostic works on this lab software combination |
| Simulator-related distributions | No metadata found for Isaac Sim, OmniDrones, PyTorch, TorchRL, TensorDict | This interpreter lacks those distribution records; external installations remain possible |

The hardware inventory supports proceeding to a small GPU runtime trial. It does not establish training throughput, safe parallel-environment count, access to all five devices, or successful physics/rendering.

## Upstream compatibility findings

The upstream [OmniDrones README](https://github.com/btx0424/OmniDrones) identifies Isaac Sim 4.1.0 and Python 3.10 as its current baseline and states that maintaining this version has become difficult. Its [installation guide](https://omnidrones.readthedocs.io/en/latest/installation.html) describes a matching older learning-library stack. Therefore an arbitrary current Isaac Sim installation is not a validated dependency upgrade.

The [Isaac Lab v1.1.0 README](https://github.com/isaac-sim/IsaacLab/blob/v1.1.0/README.md) identifies Isaac Sim 4.1 and Python 3.10. It is a candidate matching release to inspect, not a proven full ALiGn dependency lock. Upstream OmniDrones instructions that clone an unpinned Isaac Lab branch should not be copied into reproducible setup instructions.

The [Isaac Sim 4.5 migration](https://github.com/btx0424/OmniDrones/pull/94) and [5.1 migration](https://github.com/btx0424/OmniDrones/pull/106) are open pull requests at review time. In the latter, the author describes changes to rotor batching and force application as well as imports, and the discussion includes unresolved user failures. Those changes require dynamics checks; they are not proof of a validated drop-in upgrade. NVIDIA's [5.1 requirements page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) also labels that release unsupported now.

## Proposed first runtime candidate

Prefer evaluating the documented upstream baseline first to establish a reference for drone behavior:

| Component | Candidate, not yet locked |
|---|---|
| OmniDrones | Exact upstream main revision to select and record before installation |
| Isaac Sim | 4.1.0, subject to obtaining the runtime and successful lab execution |
| Python | 3.10 for this candidate |
| Isaac Lab | A matching historical release such as v1.1.0, with its exact revision checked |
| Learning libraries | Inspect the documented PyTorch 2.2.2/cu118, TorchRL 0.3.1, TensorDict 0.3.2 combination and resolve a complete compatible lock |
| Execution | One GPU and a minimal headless scene; controller/physics checks before learning |

This is a compatibility investigation target, not an installation prescription or a commitment that an older runtime is the best long-term platform. If the baseline cannot be obtained or validated, evaluate a maintained-runtime migration explicitly and account for the code/dynamics differences.

The current ALiGn package requires Python >=3.12 and uses datetime.UTC, introduced after Python 3.10. A move to 3.10 must include a deliberate code/dependency compatibility change, matching interpreter/configuration pins, and tests; editing .python-version alone would be insufficient. No interpreter or dependency change is made during this report review.

## Installation isolation

Update reviewed on 2026-09-16: the user has successfully queried host GPU 0 from an Ubuntu 22.04 container using sudo Docker and the NVIDIA runtime. Docker/toolkit package versions remain unrecorded. Isaac Sim is not installed yet. Proceed to acquire and record the candidate simulator image, then validate a minimal runtime launch. See the [setup procedure and recorded result](../docs/03-gpu-container-setup.md).

If available, a versioned container is a useful candidate for keeping older native simulator libraries separate from the host. NVIDIA documented an Isaac Sim 4.1.0 container workflow for headless servers in its [container guide](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/install_container.html). The historical image/tag's current availability and the host-driver combination still need verification; a container shares the host kernel and driver and does not automatically solve compatibility.

uv remains the Python project manager within the chosen runtime arrangement. Keep its environment isolated from simulator-managed packages and record how native paths are supplied. Do not downgrade the shared host driver or alter system/container services based on a version number alone. Existing working runtime installations should be inspected before adding another large installation.

## Required observable outcomes

- Identify an available GPU and record device selection; do not occupy all five by default.
- Resolve and record the runtime build/image digest, source revisions, Python, and dependency set.
- Launch and close a minimal simulator application successfully, retaining logs and version information.
- Instantiate a drone, advance physics, inspect finite states, reset, and close cleanly.
- Apply a known controller command and verify axis/sign, timestep, expected movement, and limits.
- Capture a short trajectory and, where supported, a rendered clip; measure memory and startup/step times.
- Document the executed commands and explain simulator, environment, controller, and policy as separate concepts.

Multi-GPU learning, large parallel batches, and final training experiments follow measured single-GPU behavior. Independent seeds can later use separate devices if allocation permits, while one training job's data-parallel execution requires its own implementation and validation.

## Candidate image and startup validation

The lab user supplied the Isaac Sim 4.1.0 digest sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173. Acquisition is confirmed; simulator execution and OmniDrones compatibility remain unverified. The startup launcher and artifact contract are described in [the smoke-test guide](../docs/04-isaac-sim-smoke.md).

## Observed shutdown failure

The supplied run 20260915T192028.862699Z-6a871ddd reached app ready with an active A4000 Vulkan device, then crashed in SimulationApp.close during the explicitly selected full-cleanup mode. It exited 1 after approximately 57.5 seconds; no final probe result was available. The revised fast-shutdown probe saves pre-close evidence and corrects the Kit log mount. A successful retry and later drone/OmniDrones validation remain outstanding. Original run artifacts are preserved unchanged under runs/isaac-sim-smoke/.

## Successful fast-mode retry

Run 20260916T010519.212064IST-5253d3c9 passed with exit 0 in 57.1569 seconds. It reported Python 3.10.14, PyTorch 2.2.2+cu118, CUDA build 11.8, one A4000, CUDA sum 1024.0, and 20 application updates. The pre-close record and probe hash match the supplied console and current script. This validates the isolated runtime boundary, not the full OmniDrones dependency set. See [lab handoff](../docs/06-lab-handoff.md) for next acceptance criteria.

## Pinned integration and current runtime record

The [runtime guide](../docs/07-omnidrones-runtime.md) records the selected OmniDrones commit, exact Docker/toolkit versions, dependency ownership, and completed image build. Isaac Lab v1.1.0 was inspected but is not needed by the basic drone/controller import path, so it remains uninstalled. The [control guide](../docs/08-single-drone-control.md) records actual calibration attempts, including the initial unresolved visual material failure. It supersedes the earlier candidate-only discussion without deleting it.
