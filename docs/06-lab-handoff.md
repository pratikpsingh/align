# Continuing ALiGn on the lab machine

Snapshot reviewed on 2026-09-16. This document combines the verified runtime boundary and a ready-to-paste agent prompt. Read newer run evidence and code before treating any snapshot as current. No new simulator execution was performed during this handoff.

## Newer integration work

The snapshot below is retained as history. See [OmniDrones runtime](07-omnidrones-runtime.md) and [single-drone control](08-single-drone-control.md) for current source pins, Python compatibility changes, build instructions, and lab attempts.

## Confirmed result

Run 20260916T010519.212064IST-5253d3c9 passed the basic Isaac Sim check on host GPU 0. The report and console progress agree, and their probe hash matches the current local script. CUDA arithmetic returned 1024.0, exactly one A4000 was visible, and 20 application updates completed. Exit status was 0. Total time was 57.1569 seconds, including about 53.0836 seconds in application startup; this is not training throughput.

| Component | Observed value |
|---|---|
| Container | Isaac Sim 4.1.0, pinned digest below |
| Runtime Python | 3.10.14 |
| Runtime PyTorch | 2.2.2+cu118 |
| PyTorch CUDA build | 11.8 |
| Shutdown mode | fast; return from close not observed |
| ALiGn time display | IST, +05:30; raw vendor log remains UTC |
| Unverified | Drone physics, OmniDrones dependencies, controller/reset, cameras, training, full cleanup |

A null probe_result is expected because fast shutdown exits before the post-close marker. The completed pre-close record plus zero container exit satisfy the documented pass contract. This does not establish that the earlier full-cleanup crash was repaired. Preserve that failed attempt too.

Display/GLFW and other warnings remained in the passing log. They are non-blocking for this particular test, not a guarantee about later features. Do not upgrade drivers or disable IOMMU solely because these warnings appear.

## Transfer checklist

Sync the new handoff and learning notes along with all current source, scripts, tests, pyproject.toml, uv.lock, .python-version, and AGENTS.md. Verify docs/, plans/, and learning/ exist on the destination; they are currently tracked in this checkout despite older notes saying otherwise. Do not copy the laptop .venv to the lab: rebuild the host environment with uv sync --locked. Generated runs/ is ignored and needs separate backup/transfer if evidence is not already on the lab.

The successful lab directory already exists at the path printed by the user's launcher. The reviewed local copy is under runs/isaac-sim-smoke/20260916T010519.212064IST-5253d3c9/, including an added assessment.json with file hashes. Do not overwrite either machine's existing run artifacts. Kit logs were not supplied for this review; retain them on the lab.

Open the JRF workspace root in Cursor so the agent can read align and the sibling reference repositories. No change to the host timezone, driver, or installed Docker stack is needed for this handoff.

## Continuation prompt

Copy the following text into the agent on the lab machine:

```text
Continue the ALiGn research project on this lab machine. Work in align/ inside the JRF workspace. The sibling folders are DMPC-to-MARL-Sim/, multi-UAV-formation/, my-mappo/, papers/, and quad-swarm-rl/. Preserve them as references; do not modify them implicitly. Do not assume that the additional reference repositories have already been audited. Discover the actual root path rather than hard-coding the previous laptop username.

Read align/AGENTS.md first, then align/docs/06-lab-handoff.md, align/plans/00-README.md, align/plans/01-scope-and-claims.md, align/plans/02-implementation-roadmap.md, align/plans/06-lab-runtime-assessment.md, and the relevant indexed learning/setup documents. Inspect actual code, Git status, and saved evidence before making changes. This handoff is a dated snapshot, not authority to override newer measurements.

Purpose and research scope:
We are implementing a maintainable OmniDrones version of the intended system in papers/2026_Distributed_MARL_Submission.pdf, informed by the student's PyBullet implementation in my-mappo/. Implement retained claims that the student did not finish, and correct identified mathematical/engineering defects. The detailed C01–C12 register is the source of acceptance criteria; implementation does not guarantee reproducing the paper's numbers. Paper-04 is an unpublished antecedent. Paper-03 comparison is deferred. Verify scientific interpretations against the actual PDFs and source when implementing the relevant component.

The core includes shared recurrent MAPPO with decentralized local actor inputs and a centralized critic, meaningful formation reward, ground-to-formation takeoff, cube/sphere/pyramid/plane target conditioning, maintenance during travel, bounded neighbor selection, a communication-cost/neighbor-budget study, inter-UAV safety and smoothing, waypoint missions, larger-swarm evaluation, commanded in-flight shape transitions, and measured policy-export feasibility. Keep one group's shape transitions in scope; multiple independently moving groups and reunification are a later extension.

Known issues in the old checkout that require deliberate correction include: formation weight zero in the default training path; recurrent samples treated as length-one sequences instead of genuine temporal unrolling; minimum-neighbor filling beyond the radius; undeclared absolute-position inputs relative to locality claims; reset-time shape sampling mistaken for in-flight changes; formation sum-versus-mean inconsistency; bounded Gaussian means mistaken for bounded sampled actions; and incomplete ground-start/waypoint pipelines. The draft's 3N-6 edge count is not a sufficient 3-D rigidity guarantee or proof of an optimal neighbor count. Observed neighbors are not equivalent to radio transmissions. Audit details in the claim register before adapting code.

Later extensions are static/dynamic obstacles, GRU versus LSTM/MLP comparisons, adaptive communication and delay/dropout studies, group split/merge, systematic reward improvements, and eventually paper-03 under a fair shared protocol. Design extension points now, but do not attempt the entire roadmap in one change.

Teaching and engineering requirements:
The user is new to OmniDrones and reinforcement learning. Before each bounded increment explain what we are building, why it is needed, and the observable outcome. Afterward explain the implementation and evidence in a topic-based learning document. Use two-digit reading-order prefixes and 00-README.md indexes in docs/, learning/, and plans/; do not label documents by implementation stage. Update links when names change. Operational instructions must be usable by another researcher. The current .gitignore excludes runs/ and .venv, but not plans/ or learning/; check actual tracking instead of following obsolete statements that those notes are ignored.

Use uv, a src/align package, thin scripts, explicit configuration, simulator-independent math/policy/metrics code, and meaningful tests. No hidden sys.path hacks, silent fallbacks, or fabricated simulator success. Record adaptations in observations/controllers/rewards as scientific changes. Use IST for ALiGn human-readable times and new run names, retaining offset-aware UTC machine fields. Native simulator logs may still use UTC; preserve their contents.

Current implementation:
The host package requires Python >=3.12, has no runtime dependencies, and uses uv_build and Ruff. It provides align doctor, machine inventory, unique run directories, atomic JSON reports, human logs, JSONL events, and CPU tests. scripts/run_isaac_sim_smoke.py runs a pinned Docker image; scripts/isaac_sim_smoke.py is an isolated Python 3.10-compatible vendor-runtime probe. The last local validation passed 31 CPU tests plus lint/format checks. This does not establish GPU physics behavior.

No ALiGn OmniDrones task, policy learner, training command, checkpoint recovery, drone video pipeline, or research result exists yet. The old reference code is not an implemented ALiGn feature.

Lab evidence:
Earlier inventory reported five RTX A4000 GPUs, each 16,376 MiB, driver 580.173.02, AMD EPYC 9654, 192 logical CPUs, about 503 GiB host RAM, Ubuntu 24.04.5, and substantial free disk. These are inventory values, not guaranteed current allocations. GPU memory is separate per device. Start on allocated GPU 0; additional GPUs can later run independent seeds/budgets. Distributed training is not automatic. Exact Docker/toolkit versions still need recording.

Docker GPU exposure and a real Isaac Sim smoke test now passed. Use this image identity:
nvcr.io/nvidia/isaac-sim@sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173
It is the inspected Isaac Sim 4.1.0 image. Do not replace it with latest without an evidence-based migration decision.

Successful run: 20260916T010519.212064IST-5253d3c9.
Start: 2026-09-16T01:05:19.212151+05:30.
Finish: 2026-09-16T01:06:16.389832+05:30.
Container exit 0; duration 57.156927968986565 seconds; application startup about 53.0836 seconds.
Bundled Python 3.10.14; PyTorch 2.2.2+cu118; torch.version.cuda 11.8; one visible NVIDIA RTX A4000; CUDA sum 1024.0; 20 application updates; shutdown_mode fast; drone_physics_tested false.
Probe SHA256: 2c96cc4b6f44cda305a6b6f71594494401745b6f474ffe0eeaff71234315bca4.

Read the run's report.json and console.log under align/runs/isaac-sim-smoke/<run-id>/; supplied copies are also align/report.json and align/console.log, but those top-level copies may be replaced in future. The local reviewer archived them and an assessment.json without altering raw content. Check matching hashes when transferring evidence. Historical runs may need copying separately because runs/ is ignored.

Interpret shutdown correctly: probe_result=null and shutdown_return_observed=false are expected for this successful fast-exit path. The flushed last_probe_progress reports passed checks before close; Docker exit 0 completes the launcher's pass contract. This does not validate full extension cleanup. The preceding run 20260915T192028.862699Z-6a871ddd crashed inside SimulationApp.close with fast_shutdown=False and exit 1. Do not reintroduce that mode as the default or describe the crash as repaired. Save artifacts/checkpoints before fast shutdown, because code after close may never execute.

The successful log still contains display/GLFW, no-windowing, IOMMU, missing optional cache, and possible fabric-version warnings. They did not prevent this smoke test, but that does not prove they are harmless for physics/rendering. Preserve and investigate if a later check fails. Do not change shared host drivers, BIOS/IOMMU, or Docker services merely to clear warnings. The log's UTC times persist despite TZ=Asia/Kolkata; ALiGn's IST reporting is working.

Immediate next task — reproducible OmniDrones runtime and one controlled drone:
1. Verify workspace/source state and existing evidence. Run the existing CPU checks. Inspect current GPU allocation and Docker/toolkit versions. Reuse the successful smoke test as evidence; rerun it if the image, source, GPU/driver, or setup differs or the artifacts are missing, not simply as ritual.
2. Inspect upstream OmniDrones installation metadata, APIs, licenses, and assets; select and record an exact Git revision compatible with the working Isaac Sim 4.1 runtime. Previous investigation identified Isaac Lab v1.1.0, TorchRL 0.3.1, and TensorDict 0.3.2 as candidates only. They are not installed/validated facts. Recheck these against the selected source; do not follow unpinned main branches or overwrite the working vendor PyTorch stack accidentally.
3. Resolve the host Python 3.12 versus simulator Python 3.10 boundary explicitly. Keep host orchestration separate or deliberately migrate the shared package after inspecting all syntax/dependencies. datetime.UTC is unavailable in 3.10. Updating only .python-version is insufficient. Choose and document a uv-managed, reproducible environment/derived-image arrangement, preserving the base image; record exactly which dependencies the vendor runtime supplies and which uv locks. Do not claim the current uv.lock reproduces simulator dependencies.
4. Implement a minimal OmniDrones integration with a documented launch command. Start with one drone and one environment. Define units, world/body frames, quaternion ordering, observation/action shapes, physics/control timesteps, controller parameters and saturation, and reset behavior. Initially aim for a calibrated velocity-level controller matching the student's intent; justify any unavoidable action-interface change. Do not silently adopt paper-03 CTBR controls.
5. Demonstrate actual reset, finite observations, correct-axis response to deterministic commands, and a small controlled hover/position or velocity test. Choose documented tolerances after basic calibration; save raw time-series data and plots. Record model/assets, seed, controller, timestep, versions, exact command, elapsed duration, and failed outcomes. Test repeated reset/controller state. Then try a small batch only after single-drone behavior is understood. Application updates alone do not satisfy this requirement.
6. If compatible rendering is straightforward, save a short video; otherwise preserve trajectory data and state that rendering remains unvalidated. Video is useful evidence but must not block the basic dynamics diagnosis. Do not train MAPPO yet or substitute fake dynamics for a real simulator check.
7. Deliver code, CPU tests, actual lab integration results, public setup instructions, and a numbered topic learning document. Summarize what works, what remains uncertain, and the next bounded task. Stop this increment once the documented one-drone acceptance is met; do not expand automatically into the whole research system.

Subsequent roadmap:
Define formation geometry/assignment and invariant metrics, build safe task/reset/reward contracts, then implement correct recurrent MAPPO. Develop recoverable training before expensive runs: save policy, critic, optimizers, normalization, RNG states, counters, configuration and source/runtime identity; validate partial writes, fallback checkpoints, and resume lineage. State whether simulator episodes restart or trajectory state is restored. Keep evaluation and rendering usable after training.

For experiments retain raw metrics and plots, training/evaluation seeds, successes and failures, formation error, collision/separation, smoothness, goal/waypoint/transition times, neighbor topology and modeled traffic, wall-clock/throughput, resource peaks, checkpoint/export sizes, parameter count, and inference latency. Source-file size can be recorded but does not establish deployment feasibility. Compare matched protocols and uncertainty across seeds. Never claim novelty, superiority, or paper-number reproduction from a working implementation alone.

Start by summarizing the confirmed runtime boundary and proposing the concrete one-drone acceptance check in plain language, then proceed with the bounded runtime/integration work. Ask only for genuinely missing access/allocation or consequential unresolved choices. Do not install/reconfigure shared system services or run long training jobs as part of this task.
```
