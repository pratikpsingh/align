# Vectorized task environment

## Status and purpose

ALiGn now has a CPU reference episode contract and a simulator-native vector task for the pinned OmniDrones runtime. The task combines the validated velocity controller, bounded local actor observations, separate centralized critic state, component reward, phase schedule, safety outcomes, and reset memory behind step/reset operations.

The CPU contract and the one-world/four-world GPU acceptance test passed. The accepted run validates cloned physics, tensor parity, partial resets, terminal masks, contact reporting, and measured probe throughput. This task does not contain a policy, rollout collector, or MAPPO update.

## Task tensors

For the four-drone baseline with four cloned environments:

| Value | Shape | Meaning |
|---|---|---|
| actor observation | [4, 4, 55] | one bounded local observation per drone |
| critic state | [4, 80] | one padded global training state per environment |
| high-level action | [4, 4, 4] | world direction XYZ plus nonnegative speed fraction |
| reward | [4, 4, 1] | per-agent weighted component sum |
| done / terminated / truncated | [4, 1] each | episode boundary masks |

The actor never receives critic state. Actions are clipped to [-1, 1], direction is normalized, and the absolute fourth value scales speed up to 0.5 m/s. The Lee controller converts this world-velocity command to four bounded rotor commands. During the ground-settle phase the motors remain idle.

All physics state, observation construction, reward calculation, and episode buffers remain on the selected CUDA device during a step. Saved CSV rows are copied to the host only as probe evidence.

The scene uses Isaac Sim's local 20 m GroundPlane primitive at /World/ground. It does not use OmniDrones' default scene helper, because that helper references a remote grid USD that cannot be resolved in the intentionally offline runtime. The single-drone and four-drone controller probes already use the same local ground implementation. The ground implementation, path, size, cloning mode, and contact setup are recorded in scene-setup.json before cloning and copied into runtime.json after successful initialization. The host report records the actual --network=none container command.

## Episode phases and target timing

Each environment has its own integer step counter:

1. ground: use the assigned ground locations;
2. takeoff: keep each drone above its ground location at target altitude;
3. formation: use the fixed assigned plane slots.

An action at step k acts against the target for step k. Reward after physics uses that same target. The returned observation uses the target for step k + 1. At a phase boundary, progress and command-smoothness memory are cleared so two different targets are not compared as one continuous signal.

## Episode outcomes

terminated=True identifies a true task boundary:

- success after the configured consecutive formation dwell;
- separation below 0.25 m;
- airborne contact above the force threshold;
- leaving the configured XYZ safety envelope;
- a nonfinite state, contact, or reward.

The reward begins penalizing separation below 0.55 m, before the 0.25 m terminal collision threshold. This gives the policy a corrective signal before the episode is irrecoverable.

truncated=True identifies only the configured time limit. A true terminal event takes precedence if both occur on one step. done is their logical OR. This distinction is required for later GAE/value bootstrapping.

A completed environment must be reset before another step. Partial reset clears only that environment's physical state, step counter, success dwell, previous target distances, and previous commands.

## CPU checks

Run:

~~~sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check .
uv run --locked ruff format --check .
~~~

The vector task checks cover fixed and finite shapes, configuration compatibility, phase-boundary memory, independent partial reset, success dwell, failure termination, time-limit truncation, raw trajectory integrity, and launcher pass criteria. These are mathematical and orchestration checks; they do not prove cloned drone physics.

## Build and GPU acceptance command

Rebuild the derived image after these source changes, then run the two-scenario probe:

~~~sh
sudo -v

uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<prepared-id> &&

uv run --locked python scripts/run_vector_task.py \
  --build-report runs/runtime-build/<prepared-id>/report.json \
  --accept-eula \
  --gpu 0
~~~

Each prepared context can be built once. After a successful build, retry only the task command. Refresh terminal sudo authentication with sudo -v when needed. A busy-device failure occurs before scenario startup; inspect the saved GPU inventory and wait for the allocated device to become idle.

The launcher verifies that the chosen GPU is idle, then starts two fresh containers:

- single: one environment, finite tensors, CPU/GPU numerical parity, and a time-limit truncation;
- batch: four cloned environments, a reset of environment 0 while the others continue, a controlled safety-envelope termination and reset of environment 1, and independent time-limit truncation of the untouched environments.

Choose an allocated idle GPU; the five cards have separate memory. --gpu 0 is not an allocation mechanism. The probe uses headless rendering and does not validate video.

## Saved evidence

Each invocation creates runs/vector-task/<id>/ with the resolved four-part configuration, build/context identities, host/GPU inventories, exact commands, and overall report. The single/ and batch/ folders each contain:

- trajectory.csv: raw per-step, per-environment, per-agent state, target, action, contact, reward, and boundary masks;
- events.jsonl: reset, forced-condition, completion, and pre-close records;
- metrics.json: checks, CPU/GPU parity errors, throughput, and peak CUDA allocation;
- host-audit.json: independent structural audit of the raw CSV;
- probe-result.json and console.log;
- copied drone assets with SHA-256 hashes, the flattened scene, controller parameters, and Kit logs.

All result files are flushed before fast shutdown. runs/ is ignored and needs separate backup.

## Acceptance limits

This probe uses deterministic actions to validate the environment interface; it does not train or evaluate a policy. The short 160-step acceptance episodes cover ground and early takeoff, while formation completion remains established by the earlier full construction run and CPU success-dwell tests. Throughput from one and four environments is a baseline measurement, not a multi-GPU result or a training-speed estimate. Rendering remains unvalidated.

## Lab evidence and cloned-view initialization

Run 20260916T154136.608722IST-14b3d22a used build 20260916T153608.887695IST-65589901. Its single scenario passed all eight simulator checks and all nine host checks with container exit 0: 160 environment steps, 640 agent rows, zero initial reset-position error, actor parity error 5.96e-8, critic parity error 1.19e-8, and reward parity error 7.23e-14. It reached the time limit with truncation rather than true termination. The measured loop took 1.47755 seconds (433.15 agent steps/s), excluding startup. This is acceptance-probe throughput including logging.

The batch scenario failed during drone initialization before producing trajectories. At 15:43:43 IST the log first reports a cloned base-link collision shape being deleted while in use and invalidating the physics tensor view; subsequent root-transform reads fail. The successful single scenario is retained inside the overall failed run.

Run 20260916T155103.036175IST-561845e8 used build 20260916T154958.642167IST-6af1f589 and tested that first retry. Its single scenario again passed, now including the required ground-contact check: 160 environment steps, 640 agent rows, zero reset error, all nine simulator checks, and all nine host checks. The measured loop took 1.42654 seconds (448.64 agent steps/s). This confirms that pre-clone contact schemas and transform preservation work for one world.

The four-world scenario still failed during base-link view construction. It reached cloned_scene_ready, then deletion of /World/envs/env_1/Hummingbird_0/base_link/collisions invalidated the tensor view. The traceback shows that the articulation view had initialized and the next operation was construction of the base-link rigid view.

Run 20260916T160508.191769IST-d4a14633 tested preserved stabilization as well. Its single scenario passed all checks at 446.73 agent steps/s, but its batch failed at the same boundary. This rules out the stabilization option as the deletion trigger. The saved Isaac Sim 4.1 source identifies the actual behavior: RigidPrimView treats track_contact_forces=True as a request to prepare sensors even when prepare_contact_sensors=False. It authors a PhysX sleep-threshold attribute on every matched cloned body after physics startup.

The new retry applies both required schemas and their zero thresholds to the source base links before cloning. OmniDrones' base-link motion view initializes with contact tracking disabled. A separate RigidContactView then attaches with preparation and stabilization changes disabled, so it creates only the contact tensor view. Physics replication stays enabled. The acceptance checks require a nonzero ground contact signal in every environment. scene-setup.json and source_scene_ready, cloned_scene_ready, drone_views_ready, and contact_view_ready events distinguish future failure boundaries. CPU tests inspect the runtime call contract and combined upstream patches; they cannot establish native PhysX handle validity.

Run 20260916T161917.081493IST-0ad28819 reached drone_views_ready without collision-shape deletion, then stopped before physics because the 4.1 RigidContactView constructor requires an explicit filter_paths_expr argument. The corrected call supplies an empty list because ALiGn needs net contact forces against all bodies, not a filtered contact matrix. This interface correction was included in the accepted run below, which reached contact_view_ready and completed the physics checks.


## Accepted vector-task run

Run 20260916T162804.441981IST-29e96228, using build 20260916T162101.780288IST-29c715e1, passed both scenarios with container exit 0 and every simulator and host-audit check true.

The single scenario recorded 640 agent transitions in 1.42565 seconds (448.92 agent steps/s). Initial reset error was zero. Maximum CPU/GPU disagreement was 5.96e-8 for actor observations, 1.19e-8 for critic state, and 7.23e-14 for reward.

The batch recorded 2,560 agent transitions in 1.60156 seconds (1,598.44 agent steps/s). Resetting environment 0 at global step 40 produced progress [0, 40, 40, 40] with zero state change in unaffected environments. Environment 1 then received a controlled safety-envelope violation, terminated at step 71, and reset independently. Environments 2 and 3 reached step 160 as time-limit truncations. All four environments observed nonzero ground contact. Maximum batch parity disagreement was 1.99e-8 for actor observations, 1.19e-8 for critic state, and 1.26e-13 for reward.

These rates measure the short deterministic probe with raw trajectory logging; they are not training throughput or policy performance. Headless rendering and video remain unvalidated.
