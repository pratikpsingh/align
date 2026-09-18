# Planning one in-flight shape command

The current OmniDrones task selects a template at reset. That does not satisfy C11's commanded in-flight shape change. A first prerequisite is an identity-preserving destination assignment: when an airborne plane becomes a pyramid, each drone needs a specific new slot. The ordinary ground-to-template assignments are unsafe to reuse directly. In the four-drone example, two ideal straight paths cross exactly halfway, giving 0 m planned separation.

`align.formations.transition` searches all 24 destination-slot permutations for four agents and chooses the shortest total-travel mapping whose **synchronized straight-line interpolation** stays above a declared clearance margin. Pairwise clearance is computed analytically: each relative path is linear in interpolation fraction, so its squared distance is a quadratic whose minimum is checked in `[0,1]`. This is an exact statement about the planned geometry, not the physical trajectories. It never renumbers drones. Exhaustive search is capped at eight agents; larger groups require a different scalable planner and are rejected rather than using an unchecked fallback.

## Reproduce the CPU plan

From `align/`:

```sh
uv run --locked python scripts/report_shape_transition.py \
  --construction-config configs/feasible-plane-construction.json \
  --task-config configs/shape-transition-task.json \
  --transition-config configs/shape-transition-plane-pyramid.json
```

The command saves `runs/shape-transition-plan/<IST-run-id>/report.json` and `planned-positions.csv`. It uses no Docker, GPU, PyTorch, or simulator. The report records full input configurations and hashes, agent IDs, source positions, destination slots, chosen slot mapping, minimum analytic separation, worst pair/interpolation fraction, total and longest agent travel, search counts, command step, and available time. The CSV samples the ideal path at 101 interpolation fractions for inspection; the analytic minimum is more precise than the samples. Runs are ignored by Git and need separate backup.

The selected command is plane → pyramid at episode step 2,600 (26 s). A separate `shape-transition-task.json` allows 4,800 steps so the plane can first form and the final 200-step success dwell can occur after the command. The construction's formation phase begins at step 1,550. The extra time is a declared new scenario, not a modification to the accepted 2,350-step learning runs.

## Accepted geometry result

CPU run `20260918T183317.950631IST-60d31af2` passed. It searched 24 assignments, found 8 with at least 0.55 m ideal-path clearance, and selected slot order `[1, 2, 3, 0]`. Its minimum separation is 0.9129 m at interpolation fraction 1/3; the longest agent path is 0.9186 m. At the configured 0.5 m/s command limit, even an ideal controller needs at least 1.84 s to cover that path. The episode has 22 s after the shape command, including the final 2 s dwell. The original ground-based destination mapping had 0 m ideal-path clearance.

## Physical reference check and one-variable retry

A dedicated reference scenario now applies the planned destination targets at step 2,600 to all four drones in one flight. It changes both actor-visible relative targets and reward targets, resets target-progress and smoothing memory across the command, clears the success dwell, and suppresses success on the initial plane. The proportional reference controller uses privileged exact targets and poses. No learned checkpoint is loaded. `shape_command_issued` events and per-drone target/action/state rows expose the actual switch. A host audit requires the **first episode** of each environment to switch at the declared step and checks the exact destination per drone; it reports first/last assigned error, minimum physical separation, success, timeout, and safety termination. A passing process is not automatically a successful mission.

The first physical run `20260918T184629.244290IST-ed69806b` passed the command and telemetry audit: four of four worlds switched at step 2,600 and the minimum observed post-command separation was 0.916 m. It did **not** complete the transition: 0/4 successes, four timeouts, and 0 safety terminations. The assigned error fell from 0.530 m just after the command to about 0.132 m by step 2,800, then rose to about 1.031 m in the final 100 steps. Three drones assigned to lower pyramid slots descended to ground level: in environment 0, drone 0 crossed its 1.25 m target near step 2,800 at 0.264 m/s downward, was at 0.694 m by step 2,900, and settled at z≈0.06 m with nonzero ground-contact force despite a +0.5 m/s upward command. The current airborne-contact trigger applies only above the configured airborne-height threshold, while crash-height is −0.05 m, so this ground contact did not produce a safety termination. Zero safety terminations does not establish a safe landing or successful mission; the controller/termination contract needs further investigation. Keep the raw failed-mission run as evidence.

A one-variable retry moves only the pyramid center from z=1.5 m to z=1.75 m. This keeps the three base targets at their starting z=1.5 m while retaining the same source shape, command step, controller, task duration, and 0.55 m planning margin. CPU plan `20260918T185525.341864IST-2fb0ce52` passed: mapping `[1, 2, 3, 0]`, 0.913 m nominal minimum clearance, and 1.132 m longest path (2.26 s ideal speed-limited lower bound). The naive mapping still crosses. This is a controller-feasibility probe, not a claim that the transition will succeed.

The raised-center check used `configs/shape-transition-plane-pyramid-raised.json`. For the next wide-footprint check, rebuild the pinned runtime with the new simulator code and run from `align/`:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<BUILD-ID> && \
uv run --locked python scripts/run_shape_transition.py \
  --build-report runs/runtime-build/<BUILD-ID>/report.json \
  --transition-config configs/shape-transition-plane-pyramid-wide.json \
  --accept-eula --gpu 0
```

The launcher checks GPU availability and writes `runs/shape-transition-reference/<IST-run-id>/` with resolved configuration, hashes, native simulator logs, raw evaluation and per-drone telemetry CSVs, event stream, simulator metrics, and the independent host audit. A failed or interrupted run remains evidence. The prepared context is single-use; do not rebuild it after it is built. Both the original and raised-center CUDA runs are validated as **failed missions**. The raised-center run `20260918T185822.625381IST-b98990a8` switched all four worlds but had 0/4 successes and four timeouts. Its final-100-step assigned error was 1.247 m, versus 1.031 m in the original run; minimum post-command separation was 0.897 m. The lower-slot drones again descended to ground even though their assigned altitude remained 1.5 m. This rejects the simple explanation that a lower destination altitude alone caused the fall.

The pinned OmniDrones `MultirotorBase.apply_action` adds a downwash force between drones. Its model amplifies downward coupling when an upper drone moves over a lower one. For the recorded raised-center flight in environment 0, a vertical-thrust approximation of the model's coupling factor for drone 0 rose from about 0.001 at step 2,750 to 0.049 at 2,800, 0.166 at 2,850, and 0.274 at 2,900 as the apex rose and the base drone fell. This is an **inference from positions and source code**, not a measured force; rotor thrust direction and magnitude vary. The wider-footprint test below addresses that hypothesis. The geometric plan remains only a nominal clearance check because real drones can move asynchronously, overshoot, or drift. Learned-policy shape changes, C04 travel, and C09 state-based waypoints remain separate work.

## Inspect the downwash hypothesis

`align.formations.downwash` reproduces the **dimensionless geometric factor** in the pinned, MIT-licensed OmniDrones `MultirotorBase.downwash` model with `kr=2`, `kz=0.3`, and an assumed vertical upper thrust. It does not measure aerodynamic force, motor thrust, or power. Both previous physical runs have a separate immutable CPU audit with all 76,800 first-episode drone rows and 19,200 snapshots: original `20260918T190948.000059IST-92eb91aa`, raised-center `20260918T190948.055930IST-859546eb`. The maximum geometric factors were about 0.310 in both. Recompute from raw trajectories with:

```sh
uv run --locked python scripts/analyze_shape_downwash.py \
  --source-run runs/shape-transition-reference/<RUN-ID>
```

The new `configs/shape-transition-plane-pyramid-wide.json` keeps the raised center and doubles only the destination's horizontal offsets. CPU plan `20260918T190828.629829IST-e5ac59e4` passes the 0.55 m clearance test and predicts an endpoint vertical-thrust coupling factor 0.0108 versus 0.2177 for the narrow pyramids. Physical reference `20260918T191219.165303IST-01ae5fca` **completed all four worlds**: each switched at step 2,600 and succeeded at step 3,151, 5.51 s after the command, with no timeout or safety termination. The last-100-step assigned RMSE was 0.0448 m and the minimum observed post-command separation was 0.9998 m. Raw telemetry shows minimum altitude 1.434 m and zero contact force after the command. Its separate downwash audit `20260918T191508.621068IST-186469af` found maximum geometric exposure 0.0171, versus about 0.310 in each failed narrow-pyramid flight. These four near-identical worlds demonstrate a deterministic reference behavior, not success across independent seeds or a learned policy; the exposure calculation is not a force measurement.

## Check sustained holding after the command

The first wide flight stopped when the destination shape met a 200-step (2 s) dwell condition. It therefore cannot show that the drones would hold the pyramid for the remaining episode. `configs/shape-transition-hold-task.json` changes only `success_dwell_steps` from 200 to 1,000 (10 s). CPU plan `20260918T191636.892505IST-40123b0d` passes. This is a separate, declared maintenance check; a timeout will remain useful evidence of drift. With the already built wide-transition image, run from `align/`:

```sh
sudo -v
uv run --locked python scripts/run_shape_transition.py \
  --build-report runs/runtime-build/20260918T191121.663254IST-c4ea864e/report.json \
  --transition-config configs/shape-transition-plane-pyramid-wide.json \
  --task-config configs/shape-transition-hold-task.json \
  --accept-eula --gpu 0
```

The host audit again records first-episode outcomes and keeps every step's positions, velocities, targets, and contact forces. Long-hold run `20260918T192402.753294IST-f7249d4c` passed the instrumentation audit but **timed out in all four worlds** under the 1,000-step dwell rule. There was no post-command contact; observed altitude stayed at least 1.352 m and minimum separation was 0.9998 m. The base drones slowly settled to about 1.353 m against 1.5 m targets. Their final-100-step assigned RMSE was 0.1275 m, above the 0.1 m success tolerance. Source-model audit `20260918T192753.821992IST-1420a27f` found a maximum geometric exposure of 0.0266. This is steady holding bias rather than a collision or a failed shape command.

`configs/shape-transition-gain-1p2-construction.json` changes only the proportional target-position-to-velocity gain from 0.8 to 1.2 s⁻¹. The wide shape, 10-second dwell, speed limit, and all safety settings remain fixed. CPU plan `20260918T192853.310200IST-4c2dadb4` passes. The reference controller's approximate steady velocity demand was 0.8×0.147≈0.118 m/s upward; at gain 1.2, the same demand corresponds to about 0.098 m position error. That calculation motivates a test but does not predict the simulator outcome. Reuse the built wide image to run this one-variable probe:

```sh
uv run --locked python scripts/run_shape_transition.py \
  --build-report runs/runtime-build/20260918T191121.663254IST-c4ea864e/report.json \
  --transition-config configs/shape-transition-plane-pyramid-wide.json \
  --task-config configs/shape-transition-hold-task.json \
  --construction-config configs/shape-transition-gain-1p2-construction.json \
  --accept-eula --gpu 0
```

The gain-probe physical outcome is pending.

The separate [flight-contact guide](39-flight-contact-termination.md) documents the correction to the earlier task's ground-contact blind spot. Its simulator result is pending and requires a new image; the wide-reference and 10-second hold checks above use the previously built task semantics.
