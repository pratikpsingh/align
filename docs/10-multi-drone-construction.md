# Multi-drone ground-to-formation construction

## Purpose and evidence status

This probe connects the tested formation geometry to four real OmniDrones Hummingbirds in one Isaac Sim environment. It checks group identity, ground reset, staged takeoff, fixed slot assignment, formation tracking, physical contacts, inter-UAV separation, dwell-based completion, and repeated reset without reinforcement learning.

The source, host launcher, offline evaluator, plotting, configuration, and CPU tests are implemented. The first lab calibration has been inspected and the schema-2 tolerances are frozen with its run ID. A clean acceptance rerun is still required. Rebuild the derived image before running because it contains the ALiGn wheel present at build time.

## Group and tensor contract

There is one group (`group_id = 0`), one environment, and four stable agent identities `0..3`. The world frame, body frame, quaternion ordering, Hummingbird model, Lee controller, 100 Hz physics/control rate, and rotor saturation match the accepted [single-drone contract](08-single-drone-control.md).

| Quantity | Shape | Meaning |
|---|---:|---|
| State instrumentation | `(1,4,23)` | Full upstream per-drone state used by the deterministic controller and logger |
| High-level action | `(1,4,4)` | Bounded `[direction_x,direction_y,direction_z,speed_scale]` |
| Target velocity | `(1,4,3)` | World-frame metres/second |
| Rotor action | `(1,4,4)` | Raw and clipped Lee-controller commands |
| Contact force | `(1,4,3)` | Net force on each base link |

The full state remains instrumentation, not the future decentralized actor observation.

## Construction sequence

The checked [configuration](../configs/multi-drone-construction.json) defines:

1. **Ground settle, 0.5 s.** Four drones start in a centred line, one metre apart, with base centers at Z = 0.06 m. Rotor throttle is explicitly reset to zero and idle commands allow the bodies to establish ground contact.
2. **Vertical takeoff, at most 5 s.** Each drone retains its ground X/Y coordinate and targets Z = 1.5 m. This separates vertical lift from horizontal rearrangement.
3. **Plane construction, at most 8 s.** A Hungarian assignment computed once from ground positions maps identities to the one-metre-spaced plane slots. Assignment remains fixed.
4. **Success dwell, 2 continuous seconds.** Assigned RMSE must be at most 0.10 m, pairwise RMSE at most 0.08 m, maximum speed at most 0.08 m/s, separation at least 0.55 m, and no airborne contact.

An outer proportional controller converts position error to a world velocity and clips the **vector norm** to 0.5 m/s. That velocity is encoded with the same four-value direction/speed interface validated in the one-drone suite. The Lee controller still produces rotor commands at every physics step.

## Safety and termination

Ground contact is expected and must be observed. Once a drone's center is above 0.25 m, a net contact force above 0.01 N is an `airborne_contact` failure. This detects any physical contact on the tracked base link; it does not identify the other collider.

Inter-UAV safety is measured separately from contacts using the minimum pair distance. The probe terminates an episode with one primary reason:

- `success`: all dwell conditions remained true continuously;
- `separation_violation`: a pair moved closer than 0.55 m;
- `airborne_contact`: an airborne base-link contact exceeded the threshold;
- `safety_envelope`: a drone left the configured XYZ envelope;
- `nonfinite`: state, action, or contact data became nonfinite;
- `timeout`: success was not achieved by the maximum step.

Two repetitions reuse the existing drone instances. Reset explicitly restores poses, velocities, rotor throttle, rotor joints, forces, and deterministic seeds. Schema 2 reports repeat differences by unit and signal: position, linear velocity, attitude quaternion component, ground and airborne angular rate, raw/applied actuator command, and rotor state. This avoids treating metres, radians/second, and normalized actuator values as one quantity.

The frozen repeat thresholds are 0.0002 m position, 0.002 m/s linear velocity, 0.0001 quaternion component, 0.02 rad/s ground angular rate, 0.01 rad/s airborne angular rate, 0.002 actuator command, and 0.001 rotor state.

## Build and run

From `align/`, first verify that the selected GPU is allocated and idle. Build a new image containing this source, then run:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py
uv run --locked python scripts/run_multi_drone.py --accept-eula --gpu 0
```

If the source context was prepared by an agent without Docker access, use its printed directory:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<prepared-id>
uv run --locked python scripts/run_multi_drone.py --accept-eula --gpu 0
```

The launcher refuses a selected GPU with an active compute process or material utilization. It mounts only the unique output and Kit-log directories, disables container networking, launches the exact built image ID, and uses fast shutdown after flushing artifacts.

## Artifacts and offline recomputation

Each `runs/multi-drone/<id>/` contains:

- `trajectory.csv`: every post-step state, fixed target, target velocity, high-level action, raw/applied rotor command, throttle, and contact magnitude for every agent;
- `resets.json` and `episodes.json`: reset state and explicit termination records;
- `metrics.json`: all acceptance checks and numerical summaries;
- `trajectory.png` and `trajectory.pdf`: XY paths, altitudes, formation error, separation, and contact force;
- `runtime.json`, localized assets, `scene.usda`, exact build/context records, structured events, console output, and native Kit logs;
- `probe-result.json` and host `report.json`: flushed simulator outcome, host GPU inventory, container exit, elapsed time, and independent host recomputation status.

Recompute without Isaac Sim:

```sh
uv run --locked python scripts/report_multi_drone.py runs/multi-drone/<id>
uv run --locked --extra reporting python scripts/report_multi_drone.py \
  runs/multi-drone/<id> --plots
```

The host launcher compares freshly recomputed metrics with the container record before reporting a pass. `runs/` is ignored and needs external backup.

## Calibration result

Run `20260916T115816.925151IST-deb9d6bb` created four drones, observed ground contact, and completed both repetitions with `success` at 11.05 simulated seconds. It saved 8,840 agent samples. Minimum separation was 0.782145 m, maximum ground contact was 0.072271 N, no airborne contact was measured, assigned RMSE during the final dwell was at most 0.041980 m, pairwise RMSE at most 0.040747 m, and speed at most 0.079619 m/s.

The original run remains **failed** because its schema-1 evaluator compared Python targets with float32 CSV values at an unjustified `1e-9` tolerance; the measured difference was `1.34e-9 m`. The same issue affected action/velocity comparison by at most `3.00e-8 m/s`. Schema 2 uses a `1e-6` serialization-only tolerance and has a regression test that round-trips through float32.

The old mixed-unit reset comparison also reported `0.009132`, localized to ground-contact angular rate. Unit-specific maxima were 0.0000311 m position, 0.000951 m/s linear velocity, 0.0000213 quaternion component, 0.009132 rad/s ground angular rate, 0.003433 rad/s airborne angular rate, 0.000541 actuator command, and 0.000138 rotor state. These measurements set the frozen thresholds above.

Offline derived report `runs/multi-drone-reports/20260916T120616.341947IST-09b7f54d/` passes all 30 corrected checks and contains both plots. It does not change the original run status. The next simulator run must use schema 2 and pass inside the container, match host recomputation, save both plots, and exit zero.

## Current limitations

The probe has one group, one environment, four drones, and a plane target. It exercises the geometry and controller boundary but has no learned policy, local observation, reward, obstacle, waypoint, shape transition, or communication model. Contact attribution, aerodynamic fidelity, larger batches, and video rendering remain unvalidated. The frozen rerun is still needed before this becomes accepted physical evidence.
