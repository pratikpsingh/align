# Deterministic single-drone control

## Purpose and evidence status

The probe creates one real OmniDrones Hummingbird in one Isaac Sim environment and applies deterministic control. It tests the physical interface before RL adds another source of uncertainty. The acceptance configuration is frozen from calibration run `20260916T015409.084496IST-8edb58c1`. See the result record below; CPU fixtures are not simulated flight evidence.

## Scientific interpretation

The submission's Section III-D defines `[direction_x, direction_y, direction_z, speed_scale]`, interpreted as a velocity target through low-level PID. The local student implementation normalizes the direction and multiplies it by `SPEED_LIMIT * abs(speed_scale)` in `my-mappo/onpolicy/envs/gym_pybullet_drones/envs/BaseRLAviary.py`.

ALiGn retains that four-value direction/speed mapping and clips all four inputs to [-1,1] before conversion. Zero direction gives zero velocity. The maximum **vector norm**, not each separate velocity component, is 0.5 m/s in this probe. `[1,0,0,0.5]` therefore requests world-X velocity 0.25 m/s. This is a deterministic interface check, not a choice of future policy distribution or PPO log-probability convention.

The low-level adaptation is explicit: upstream Hummingbird plus `LeePositionController` replaces the student's Crazyflie/PyBullet PID. The supplied controller supports world velocity directly and has model-specific gains; the repository does not supply corresponding Lee gains for Crazyflie. This establishes a calibrated OmniDrones control baseline, not matching old drone dynamics or reproducing paper results. It does not adopt paper-03's thrust/body-rate policy interface.

## Contract

| Item | Meaning |
|---|---|
| World | Local right-handed Cartesian frame; Z upward; X/Y are arbitrary local axes, not declared geographic directions |
| Body | Right-handed frame; heading +X and thrust/up +Z; body-to-world attitude quaternion |
| Units | metres, seconds, metres/second, radians, radians/second, kilograms, newtons |
| Quaternion | `[w,x,y,z]`; identity `[1,0,0,0]`; the upstream view converts PhysX's internal ordering |
| Upstream state | `(1,1,23)`: position 3, quaternion 4, world linear/angular velocity 6, heading 3, up 3, normalized rotor state 4 |
| Controller observation | First 13 values, shape `(1,1,13)`; this is full-state instrumentation, not a future local actor observation |
| High-level action | Four direction/speed values; target world velocity shape `(1,1,3)` |
| Actuator action | Four normalized rotor thrust commands, `(1,1,4)`; explicit clipping to [-1,1] before application; raw and applied commands saved |
| Physics and control | Both 0.01 s (100 Hz), one control update per physics step; no decimation |
| Gravity | `[0,0,-9.81]` m/s² |
| Reset | 1.5 m altitude, zero world linear/angular velocities, prescribed yaw, reset rotor/joint/controller-related state |
| Yaw | Held at the episode's reset yaw; this fixed reference is an explicit simplification of the student's current-yaw target |

Velocity cases set no fixed position target, so the Lee controller uses the current position: position error is zero and velocity drives the motion. Zero velocity brakes the drone; it does not return it to its initial position. The hover case alone holds `[0,0,1.5]` and starts offset by `[0.15,-0.10,0.08]` m to test recovery.

The simulator measured base-link mass as 0.68 kg; the YAML/controller mass 0.716 kg includes the rotor bodies. Upstream reset throttle uses base-link mass, so its initial rotor equilibrium is approximate for the whole articulation; settling and measured trajectories expose the resulting transient.

The controller is stateless: gains are fixed, and it has no PID integral accumulator. The rotor model is stateful. Its throttle is reset to the upstream hover equilibrium, and joint positions/velocities, body pose/velocity, and force buffers are reset on the existing drone instance. Recreating a Python controller each episode would not by itself test that contract.

## Parameters and limitations

The upstream Hummingbird parameter file specifies mass 0.716 kg, diagonal inertia `[0.007,0.007,0.012]` kg m², four arms of 0.17 m, and rotor maximum angular velocity 838 rad/s. The supplied gains are position `[4,4,4]`, velocity `[2.2,2.2,2.2]`, attitude `[0.7,0.7,0.035]`, angular rate `[0.1,0.1,0.025]`. Raw gains and the controller's derived mixer/gains are saved alongside actual simulator mass/inertia.

The pinned rotor implementation interpolates throttle by 0.43 **per controller step**, not a calibrated time constant in seconds. Changing timestep therefore changes actuator dynamics. Its sampled noise is multiplied by zero. The drone's additional drag coefficient initializes to zero despite the YAML's nonzero drag value; rigid-body damping remains active (default 0.2 linear/angular). These are observed source behaviors, not a claim of aerodynamic realism. Randomization, wind, contacts, takeoff, aggressive attitude maneuvers, and multiple environments are outside this acceptance check.

## Episodes and frozen acceptance tolerances

Two repetitions each of five cases give ten episodes and 6,000 physics steps / 60 simulated seconds:

- Hover: six seconds of position recovery.
- X, Y, Z: one second settling, three seconds at 0.25 m/s, two seconds braking.
- X after a 90° yaw reset: same world-X command to test frame interpretation.

The saved configuration defines all durations and tolerances. Frozen thresholds are final hover error <=0.01 m, final speed <=0.01 m/s, commanded-axis displacement >=0.50 m, cross-axis displacement <=0.01 m, and mean commanded-axis velocity within 0.04 m/s of target. Velocity and final-state measurements use the final 0.5 s of the applicable segment. Repeated full trajectories must differ by no more than 0.0002 in the logged position/velocity/quaternion/rate/actuator fields; this is an explicitly mixed-unit diagnostic, not a physical distance metric. Initial reset-state error must be <=1e-6.

These thresholds were tightened after inspecting the calibration trajectories, before the acceptance run. The original provisional configuration is retained in that calibration run, and `calibration_run_id` records its lineage. Every metric and boolean decision is saved. Missing, reordered, or nonfinite trajectory samples fail reporting. A failed metric is not repaired by changing its label or discarding its run.

## Execute and inspect

Follow the [runtime setup](07-omnidrones-runtime.md), then:

```sh
sudo -v
uv run --locked python scripts/run_single_drone.py --accept-eula --gpu 0
```

The launcher saves configuration, selected GPU and process inventory, command, source/build identities, elapsed time, and container exit. Inside the container:

- `trajectory.csv`: every post-step observation, simulation time, direction/speed input, target velocity, raw/clipped rotor command, and rotor throttle.
- `resets.json`: every initial state and reset error.
- `runtime.json`: model parameters, gains, actual mass/inertia, versions, package origins, and asset hashes.
- `assets/` and `scene.usda`: local model files and a flattened scene snapshot. Indexed asset copies support audit; the flattened scene preserves the complete scene description.
- `metrics.json`, `trajectory.png`, `trajectory.pdf`: decisions and plots derived from raw samples.
- `events.jsonl`, `console.log`, `kit-logs/`: structured ALiGn events and preserved native logs.
- `probe-result.json`: flushed pre-close outcome, performance measurements, and failures.

A pass requires both completed physics/metric evidence and container exit 0. Fast shutdown may not return; the probe saves all data first. Python errors still fail the host report even if fast shutdown returns exit 0. The earlier full-cleanup crash remains unresolved.

Rendering and video are currently unvalidated. Trajectory plots are measurement visualizations, not camera frames. Do not infer contact correctness, large-swarm throughput, training stability, or deployment feasibility from this flight suite. Step throughput includes Python logging and GPU-to-CPU transfers and is not training throughput.

## Lab result record

First attempt `20260916T014731.351182IST-2fcf2b76` created and initialized the drone but failed before stepping physics: the upstream USD referenced a remote MDL plastic material. The offline run correctly rejected unresolved assets. Its log, model/controller metadata, and failure are preserved.

The retry explicitly localizes visual materials: remove the two material prims and their bindings, retain untextured geometry, and compare every nonmaterial authored property before and after. `hummingbird-local.usda`, original/local hashes, and the equality check are saved. This changes appearance only and is not a silent missing-asset fallback. The localized model completed all ten episodes in the following calibration attempt.

Calibration run `20260916T015409.084496IST-8edb58c1` completed 6,000 physics steps in 27.18 seconds of loop wall time (including logging), but failed its overall artifact contract during PDF export because vendor matplotlib lacked fontTools. It also used the earlier root-only JSON mode. The PNG and CSV are intact. Its `host-trajectory-review.json` recomputes measurements using CSV plus the actual reset-error events. All provisional physics criteria pass; **the overall run remains failed**.

Measured calibration outcomes: maximum hover tail error 0.000143 m, axis displacements 0.59129–0.59357 m during the three-second command, command-tail velocities 0.22887–0.22919 m/s, maximum cross-axis displacement 0.000854 m, maximum final-tail speed 0.005419 m/s, and maximum repeated-trajectory discrepancy 0.0000177. All reset errors were zero at recorded precision. The X command after a 90° yaw reset still moved along world X.

A useful low-speed approximation explains the measured speed deficit: with velocity gain 2.2 and linear damping 0.2, balancing controller acceleration against damping predicts `v ≈ 2.2/(2.2+0.2) × 0.25 = 0.22917 m/s`. This is an interpretation of the measured baseline, not an independently identified aerodynamic model. No integral or damping feedforward correction was added.

The corrected runtime adds locked fontTools 4.53.1 for PDF export and makes exported probe JSON host-readable.

Final acceptance run `20260916T101342.169928IST-39fd6f42` passed all 62 frozen decisions over 6,000 samples. It began at `2026-09-16T10:13:42.172066+05:30`, completed in 96.132 seconds, and exited the container with code 0. The flushed probe records `drone_physics_tested: true`; the host independently recomputed the metrics from `trajectory.csv` and obtained the same pass. Both `trajectory.png` and `trajectory.pdf` are present. The measured values remain consistent with calibration: maximum hover error 0.000143 m, commanded-axis displacements 0.59129–0.59357 m, command-tail velocity about 0.229 m/s, maximum cross-axis displacement 0.000854 m, and repeat discrepancy below 0.0000177.

## Recompute after the simulation has ended

```sh
uv run --locked python scripts/report_single_drone.py runs/single-drone/<id>
uv run --locked --extra reporting python scripts/report_single_drone.py runs/single-drone/<id> --plots
```

Each command creates a separate report directory under `runs/single-drone-reports/`; original flight artifacts are preserved. Plotting is optional and uses the host reporting lock. The host launcher also compares the saved container metrics with a fresh evaluation of the raw CSV.

## Current operational baseline

The accepted derived image and exact build inputs are recorded in the final run. Re-run the same command under **Execute and inspect** after checking GPU allocation. Fresh machine inventory for the acceptance session is `runs/diagnostics/20260916T101219.514424IST-d2b1397b/`. Keep the earlier failures and calibration run: they explain the offline-asset, PDF-dependency, and permission corrections. Because `runs/` is ignored, copy the accepted run directory to the project's external backup along with the tracked source revision.
