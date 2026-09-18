# Matched takeoff controller replay

A frozen actor's seed-41 update-4 flight in [the revised-image telemetry run](../runs/policy-telemetry/20260919T004809.622662IST-b996b218/evaluation/policy-telemetry.csv) lost altitude during takeoff. Drone 2 was commanded upward at its first post-ascent ground contact. The contact latch correctly terminated the flight, but the recorded actor command alone does not identify the physical cause. This diagnostic replays that flight's **saved world-frame velocity requests**, one 0.01-second controller/physics step at a time. There is no actor inference, reward optimization, or training in these runs.

The pinned OmniDrones `MultirotorBase.apply_action` computes rotor thrust and, for multiple drones, applies a simplified downwash force to the base link. The source is frozen under `.runtime/sources/OmniDrones/omni_drones/robots/drone/multirotor.py`; ALiGn does not modify it. The diagnostic uses the same source training configuration and Lee velocity controller in all arms:

| Arm | Drones and commands | Variable changed |
|---|---|---|
| `four` | Four at the source launch positions; all four get their saved commands | Reference replay |
| `isolated` | Drone 2 gets its saved commands; three passive drones are parked far away | Nearby active neighbors removed, while the scene still contains four bodies |
| `no_downwash` | Four at source positions; all get saved commands | Only the vendor downwash force function returns zero |

The isolated arm is **one active drone**, not a one-body simulator world. This keeps the Hummingbird view size and physics setup matched. The `no_downwash` arm changes a simulator model term for diagnosis; it is not a proposed production fix or a physical measurement of aerodynamic force.

## Lab command

Prepare and build a **new** image after these code changes. From `align/`:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<new-prepared-id>
uv run --locked python scripts/run_takeoff_replay.py \
  --source-run runs/policy-telemetry/20260919T004809.622662IST-b996b218 \
  --build-report runs/runtime-build/<new-prepared-id>/report.json \
  --accept-eula --gpu 0
```

Replace `<new-prepared-id>` with the printed directory name. The prepared context is single-use. The launcher checks that GPU 0 is idle and the image is available. It runs each arm in a separate simulator container, then audits all raw CSVs on the host. The source training and telemetry runs are read only. Docker authentication is required in the user's terminal on this machine; no password belongs in a report or chat. Expect three simulator startups and about 10.17 simulated seconds per arm. Do not equate simulator seconds with wall time.

## Artifacts and interpretation

`runs/takeoff-replay/<ID>/commands.csv` is the extracted first episode from world 0, with 1,017 steps and four commands per step for this source. Its hash, source telemetry hash, checkpoint identity, source image, new image, and exact Docker commands are in `report.json`. Each arm saves `trajectory.csv`, `metrics.json`, `probe-result.json`, console and native Kit logs, scene/model/assets, and GPU metadata. `altitude-and-force.svg` plots drone 2's altitude and the vendor model's vertical base-link force; its numerical source is the CSV, not the plot.

The host audit requires complete per-step/per-drone rows, finite values, and exact focal command parity. In the isolated arm, only the three parked drones have their commands replaced by zero and their motors turned off. It reports first ascent and contact, minimum vertical speed, final altitude, mean rotor thrust, and maximum downward model force. A separate flag asks whether the four-drone replay reproduces the source contact within 50 steps and has altitude RMSE ≤0.15 m over the first 750 steps. These are **declared diagnostic tolerances**, not calibrated flight tolerances. If the reference replay fails this check, the other arms cannot explain the saved flight.

If the reference replay reproduces contact while both modified arms avoid it, the report marks modeled downwash as supported as necessary **for this simulator replay**. Other outcomes remain inconclusive and require inspection. In particular, removing a force term from a simulator does not measure real airflow, and command replay cannot establish that PPO learned a safe control law. The actor's formation task remains open until independent frozen evaluations succeed safely.

CPU checks: `uv run --locked python -m unittest tests.test_takeoff_replay`. The accepted physical evidence is recorded below. `runs/` is ignored by Git and needs separate backup.

## Accepted physical comparison, 2026-09-19 IST

[Run `20260919T011921.157537IST-01376539`](../runs/takeoff-replay/20260919T011921.157537IST-01376539/report.json) passed all three simulator arms and the independent host CSV audit on image `sha256:ddffc9fd2deed54fdefff320d02e1babc03983553f0191d9e8408eaf21e16280`. The 1,017-step four-drone replay contacted at zero-based step 1,016, exactly matching the saved source. Its altitude RMSE against the source for the first 750 steps was `0.0000539 m`, well within the declared `0.15 m` fidelity gate. Model/controller parameters matched the source. Both other arms stayed airborne with no post-ascent contact and ended at `1.2331 m` altitude.

At step 900, the four-drone arm measured `−0.948 N` vertical vendor model force, `7.475 N` total rotor thrust, `−0.164 m/s` vertical velocity, and `0.959 m` altitude. At step 1,000 the model force had reached `−2.993 N`, rotor thrust `9.272 N`, vertical velocity `−1.241 m/s`, and altitude `0.264 m`. At the last step it contacted the ground. The isolated and no-downwash arms had zero recorded base-link model force, near-identical trajectories, and positive vertical velocity at the last step. The host report sets `baseline_reproduced_with_declared_tolerance: true` and `modeled_downwash_causal_support: true`.

This is strong causal evidence **within the pinned simulator and this fixed command sequence**: turning off the vendor downwash term prevented this contact, while leaving the four-drone layout and commands otherwise fixed. It does not validate turning downwash off for training or deployment, estimate real-world aerodynamic loads, or establish that the learned actor safely maintains formation. The appropriate next intervention is to prevent the lower drone from entering the modeled wake while preserving the physical force model, then replay/evaluate that intervention with safety and mission metrics.
