# Contact detection after takeoff begins

A ground contact is expected while a drone is parked. Once that drone rises above the configured `airborne_height_m` (0.25 m in the feasible plane task), any later contact force above `contact_force_threshold_n` (0.01 N) is unsafe for the rest of that episode. The task stores this **per-drone, per-world latch** and clears it on reset. The formation phase continues to guard all contacts, including a drone that never reached 0.25 m. Reason code 3 remains `flight_contact`.

The earlier phase-aware rule used only current height during takeoff. If a drone climbed above 0.25 m and fell back to the ground before formation, current height dropped below the threshold and the contact was temporarily ignored. A four-drone positive-speed policy later produced many contact failures at the first formation step. The latch closes the logical gap; saved evaluation rows alone do not yet prove those particular drones crossed the takeoff threshold before falling.

The host [episode ledger](../src/align/tasks/environment.py) and live [OmniDrones task](../src/align/simulation/vector_task.py) implement the same state transition. The [CPU regression](../tests/test_task_environment.py) checks initial ground contact, ascent, a fall in the takeoff phase, and partial reset isolation. The [matched action result](41-nonnegative-policy-speed.md) remains on its original image and must not be reinterpreted as if it had used this latch.

## Validation

From `align/`:

```sh
uv run --locked python -m unittest tests.test_task_environment tests.test_action_interface_comparison
uv run --locked ruff check src/align/tasks/environment.py src/align/simulation/vector_task.py tests/test_task_environment.py
```

The CPU checks passed. The following physical sequence is **prepared but not yet executed**. It uses the immutable old-image policy checkpoint, then builds context `20260918T222825.144613IST-3e73a789`, flies the previously accepted wide-pyramid reference, and replays the same unsafe checkpoint on the new image:

```sh
sudo -v
uv run --locked python scripts/run_policy_telemetry.py \
  --source-run runs/plane-baseline-training/20260918T220124.368475IST-7a5bed66 \
  --build-report runs/runtime-build/20260918T202958.790600IST-4c23a0a2/report.json \
  --policy-seed 41 --evaluation-update 4 --accept-eula --gpu 0
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/20260918T222825.144613IST-3e73a789
uv run --locked python scripts/run_shape_transition.py \
  --build-report runs/runtime-build/20260918T222825.144613IST-3e73a789/report.json \
  --transition-config configs/shape-transition-plane-pyramid-wide.json \
  --accept-eula --gpu 0
uv run --locked python scripts/run_policy_telemetry.py \
  --source-run runs/plane-baseline-training/20260918T220124.368475IST-7a5bed66 \
  --build-report runs/runtime-build/20260918T222825.144613IST-3e73a789/report.json \
  --policy-seed 41 --evaluation-update 4 --accept-eula --gpu 0
```

Run each command only if the preceding one passes. Compare raw `policy-telemetry.csv` heights, forces, commanded velocities, and first terminal steps in the old and new replay. Require the new-image reference to retain safe formation completion, not merely `report.json` status `passed`. A passing container establishes integration; a safe mission outcome must be judged separately. Preserve the prior image and failed policy run as immutable evidence. Docker access currently requires the user's terminal; the agent session cannot authenticate sudo or open `/var/run/docker.sock`.
