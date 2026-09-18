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

The CPU checks passed. The old-image replay, revised-image safe reference, and same-policy revised-image replay have completed. The revised image is sha256:a8b2ff3b54c23f4c44fbe1c38232bcdd9a33029b9caf50479878b4716cba7a4f. Its [wide-pyramid reference](../runs/shape-transition-reference/20260919T004010.874376IST-150d2f91/report.json) completed successfully in four cloned worlds with zero contact terminations and 0.782 m minimum separation. This checks that the new latch did not reject the known-safe flight.

Run 20260919T003146.210187IST-5b43f7a3 passed its telemetry and exact-checkpoint checks, and the [host action audit](../runs/policy-telemetry-analysis/20260919T003558.559307IST-d45ca96f/report.json) passed. This was **not** a safe-flight pass: all four near-identical worlds ended with flight contact at the first formation step (episode step 1,551). Drone 2 in every world first rose above 0.25 m at step 247, then contacted the ground during takeoff at step 1,017 (1,018 in world 2). At first contact its height was about 0.060 m, force was 0.68–1.02 N, and the old rule still reported reason 0. World 0 commanded +0.191 m/s vertically at that contact; a positive command did not prevent the fall. These four cloned worlds share a seed and are not independent trials. The [raw telemetry](../runs/policy-telemetry/20260919T003146.210187IST-5b43f7a3/evaluation/policy-telemetry.csv) and [episode rows](../runs/policy-telemetry/20260919T003146.210187IST-5b43f7a3/evaluation/evaluation.csv) preserve the observation. The [reproducible host audit](../src/align/tasks/policy_telemetry.py) now reports each drone's first post-ascent contact and resets that history per episode. A policy that terminates before formation can produce a valid telemetry capture: the host accepts that narrow case only when the checkpoint/data checks pass and contact termination is recorded. Its report labels the capture as an early safety termination; it does not call the flight successful. The accepted old-image host audit is [here](../runs/policy-telemetry-analysis/20260919T004646.129304IST-342313a6/report.json). Focused CPU checks cover early contact, per-world reset isolation, and rejection of other failed checks.

The revised-image [replay artifacts](../runs/policy-telemetry/20260919T004809.622662IST-b996b218/report.json) retain the exact checkpoint and raw CSVs. In all four cloned worlds, the first post-ascent ground contact of drone 2 was unchanged: episode step 1,017 (1,018 in world 2), at about 0.060 m and 0.68–1.02 N. The revised task terminated immediately with reason 3 in each world; it also terminated four subsequent episodes during takeoff. The pre-contact 4,064 environment rows and 16,256 drone rows have identical CSV field values in the old and revised replays. This isolates the changed termination rule from earlier trajectories in this paired run.

The original revised-image launcher report says failed because the older image required a formation-phase row even when safety correctly ended the flight during takeoff. Its fast shutdown returned container exit 0. The [independent CPU audit](../runs/policy-telemetry-analysis/20260919T005118.273685IST-700ff62a/report.json) passed, preserving the failed source report and verifying its checkpoint, CSV integrity, eight first post-ascent contact events, and early-safety capture mode. The host launcher now accepts this narrow case when every other check passes. A valid recording does not make the policy successful.

To reproduce the same unsafe-policy flight on this image:

```sh
sudo -v
uv run --locked python scripts/run_policy_telemetry.py \
  --source-run runs/plane-baseline-training/20260918T220124.368475IST-7a5bed66 \
  --build-report runs/runtime-build/20260918T222825.144613IST-3e73a789/report.json \
  --policy-seed 41 --evaluation-update 4 --accept-eula --gpu 0
```

The physical rule is accepted for this paired fall and the known-safe reference. It does not explain why the actor-controlled drone fell despite an upward command. Preserve both images and their raw runs. Docker access currently requires the user's terminal; the agent session cannot authenticate sudo or open /var/run/docker.sock.
