# Fixed-ID group waypoint routes

C04 requires the swarm to maintain its assigned formation while it travels. C09 requires a complete waypoint mission rather than a helper that returns one intermediate point. The submission describes switching to the next waypoint when agents are sufficiently close but does not give a numeric group switching rule. The student's PyBullet `BaseAviary._calculateNextStep` computes one drone's intermediate point; it does not implement a synchronized mission. ALiGn defines an explicit route contract and now connects its fixed-ID targets to the simulator's deterministic reference controller.

`align.tasks.waypoints` translates the **same assigned slots** through an evenly spaced sequence of group centers. Drone IDs and relative slot offsets stay fixed. The current 3 m protocol starts after formation construction at step 2,600 and moves the group center along world +X from `(0, 0, 1.5)` to `(3, 0, 1.5)` m. The waypoint arm has four legs of 0.75 m; the direct arm has one 3 m leg. The only difference between their route configuration files is `maximum_leg_length_m`. A group advances at most one leg per step, and only after **every** drone has stayed within 0.15 m of its own slot for 20 consecutive 0.01 s steps, the minimum pair separation is at least 0.55 m, and no drone exceeds 0.2 m/s. A failed gate clears the dwell counter. Route state resets per episode.

This is a high-level group command. At a live route command, the task replaces each actor-visible assigned-target vector and reward target with the current translated slot. The target is not an extra neighbor observation. This integration currently runs only with privileged exact-pose reference control; a trained actor has not executed a route. A real decentralized deployment would need to communicate the synchronized waypoint index, and that traffic must be counted under C07. The CPU planner rejects targets outside the task's safety envelope, impossible ideal speed/dwell deadlines, malformed config, and routes over 128 legs. These checks do not prove obstacle clearance or dynamic flight feasibility.

## Reproduce the CPU plans

From `align/`:

```sh
uv run --locked python scripts/report_waypoint_route.py \\
  --construction-config configs/feasible-plane-construction.json \\
  --task-config configs/waypoint-task-48s.json \\
  --route-config configs/waypoint-route-3m.json

uv run --locked python scripts/report_waypoint_route.py \\
  --construction-config configs/feasible-plane-construction.json \\
  --task-config configs/waypoint-task-48s.json \\
  --route-config configs/waypoint-route-3m-direct.json
```

Each run writes an immutable `runs/waypoint-plan/<IST-run-id>/report.json` and `assigned-waypoints.csv` with every fixed-ID target. Accepted CPU reports `20260918T194225.716860IST-c4fbb2e0` (four waypoints) and `20260918T194225.806203IST-df33a71a` (direct) each cover 3 m. Both have a 6 s **ideal lower bound** from the 0.5 m/s speed limit; the waypoint arm adds at least 0.8 s of dwell, versus 0.2 s direct, within the 22 s available after the command. No drone flight or learned policy was tested by these reports.

The current task safety envelope allows this 3 m +X route with the four-drone plane slots. A 5 m +X route would place a slot beyond the configured 5 m horizontal safety bound, so 5/10/20 m evaluation needs a declared larger world and safety envelope plus physical calibration. Do not silently accept a CPU path as a physical route.

## Physical reference check

The route launcher uses one pinned simulator image, four drones per world, four cloned worlds, the same task seed, controller gain, reward, and 48 s episode limit for both arms. It changes only `maximum_leg_length_m` in the route config. `configs/waypoint-task-48s.json` is a named copy of the 48 s task used for the shape-reference flight. The low-level action is a bounded world-frame velocity command. This reference knows exact target and pose; it is a feasibility check, not a learned or decentralized mission.

After building a new runtime image that contains the route code, from `align/` run the four-leg arm and then the direct arm with the **same build report**:

```sh
uv run --locked python scripts/run_waypoint_route.py \
  --build-report runs/runtime-build/<new-build-id>/report.json \
  --route-config configs/waypoint-route-3m.json \
  --accept-eula --gpu 0

uv run --locked python scripts/run_waypoint_route.py \
  --build-report runs/runtime-build/<new-build-id>/report.json \
  --route-config configs/waypoint-route-3m-direct.json \
  --accept-eula --gpu 0
```

Each `runs/waypoint-reference/<IST-run-id>/` saves the resolved config, image/source metadata, native simulator log, `evaluation/evaluation.csv`, per-drone `evaluation/policy-telemetry.csv`, and per-world `evaluation/waypoint-progress.csv`. The route audit checks the command at step 2,600, every active fixed-ID target, index changes, all-drone arrival dwell, completion, and safety outcomes from those rows. `report.json` contains the audit and outcome counts. A `passed` launcher result establishes a complete, internally consistent measurement; inspect `audit.success_count` and `audit.route_completion_count` to judge flight success. Failed and timed-out attempts remain evidence.

The deterministic physical direct-versus-waypoint comparison below has passed; learned-policy route evaluation remains pending. The current simulator target switching is reference-only. A real decentralized controller would need to distribute the synchronized waypoint index and count those bytes under C07.

After both runs finish, compare their first-episode outcomes without retraining:

```sh
uv run --locked python scripts/compare_waypoint_routes.py \
  --waypoint-run runs/waypoint-reference/<four-leg-run-id> \
  --direct-run runs/waypoint-reference/<direct-run-id>
```

The comparison refuses different simulator images, resolved task bundles, controller/task inputs, endpoints, or any route-config difference beyond leg length. It saves `runs/waypoint-comparison/<IST-run-id>/paired-worlds.csv` with each world's completion, success, safety reason, post-command formation error/separation, and elapsed time. These four worlds share one seed and almost identical initial conditions; they are not four independent trials. The report therefore describes this deterministic comparison without claiming a statistically reliable waypoint advantage.

## Accepted 3 m reference result (2026-09-18 IST)

Built image `sha256:91011c604be94b708bd7795d65c2d9fa1b6f1f7fad2f3af77adf8eef58dbe7f8` from prepared context `20260918T202958.790600IST-4c23a0a2`. Four-leg run `20260918T203136.267774IST-11301294` and direct run `20260918T203409.683700IST-ea8e4b7a` both returned container exit 0 and simulator `waypoint_route_physics_tested=true`. Each commanded four worlds at step 2,600, completed its route in 4/4 worlds, and finished the ordinary success dwell in 4/4, with zero safety terminations. The host audit verified every commanded target and all gate transitions. Matched comparison `20260918T203648.005391IST-1e1603cc` passed provenance and single-variable checks.

| First-episode measure | Four 0.75 m legs | Direct 3 m leg |
|---|---:|---:|
| Time from command to successful outcome | 11.88 s | 9.66 s |
| Minimum post-command separation, world 0 | 0.9998 m | 0.9998 m |
| Mean post-command pairwise RMSE, world 0 | 0.0000258 m | 0.0000875 m |
| Success and safety outcomes | 4/4 success, 0 safety | 4/4 success, 0 safety |

The waypoint route was 2.22 s slower in every near-identical cloned world, consistent with intermediate settling and dwell. Both arms preserved separation and succeeded, so this experiment does not show that waypoints improve safety or mission success. Pairwise error was slightly smaller with waypoints, but the absolute difference is tiny. These are deterministic reference flights sharing one seed, not independent learned-policy trials. The 5/10/20 m distance claims and trained-actor route remain open.
