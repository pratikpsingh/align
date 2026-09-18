# Eight-drone plane reference

This is the first bounded C10 size check. It uses eight Hummingbird drones in one simulator world, fixed identities, a 1 m-spaced launch line, a plane target, and the same 0.5 m/s calibrated velocity controller. The actor's local observation stays 55 values; the existing centralized critic has exactly eight agent slots. The reference controller uses exact target positions and does **not** test a learned policy or size generalization.

[Configuration](../configs/eight-drone-reference-plane.json) changes only `num_agents: 4 → 8` and formation timeout `8 → 12 s` relative to the calibrated four-drone construction config. The farther eight-drone target requires up to 2.741 m of horizontal travel. In an ideal saturated proportional model, this needs roughly 6.5 s to reach a 0.1 m error, plus the 2 s success dwell; the 12 s timeout leaves room for physical lag. This is a scheduling calculation, not a simulator result. Ground |x| is at most 3.5 m within the ±5 m safety envelope. The fixed takeoff-to-plane mapping has 0.707 m nominal minimum synchronous straight-line separation, above the 0.55 m planned margin. Physical trajectories can deviate and must be checked separately.

CPU geometry and shape checks:

```sh
uv run --locked python -m unittest tests.test_eight_drone_reference
```

The latch image is built, its four-drone safe reference passed, and an unsafe actor replay confirmed immediate post-ascent contact termination. To reproduce the physical eight-drone reference, run from `align/`:

```sh
sudo -v
uv run --locked python scripts/run_multi_drone.py \
  --config configs/eight-drone-reference-plane.json \
  --build-report runs/runtime-build/20260918T222825.144613IST-3e73a789/report.json \
  --accept-eula --gpu 0
```

The launcher retains an older “four-drone” help description but reads `num_agents` from this explicit config. Inspect the new `runs/multi-drone/<run-id>/report.json`, `metrics.json`, `trajectory.csv`, contact forces, min separation, target error, termination reasons, reset checks, GPU memory, and elapsed time. A passed process must still contain actual safe formation completion before the reference is accepted. Do not extrapolate from eight drones to 16/32/64/128: the current 1 m-spaced line exceeds the ±5 m envelope at 16, and the critic has only eight slots. Larger sizes require new safe launch geometry and an actor-only evaluator.

## Accepted physical result

[Run 20260919T005232.351407IST-3c858751](../runs/multi-drone/20260919T005232.351407IST-3c858751/report.json) used the pinned revised image and passed all controller, takeoff, tracking, separation, contact, and repeated-reset checks. Two sequential episodes in one world each reached success at step 1,459 (14.59 s), inside the 1,750-step deadline. The saved [raw trajectory](../runs/multi-drone/20260919T005232.351407IST-3c858751/trajectory.csv) independently yields 0.66305 m minimum post-ground separation, zero guarded airborne contact force, and 0.02304 m maximum final-200-step assigned RMSE for the first episode; the second differs only slightly. The reset-repeat position difference was at most 0.000077 m, below the configured 0.0002 m tolerance.

This establishes that the declared eight-drone reference scenario is physically feasible under privileged exact-target control. The two repeats share the same setup and are not independent seeds. No trained actor flew this scenario, and no 16/32/64/128-drone evidence follows from it. Further size tests need an actor-only frozen-policy evaluator and safe launch geometry.

Status: CPU and physical eight-drone reference accepted; learned-policy size generalization open.
