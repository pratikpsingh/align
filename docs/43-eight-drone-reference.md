# Eight-drone plane reference

This is the first bounded C10 size check. It uses eight Hummingbird drones in one simulator world, fixed identities, a 1 m-spaced launch line, a plane target, and the same 0.5 m/s calibrated velocity controller. The actor's local observation stays 55 values; the existing centralized critic has exactly eight agent slots. The reference controller uses exact target positions and does **not** test a learned policy or size generalization.

[Configuration](../configs/eight-drone-reference-plane.json) changes only `num_agents: 4 → 8` and formation timeout `8 → 12 s` relative to the calibrated four-drone construction config. The farther eight-drone target requires up to 2.741 m of horizontal travel. In an ideal saturated proportional model, this needs roughly 6.5 s to reach a 0.1 m error, plus the 2 s success dwell; the 12 s timeout leaves room for physical lag. This is a scheduling calculation, not a simulator result. Ground |x| is at most 3.5 m within the ±5 m safety envelope. The fixed takeoff-to-plane mapping has 0.707 m nominal minimum synchronous straight-line separation, above the 0.55 m planned margin. Physical trajectories can deviate and must be checked separately.

CPU geometry and shape checks:

```sh
uv run --locked python -m unittest tests.test_eight_drone_reference
```

After the prepared latch image `runs/runtime-build/20260918T222825.144613IST-3e73a789` is built and its four-drone safety regression passes, run the physical reference from `align/`:

```sh
sudo -v
uv run --locked python scripts/run_multi_drone.py \
  --config configs/eight-drone-reference-plane.json \
  --build-report runs/runtime-build/20260918T222825.144613IST-3e73a789/report.json \
  --accept-eula --gpu 0
```

The launcher retains an older “four-drone” help description but reads `num_agents` from this explicit config. Inspect the new `runs/multi-drone/<run-id>/report.json`, `metrics.json`, `trajectory.csv`, contact forces, min separation, target error, termination reasons, reset checks, GPU memory, and elapsed time. A passed process must still contain actual safe formation completion before the reference is accepted. Do not extrapolate from eight drones to 16/32/64/128: the current 1 m-spaced line exceeds the ±5 m envelope at 16, and the critic has only eight slots. Larger sizes require new safe launch geometry and an actor-only evaluator.

Status: CPU checks passed; physical eight-drone result pending.
