# Flight-phase contact termination

The vector task uses base-link contact force in newtons to detect collisions. Ground contact is expected during the initial ground-settle phase and while the drone is still low during takeoff. Once the formation phase begins, any base-link contact above `contact_force_threshold_n` is unsafe even if a fallen drone's center has dropped below `airborne_height_m`. The earlier rule checked only height, so three drones that fell to z≈0.06 m in the narrow shape-transition runs contacted the ground and timed out without a safety termination.

The CPU task and simulator task now apply the same guard: `contact_force > threshold` and (`z > airborne_height` or `phase == formation`). A guarded contact contributes the configured contact penalty and terminates the episode with reason code 3, now named `flight_contact`. Numeric code 3 remains compatible with existing outcome-count fields. Historical reports and native logs retain the earlier `airborne_contact` label and behavior; do not reinterpret them. The first ground/takeoff contact remains allowed below the height threshold. Separation and envelope checks still have their own reasons.

The CPU contract test in `tests/test_task_environment.py` checks a 0.02 N contact at z≈0.06 m across ground, takeoff, and formation: only formation terminates. A read-only reclassification of the saved narrow flights finds the first newly guarded contact at step 2,981 in all four original worlds (z≈0.06 m, force≈0.52–0.54 N) and step 3,044 in all four raised-center worlds (force≈0.50–0.52 N). No post-command guarded contact appears in either wide-pyramid run. This retrospective prediction was tested on the rebuilt image. The physical acceptance result is recorded below; the old runs retain their original semantics.

To prepare a fresh image from `align/`:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
```

Build its new prepared context once, then run the existing narrow reference with `scripts/run_shape_transition.py --transition-config configs/shape-transition-plane-pyramid-raised.json` and the new `--build-report`. The prior narrow runs remain the baseline; they were built before this safety correction. GPU availability and Docker access are checked by the launcher. Logs, trajectories, and report are saved under `runs/shape-transition-reference/`; back up ignored `runs/` separately.

## Accepted simulator regression (2026-09-18 IST)

Run `20260918T213058.557715IST-8368f6b4` reused built image `sha256:91011c604be94b708bd7795d65c2d9fa1b6f1f7fad2f3af77adf8eef58dbe7f8` and the raised-pyramid configuration. Container exit was 0; the raw host audit passed. All four first episodes ended at step 3,044 with reason 3 `flight_contact`; none timed out. In world 0, drones 0 and 3 were at z `0.06013` and `0.06012` m with base-link contact force `0.5163` and `0.5122` N at the terminal step. At step 3,043, drone 0 was at z `0.06199` m with zero contact force and no termination. At ground-phase step 1, the same drone had `0.0661` N contact at z `0.06001` m and reason 0, confirming intentional ground contact remains allowed. The other three worlds had the same terminal step and reason with small numerical differences.

Read [the immutable report](../runs/shape-transition-reference/20260918T213058.557715IST-8368f6b4/report.json) with `evaluation/evaluation.csv` and `evaluation/policy-telemetry.csv`. This confirms the safety rule for this failure mode. It does not establish that the narrow pyramid can be held safely; the failed flight remains a negative shape-transition result.

A later [takeoff-contact latch](42-takeoff-contact-latch.md) remembers each drone after it first rises above the threshold, so a fall during takeoff is also unsafe. That extension has passed CPU checks and awaits a separate physical replay.
