# Local wake-clearance command probe

The accepted [matched takeoff replay](44-takeoff-controller-replay.md) reproduced the learned actor's ground contact and attributed that fixed-command simulator fall to the pinned OmniDrones downwash term. Turning that term off is a diagnostic, not a flight solution. This command probe keeps the term active and changes only drone 2's requested world-frame velocity when it is below a nearby drone during takeoff. It does not retrain or modify the source checkpoint.

The CPU [guard](../src/align/tasks/wake_guard.py) uses relative drone positions and neighbor-requested commands within the source actor's declared 1.5 m sensing radius. It triggers only after drone 2 is at least 0.25 m high and another drone is 0.1–1.5 m above and less than 0.9 m away horizontally. It examines eight lateral directions over a 0.8 s constant-velocity prediction, rejects candidates with a predicted pair distance below 0.55 m, and selects the largest predicted horizontal clearance from the overhead drone. It retains the actor's requested vertical component and caps the whole command at 0.5 m/s. If no candidate qualifies, it leaves the command unchanged and records an unresolved step. These are explicit *candidate* parameters, not calibrated safety guarantees: the prediction omits dynamics and other agents' future policy reactions.

## Run one additional physical arm

Prepare and build a new image after this code change. From `align/`:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<new-prepared-id>
uv run --locked python scripts/run_takeoff_replay.py \
  --source-run runs/policy-telemetry/20260919T004809.622662IST-b996b218 \
  --baseline-run runs/takeoff-replay/20260919T011921.157537IST-01376539 \
  --build-report runs/runtime-build/<new-prepared-id>/report.json \
  --include-wake-guard --accept-eula --gpu 0
```

The launcher verifies the source telemetry, baseline command trace and configuration hashes, checkpoint identity, model/controller parameters, and current GPU availability. It re-audits the saved three-arm baseline but starts only one new simulator container for `wake_guard`. The original baseline and source runs remain immutable; back up all three ignored run directories together.

The new `wake_guard/trajectory.csv` records every saved request, actual guarded command, pre/post position, rotor output, vendor model force, and contact. The host independently recomputes every guard command from the saved pre-step positions, counts interventions/unresolved conflicts, and checks contact and minimum takeoff separation. `wake_guard.candidate_accepted` requires at least one intervention, zero unresolved steps, zero focal post-ascent contacts, and ≥0.55 m observed pair separation. A process `passed` means the data were complete and audited, even if the candidate itself was rejected. The SVG overlays the source, three baseline arms, and candidate altitude/force.

This is an **open-loop command replay**: the other drones continue receiving their saved commands, and the actor does not observe the altered trajectory. If the candidate passes, the next test is to integrate this declared guard into a separate frozen-policy evaluation arm, where observations and recurrent actor state evolve with the changed flight. Only then can we assess whether it helps the actual learned policy without increasing other safety failures or harming formation progress.

CPU checks: `uv run --locked python -m unittest tests.test_wake_guard tests.test_takeoff_replay`. The physical candidate passed the independent saved-trace audit described below.

## Accepted open-loop evidence and host-report repair

The first physical guard arm is [run `20260919T013621.050067IST-5e728518`](../runs/takeoff-replay/20260919T013621.050067IST-5e728518/report.json). Its simulator probe passed and saved all 1,017 steps, but the original host launcher reported `failed` after simulation because it mistakenly looked for `minimum_separation_m` under `task` instead of `construction`. The code now reads the correct configuration section. The immutable source report and raw files were retained. A new CPU-only [analysis run `20260919T013845.620755IST-45631a55`](../runs/takeoff-replay-analysis/20260919T013845.620755IST-45631a55/report.json) passed after rechecking source/baseline hashes, every command, simulator counts, contact, and separation. Reproduce that audit without Docker:

```sh
uv run --locked python scripts/analyze_takeoff_replay.py \
  --source-run runs/takeoff-replay/20260919T013621.050067IST-5e728518
```

The guard was active for 82 steps, had zero unresolved choices, and kept drone 2 airborne with no post-ascent contact. Its final altitude was `1.21527 m`; minimum observed takeoff pair separation was `0.71733 m`, above the `0.55 m` threshold. Maximum downward vendor model force on drone 2 was only `0.00530 N`, versus `3.02008 N` in the matched four-drone baseline. The vendor force model remained enabled. The independent report sets `candidate_accepted: true` **for this open-loop command replay**.

That physical image used a provisional 2.0 m candidate-scoring radius. The source actor's observation radius is 1.5 m, so the current guard uses 1.5 m. Recomputing **every** saved guarded command from the run's pre-step positions with the corrected 1.5 m rule produced an exact match: the extra 0.5 m of provisional visibility changed no command in this flight. Therefore a repeat GPU run is not needed to interpret this specific trace. The unused prepared 1.5 m context `20260919T013634.079365IST-f395a936` remains available for an independent reproduction if desired; it is not a built or validated simulator image.

This result does not yet show that a closed-loop actor stays safe or completes formation. Its future observations and LSTM state would change when the guard redirects it. The next bounded acceptance is a frozen-policy replay with the guard explicitly enabled and both original request and modified command logged, compared against the same checkpoint without the guard. Report all contacts, pair separations, formation exposure/error, and guard intervention counts before treating the rule as useful beyond this command sequence.
