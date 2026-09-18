# Frozen-policy wake-guard evaluation

The [open-loop command replay](45-local-wake-clearance.md) showed that a lateral escape kept drone 2 airborne for one saved takeoff sequence. A frozen-policy evaluation answers a different question: after each altered velocity command, the simulator returns the new observation to the same recurrent actor, which then chooses its next action. The checkpoint and LSTM update are unchanged; only the declared takeoff command intervention differs.

## Contract and use

[The task integration](../src/align/simulation/vector_task.py) accepts `--wake-guard` only for a four-drone evaluation with per-drone telemetry, a 1.5 m actor neighbor radius, and 0.5 m/s command limit. The rule operates only during takeoff and only on drone 2. It preserves the actor's requested vertical velocity and passes the altered world-frame velocity to the same Lee controller. No training or checkpoint update occurs.

The guard considers only neighbors within 1.5 m by position, but it also uses those neighbors' current requested velocities in its 0.8 s prediction. In this diagnostic those commands are available inside the centralized simulator step. A deployed version would need a declared neighbor-command communication mechanism, measured freshness, and sensing accuracy; local position range alone does not establish that interface. The fixed drone ID and hand-selected thresholds are also experimental choices.

The guarded CSV extends the existing telemetry schema with the actor's requested velocity and two flags. `command_v*` remains the velocity actually sent to the controller. The [host audit](../src/align/tasks/policy_telemetry.py) recomputes the actor request, guard choice, rewards, geometry, and rotor saturation from raw rows. The [paired comparator](../src/align/runtime/policy_guard_comparison.py) requires identical training source, checkpoint hash, evaluation image, seed, update, and task config; it accepts an intact early-contact capture as negative evidence rather than discarding it.

From `align/`, prepare one image after source changes, then build it and run both arms. Use the same `<build-id>` in all three commands:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepared-run runs/runtime-build/<build-id>
uv run --locked python scripts/run_policy_telemetry.py \
  --source-run runs/plane-baseline-training/20260918T220124.368475IST-7a5bed66 \
  --build-report runs/runtime-build/<build-id>/report.json \
  --policy-seed 41 --evaluation-update 4 --accept-eula --gpu 0
uv run --locked python scripts/run_policy_telemetry.py \
  --source-run runs/plane-baseline-training/20260918T220124.368475IST-7a5bed66 \
  --build-report runs/runtime-build/<build-id>/report.json \
  --policy-seed 41 --evaluation-update 4 --wake-guard --accept-eula --gpu 0
```

Run the second arm even if the first reports `failed`: a ground-contact run can have complete, independently auditable telemetry but miss the original formation-entry process gate. Then compare the two **new** replay run IDs:

```sh
uv run --locked python scripts/compare_policy_guard.py \
  --baseline-run runs/policy-telemetry/<baseline-id> \
  --guarded-run runs/policy-telemetry/<guarded-id>
```

The comparison writes an immutable `runs/policy-guard-comparison/` report with hashes, first post-airborne contact events, guard activity, minimum non-ground pair separation, formation exposure/error, and outcomes. A comparator `passed` means data and pairing are valid; decide safety and mission usefulness from those measurements. The same held-out checkpoint should also be tested over independent seeds before adopting the rule. Preserve all ignored run directories and the prepared build context in backup.

CPU validation: `uv run --locked python -m unittest tests.test_guarded_policy_telemetry tests.test_policy_guard_comparison`. The paired GPU result and independent comparison are accepted below.


## Same-image physical result, 2026-09-19 IST

The prepared image `20260919T014804.061440IST-3f9ca840` built as `sha256:81fc210ae2888467211106c78eedb8f4db26f334804bb4133284623127b26241`. The baseline [run `20260919T015013.905042IST-20bb0cd3`](../runs/policy-telemetry/20260919T015013.905042IST-20bb0cd3/report.json) and guarded [run `20260919T015207.324862IST-48221710`](../runs/policy-telemetry/20260919T015207.324862IST-48221710/report.json) used the same update-4 seed-41 checkpoint and 9,400 environment steps each, with zero optimizer updates. The [immutable paired comparison `20260919T015609.579036IST-5a799800`](../runs/policy-guard-comparison/20260919T015609.579036IST-5a799800/report.json) independently re-audited both CSVs, hashes, identities, commands, geometry, reward, contacts, and outcomes.

| Measurement | No guard | Takeoff guard |
|---|---:|---:|
| Post-airborne contact events | 8 | 4 |
| First contact episode step | 1,016–1,020 | 1,820–1,828 |
| Formation environment rows | 0 | 1,102 |
| Minimum non-ground pair separation | 0.654 m | 0.772 m |
| Guard interventions / unresolved choices | 0 / 0 | 523 / 0 |
| Success outcomes | 0 | 0 |

The baseline's simulator evaluation missed its formation-entry check because all four worlds contacted the ground during takeoff, but its host status was `passed` under the explicit early-contact capture rule after the raw audit. The guarded arm reached formation, then drone 2 contacted the ground in each world. In world 0, drone 2 descended from about 1.833 m to 0.060 m during 278 formation steps; its mean requested vertical velocity was +0.070 m/s while mean measured vertical velocity was −0.636 m/s. This is a command/response mismatch, not yet a force attribution. The guarded formation-phase assigned and pairwise RMSEs were 1.639 m and 1.877 m; the baseline has no formation rows and therefore no comparable formation error. Neither arm completed the task.

The same guard rule, evaluated without applying it on the guarded trace's formation rows, would have triggered only 9 times in 1,102 formation environment steps. Extending this particular condition from takeoff to formation is therefore unlikely to explain or repair the later fall by itself; that is an inference from saved one-step geometry, not a counterfactual flight. The next physical diagnosis should replay the post-takeoff commands and record the vendor force contribution and controller thrust, then test one declared change at a time. The present rule is a **partial safety improvement in this pinned simulator**, not a learned-formation solution or a deployable safety guarantee.
