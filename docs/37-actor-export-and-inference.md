# Actor export and fresh-process inference

The shared recurrent actor can be packaged separately from the centralized training critic. An accepted learner checkpoint is still read **during export**, after its manifest and SHA-256 have been checked. The output artifact contains only actor weights, actor configuration, observation configuration, and an input/action interface. The verifier is a new process with a read-only mount of that artifact and no mount of the learner checkpoint or source run.

This is a vendor-PyTorch CPU export. It does not start Isaac Sim, run a drone, convert the model to an embedded format, or validate onboard deployment. The inner drone velocity controller is also outside the artifact. Use [the observation contract](12-local-observation-contract.md) to construct local inputs and [the control contract](08-single-drone-control.md) for command semantics.

## Build and export

After host checks, prepare a new runtime context because the image must include `align.policies.actor_export` and `actor_inference`:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
```

Save the printed `runs/runtime-build/<BUILD-ID>` path. In a terminal with Docker access:

```sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<BUILD-ID>
uv run --locked python scripts/export_actor.py \
  --source-run runs/plane-baseline-training/20260918T172025.554059IST-f5bd6176 \
  --build-report runs/runtime-build/<BUILD-ID>/report.json \
  --policy-seed 41 --evaluation-update 4 --accept-eula
```

The exporter uses CPU only; it does not reserve a GPU. It verifies that the selected source run passed and that the exact update-4 checkpoint payload matches its manifest before deserializing it. The output is a new `runs/actor-export/<IST-run-id>/`. Use `--docker direct` only in a session with direct Docker socket access. Other seed/update choices require a valid checkpoint in the selected source run.

## Artifact contract

`artifact/actor.pt` is a PyTorch state dictionary, not a serialized Python module. `actor-config.json` defines the shared LSTM actor dimensions, distribution bounds, and hidden size. `observation-config.json` defines local neighbor capacity/radius and feature normalization scales. `interface.json` declares `[batch, 1, 55]` float32 observation and `[batch, 1, 4]` bounded action shapes for the current policy, hidden/cell tensors `[layers, batch, hidden]`, reset mask behavior, velocity limit, and controller timestep. The 55 input values are six self/target features, seven six-value neighbor slots, then seven masks; a smaller swarm pads missing slots and retains the same dimension.

The first three action values form a world-frame direction, normalized if nonzero. The fourth sets the **magnitude** of commanded speed through `abs(action[3]) × max_speed_m_s`; its sign does not reverse the direction. The drone's inner velocity controller remains necessary. A new episode zeros LSTM hidden/cell state or applies memory mask zero; continuing steps use mask one. Deterministic inference applies the actor's bounded tanh transform to the latent mean. The critic and optimizer are not needed for this inference path.

`artifact/report.json` records source checkpoint identity, file hashes, parameter count, tensor bytes, artifact bytes, and CPU batch-1/batch-8 `actor.act` latency with warmup and 200 measured calls by default. Its exporter process peak RSS includes full learner-checkpoint deserialization. `verification.json` records fresh-process RSS before actor construction, after loading, and after inference; these are whole vendor-Python process values, not isolated neural-network allocations. The verifier compares exact action/hidden/cell hashes, bound checks, and episode-reset behavior. `report.json` at the run root joins both process statuses and hashes. Preserve these raw files and the build context because `runs/` is ignored by Git.

## Interpretation and limits

A passed export establishes a portable **ALiGn/PyTorch actor artifact in the pinned image**, with measured desktop CPU cost. Its observation builder, command decoder, and controller still need to be deployed alongside it. Batch-1 CPU timing is not a flight-loop deadline guarantee; Python overhead and machine load are included, while sensing and control are excluded. The current configuration includes critic dimension metadata because `RecurrentPolicyConfig` also describes training, but no critic parameters or critic inference module are in the artifact. A compressed or reduced-precision export, equal-input action/flight comparison, and actual onboard timing/RAM/power measurements remain open C12 evidence.

## Accepted vendor-PyTorch result (2026-09-18 IST)

The source checkpoint was seed 41, update 4 from `20260918T172025.554059IST-f5bd6176`. The derived image `20260918T180717.335084IST-70d222e0` passed build/stack verification without changing vendor PyTorch. Actor-export runs `20260918T181000.134412IST-0a344ef7` (during a GPU evaluation) and `20260918T181821.229353IST-a87fa712` (after it finished) both passed export, fresh-process artifact-only inference, exact action/LSTM parity, bounds, and episode reset. Both produced the same actor SHA-256 `fbeaddefbf94e6e1ba6cb5e2040f817883c47016a0ac1592bf4d892a146fd629`.

| Measurement | Accepted value |
|---|---:|
| Trainable parameters | 542,838 |
| Parameter tensor bytes (float32) | 2,171,352 |
| `actor.pt` bytes | 2,176,931 |
| LSTM hidden + cell bytes per agent | 2,048 |
| Idle host CPU batch-1 median / p95 `actor.act` | 393.8 / 405.5 µs per call |
| Idle host CPU batch-8 median / p95 `actor.act` | 545.3 / 561.6 µs per call |
| Fresh verifier RSS before model / after load / after inference | 316.5 / 331.7 / 336.8 MB |

The fresh verifier's RSS increase from its already imported PyTorch baseline to after inference was about 20.3 MB. That delta includes actor construction and runtime allocations; it is **not** a precise neural-network-only working-set measurement. The exporter itself deserialized the full learner checkpoint before discarding the critic, so its peak RSS is a different, less relevant figure. Batch-8 timing is per **call for eight agents**, not per agent. The first run's batch-1 median was 390.9 µs under concurrent simulator work; the idle repeat was 393.8 µs, so no improvement should be inferred from that tiny difference. Physical controller execution, communication, target hardware, and compressed precision are unvalidated.
