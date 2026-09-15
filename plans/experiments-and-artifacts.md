# Experiments, measurements, plots, and simulation artifacts

## Record every run

Every training, evaluation, profiling, and rendering attempt needs an identity, configuration, provenance, start/end/status records, and its available results. A failed run still has a useful result: failure type, last valid progress, logs, and consumed resources. Use null with a reason for unavailable measurements; never substitute zero.

Separate debug/smoke runs from research runs. A short smoke run demonstrates that a path executes; it does not establish a paper claim. Preserve negative results and unsuccessful seeds.

## Proposed artifact layout

~~~text
runs/<run-id>/
  manifest.json                 # identity, provenance, schemas, source/runtime
  config.resolved.yaml
  source/                       # allowlisted snapshot/patch and checksums
  attempts/<attempt-id>/
    launch.json
    status.json
    events.jsonl
    console.log
    train_metrics.jsonl
    system_metrics.jsonl
  checkpoints/
    <immutable-checkpoint>.*
    latest.json
    best.json
  exports/<checkpoint-id>/      # actor, preprocessing, input/action contract
  evaluations/<suite-id>/<checkpoint-id>/<eval-id>/
    manifest.json
    episodes.csv
    trajectories/              # compressed arrays and metadata
    videos/
  profiles/<profile-id>/
  reports/<report-id>/
    summary.json
    tables/
    figures/                   # PNG plus vector/PDF where appropriate
    report.md
~~~

This layout is a proposal. Version schemas and write readers independently of exact filenames. CSV/JSON are useful for summaries; compressed numeric arrays or a documented columnar format suit trajectories. Save units, field definitions, missing-value rules, and identifier mappings with the data. Avoid opaque Python-only objects as the sole long-term record.

## Provenance and reproducibility

Save command arguments, resolved configuration/hash, code revision and dirty state, source snapshot/patch, dependency lock hash, simulator/runtime builds, assets/scene versions, seeds and RNG policy, controller/observation/action/reward/metric schemas, and start/end timestamps.

Record OS, CPU, RAM, per-GPU model/VRAM, driver, CUDA/runtime versions, selected device, device visibility, and thread settings. Multiple GPUs do not automatically combine into one memory pool; record actual device allocation. Use an allowlist of useful environment settings rather than dumping credentials or the entire shell environment.

Identify parent checkpoints, run attempts, training/evaluation roles, and whether evaluation uses deterministic or sampled actions. Save the normalizer freeze setting, recurrent reset convention, suite version, and scene/mission seeds.

## Sample and time accounting

Define counters precisely:

- Environment transitions: the number of valid environment control-step transitions collected, summed across parallel environments.
- Agent transitions: the number of valid acting-agent transitions. With constant N, this is N times environment transitions.
- Policy/control/physics steps: separately named when their frequencies differ.
- Simulated time: per environment/episode, plus an explicitly labeled aggregate if used.
- Wall time: initialization, collection, optimization, evaluation, checkpointing, rendering, reporting, and total active run time, with overlapping timers identified.
- Recovery compute: time spent in abandoned attempts/updates remains in the actual compute total even if those samples are excluded from the active learning curve.

A vectorized step with 1,024 environments is not one training sample. The draft's 25 million timesteps must be interpreted and documented before a budget comparison; do not silently equate it with agent transitions.

## Required metric families

| Family | Save and report |
|---|---|
| Reward | Total and each raw/weighted term, per-agent/group aggregation, episode return, clipping/normalization behavior |
| Formation | Legacy aligned SSE for audit; aligned mean squared error and RMSE; target tracking error; pairwise geometry error; group/template/scale |
| Navigation | Goal/waypoint distance, arrival and dwell success, episode duration, path length, timeout/termination reason, transition settling time |
| Safety | Minimum separation, threshold violations and duration, distinct contact events, fraction of episodes with collisions, obstacle/inter-UAV categories |
| Control | Sampled/executed actions, saturation/clipping rates, action changes with units/timestep, physical actuator/rotor variation where available, speed/tilt limits |
| Communication | Observed neighbors, discovered candidates, directed/undirected graph definition, range, refresh rate, transmitted/received packets/bytes, age/dropout/delay, topology diagnostics |
| Learning | Actor/critic losses, entropy, approximate KL, PPO clip fraction, explained variance, gradient norms, learning rate, value/advantage statistics, invalid/NaN events |
| Performance | Environment and agent transitions/s, collection/update latency, total training time, GPU utilization, CPU RSS, GPU allocated/reserved memory, device memory usage |
| Model/export | Actor/critic parameter counts, trainable counts, dtype, serialized bytes, checkpoint bytes, recurrent-state size, single-agent inference latency, peak working memory |
| Software footprint | Allowlisted source-file bytes, total source bytes, nonblank/source line-count definition, source archive bytes, dependency/runtime footprint separately |

Source size documents the implementation footprint; it is not a measure of flight quality or efficiency. Policy export size differs from a resume checkpoint that also contains optimizer and critic state. Runtime installation size differs from both.

For model profiling, record batch size, device, dtype, thread count, warm-up, repetitions, and timing method. Report median and tail latency such as p95. Synchronize GPU timing appropriately. Profile deployed single-agent behavior separately from large-batch GPU throughput.

Log process-specific framework memory and whole-device memory separately. Device utilization may include other processes. Optional energy estimates may integrate sampled GPU power over time; label them GPU-only estimates and report missing sensor support. Do not present them as whole-machine or UAV battery energy.

## Metric definitions that must stay consistent

Aligned formation error removes the explicitly chosen nuisance transforms. The student's Procrustes-style residual removes translation/rotation but not scale; preserve that definition for its audit metric. State whether UAV-to-slot assignment is fixed or optimized. An assigned-slot tracking metric and a permutation-invariant shape metric answer different questions.

Record aligned SSE as a sum and MSE=SSE/N for each group, with RMSE in distance units. When normalizing by formation scale, save that scale and the dimensional result too. For multiple groups, report per-group errors and the declared aggregate. Do not use communication-edge removal to remove evaluated formation constraints.

Define a collision event using contact onset or an explicit distance-event rule. A single sustained contact across ten frames is not necessarily ten collisions. Also record violation duration. Mission success requires the declared combination of arrival, formation tolerance, settling/dwell, and safety; freeze its numeric thresholds before confirmatory comparisons.

Report outcomes over all episodes and optionally a separately labeled successful-only subset. Never hide failed flights by showing only successful-episode formation error. Document where a paper's metric requires a different population.

## Experiment protocol

Begin with small debugging runs and hardware capacity measurements. For confirmatory comparisons, plan multiple independent training seeds; five is a useful starting budget to review after profiling, not a statistical guarantee. Use common evaluation scenes/seeds across methods where appropriate.

Keep validation and final test suites separate. Select a best checkpoint using a declared validation rule, such as mission success followed by formation error, rather than retrospectively choosing the most attractive test video. Latest recovery checkpoint and best evaluation checkpoint have different purposes.

Show uncertainty across independent training runs and identify evaluation episode counts separately. Hundreds of episodes from one trained policy are not hundreds of independent training replications. Use seed-level summaries or an appropriate hierarchical/cluster bootstrap. State the procedure and sample counts. Never treat adjacent timesteps as independent evidence.

Record allocated sample budgets, early stopping, failures, retries, and hardware limits. Compare both sample efficiency and measured compute where relevant. Train-per-budget communication comparisons and fixed-policy neighbor-removal evaluations must remain separate experiment families.

## Mapping the student submission to deliverables

The [submission](../../papers/2026_Distributed_MARL_Submission.pdf) is an experiment-design reference. New results must be labeled as ALiGn/OmniDrones measurements, not asserted reproductions of its numbers.

| Submission item | Planned artifact |
|---|---|
| Figures 1–2: architecture | Diagram matching the implemented encoder, actor memory, critic, information flow, and measured dimensions/parameters |
| Figure 3: normalized reward components | Raw and weighted component curves for N=8; any display normalization is defined and preserves access to raw values |
| Figure 4: formation error during training | Evaluation/training error curves with metric definition, sample axis, seeds, and uncertainty |
| Figure 5: smoothing with/without penalty | Matched ablation, executed-action smoothness, formation/navigation/safety effects |
| Figure 6: error versus neighbors | Budget-versus-quality curves; distinguish retraining from fixed-policy inference ablation; add traffic cost |
| Table I: N=32, four shapes, 3 m | Per-template formation and smoothing plus success/collision/time measurements for the same frozen actor |
| Table II: N=8/16/32/64/128, 5 m | With/without smoothing, formation/safety/smoothness, feasible scene geometry, runtime/memory; record inability to run a setting |
| Table III: 3/5/10/20 m waypoint study | Direct versus waypoint execution, N=8, stated spacing and training-distance distribution, completion/error/time |
| Tables IV–VI: memory/deployment | Actual actor layer/parameter/byte table, measured runtime memory, explicit analytical assumptions; firmware figures only from a real target build |

The submission describes 25 million training timesteps and 600-run averages for some evaluations. Treat these as protocol details to resolve, not evidence available in the source checkout. A replication-oriented experiment can use 600 evaluation episodes per declared condition, with the breakdown across training seeds shown. A smaller pilot must be labeled a pilot.

Save plotting inputs and report configuration next to each figure. Export readable PNG and a suitable PDF/SVG for papers. Include units, seed/episode counts, definition of bands, and smoothing window. Preserve unsmoothed data. Do not use independent per-run min–max scaling that obscures reward magnitude differences.

## Simulation and videos after training

Yes: a saved policy can be loaded later to generate a new evaluation flight and record an OmniDrones video. It needs its preprocessing, actor/recurrent configuration, action/controller contract, mission/scene configuration, and compatible simulator runtime. Freeze learning and label deterministic versus sampled actions.

There are two useful products:

- A fresh policy evaluation in OmniDrones: produces new physics, metrics, trajectories, and optional video. The lab GPU/runtime is needed; headless means no interactive window, not no GPU.
- Playback of saved trajectories: uses recorded poses and scene state to recreate a presentation or a lightweight 3-D plot. A laptop can play MP4 files and typically generate lightweight trajectory plots without running OmniDrones physics. Label playback as recorded motion.

Save time-indexed UAV IDs, group membership, positions, orientations, velocities, targets/assignments, obstacle states, mission events, sampled/executed actions, validity/reset flags, terminal reasons, and the sample interval. Retain scene/asset identities and camera settings. Add communication edges/message age where communication overlays are planned.

Action replay alone may diverge because simulator execution is not guaranteed deterministic. Pose/state playback is useful for consistent presentation, but it is not a fresh policy evaluation. If complete physical replay is claimed, validate its full-state requirements separately.

Record representative fixed evaluation seeds, including failure cases. Use a readable camera, per-UAV/group colors, target/obstacle overlays, elapsed time, and mission labels. Offer top/side views and trajectory traces when they clarify behavior. Save the video frame rate and simulation-time mapping; rendering speed is not policy inference speed.

Avoid rendering every parallel environment throughout training. Save lightweight metrics for all runs, detailed trajectories for configured evaluation episodes, and videos for a standard small probe set. More videos can be generated later from saved policies or trajectories. Scalar reward logs alone cannot reconstruct a flight.

## Retention and backup

Estimate storage from a pilot, number of seeds, checkpoint retention, trajectory sampling, and video settings. Keep evaluation summaries and raw plot inputs even if redundant dense training traces are pruned. Record any downsampling and retention rules.

Back up immutable checkpoints, manifests, source/configuration, final evaluation data, and selected videos to a separate configured destination. Verify checksums and periodically test loading a copied checkpoint. Never claim backup exists until a destination and successful copy/verification are recorded.
