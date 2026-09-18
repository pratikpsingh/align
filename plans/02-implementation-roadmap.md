# Implementation roadmap

## Working method

Build a usable, verified increment at a time. Each increment includes its code, meaningful checks, public documentation, a topic-based learning explanation, and evidence from a runnable example or experiment. The headings below name capabilities, not document installments. Completion boxes record implemented work; simulator and research acceptance still require their stated evidence.

CPU verification and simulator verification are separate evidence. The laptop can support mathematical utilities, model/buffer checks, configuration, reporting, and much of checkpoint testing. Flight behavior, throughput, rendering, and GPU integration need the validated lab runtime.

## Runtime and project foundation

Dependencies: access to the current scaffold and the lab machine's hardware/runtime information.

- [x] Receive lab inventory: GPU models and per-device VRAM, driver, OS, host RAM, and filesystem capacity. GPU allocation/container access and simulator-build compatibility still need verification.
- [x] Select an OmniDrones revision and compatible Isaac Sim/Python/PyTorch/TorchRL/TensorDict combination. Review licenses and asset requirements; see [pinned runtime](../docs/07-omnidrones-runtime.md).
- [x] Validate uv integration in a derived image while preserving vendor PyTorch. Audit shared Python >=3.10 compatibility and maintain separate host/additional-runtime locks.
- [x] Establish the src package, validated diagnostic configuration, CLI, offline logging, and machine-readable inventory reports. Training configuration remains future work; the pinned simulator compatibility and vector-task checks have passed.
- [x] Run the isolated headless Isaac Sim startup/CUDA check on the lab; fast-mode run 20260916T010519.212064IST-5253d3c9 passed.
- [x] Integrate pinned OmniDrones dependencies and run an actual drone observation/action/reset check. Record exact commands, trajectories, and outcomes; accepted run `20260916T101342.169928IST-39fd6f42`.

Acceptance: reproducible setup from a clean checkout; CPU imports do not initialize Isaac Sim; GPU preflight gives actionable errors; versions and hardware are recorded. Public installation instructions and a learning document explaining the runtime are required.

## Task contracts and formation geometry

Dependencies: configuration foundation; GPU runtime for integration checks.

- [x] Specify task observation/action frames, units, tensor shapes, masks, control frequency, controller state, and limits. The controller is physically accepted; audit `20260916T125326.859172IST-96b8931e` validates the 55-value masked local actor input and separate 80-value critic state on 8,840 agent steps.
- [x] Implement tested cube/sphere/pyramid/plane templates, target assignment, formation metrics, and group-aware data structures. The first group contract uses stable identities and fixed assignment.
- [x] Implement reset, ground takeoff, goal tracking, dwell-based success, time limits, and explicit termination reasons. Accepted four-drone plane run `20260916T121136.980227IST-48963b76` passed 30/30 checks over two successful episodes.
- [x] Integrate a calibrated low-level controller and validate commanded motion before policy learning.
- [x] Implement reward components as separately logged quantities, including an active formation term and inter-UAV safety. Audit `20260916T122706.141515IST-e549a070` produced 8,840 finite component rows from the accepted physical trajectory; online learning integration remains future work.
- [x] Validate the one/four-environment vector task on the lab GPU. Run 20260916T162804.441981IST-29e96228 passed cloned physics, contact sensing, tensor parity, partial-reset isolation, true termination, time-limit truncation, raw trajectory audits, and measured probe throughput.

Acceptance: simple controlled motion behaves in the correct axis/units; reset and target placement are reproducible; analytical geometry examples match metrics; physical contacts and distance violations are distinct. Include diagrams or small numerical examples in the learning explanation.

## Recurrent MAPPO and decentralized observations

Dependencies: stable task contracts and reward/termination semantics.

- [x] Implement a shared actor over the fixed-capacity masked neighbor observation, LSTM memory, and an explicitly transformed continuous-action distribution. Run `20260916T183350.312358IST-f57f1110` passed all 18 vendor-PyTorch CUDA checks.
- [x] Keep centralized critic inputs separate from actor inputs. Normalize critic observations using saved training statistics.
  - [x] Actor and critic APIs, recurrent state, inputs, and parameters are separate; the actor accepts only the 55-value local vector and the critic accepts the 80-value training state.
  - [x] Implement, checkpoint, and restore critic-only active-group normalization with frozen warmup statistics. Run `20260917T125147.124653IST-920dcb9c` passed two-seed training and fresh evaluation; explained variance improved sharply, while formation metrics remained mixed and no evaluation succeeded.
  - [x] Measure active critic-group clipping, mean shift, and variance ratio on every rollout; verify exact transform reproduction and host-audit raw tables. Run `20260917T131555.325446IST-648f12e3` passed twenty updates and six exact evaluations. Position/target scales remained stable, while velocity clipping reached 18.1 percent; no formation evaluation succeeded.
  - [x] Validate the calibrated `0.025` critic denominator floor on the same ten-update, two-seed GPU protocol. Host calibration run `20260917T135710.911636IST-36a377ac` selected the smallest candidate that bounded all observed velocity extrema without changing position or target scaling. GPU run `20260917T140141.720844IST-685b2854` passed with zero velocity clipping, positive final critic explained variance for both seeds, mixed flight metrics, and no successful evaluations.
  - [x] Implement semantic diagnostics for all 80 critic inputs, active-slot versus padding checks, saturation, value/return alignment, and host-audited raw summaries. Run `20260917T121849.878540IST-4cabd9a6` passed and selected a critic-only active-feature group-normalization experiment with frozen checkpointed statistics.
- [x] Implement rollout collection, GAE, PPO updates, actual temporal sequence unrolling, and per-agent recurrent resets.
  - [x] Define and CPU-validate time-major rollout storage, separate bootstrap/trace/reset masks, GAE, initial LSTM states, padding masks, and episode-safe chunks.
  - [x] Collect a device-resident live rollout with per-drone actor memory, per-environment critic memory, partial resets, episode-safe chunks, and CPU-reference parity. Accepted run `20260916T190924.021233IST-667b7fa8` passed all device and host checks.
  - [x] Implement and validate masked recurrent PPO losses and optimizer updates. CUDA run `20260916T210604.064887IST-87a056c2` passed all 14 finite-update, clipping, team-advantage, and padding-invariance checks on a synthetic sequence fixture. Bounded live task run `20260916T220631.683824IST-5f89b3be` then completed two updates across separate simulator processes.
- [x] Preserve sequence ordering, chunk boundaries, initial recurrent states, and valid-sample masks during minibatching. CPU chunks are episode-safe; the CUDA actor/critic exactly matched full versus two-chunk evaluation.
- [x] Handle true termination and time-limit truncation correctly, using the final observation where bootstrap is appropriate. The accepted collector observed both boundary types and verified zero terminal versus finite truncation bootstraps.
- [x] Demonstrate finite gradients/updates, episode isolation, and dependence on prior observations in a meaningful memory check.
  - [x] The CUDA tensor probes demonstrated finite nonzero gradients and updates, exact reset isolation, measurable dependence on earlier observations, and update invariance to corrupted padding. The bounded task-connected run demonstrated finite live updates and changed actor/critic parameters. Two-seed run `20260916T230826.067057IST-2534c4be` completed three updates and fresh-process evaluation per seed; matched-rollout run `20260916T233640.753380IST-74e912b3` selected critic rate `1e-5`; repeated run `20260916T234710.834029IST-0733818e` kept all six consecutive updates within value-clip guidance and revalidated fresh-process evaluation. A ten-update, two-seed curve with exact update 0/5/10 evaluation is implemented. Run `20260917T001738.329486IST-5f61f50e` preserved both completed training seeds and all seed-41 evaluations, but timed out when seed 73's update-0 Isaac Sim process stalled during startup; immutable missing-evaluation recovery then passed as run `20260917T103636.493092IST-3c169a39`. The accepted curve showed a small non-monotonic assigned-RMSE improvement, no formation successes, and persistently negative explained variance.

Acceptance: recurrence is trained across time rather than as independent length-one samples; the actor receives no undeclared global data; reset memory cannot leak between episodes. A small simulator learning run produces interpretable diagnostics, without claiming final performance.

## Recoverable runs and evidence collection

Dependencies: learner state definitions. Develop this alongside the learner, before expensive runs.

- [ ] Implement run identities, immutable configuration, source/runtime manifests, event logs, and raw metric tables.
- [x] Implement atomic checkpoints, pre-publication deserialization, integrity verification, retained previous checkpoints, parent-checkpoint lineage, and the explicit `training_resume_with_environment_reset` mode. CUDA run `20260916T214543.893138IST-10cdc57d` passed all 18 checks.
- [ ] Separate resume, policy warm start, evaluation, rendering, and reporting.
- [ ] Test interruption, partial writes, corrupt latest checkpoints, repeated restarts, and counter/plot consistency.
  - [x] Host tests cover abrupt subprocess exit, partial and failed writes, malformed pre-publication payloads, corrupt newest fallback, and configuration mismatch. The CUDA probe matched the exact next update, advanced counters once, and loaded the fallback in a fresh process. Task-connected attempt metrics and plot supersession remain pending.
  - [x] Split a declared learning budget into fresh simulator processes at committed update boundaries. Run `20260917T144058.713679IST-643868db` passed 24 total updates across six training processes with exact ranges, checkpoint ID/hash continuity, frozen normalization continuity, one warmup per seed, and exact counters.
  - [x] Recover an incomplete multi-segment host run into a separate immutable run and fault-inject a live mid-rollout interruption. Source run `20260917T195211.368696IST-61fd6fc0` stopped update 5 after 64 steps; recovery `20260917T200314.721203IST-6c0eec2f` passed with an unchanged source tree and exact joined counters.
- [x] Produce a report from an interrupted-and-resumed example and explain any lost rollout progress.
  - [x] The synthetic recovery report explicitly discards partial rollout and recurrent state. A bounded live two-process run restored update 1 in a new Isaac Sim process, labeled four unfinished environment episodes as abandoned, restarted recurrent memory, and committed update 2. The later segmented acceptance discarded 256 environment transitions and 1,024 agent transitions from a fault-injected partial update, restored checkpoint 4 in a separate run, and completed without gaps or duplicate updates.

Acceptance: an interrupted run resumes from its latest valid committed state; no silent resetting of optimizer or normalization; reports identify attempts and abandoned work. See [recovery](07-training-recovery.md) and [artifacts](08-experiments-and-artifacts.md).

## Core flight and communication experiments

Dependencies: verified recurrent learning, recoverable runs, and a frozen basic evaluation protocol.

- [ ] Train one target-conditioned actor on the selected formation templates and ground-to-formation task.
  - [x] The deterministic four-template scheduler, unchanged local target conditioning, per-template raw metrics, recurrent PPO/checkpoint path, and frozen evaluation path passed live run `20260917T232002.612950IST-6bc40009` across two seeds, eight optimizer updates, and six exact-checkpoint evaluations.
  - [ ] Establish successful, stable formation behavior. The matched two-arm, two-seed 12-update run passed but all 64 evaluated environment episodes timed out; final plane assigned error remained about 1.38 m versus a 0.10 m tolerance. Diagnose formation-phase trajectories, reward contributions, and controller/action behavior before increasing the budget.
  - [x] Implement and execute an equal-budget plane-specialist versus four-template-generalist protocol. GPU arms `20260918T105406.233167IST-d74f8b47` and `20260918T111606.650830IST-7fb24d49` passed; comparison `20260918T115133.727696IST-e9186486` matched image/source/configuration and audited 48 training and 16 evaluation CSVs. Per-seed plane exposure was 36,864 versus 9,216 rows, so the result is equal-total-update rather than equal-plane-exposure.
  - [x] Diagnose saved frozen-policy formation phases without simulator replay. Run `20260918T120338.089896IST-b257083e` passed 64 reset-safe episode audits; final specialist assigned error changed only 1.6108→1.6083 m during formation while pairwise error worsened. Per-component rewards and realized controller motion remain unobserved; see [the evidence register](09-unresolved-evidence.md).
- [ ] Enforce hard communication range/budget masks and record neighbor topology and message age.
- [ ] Implement an explicit communication accounting model and neighbor-budget study, separating observation count from transmitted bytes.
- [ ] Implement waypoint advancement, settling/completion conditions, and direct-versus-waypoint evaluation.
- [ ] Add commanded in-flight template transitions, feasible target motion, and transition-specific metrics.
- [ ] Evaluate the frozen actor on multiple swarm sizes; record infeasible geometry, memory limits, and failures explicitly.
- [ ] Produce smoothing, communication, recurrence-correctness, and formation-reward ablations with controlled budgets.

Acceptance: C01–C11 in [the claim register](01-scope-and-claims.md) have traceable code/configuration, checks, and result artifacts. Results may falsify a hypothesis; implementation completion does not imply a positive result.

## Policy export and feasibility measurements

Dependencies: a frozen actor and verified preprocessing/action contracts.

- [ ] Export the actor, normalizers, recurrent-state specification, and input/action schema independently of the centralized critic.
- [ ] Measure actual parameter counts, artifact bytes, inference latency, working memory, and recurrent memory.
- [ ] Evaluate any supported reduced-precision/compressed export against the original on the same observations and flight suite.
- [ ] Distinguish desktop measurements, analytical embedded estimates, and measurements on actual target hardware.

Acceptance: C12 has measured artifacts and clearly labeled limits. Do not present estimated firmware/RAM as an onboard deployment result.

## Research extensions

Dependencies: stable core results and preserved comparison configurations. Extend one research variable at a time before testing combinations.

| Capability | Implementation focus | Evidence |
|---|---|---|
| Static/dynamic obstacles | Geometry, motion models, observable obstacle state, local sensing limits, contact and near-miss metrics | Fixed scene suites, motion seeds, collision-free arrival, generalization to unseen layouts |
| GRU/MLP comparison | Interchangeable memory modules, recurrent-state contracts, matched encoders | Equal-width and matched-parameter comparisons, memory/latency, delayed/dropout observations |
| Adaptive communication | Local scheduling, message discovery/freshness, event or learned policies | Mission quality versus modeled/measured traffic, delay/dropout robustness |
| Split/merge missions | Membership, target reassignment, separate routes, rendezvous, all-drone collision avoidance | Group-specific formation, split/merge completion, reassignment cost, safety |
| Reward improvements | Normalization, potential shaping, constraints and weight sensitivity | Task metrics and ablations; reward magnitude alone is insufficient |
| Paper-03 comparison | Recover benchmark details and adapt both methods to a shared protocol | Comparable controllers, training budgets, tasks, metrics, seed accounting, compute |

## Release readiness for research use

- [ ] Installation, training, resume, evaluation, and video examples run from a fresh documented setup.
- [ ] A frozen example checkpoint and its evaluation report can be identified without guessing filenames.
- [ ] Every reported figure/table can be regenerated from retained data and a recorded command/configuration.
- [ ] Public docs explain limitations; learning documents explain the implemented mechanisms.
- [ ] A result audit distinguishes implemented capabilities, simulator-validated behavior, and supported research claims.
- [ ] Run artifacts and the ignored personal documentation have an explicit backup procedure.

Time estimates and final experiment budgets will follow runtime smoke tests and measured collection/update/checkpoint timings. Do not estimate months of GPU work from another simulator's reported throughput.

## Single-drone evidence update, 2026-09-16

Final run `20260916T101342.169928IST-39fd6f42` passed all 62 frozen checks over ten episodes / 6,000 real physics samples. It confirmed correct axes, hover recovery, repeated-reset agreement, host metric recomputation, both trajectory plots, and container exit 0. See [the exact attempts and accepted evidence](../docs/08-single-drone-control.md). Formation geometry utilities, four-drone ground-to-plane construction, and the audited task reward contract are implemented. Formation learning has not started.
