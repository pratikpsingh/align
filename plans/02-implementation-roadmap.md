# Implementation roadmap

## Working method

Build a usable, verified increment at a time. Each increment includes its code, meaningful checks, public documentation, a topic-based learning explanation, and evidence from a runnable example or experiment. The headings below name capabilities, not document installments. Completion boxes record implemented work; simulator and research acceptance still require their stated evidence.

CPU verification and simulator verification are separate evidence. The laptop can support mathematical utilities, model/buffer checks, configuration, reporting, and much of checkpoint testing. Flight behavior, throughput, rendering, and GPU integration need the validated lab runtime.

## Runtime and project foundation

Dependencies: access to the current scaffold and the lab machine's hardware/runtime information.

- [x] Receive lab inventory: GPU models and per-device VRAM, driver, OS, host RAM, and filesystem capacity. GPU allocation/container access and simulator-build compatibility still need verification.
- [x] Select an OmniDrones revision and compatible Isaac Sim/Python/PyTorch/TorchRL/TensorDict combination. Review licenses and asset requirements; see [pinned runtime](../docs/07-omnidrones-runtime.md).
- [x] Validate uv integration in a derived image while preserving vendor PyTorch. Audit shared Python >=3.10 compatibility and maintain separate host/additional-runtime locks.
- [x] Establish the src package, validated diagnostic configuration, CLI, offline logging, and machine-readable inventory reports. Training configuration and simulator compatibility checks remain future work.
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

Acceptance: simple controlled motion behaves in the correct axis/units; reset and target placement are reproducible; analytical geometry examples match metrics; physical contacts and distance violations are distinct. Include diagrams or small numerical examples in the learning explanation.

## Recurrent MAPPO and decentralized observations

Dependencies: stable task contracts and reward/termination semantics.

- [ ] Implement a shared actor with a fixed-capacity masked or set-based neighbor encoder, LSTM memory, and an explicitly defined continuous-action distribution.
- [ ] Keep centralized critic inputs separate from actor inputs. Normalize observations using saved training statistics.
- [ ] Implement rollout collection, GAE, PPO updates, actual temporal sequence unrolling, and per-agent recurrent resets.
- [ ] Preserve sequence ordering, chunk boundaries, initial recurrent states, and valid-sample masks during minibatching.
- [ ] Handle true termination and time-limit truncation correctly, using the final observation where bootstrap is appropriate.
- [ ] Demonstrate finite gradients/updates, episode isolation, and dependence on prior observations in a meaningful memory check.

Acceptance: recurrence is trained across time rather than as independent length-one samples; the actor receives no undeclared global data; reset memory cannot leak between episodes. A small simulator learning run produces interpretable diagnostics, without claiming final performance.

## Recoverable runs and evidence collection

Dependencies: learner state definitions. Develop this alongside the learner, before expensive runs.

- [ ] Implement run identities, immutable configuration, source/runtime manifests, event logs, and raw metric tables.
- [ ] Implement atomic checkpoints, integrity verification, retained previous checkpoints, restart lineage, and a clearly stated simulator recovery mode.
- [ ] Separate resume, policy warm start, evaluation, rendering, and reporting.
- [ ] Test interruption, partial writes, corrupt latest checkpoints, repeated restarts, and counter/plot consistency.
- [ ] Produce a report from an interrupted-and-resumed example and explain any lost rollout progress.

Acceptance: an interrupted run resumes from its latest valid committed state; no silent resetting of optimizer or normalization; reports identify attempts and abandoned work. See [recovery](07-training-recovery.md) and [artifacts](08-experiments-and-artifacts.md).

## Core flight and communication experiments

Dependencies: verified recurrent learning, recoverable runs, and a frozen basic evaluation protocol.

- [ ] Train one target-conditioned actor on the selected formation templates and ground-to-formation task.
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
