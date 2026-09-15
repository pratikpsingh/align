# Scope, claims, and source evidence

## Source snapshot

The review concerned these local sources:

- [Student submission](../../papers/2026_Distributed_MARL_Submission.pdf): ALiGn: Adaptive Local Coordination for Global Harmony in IoT-Edge UAV Swarms.
- [my-mappo](../../my-mappo/README.md), revision 2b7e74a32e79ca3d3d7095509e5bc286bba5fab4.
- [multi-UAV-formation](../../multi-UAV-formation/README.md), revision c4f1a6b73cde79ddbad96d7956475d3fb929c3ba.
- [Paper-04](../../papers/04-UAV-Distribution-Formation-Control-with-Improved-PPO-Algorithm.pdf), the MA-LSTM-PPO antecedent.
- [Paper-03](../../papers/03-Multi-UAV-Formation-Control-with-Static-and-Dynamic-Obstacles.pdf), an obstacle/formation benchmark for later comparison.

The submission includes established descriptions, unfinished annotations, proposed mechanisms, and future work. The available checkout does not establish which code generated its tables. No matching checkpoints or result datasets were found in the two inspected repositories. This is a reproducibility limitation, not a finding that the reported numbers are false.

## Retained capability register

IDs identify requirements and experiments; they do not imply implementation order. All ALiGn implementation and experimental validation below remain pending.

| ID | Requirement | Source evidence or gap | Acceptance evidence |
|---|---|---|---|
| C01 | Shared recurrent MAPPO actor and centralized critic | Submission IV-B; LSTM implementation exists | Correct temporal unrolling, actor/critic input isolation, finite updates, reproducible learning runs |
| C02 | Formation reward actively affects learning | Submission IV-A/IV-D; default training sets its weight to zero | Logged weighted contribution and a controlled reward ablation |
| C03 | Ground-to-formation construction | Submission IV-F/V-A; current reset starts around the formation template | Ground initialization, safe takeoff, assigned target tracking, formation dwell/completion results |
| C04 | Formation maintenance during travel | Central task throughout submission | Shape error, arrival, separation, flight stability, and completion time across the entire trajectory |
| C05 | One trained policy handles cube, sphere, pyramid, and plane | Submission IV-D/V-A; per-episode template sampling exists | Explicit target conditioning and evaluation of the same frozen actor across the template set |
| C06 | Bounded, dynamically selected local neighbor inputs | Submission III-C/IV-C; radius and padding exist with caveats | Hard range/budget checks, masks, local-input audit, neighbor-count/geometry measurements |
| C07 | Communication cost and neighbor-count study | Draft IV-C introduces optimization and a communication term, absent from drone training | Corrected graph assumptions, budget sweeps, actual/proxy cost labels, performance-cost trade-offs |
| C08 | Stable actions and collision avoidance between UAVs | Submission IV-A/V-B; reward penalties exist | Physical separation/contact measures, action/rotor smoothness, smoothing ablation |
| C09 | Long-distance waypoint execution | Submission IV-E/V-B; claimed execution pipeline not found | Fixed waypoint protocol, 3/5/10/20 m evaluation, completion and error with/without waypoints |
| C10 | Generalization to larger swarms | Submission Tables I/II | Train with 8; evaluate 8/16/32/64/128 where scenarios are feasible, document hardware limits and failures |
| C11 | Commanded in-flight shape changes | Claimed in introduction/problem statement; reset sampling does not implement it | Mid-episode command, explicit destination slots, transition trajectories, recovery time and collision-free completion |
| C12 | Deployment feasibility and compression assessment | Submission V-D, Tables IV-VI; runtime deployment remains future work | Actual actor counts/bytes, recurrent-state and working-memory measurements, latency, validated compressed-export comparison |

Larger-swarm evaluation must not require the training critic to accept a different swarm size. The decentralized actor export should be independently usable, with a fixed-capacity or set-based input contract. A policy trained for maximum k=7 must not silently change input dimension when evaluated on a smaller group.

## Extensions after the core capability suite works

| Extension | Scope and relation to retained claims |
|---|---|
| Static and dynamic obstacles | Adds environmental obstacle observations, scene generation, and avoidance; distinct from C08's inter-UAV collisions |
| GRU and memory-free comparison | Builds on C01; compare MLP/GRU/LSTM with matched encoders and training budgets |
| Advanced communication scheduling | Builds on C06/C07; learned/event-driven selection, message freshness, delay/dropout robustness, and network-aware cost |
| Multiple groups and reunification | Builds on C11; explicit membership, separate group paths, rendezvous, assignment, and inter-group safety |
| Systematic reward optimization | Builds on C02/C08; normalize terms, compare shaping and constrained formulations, quantify trade-offs |
| Paper-03 comparison | Reproduce/adapt its method later and use a common evaluation environment and protocol |
| Physical onboard deployment | Requires hardware-specific inference implementation and measured flight validation beyond analytical feasibility |

Obstacles and group operations should have extension points in the architecture. Do not delay basic logging, tests, recovery, or documentation until these extensions are added.

## Known issues to correct

- [Reward wrapper](../../my-mappo/onpolicy/envs/pybullet_drone_env.py) defaults to w_form=0; [training entrypoint](../../my-mappo/onpolicy/scripts/train/train_pybullet_drones.py) does not override it. A logged formation metric does not establish a formation-learning objective.
- [Recurrent buffer](../../my-mappo/onpolicy/utils/shared_buffer.py) repeats the chunk's initial state across all timesteps. [Policy](../../my-mappo/onpolicy/models/ma_lstm_policy.py) treats flattened samples as length-one sequences. Preserve rollout state and implement actual masked sequence training.
- [Observation construction](../../my-mappo/onpolicy/envs/gym_pybullet_drones/envs/BaseRLAviary.py) includes absolute position, and minimum-neighbor filling can reach outside the configured radius. Decide and enforce the intended measurement assumptions.
- [Formation initialization](../../my-mappo/onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py) samples positions around an already defined formation. Its dynamic mode samples a template at reset, not during flight.
- The generic PID intermediate-target helper is not evidence of the submission's complete waypoint experiment.
- [Formation utility](../../my-mappo/onpolicy/utils/formation.py) returns the sum of squared alignment residuals despite comments describing a mean. Keep a clearly named legacy sum for audit and an explicitly normalized measure for comparisons across swarm sizes.
- A Gaussian action mean bounded by tanh does not itself bound Gaussian samples. Specify the distribution, action transformation, limits, and log-probability convention.
- Ground construction, waypoint changes, success thresholds, velocity settling, timeout termination, and action/control frequency must have one consistent configuration.
- The draft's 3N-6 edge count is not a sufficient 3-D rigidity condition or a proof of the policy's optimal neighbor count.
- The MPC expression in the draft is not an implemented MAPPO component. If retained as motivation, explain its role and replace separation minimization with desired-separation error.

## What migration means

Retain algorithmic intent and verified mathematical utilities. Replace PyBullet world management, action/controller plumbing, process-based rollout orchestration, and NumPy-oriented hot paths with a validated OmniDrones integration. Rewrite contracts where needed rather than disguising the old environment with new imports.

A controller interface change is a scientific change. To preserve the student's intent, initially target velocity direction/speed through an explicitly calibrated low-level controller if supported. Verify mapping, frequency, limits, frames, and controller behavior. If another interface is necessary, record it as an adaptation and isolate its effect; do not silently switch to paper-03's CTBR commands.

Old checkpoints are not assumed portable across simulator dynamics, observation schemas, or controllers. Their main value is historical evaluation when the original runtime can be reconstructed.

## Attribution and research framing

Paper-04 remains an intellectual antecedent even if it is a preprint and paper-03 comparison is deferred. Being a preprint alone does not prohibit citation; describe its status accurately.

[Paper-01](../../papers/01-Decentralized-Control-of-Quadrotor-Swarms.pdf) already studies local-neighbor policies and multiple formation scenarios. [Paper-02](../../papers/02-Collision%20-Avoidance-and-Navigation-for-a-Quadrotor-Swarm.pdf), Figure 5, already compares range and nearest-neighbor observations. Paper-03 already uses action smoothing. Split/merge behavior also has prior work.

The proposed research direction is the measured relationship between communication, recurrent memory, formation/group reconfiguration, and obstacle navigation. Do not claim novelty or superiority from combining feature names.
