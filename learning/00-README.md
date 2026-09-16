# Learning ALiGn

Read this index first, then follow the two-digit filename prefixes in ascending order. Each folder has its own reading sequence; numbers describe reading order, not implementation stages.

These notes explain the project by topic. They will grow alongside the implementation, with worked examples and links to the actual code. They are personal learning material and are tracked in the current checkout.

## Available reading

| Topic | What it explains |
|---|---|
| [Project overview](01-project-overview.md) | UAV formation control, simulation, policies, MAPPO, local observations, memory, and the migration |
| [Runnable project and diagnostic](02-project-foundation.md) | What we built, how uv runs it, how to read the report, and the checks behind it |
| [Lab report and runtime compatibility](03-lab-runtime.md) | What the five GPUs mean, how simulator dependencies fit together, and why we need a real drone test |
| [Simulator startup](04-simulator-startup.md) | Why we test the image, CUDA arithmetic, and application lifecycle before drone behavior |
| [Training, results, and recovery](05-training-results-and-recovery.md) | What training saves, how to read results, what a checkpoint restores, and how videos can be made later |
| [From startup to drone control](06-from-startup-to-drone-control.md) | What passed, why controller/reset checks come next, and how to continue on the lab |
| [Runtime and controller contract](07-runtime-and-controller-contract.md) | Velocity commands, rotor forces, reset memory, pinned environments and flight evidence |
| [Formation geometry and assignment](08-formation-geometry-and-assignment.md) | Shape slots, minimum spacing, one-to-one assignment, and comparable error metrics |
| [Ground-to-formation construction](09-ground-to-formation-construction.md) | Stable identities, staged takeoff, velocity tracking, contact/separation safety, and success dwell |
| [Task rewards and aggregation](10-task-rewards-and-aggregation.md) | Active formation and safety terms, means versus sums, timestep scaling, memory, and audit evidence |
| [Local observations and neighbor masks](11-local-observations-and-neighbor-masks.md) | Actor locality, hard range, fixed budgets, padding masks, normalization, and critic separation |
| [Vectorized task state and resets](12-vectorized-task-state-and-resets.md) | Cloned worlds, transition ordering, per-environment memory, partial reset, and terminal masks |
| [Recurrent rollouts and time masks](13-recurrent-rollouts-and-time-masks.md) | Ordered sequences, LSTM state, bootstrap/trace/reset masks, padding, and chunk boundaries |
| [Shared recurrent policy and bounded actions](14-shared-recurrent-policy-and-bounded-actions.md) | Actor/critic separation, real temporal memory, transformed Gaussian actions, gradients, and evidence limits |
| [Live recurrent collection](15-live-recurrent-collection.md) | Device rollout storage, team-value lanes, final-state bootstrap, partial resets, and reference parity |
| [Recurrent PPO updates](16-recurrent-ppo-updates.md) | Probability ratios, policy/value clipping, advantage normalization, padding invariance, and update diagnostics |
| [Training recovery with an environment reset](17-training-recovery.md) | Complete learner state, Adam/RNG continuity, atomic publication, fallback, lineage, and reset semantics |
| [Task-connected training loop](18-task-connected-training-loop.md) | How live drone rollouts, recurrent PPO, two-process resume, counters, and raw diagnostics fit together |

The [planning index](../plans/00-README.md) contains implementation requirements and research decisions. The runnable package and diagnostic are implemented and checked on the lab. Simulator startup/CUDA, the deterministic single-drone suite, and the four-drone construction suite have passed. Formation geometry, task rewards, bounded local observations, and recurrent rollout semantics are CPU-validated; rewards and observations were audited on the accepted physical trajectory. The cloned vector task passed its one-world and four-world GPU acceptance runs. The shared LSTM actor and centralized recurrent critic, live device collector, masked recurrent PPO optimizer update, and reset-mode learner recovery passed their CUDA contracts. A bounded two-process task-connected training run has now passed; sustained stable learning and independent evaluation remain pending.

## How future explanations will work

Each implemented topic will connect its purpose, mathematical idea, small example, code/configuration, validation, and limitations. Runnable examples will show tested commands and expected outputs. CPU checks and actual flight checks will be identified separately.

Planned topics include training stability and evaluation, communication accounting, experiment analysis, waypoint missions, formation transitions, obstacle sensing, and group coordination. Create these when there is substantive material to explain; do not add empty placeholder lessons.

For each topic, try to answer: what enters the component, what does it compute, what leaves it, and how would we know it is wrong? This keeps the explanation connected to behavior rather than only naming algorithms.

Operational setup/usage documentation will also live in tracked docs/ when implemented so another researcher can use the project without these learning notes.
