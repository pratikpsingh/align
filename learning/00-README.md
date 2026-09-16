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

The [planning index](../plans/00-README.md) contains implementation requirements and research decisions. The runnable package and diagnostic are implemented and checked on the lab. Simulator startup/CUDA, the deterministic single-drone suite, and the four-drone construction suite have passed. Formation geometry, task rewards, and bounded local observations are CPU-validated; rewards and observations were also audited on the accepted physical trajectory. Training remains unimplemented.

## How future explanations will work

Each implemented topic will connect its purpose, mathematical idea, small example, code/configuration, validation, and limitations. Runnable examples will show tested commands and expected outputs. CPU checks and actual flight checks will be identified separately.

Planned topics include vectorized task environments, recurrent policies, PPO/GAE, communication accounting, checkpoint recovery, experiment analysis, waypoint missions, formation transitions, obstacle sensing, and group coordination. Create these when there is substantive material to explain; do not add empty placeholder lessons.

For each topic, try to answer: what enters the component, what does it compute, what leaves it, and how would we know it is wrong? This keeps the explanation connected to behavior rather than only naming algorithms.

Operational setup/usage documentation will also live in tracked docs/ when implemented so another researcher can use the project without these learning notes.
