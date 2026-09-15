# Learning ALiGn

These notes explain the project by topic. They will grow alongside the implementation, with worked examples and links to the actual code. They are personal learning material and are intentionally ignored by Git.

## Available reading

| Topic | What it explains |
|---|---|
| [Runnable project and diagnostic](project-foundation.md) | What we built, how uv runs it, how to read the report, and the checks behind it |
| [Project overview](project-overview.md) | UAV formation control, simulation, policies, MAPPO, local observations, memory, and the migration |
| [Training, results, and recovery](training-results-and-recovery.md) | What training saves, how to read results, what a checkpoint restores, and how videos can be made later |

The [planning index](../plans/README.md) contains implementation requirements and research decisions. The runnable package and diagnostic are implemented and checked on the laptop. Simulator and training behavior remain planned; each explanation distinguishes current code from future work.

## How future explanations will work

Each implemented topic will connect its purpose, mathematical idea, small example, code/configuration, validation, and limitations. Runnable examples will show tested commands and expected outputs. CPU checks and actual flight checks will be identified separately.

Planned topics include runtime setup, formation geometry, observations and coordinate frames, low-level control, recurrent policies, PPO/GAE, reward design, neighborhood graphs, checkpoint recovery, experiment analysis, waypoint missions, formation transitions, obstacle sensing, and group coordination. Create these when there is substantive material to explain; do not add empty placeholder lessons.

For each topic, try to answer: what enters the component, what does it compute, what leaves it, and how would we know it is wrong? This keeps the explanation connected to behavior rather than only naming algorithms.

Operational setup/usage documentation will also live in tracked docs/ when implemented so another researcher can use the project without these ignored notes.
