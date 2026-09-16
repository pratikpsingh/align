# ALiGn implementation and research plan

Read this index first, then follow the two-digit filename prefixes in ascending order. Each folder has its own reading sequence; numbers describe reading order, not implementation stages.

Prepared: 2026-09-15

Status: the uv-managed package, machine diagnostic, local logging, and report artifacts are implemented and checked on CPU. Public [installation](../docs/01-installation.md), [diagnostic](../docs/02-diagnostics.md), and [development](../docs/05-development.md) instructions now exist. Python 3.12 remains the development pin; the isolated Isaac Sim startup/CUDA probe, deterministic controller checks, one/four-world OmniDrones task acceptance, recurrent learner components, and reset-mode checkpoint recovery have passed on the lab. Task-connected simulator training remains unimplemented.

## Direction

Build a maintainable OmniDrones implementation of the student's intended formation-control system. Preserve the useful recurrent MAPPO, local-neighbor, formation-template, smoothing, and waypoint ideas; implement the missing retained claims; correct known defects. Establish measurements and recoverable training as part of the foundation.

Paper-03 is a useful simulator/code reference, but reproducing its algorithm and reported results is deferred. Its legacy dependency stack does not automatically become the runtime choice for ALiGn.

## Documents

| Document | Purpose |
|---|---|
| [Scope and claims](01-scope-and-claims.md) | Current focus, deferred work, evidence from the student submission and repository |
| [Implementation roadmap](02-implementation-roadmap.md) | Dependencies, implementation work, validation, and documentation deliverables |
| [Research design](03-research-design.md) | Observations, actions, recurrence, neighbors, rewards, and shape/group behavior |
| [Architecture and quality](04-architecture-and-quality.md) | Intended file structure, module contracts, configuration, and engineering practices |
| [Setup and documentation](05-setup-and-documentation.md) | Lab/laptop workflows, uv/runtime compatibility, public docs, and teaching workflow |
| [Lab runtime assessment](06-lab-runtime-assessment.md) | Received hardware evidence, upstream compatibility findings, and candidate simulator setup |
| [Training recovery](07-training-recovery.md) | Power-loss recovery, checkpoint contents, atomic writes, and restart semantics |
| [Experiments and artifacts](08-experiments-and-artifacts.md) | Run records, metrics, graphs, submission figures/tables, and simulation playback |

Begin the conceptual reading at [Project overview](../learning/01-project-overview.md). The [learning index](../learning/00-README.md) will grow with implementation. [Project working agreement](../AGENTS.md) records ongoing development requirements.

## Decisions already made

- OmniDrones is the target simulator. uv is the Python project manager.
- Complete the retained student-system capabilities before making paper-03 comparison the primary objective.
- Include ground-to-formation construction, multiple 3-D templates, bounded local observations, waypoint navigation, correct recurrent learning, and commanded in-flight shape changes.
- Include a reproducible neighbor-budget study and an explicit communication cost model. Do not assume six neighbors is a proven optimum.
- Develop static/dynamic obstacles, GRU comparisons, advanced communication scheduling, and multiple-group split/merge behavior as extensions with their own evidence.
- Log and export results for every experiment. Recoverable checkpoints are required before long training.
- Explain implementation changes in topic-based learning documents. Installation and usage must be understandable by another researcher.
- Keep plans/ and learning/ available when transferring work; the current checkout tracks them. Keep operational documentation in tracked docs/ when the corresponding functionality exists.

## What must be resolved with evidence

| Question | How to resolve it |
|---|---|
| Which OmniDrones/Isaac Sim/Python versions? | Validate a version pairing on the lab machine and pin the exact revision/build |
| Lab hardware allocation/runtime access? | Inventory received: five A4000 GPUs with approximately 16 GiB each and 503 GiB host RAM; confirm available allocation, external simulator installation, and GPU-container access |
| Can simulator state be fully restored? | Test supported state serialization, including controllers and internal dynamics; select and label the proven recovery mode |
| Can the original submission runs be reconstructed? | Recover the student's actual run configurations, evaluation scripts, and checkpoints if available |
| Which metric thresholds define success? | Choose numerical values after basic calibration, before comparison experiments; freeze them in evaluation configuration |
| How does each actor obtain relative positions and targets? | Define the measurement/reference-frame assumptions and prohibit undeclared global inputs |
| Does neighbor selection save transmissions? | Specify sensing, discovery, message delivery, and broadcast/unicast accounting |

The lab diagnostic received on 2026-09-15 identifies five RTX A4000 devices with 16,376 MiB each, approximately 503 GiB of host RAM, and NVIDIA driver 580.173.02. These are separate GPU memories, not one pooled allocation. See the [lab runtime assessment](06-lab-runtime-assessment.md) for exact evidence and remaining compatibility checks. The student's documented RTX PRO Blackwell 4000 machine is different hardware.

## Completion criteria

A capability is complete only when its implementation, meaningful checks, public usage instructions, learning explanation, and result artifacts are available. Mark separately whether it is implemented, checked on CPU, checked in OmniDrones, or experimentally supported.

Successful training behavior and published numerical results cannot be promised in advance. Report measured outcomes, including negative results and unsuccessful experiments.

## Updating this plan

Maintain a dated decision record below when scope or assumptions change. Record evidence and affected configuration/metric versions. Preserve earlier experiment identities.

| Date | Decision | Evidence/status |
|---|---|---|
| 2026-09-15 | Adopt the scope and documentation/recovery requirements in this directory | Agreed direction from the project discussion; runtime validation pending |

Sources: [student submission](../../papers/2026_Distributed_MARL_Submission.pdf), [my-mappo](../../my-mappo/README.md), [paper-03 repository](../../multi-UAV-formation/README.md), [NVIDIA RTX A4000 specifications](https://www.nvidia.com/en-us/products/workstations/rtx-a4000/).

Implementation record, 2026-09-15: completed the runnable diagnostic foundation and its topic-based learning explanation; retained simulator/version selection and real-GPU checks as unresolved work.

Current handoff, 2026-09-16: [verified simulator result and next lab task](../docs/06-lab-handoff.md). Start with a pinned OmniDrones environment and controlled single-drone/reset checks, not policy training.

Runtime continuation, 2026-09-16: the [pinned OmniDrones integration](../docs/07-omnidrones-runtime.md) and [deterministic controller check](../docs/08-single-drone-control.md) now exist. Read their current evidence before interpreting the older status paragraphs above. The shared package is Python >=3.10 compatible while host orchestration stays on 3.12.
