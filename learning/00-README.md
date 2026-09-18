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
| [Training stability and policy evaluation](19-training-stability-and-policy-evaluation.md) | Why repeated seeds, post-update diagnostics, longer phase coverage, and fresh-process deterministic evaluation are separate requirements |
| [Critic values and matched optimizer updates](20-critic-values-and-matched-updates.md) | Value targets, clipping, recurrent reconstruction, identical-rollout comparisons, and normalization prerequisites |
| [Learning curves and checkpoint milestones](21-learning-curves-and-checkpoint-milestones.md) | Baseline, midpoint, and final evaluation; exact historical checkpoint selection; and careful trend interpretation |
| [Frozen critic input normalization](22-frozen-critic-input-normalization.md) | Active-slot group statistics, why they freeze before PPO, checkpoint/evaluation behavior, and warmup accounting |
| [Critic distribution shift](23-critic-distribution-shift.md) | Per-update clipping, mean drift, variance ratios, and why frozen warmup statistics can become stale |
| [Normalization denominator floors](24-normalization-denominator-floors.md) | Preventing small warmup variance from over-amplifying critic inputs while preserving raw drift measurements |
| [Segmented training and process restarts](25-segmented-training-and-process-restarts.md) | Splitting one learning lineage across fresh simulator processes without repeating warmup or carrying stale flight memory |
| [Interrupted rollouts and immutable recovery](26-interrupted-rollouts-and-immutable-recovery.md) | Discarding uncommitted live trajectories, copying a verified checkpoint prefix, and preserving the interrupted source as evidence |
| [Target conditioning across formations](27-target-conditioning-across-formations.md) | How one local actor distinguishes cube, sphere, pyramid, and plane commands through assigned-target vectors and a deterministic schedule |
| [Comparing a specialist with a formation generalist](28-matched-training-comparisons.md) | How matched total updates, unequal template exposure, exact checkpoints, and paired descriptive metrics fit together |
| [Why formation-phase measurements matter](29-phase-averages-and-control-evidence.md) | How whole-episode averages hide target-change response, and what action/reward evidence remains missing |
| [From a target to measured motion](30-target-command-and-motion.md) | How to compare a target, actor action, controller command, measured response, and reward contributions |
| [Why formation deadlines matter](31-episode-deadlines-and-frozen-policy-tests.md) | How success dwell limits available time and how a fixed-checkpoint timing test separates schedule effects from retraining |
| [A simple controller as a feasibility check](32-reference-control-and-task-feasibility.md) | Why privileged target-following control helps distinguish task feasibility from learned policy behavior |
| [Neighbor topology and packet cost](33-neighbor-topology-and-packet-cost.md) | Directed local links, declared packet schemes, a worked byte-count example, and why observation count is not radio traffic |
| [Feasible timing and training exposure](34-feasible-timing-and-training-exposure.md) | Why a rollout must reach the formation phase and how to interpret a matched long-schedule probe |
| [Testing a formation-reward weight](35-formation-reward-weight-tests.md) | How one reward weight changes learning, a numerical example, and why flight and safety must be compared separately |

| [Actor artifacts and recurrent memory](36-actor-artifacts-and-memory.md) | Which parts of a training checkpoint a drone needs, how memory resets, and what size/latency measurements mean |
| [Shape changes and path clearance](37-shape-changes-and-path-clearance.md) | Why templates need new fixed-ID assignments and how ideal-path clearance is checked |
| [Contact after a fall](38-flight-contact-after-a-fall.md) | Why flight phase changes the meaning of ground contact and how termination is checked |
| [Waypoint progress and group travel](39-waypoint-progress-and-group-travel.md) | Why waypoints move a shared center, how the group advances, and how saved flight rows verify the route |
| [Speed actions and exploration](40-speed-actions-and-exploration.md) | Why the absolute-value speed scale has redundant signs and how to test a nonnegative policy output |
| [Why a drone remembers that it took off](41-airborne-state-and-contact.md) | Per-drone contact memory, reset isolation, and why current height alone misses a fall during takeoff |
| [What changes when a swarm grows to eight drones](42-scaling-a-swarm-reference.md) | Fixed local actor width, critic capacity, launch geometry, travel deadline, and physical limits |

The [planning index](../plans/00-README.md) contains implementation requirements and research decisions. The runnable package and diagnostic are implemented and checked on the lab. Simulator startup/CUDA, the deterministic single-drone suite, and the four-drone construction suite have passed. Formation geometry, task rewards, bounded local observations, and recurrent rollout semantics are CPU-validated; rewards and observations were audited on the accepted physical trajectory. The cloned vector task passed its one-world and four-world GPU acceptance runs. The shared LSTM actor and centralized recurrent critic, live device collector, masked recurrent PPO optimizer update, and reset-mode learner recovery passed their CUDA contracts. A bounded two-process task-connected training run has passed. Multi-seed stability calibration and fresh-process deterministic evaluation passed on the lab GPU. Matched-rollout calibration selected critic rate 1e-5, and the repeated three-update run kept all six critic updates within value-clip guidance. The recovered ten-update, two-seed checkpoint learning curve passed exact update 0/5/10 evaluation. It showed a small non-monotonic assigned-error improvement, no formation success, and persistently negative explained variance. Semantic critic diagnostics then passed on two seeds, ruled out padding and input saturation, and found velocity variation roughly ten to fourteen times smaller than position and target variation. Frozen active-group critic normalization then passed with separate warmup accounting, checkpoint restoration, twenty total updates, and exact 0/5/10 evaluation. Critic explained variance improved markedly, but velocity statistics drifted into clipping. A calibrated `0.025` denominator floor then passed a two-seed ten-update curve with zero velocity clipping and positive final critic explained variance; formation metrics remained mixed, no evaluation succeeded, and reliable formation learning is not established. A 12-update segmented run then passed three fresh simulator training processes per seed with exact checkpoint/hash and normalization continuity; flight metrics remained non-monotonic and no evaluation succeeded. A deliberate live interruption then stopped seed 41 during update 5 after 64 rollout steps; immutable recovery from checkpoint 4 passed with exact 1-through-12 counters for both seeds, no duplicate updates, one normalization warmup per seed, six exact-checkpoint evaluations, and identical before/after hashes for all source artifacts. The bounded four-template probe then passed two-seed live collection, eight recurrent PPO updates, and six exact-checkpoint evaluations with audited cube, plane, pyramid, and sphere coverage. Assigned-position error improved modestly, pairwise error worsened, and all 24 evaluated episodes timed out, so reliable multi-template learning remains open. A later matched 12-update plane-specialist versus four-template-generalist experiment passed on the same GPU image and exact checkpoint protocol. Its raw audit covered 48 training and 16 evaluation CSVs; all 64 completed evaluation episodes timed out, so the next investigation is formation-phase control and reward behavior rather than a larger unexamined training budget.

## How future explanations will work

Each implemented topic will connect its purpose, mathematical idea, small example, code/configuration, validation, and limitations. Runnable examples will show tested commands and expected outputs. CPU checks and actual flight checks will be identified separately.

Planned topics include training stability and evaluation, communication accounting, experiment analysis, waypoint missions, formation transitions, obstacle sensing, and group coordination. Create these when there is substantive material to explain; do not add empty placeholder lessons.

For each topic, try to answer: what enters the component, what does it compute, what leaves it, and how would we know it is wrong? This keeps the explanation connected to behavior rather than only naming algorithms.

Operational setup/usage documentation will also live in tracked docs/ when implemented so another researcher can use the project without these learning notes.
