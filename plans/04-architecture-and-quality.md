# Architecture and engineering quality

Status: intended overall design. The src package and align doctor are implemented; other command names and most domain modules below remain proposed. See [current development structure](../docs/05-development.md) for existing files.

## Package structure

Create modules when their behavior is implemented; avoid a tree of empty placeholders.

~~~text
align/
  pyproject.toml
  uv.lock
  .python-version
  README.md
  AGENTS.md
  src/align/
    cli.py
    config.py
    runtime/           # runtime discovery, simulator lifecycle, compatibility
    envs/              # OmniDrones adapter and environment contracts
    formations/        # templates, assignments, geometric errors
    missions/          # takeoff, goals, waypoints, transitions, group membership
    neighbors/         # sensing, selection, masks, messages, cost accounting
    models/            # encoders, recurrent actor, centralized critic
    algorithms/        # PPO losses, GAE, normalization
    training/          # rollout buffer, runner, checkpoint/restart
    evaluation/        # suites, episode summaries, comparison protocols
    metrics/           # metric definitions and aggregation
    artifacts/         # run manifests, structured logs, exports
    visualization/     # plots, trajectory playback, simulator video
  configs/
    envs/
    missions/
    models/
    rewards/
    experiments/
  tests/
    unit/
    integration/
    regression/
  docs/                # tracked installation, usage, design, metric references
  plans/               # ignored project planning
  learning/            # ignored topic explanations
  runs/                # generated results; add ignore rule when implemented
~~~

Small modules can initially remain single files. Do not force this tree when a simpler structure has clear ownership. Notebooks may explore results but must not become the only implementation of training or figure generation.

## Ownership and dependencies

The environment owns physics state and reset/step semantics. Mission logic owns target schedules and completion rules. Geometry utilities compute errors without starting a simulator. Neighbor logic defines what can be observed and communicated. Models consume declared tensors; the learner owns optimization. Artifact/report code reads versioned records rather than reaching into mutable runner internals.

Simulator imports must be lazy and confined to the runtime/environment boundary. A CPU user must be able to inspect configuration, test geometry, load supported policy exports, and regenerate plots without importing Isaac Sim. CPU mathematical checks are not a substitute for flight validation.

Avoid copying the entire student framework. Adapt small, understood components with attribution and tests. Keep upstream OmniDrones modifications minimal and pinned; maintain a documented patch/fork only if the public extension API is insufficient.

## Data contracts

- Use explicit shapes: E environments, N agents, K maximum neighbors, T rollout steps, D feature dimensions. State which axes appear in every public buffer/model interface.
- Specify units and frames for position, velocity, attitude, angular velocity, targets, obstacle state, and controller outputs. Record world/body transforms and yaw/reference assumptions.
- Preserve fixed-capacity slots or use an invariant set encoder with explicit validity masks. Zero padding is not evidence that a neighbor exists.
- Separate actor observations from training-only global critic state. Keep agent identity/assignment/group information explicit where required.
- Define action sampling, transformation, scaling, saturation, and log-probability conventions together. Record sampled and executed actions if clipping or controller saturation changes them.
- Distinguish termination, truncation, invalid/padded samples, per-agent inactivity, and environment reset. Define how each affects GAE and recurrence.
- Separate simulation timestep, controller timestep, policy timestep, and metric/video sampling timestep.
- Keep tensors on the selected device in rollout hot paths. Batch host transfers for logging and avoid per-drone Python loops where practical.

## Configuration

Use one validated schema for runtime, environment, mission, model, algorithm, rewards, evaluation, logging, checkpointing, and seeds. Reject unknown keys and invalid combinations. Give configurations a schema version and save the fully resolved values plus a canonical hash.

CLI overrides must appear in the resolved configuration and launch record. No hidden changes based on a hostname, current directory, or GPU count. Derive dependent dimensions explicitly, and validate them before starting a long run.

Swarm size, number of environments, communication range/budget, obstacle distribution, controller limits, formation scale, reward weights, and success thresholds must be visible configuration. Separate debugging configurations from research configurations.

## Proposed command responsibilities

| Command concept | Responsibility |
|---|---|
| align doctor | Inspect runtime compatibility, hardware, paths, assets, and writable output storage |
| align train | Start a run from a validated configuration; explicit resume or warm-start options |
| align evaluate | Run a frozen policy on a versioned suite and save raw outcomes |
| align render | Record a new simulator evaluation or replay an existing trajectory, labeled accordingly |
| align report | Build figures/tables from retained records, without retraining |
| align profile | Measure model, memory, latency, and throughput under recorded conditions |

Exact syntax belongs in public docs only when implemented and tested. Training should use the same library interfaces as evaluation instead of duplicating model loading and preprocessing.

## Logging and errors

Use Python logging for readable console/file messages and structured JSONL events for machine processing. Include timestamps, run/attempt IDs, update and sample counters, severity, and event type. Log resolved settings once, periodic progress at controlled intervals, and failures with actionable context. Do not flood logs with one message per agent per step.

All essential logging and reporting must work offline. TensorBoard is a useful optional view over saved records; an online service must not be required. Do not log unrestricted environment variables, credentials, or tokens.

Fail clearly on incompatible checkpoints, invalid observations, NaN losses, missing assets, or unsupported runtime versions. Narrow exception handling should add context or recover under an explicit policy; it must not silently replace physics, rewards, or model state.

## Checks and maintainability

Use a formatter/linter and focused automated checks. Test behavior with scientific consequences: geometry normalization, radius/budget enforcement, permutation/masking invariants, temporal gradient flow, hidden-state resets, timeout bootstrapping, reward contributions, checkpoint recovery, and seed-level aggregation.

GPU integration checks cover controller motion, contact/reset behavior, observation/action contracts, short training/evaluation, and rendering. Keep them separately selectable from fast CPU checks. Record runtime/hardware for GPU checks.

Use small deterministic fixtures where possible and tolerance-based assertions where floating-point physics requires them. Do not write tests that merely repeat an implementation formula without an independent expected result. Do not rerun expensive full training after a documentation-only change.

Public functions need concise purpose, shape/unit contracts, and relevant assumptions. Explain non-obvious mathematical decisions beside the code and in the matching learning topic. Avoid speculative abstractions, hard-coded paths, sys.path edits, and broad compatibility hacks.

## Versioning and traceability

Record the source revision, dirty-worktree state, dependency lock hash, simulator build/revision, configuration/schema versions, and metric definitions in every run. For uncommitted research code, retain an allowlisted source snapshot or patch with checksums; a Git hash alone cannot identify uncommitted changes. Exclude secrets, virtual environments, and large generated data.

Observation/action/reward/checkpoint schema changes require explicit compatibility decisions. Resume must not silently reinterpret old tensors or reward histories. Preserve old run records and identify a new run when the experiment definition changes.
