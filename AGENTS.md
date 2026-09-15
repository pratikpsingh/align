# Project working agreement

## Objective

Implement ALiGn in OmniDrones using uv, carrying forward the verified ideas and completing the retained claims in the student's Distributed MARL submission. Correct known mathematical and implementation inconsistencies. Extend the system with obstacle avoidance, GRU comparisons, communication optimization, and formation reconfiguration. Reproducing paper-03's results is deferred.

The local planning index is [plans/00-README.md](plans/00-README.md). The user intentionally ignores plans/ and learning/ in Git. These directories may be absent in another checkout; runtime behavior, installation instructions, and scientific definitions must also be documented in tracked docs/ as they become implemented.

## Development and teaching

- Assume no prior knowledge of OmniDrones or reinforcement learning. Before each implementation increment, explain what will be built, why it is needed, and its observable outcome. Introduce terminology through a small UAV example; afterward connect the code, validation, and result to that explanation. Teach concepts as they become relevant rather than requiring a separate course first.
- Work in small, understandable increments within the user's authorized scope.
- For every substantive implementation increment, create or update a topic-based learning document in learning/. Explain the problem, concepts, data flow, relevant code, a worked example, validation, and limitations.
- Name and write all documents by topic. In docs/, learning/, and plans/, use two-digit filename prefixes for reading order, with 00-README.md as the index, and maintain links when names change. Do not label documents by implementation stage or present them as numbered implementation installments.
- Update tracked setup and usage documentation whenever commands, configuration, dependencies, or behavior change. Clearly distinguish proposed interfaces from tested commands.
- Keep the source repositories and PDFs as references. Do not silently modify my-mappo or multi-UAV-formation while implementing align.
- Preserve attribution and applicable licenses when adapting code. Describe corrected or adapted behavior accurately rather than claiming exact reproduction.

## Engineering

- Use a src/align package, thin entrypoints, explicit configuration, and clear module ownership. Add modules when needed rather than filling the repository with empty placeholders.
- Use uv for Python project/dependency management. Select Python to match the validated OmniDrones/Isaac Sim runtime; the initial Python 3.12 setting is provisional.
- Isolate simulator imports and lifecycle from mathematical utilities, policies, metrics, and report generation. CPU checks must not require Isaac Sim.
- Avoid hard-coded machine paths, sys.path modifications, broad silent fallbacks, and hidden configuration overrides.
- Keep rollout tensors on the selected device where practical. Specify units, coordinate frames, dimensions, masks, and action meaning.
- Verify meaningful invariants: recurrent temporal training, episode boundaries, locality, reward calculations, simulator integration, checkpoint recovery, and metric aggregation.
- Record what was tested locally and what still needs lab GPU validation. Do not substitute mock dynamics for evidence of OmniDrones behavior.

## Research records

- Every run has an immutable identity, resolved configuration, source identity, runtime/hardware metadata, seeds, status, structured logs, and machine-readable results.
- Save raw metrics alongside plots; show sample counts and distinguish training seeds from evaluation episodes. Track failures and interrupted work.
- Recovery must restore training state from a validated checkpoint, preserve run lineage, and state whether simulator trajectories restart or continue.
- Keep evaluation, visualization, and reporting usable after training. Retain a policy export, its preprocessing/configuration, and representative trajectory/video artifacts.
- Retain historical runs. Changes to rewards, observations, actions, or algorithms require explicit experiment lineage.
- No mandatory online logging account or network connection for core logging, saving, or reporting.
- Treat unverified claims as hypotheses. A lower reward loss, fewer observed neighbors, or a smaller source file does not establish better flight, less radio traffic, or deployability.

## Documentation status

This agreement describes requirements. It does not assert that a simulator, training command, checkpoint system, or research result already exists.
