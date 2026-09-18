# ALiGn documentation reading order

Start here, then follow the numbered files in order. The prefixes describe reading order, not implementation stages.

1. [Installation](01-installation.md): prepare the uv environment and understand laptop/lab requirements.
2. [Diagnostics](02-diagnostics.md): collect machine information and interpret the report.
3. [GPU container setup](03-gpu-container-setup.md): prepare Docker and NVIDIA access on the lab machine.
4. [Isaac Sim startup check](04-isaac-sim-smoke.md): test the candidate simulator and save evidence.
5. [Development](05-development.md): understand the package structure, coding workflow, and local checks.

6. [Lab handoff](06-lab-handoff.md): verified simulator result and a detailed agent continuation prompt.

7. [OmniDrones runtime](07-omnidrones-runtime.md): exact source/dependency pins, Python boundary, image build and backup.
8. [Single-drone control](08-single-drone-control.md): frames, actions, controller/reset contract, measurements and lab evidence.
9. [Formation geometry](09-formation-geometry.md): shape templates, assignment, metrics, configuration, and report command.
10. [Multi-drone construction](10-multi-drone-construction.md): group identities, ground takeoff, safety, dwell success, and lab command.
11. [Task reward contract](11-task-reward-contract.md): active formation and safety terms, aggregation, configuration, and trajectory audit.
12. [Local observation contract](12-local-observation-contract.md): bounded actor inputs, radius and budget rules, masks, critic separation, and topology audit.
13. [Vectorized task environment](13-vectorized-task-environment.md): step/reset semantics, cloned worlds, tensor shapes, episode masks, GPU acceptance, and artifacts.
14. [Recurrent rollout contract](14-recurrent-rollout-contract.md): temporal storage, GAE masks, LSTM states, episode-safe chunks, and CPU report.
15. [Shared recurrent actor and centralized critic](15-recurrent-policy-contract.md): policy architecture, bounded action distribution, CUDA acceptance, and usage.
16. [Device recurrent rollout collector](16-device-recurrent-collector.md): live policy/task integration, boundary bootstrapping, partial resets, CUDA/reference parity, and artifacts.
17. [Recurrent MAPPO optimizer update](17-recurrent-ppo-update.md): masked PPO/value losses, gradient clipping, padding invariance, CUDA evidence, and limits.
18. [Training recovery and checkpoint integrity](18-training-recovery.md): atomic publication, complete learner state, fallback, fresh-process loading, reset semantics, and lab evidence.
19. [Bounded task-connected training](19-task-connected-training.md): two-process live collection and PPO, checkpoint lineage, raw metrics, resume audit, and stability limits.
20. [Training stability and deterministic evaluation](20-training-stability-and-evaluation.md): multi-seed update calibration, post-update diagnostics, fresh-process checkpoint evaluation, and artifacts.
21. [Critic scale and matched-rollout calibration](21-critic-scale-calibration.md): value-target distributions, identical-batch learning-rate comparisons, raw samples, and selection guidance.
22. [Bounded learning curves and checkpoint evaluation](22-bounded-learning-curves.md): exact milestone loading, multi-seed trends, budgets, and artifacts.
23. [Frozen active-group critic normalization](23-frozen-critic-normalization.md): warmup statistics, active-slot transforms, checkpoint recovery, evaluation, and acceptance.
24. [Critic distribution drift](24-critic-distribution-drift.md): per-update clipping and scale drift, host-audited tables, and the normalized learning-curve command.
25. [Critic normalization denominator floor](25-critic-normalization-floor.md): historical candidate calibration, schema-2 transform, compatibility, and GPU acceptance.
26. [Segmented training](26-segmented-training.md): fresh-process update ranges, exact checkpoint lineage, normalization continuity, and bounded acceptance.
27. [Interrupted segmented-training recovery](27-interrupted-segment-recovery.md): partial-rollout fault injection, immutable source evidence, checkpoint-prefix recovery, and audit rules.
28. [Target-conditioned multi-template training](28-target-conditioned-template-training.md): deterministic cube/sphere/pyramid/plane scheduling, local target conditioning, and per-template evidence.
29. [Matched formation-training comparison](29-matched-template-comparison.md): equal-budget plane specialist and four-template generalist protocol, commands, audits, and result limits.
30. [Formation-phase learning diagnosis](30-formation-phase-diagnosis.md): CPU-only phase windows, action/separation checks, accepted result, and observability gaps.

For beginner explanations, begin with the [learning index](../learning/00-README.md). For research scope and intended implementation, use the [planning index](../plans/00-README.md). Setup instructions distinguish CPU contracts, implemented simulator behavior, and pending research validation.
