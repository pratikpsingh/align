# Understanding training records, recovery, and simulation playback

Status: conceptual guide to the planned implementation. The examples below are arithmetic illustrations, not measured ALiGn results.

## What a training run produces

A run is one identified experiment with a particular configuration, source version, runtime, and seed. During training it collects rollouts, updates the networks, and records progress. It should produce a usable saved policy, checkpoints for resuming, raw measurements, and a report even if the experiment fails.

One run may have several attempts if the process restarts. The run identity explains the scientific experiment; attempt identities explain interruptions and compute actually consumed.

## Counting experience correctly

Imagine 8 parallel environments, each containing 8 UAVs. Collect 240 policy steps from each environment, with all agents active throughout.

- Environment transitions: 8 times 240 = 1,920.
- Agent transitions: 8 times 8 times 240 = 15,360.
- At 30 policy decisions per second, each environment contributes 8 seconds of simulated flight.

These are three descriptions of the same rollout, not interchangeable training budgets. Wall-clock collection time must be measured. Faster hardware changes elapsed time; it does not change the mathematical number of transitions.

The actual project frequencies and parallel-environment count will be configuration values. They need not equal this example.

## A model file is not a full checkpoint

The policy weights are the learned numerical parameters. For inference, the policy also needs the correct observation normalization, architecture, memory shape, and action interpretation.

Continuing training needs more: the critic, optimizer state, schedules, normalizers, randomness, progress counters, and any task/curriculum state. An optimizer may keep moving averages from previous gradients; loading only weights throws away that learning state.

Continuing an unfinished flight requires even more physical and controller state. Therefore the project will distinguish training resume with fresh episodes from verified full-state continuation. Both are different from warm-starting a new experiment with old weights.

## What happens after a power cut

Suppose the last valid checkpoint contains completed update 120. The process completes updates 121 and 122 but loses power before another checkpoint is committed. Recovery starts from update 120. The later work may remain in logs, but it is not part of the resumed learner state and must be labeled accordingly.

If recovery resets the environments, it must also reset episode memory and partial rollout state consistently. Restoring an LSTM's memory from an old flight into a newly reset world would give the policy misleading history.

Atomic writes and keeping older checkpoints protect against a half-written newest file. Backups to separate storage protect against a different problem: losing the disk or machine. Neither can save work that was never successfully persisted.

The proposed initial saving target is every five active minutes at safe update boundaries, adjustable after timing measurements. It is not a promise of zero lost work. The detailed contract is in [training recovery](../plans/07-training-recovery.md).

## Reading a training graph

Ask what the horizontal axis counts: environment transitions, agent transitions, updates, or wall time. Ask whether the curve is training reward or evaluation performance. Ask how many independent trained policies contributed to the curve and what its uncertainty band means.

A smooth reward curve may be an average over episodes or a moving average over time. Keep raw data and label smoothing so short failures are not hidden. Comparing methods requires compatible axes, task conditions, and metric definitions.

Formation-error normalization matters. Suppose every drone has an aligned residual of 0.1 m. Squared error per drone is 0.01 square metres. Eight drones give SSE=0.08; sixteen give SSE=0.16. Mean squared error remains 0.01 and RMSE remains 0.1 m. Doubling this sum did not mean each drone became less accurate.

Collision statistics also need definitions. Ten consecutive contact frames may describe one sustained collision. Save both event count and duration where useful, and always report the fraction of episodes affected.

## Seeds and uncertainty

A training seed controls random choices such as initialization and sampled experience. Different training seeds produce independently trained policies. Evaluation seeds select test scenes or other stochastic conditions for a frozen policy.

Running 600 evaluation episodes for one policy gives evidence about that policy's behavior over those episodes. It does not replace training multiple policies to understand learning variability. Reports must state both counts and use an uncertainty calculation appropriate to the experiment.

## Time, memory, and file sizes

Total training time includes collection and optimization, and may also include evaluation/checkpoint overhead. Save component timings so a slower run can be explained. Resume records should include time spent on work later lost, since the GPU still performed that work.

Keep these sizes separate:

- Source-file bytes and line counts describe code footprint.
- Actor parameter count and weight bytes describe the policy representation.
- Checkpoint bytes include extra training state.
- Runtime working memory includes activations, buffers, simulator data, and other allocations.
- Simulator/dependency installation size describes storage required to install the system.

A smaller Python file is not necessarily a smaller or faster policy. A small exported model may still need substantial runtime memory. Measure the quantity that matches the research claim.

## Making a simulation video later

A trained policy can be loaded into the compatible simulator and evaluated again, with video recording enabled. This creates a new flight. Save the scene, policy/preprocessing configuration, and evaluation seed so its conditions are known.

A recorded trajectory can also be replayed or plotted without retraining. Save positions, orientations, timestamps, targets, obstacle state, and identities during selected evaluations. The laptop can play an exported MP4; a lightweight 3-D trajectory visualization can often run without the GPU simulator.

These outputs answer different questions. A fresh evaluation shows what the frozen policy does in a new simulator rollout. Trajectory playback presents recorded motion. Replaying actions alone may not exactly reproduce an old flight because simulation can diverge.

Scalar rewards and final positions are insufficient to reconstruct a complete video. This is why trajectory recording must be designed before the interesting experiments are run.

## What to inspect after a run

Start with its status, resolved configuration, checkpoint identity, and episode summary. Then inspect success/collision/formation/navigation metrics and their distributions. Use reward and optimizer diagnostics to investigate why learning behaved that way. Finally check representative videos and resource measurements.

Reports should link these records together rather than requiring a researcher to guess which checkpoint produced a graph. See [experiments and artifacts](../plans/08-experiments-and-artifacts.md) for the planned directory structure and measurement definitions.
