# Learning curves and checkpoint milestones

## The question this experiment answers

A single PPO update can tell us whether gradients are finite and whether an
optimizer step is too large. It cannot tell us whether the policy improves with
experience.

A learning curve measures the same policy lineage at several training
milestones. In this experiment, each seed has three snapshots:

~~~text
initialized policy       midpoint policy        final policy
     update 0  ----------->  update 5  ----------->  update 10
        |                        |                       |
 deterministic test       deterministic test      deterministic test
~~~

The evaluations use the same task duration and deterministic action rule. This
makes update count the main changed variable within a seed.

## A small UAV example

Imagine four drones starting on the ground and trying to form a plane. An
untrained actor may complete takeoff but leave the drones about 1.4 metres from
their assigned targets.

After five updates, suppose assigned RMSE becomes 1.2 metres. After ten updates,
suppose it becomes 0.9 metres while minimum separation stays above the safety
threshold. That pattern is more useful than a lower training loss alone because
it measures actual simulated flight behavior.

A different result is also informative. If RMSE stays near 1.4 metres and the
critic explained variance remains negative, the run shows that this small
budget did not produce measurable learning. It does not justify hiding the run
or describing the code as failed.

## Why update 0 is evaluated

Without an initialized-policy baseline, the final metric has no local
reference. A final RMSE of 1.3 metres could be better or worse than the policy
produced by random initialization.

Update 0 answers: what did this exact actor do before an optimizer step? The
checkpoint is created before the first physical rollout, so it preserves the
same initialization used by subsequent updates.

## Why checkpoints must be selected exactly

The checkpoint store normally loads the newest valid checkpoint for recovery.
That behavior is correct when resuming training, because progress should move
forward.

Milestone evaluation has a different requirement. It asks for one historical
snapshot. ALiGn therefore selects a checkpoint by its exact completed-update
counter and verifies its manifest and payload. A missing or corrupt requested
checkpoint fails that evaluation instead of falling to another update.

## Training and evaluation data are different

Training uses stochastic actions to explore. Its batches also come from a
policy that changes after each update. Training reward is useful for diagnosing
the batch but is not a controlled policy comparison.

Evaluation uses deterministic actor means, performs no optimizer update, and
checks that actor parameters remain unchanged. Compare milestone evaluation
metrics with each other. Do not treat stochastic training reward and
deterministic evaluation reward as interchangeable measurements.

## Reading the reported values

The training summary includes:

- post-update value clip fraction, which should remain controlled at the
  selected critic rate;
- explained variance, which indicates whether value predictions track
  differences in observed returns;
- value loss and critic gradient norm;
- policy KL and clip fraction, which show actor update size.

Each milestone summary includes:

- team reward;
- assigned formation RMSE;
- pairwise shape RMSE;
- minimum inter-UAV separation;
- success, safety-failure, and time-limit counts.

A useful learning signal should appear in task metrics across both seeds.
Reward alone is insufficient because reward combines several components.

## Budget and limitations

The default run uses two seeds and ten updates per seed. Each update contains
3,072 environment transitions and 12,288 agent transitions. This is larger than
the optimizer smoke tests but still small for reinforcement learning.

The result can choose the next engineering experiment. It cannot establish
convergence, sample efficiency, superiority, or performance across swarm sizes.
Rendering and video are separate validation tasks.


## Recovery after an evaluation process stalls

Training and evaluation are separate processes. A simulator startup hang during
an evaluation does not invalidate an already committed training checkpoint.
Repeating training would create a new stochastic trajectory and would no longer
answer the question about the original policy lineage.

ALiGn therefore treats recovery like completing missing measurements in a lab
notebook. It verifies the original checkpoints, keeps the source directory
read-only, reuses evaluations that already passed, and runs only the missing
milestones. The recovery report records the origin of each result so the joined
curve remains auditable.

The sudo keepalive solves a different operational problem. Docker commands use
non-interactive sudo after the operator runs `sudo -v`. Long experiments can
outlive that authorization window. Refreshing the existing timestamp lets the
launcher remove a container after a timeout; it does not acquire new privilege
or retain a password.

Run `20260917T001738.329486IST-5f61f50e` demonstrates the distinction. Both
seeds completed training and seed 41 completed all milestone evaluations. Seed
73's first evaluation stalled before the application-started marker. Recovery
run `20260917T103636.493092IST-3c169a39` reused those valid artifacts and added
only the three missing seed-73 evaluations.

## What the accepted curve says

The update-5 policies reduced mean assigned-target error by 4.69% relative to
the initialized policies. At update 10, the reduction was only 3.33%. Reward
followed the same pattern: it improved most at update 5 and then gave back part
of that gain. This is a small learning signal, but it is not monotonic.

Pairwise shape error increased by 4.76% at update 10. All 24 evaluated episodes
ended at the time limit, so no checkpoint demonstrated formation completion.
The critic's explained variance remained strongly negative even though value
loss fell and value clipping stayed at zero. Together, these measurements say
that the pipeline can train and compare policies, while this reward/critic
configuration has not yet produced dependable task learning.

The next experiment should improve how the critic represents and scales its
inputs and returns, then repeat this same bounded curve. Simply increasing the
number of updates would spend more simulation time without resolving the
observed critic diagnostic.
