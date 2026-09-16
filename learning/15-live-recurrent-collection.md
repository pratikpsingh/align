# Live recurrent collection

## What we built

The collector is the bridge between simulation and learning. At every task
step it sends each drone's local observation through the shared actor, sends
the environment's global state through the centralized critic, executes the
bounded actions, and stores the resulting transition.

The accepted check ran four simulated worlds with four drones each for 128
steps. That gives 512 environment transitions and 2,048 agent transitions.
The policy was randomly initialized from seed 29 so this run tested data
correctness rather than flight skill.

## Two different recurrent batches

The actor and critic do not have the same batch structure:

~~~text
actor memory:  environment x drone x LSTM state
critic memory: environment         x LSTM state
~~~

The actor needs separate memory for every drone because their observation
histories differ. The critic produces one cooperative team value, so it needs
one memory lane per simulated world. Actor parameters are still shared across
all drones.

## Why the final observation matters

Imagine an episode stops only because it reached its 80-step limit. The world
was not necessarily in a physically terminal state. The return estimate may
therefore use the critic value of the final observation.

Resetting first would replace that final observation with the next episode's
initial state. The stored value would then describe the wrong state. ALiGn
evaluates the final state first, stores its bootstrap value, and only then
resets the environment and its LSTM memories.

A real terminal event is different. The accepted check deliberately moved one
environment outside its safety boundary. That transition stored zero bootstrap
value because there is no continuation beyond a true termination.

## What partial reset isolation means

At step 30, only environment 1 terminated. Environments 0, 2, and 3 continued.
A vector simulator must reset one lane without modifying the others.

The check saved the unaffected observations, critic states, and actor/critic
memories around each reset. Every measured difference was exactly zero. The
completed lane's hidden and cell states were zero before the next policy call.

## From time-major storage to training chunks

The device buffer first records a continuous time-major rollout. It then cuts
each actor trajectory and critic trajectory at episode boundaries and at the
configured maximum chunk length.

Four actor trajectories exist for every critic trajectory. In the accepted
run this produced 132 actor chunks and 33 critic chunks. Every chunk stores the
memory present at its first timestep and a mask identifying real rows versus
padding.

The device chunk metadata matched the independent CPU implementation. Running
the policy again over the chunks reproduced stored actor log probabilities and
critic values within floating-point tolerance.

## Why compare CUDA with a CPU reference

The CUDA implementation is fast and uses preallocated tensors, but tensor
layout errors can still produce plausible numbers. The pure Python reference
is slower and simpler. Comparing the two asks whether they implement the same
math from separate representations.

The largest GAE difference was about 0.000000343 and the largest return
difference about 0.000000358. Those small differences are consistent with
32-bit floating-point arithmetic. Exact metadata comparison also established
that both implementations split episodes at the same timesteps.

## The axis-order failure we found

Recurrent critic memory is stored with axes:

~~~text
time, layer, environment, hidden
~~~

The episode-boundary mask has axes:

~~~text
time, environment
~~~

The first live attempt put the layer axis first before applying the mask, so
the shapes could not match. The corrected check swaps the layer and environment
axes, producing `time, environment, layer, hidden`. A regression test
now checks that exact operation.

The rollout itself had completed correctly. Keeping the failed run made the
distinction visible instead of hiding it.

## What the accepted run establishes

The accepted run shows that:

- live local observations reach the actor and global states reach the critic;
- sampled actions stay inside the controller contract;
- stored log probabilities and values can be reproduced from sequence chunks;
- true termination and time-limit truncation use different bootstrap rules;
- LSTM memory resets at boundaries without affecting other worlds;
- team reward is consistently the mean of per-agent totals;
- device GAE and returns agree with the CPU reference;
- the saved CSV and binary artifacts are complete and hash-verified.

It does not show that the policy learned, that PPO is correct, or that a
random policy completes the formation mission.

## What comes next

The next component consumes these chunks in one recurrent MAPPO optimizer
update. It must apply the valid-row mask to every reduced loss, compare new and
old action log probabilities, clip policy and value changes, and report useful
diagnostics such as approximate KL divergence, clip fraction, entropy, gradient
norms, and explained variance.

Before long training, checkpoint recovery must save both networks, both
optimizers, normalization state if enabled, RNG states, update and environment
counters, immutable configuration, and run lineage.
