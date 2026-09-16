# Shared recurrent policy and bounded actions

## What we built

A policy turns observations into actions. ALiGn now has the two neural networks needed for centralized training and decentralized execution:

- one shared actor that each drone runs from its own 55 local values and its own LSTM memory;
- one centralized critic that reads the 80-value global training state and estimates the cooperative team's future return.

Sharing means there is one set of actor weights. Four drones create four different sequences and memories, but all four call the same function. This avoids learning a separate controller tied to each drone identity.

## Why the actor and critic see different information

During training, the critic can use all drone positions, velocities, targets, and validity masks. This gives it a clearer view of whether the team is doing well. The actor cannot use that global vector because it must still work when each drone has only local information.

You can picture the separation as:

~~~text
local observation + actor memory -> shared actor -> bounded action

global training state + critic memory -> centralized critic -> team value
~~~

The critic is a training aid. A future exported controller contains the actor, its input preprocessing, and its hidden/cell state contract.

## What the LSTM remembers

At one timestep, an LSTM receives the current feature vector plus hidden state `h` and cell state `c`. It produces an output and the next pair of states. ALiGn processes real sequences in order:

~~~text
x0 -> (h1, c1) -> x1 -> (h2, c2) -> x2
~~~

It does not flatten these into unrelated one-step examples. A zero memory mask before `x1` means `x1` starts a new episode and cannot depend on `x0`. A zero valid mask means the row is padding; it contributes no output and cannot advance memory.

The acceptance check used two histories that became identical after a reset. Their later outputs matched exactly. When the reset was removed, the final outputs differed, which shows that earlier observations really can affect a later action.

## Why `tanh(mean)` was insufficient

Suppose a Gaussian has bounded mean `tanh(mu) = 0.9` but standard deviation `0.6`. Ordinary samples can still be `1.3` or `-1.2`. Bounding only the mean does not bound what the controller receives.

ALiGn first samples an unconstrained number `z`, then computes `a = tanh(z)`. The transformed action cannot leave `[-1, 1]`. This changes probability density: values near a boundary are compressed more than values near zero. PPO therefore needs the Jacobian correction in the action log probability. Without it, collection and optimization would disagree about how likely an executed action was.

The CUDA probe sampled 6,144 scalar action components. The smallest was about -0.990 and the largest about 0.989. Recomputing their corrected log probabilities agreed within `4.768e-6`.

## What full-sequence/chunk equality means

Training divides a rollout into manageable chunks. Evaluating six timesteps at once must give the same result as evaluating the first three, carrying the resulting memory, and evaluating the next three.

Both actor and critic matched exactly in the accepted probe, including final hidden and cell state. This establishes that chunking itself does not change the recurrent computation. It does not yet establish a correct PPO loss or optimizer update.

## Reading the size numbers

The actor has 542,838 trainable parameters. Stored as 32-bit floating-point values, those parameters occupy 2,171,352 bytes, about 2.07 MiB. LSTM weights dominate the total.

This is a precise desktop model measurement. It is not an onboard deployment result. Export formats may add metadata; inference also needs activations, LSTM memory, input buffers, and library workspace. Reduced precision must be tested against the original actions and flight behavior.

## What the accepted run proves

The vendor PyTorch/CUDA run proves:

- the actor and critic accept the intended tensor shapes;
- actor and critic parameters are separate;
- temporal chunks and reset masks behave correctly;
- the shared actor is independent across batch entries;
- stochastic actions are bounded and have consistent finite log probabilities;
- gradients reach every trainable actor and critic tensor;
- CPU and CUDA outputs agree within the declared tolerance.

The probe did not start Isaac Sim and used random tensors. It proves the neural mathematics and runtime compatibility, not flight performance or learning.

## What comes next

A collector must connect four pieces already present in the project: the vector task, local/global observations, the recurrent policy, and rollout storage. It must reset actor memory for only the drones in finished environments, reset one critic memory per finished environment, and evaluate the final pre-reset observation when a time limit needs bootstrapping.

Only after that connection has tensor parity should PPO optimization be added. This order makes a learning failure easier to locate: task transition, memory handling, stored probability, return calculation, or optimizer.
