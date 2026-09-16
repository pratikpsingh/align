# Recurrent rollouts and time masks

## What a rollout is

A rollout is a time-ordered record of what the policy saw, what it remembered, what it did, and what happened next. For four simulated worlds with four drones each, one environment step contributes sixteen agent transitions.

An LSTM has two memories: a hidden state and a cell state. Saving observations without these states is like saving the pages of a calculation while discarding the values carried from one page to the next. Training could no longer reproduce what the recurrent policy computed during flight.

## Why flattening breaks recurrence

Suppose a drone sees a neighbor moving closer over three observations. A proper LSTM receives them in order:

~~~text
observation 0 -> memory 1 -> observation 1 -> memory 2 -> observation 2
~~~

If all three samples are flattened and presented as independent sequences of length one, every sample starts from an unrelated initial memory. The network exists in the code, but training never teaches it to use history. This was the central temporal problem in the student's implementation.

ALiGn keeps consecutive samples together and records the memory at the start of each training chunk.

## A worked boundary example

Use `gamma = 0.5`, `lambda = 1`, and zero current value estimates.

1. An ordinary step gives reward 1 and has final value 2.
2. The next step gives reward 3 and truly terminates.
3. A new episode reaches a time limit with reward 5 and final value 4.

The final step's delta is:

~~~text
5 + 0.5 * 4 = 7
~~~

It bootstraps because a time limit does not mean the physical task entered a terminal state. Its advantage does not flow into the preceding episode.

The true terminal step has advantage 3 and no bootstrap. The first step includes the next advantage because it is in the same episode:

~~~text
(1 + 0.5 * 2) + 0.5 * 3 = 3.5
~~~

The resulting advantages are `[3.5, 3.0, 7.0]`.

## Why three masks are needed

One Boolean called `done` is insufficient for recurrent MAPPO:

- the bootstrap mask decides whether the final observation has future value;
- the GAE trace mask decides whether an advantage may propagate backward across this boundary;
- the recurrent mask decides whether LSTM memory belongs to the next action.

At a time limit, these are `1, 0, 0`. At a true termination they are `0, 0, 0`. At an ordinary transition they are `1, 1, 1`.

## Episode-safe chunks

Long rollouts are divided into shorter chunks for training. A chunk may start in the middle of an episode, so it needs the hidden and cell state that existed at that exact timestep. It may end early at a terminal boundary. Remaining rows are padding with a zero valid-sample mask.

A chunk must never contain the end of one episode followed by the start of another. Otherwise gradients could teach the LSTM that the new flight continues the previous flight's memory.

Shuffling whole chunks is safe because order remains intact inside each chunk. Shuffling individual timesteps is not safe for recurrent training.

## How this connects to ALiGn

`align.learning.rollout` is the pure Python reference. It stores actor observations separately from centralized critic state, enforces bounded stored actions, validates recurrent resets, computes GAE, and creates reproducibly shuffled chunks.

The checked production dimensions match the accepted OmniDrones vector task. The report command uses a small worked example because the purpose is mathematical verification rather than simulator throughput.

## What this establishes

The current checks establish a correct storage and masking contract. They do not establish that a neural LSTM uses memory, that PPO updates are correct, or that a learned swarm flies well.

The next topic will cover the shared actor, centralized critic, and bounded action distribution. Full-sequence and chunked evaluations must agree when given the same initial recurrent state and masks.
