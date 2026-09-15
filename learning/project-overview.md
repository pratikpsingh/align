# Understanding the project

## What ALiGn is trying to do

Several UAVs must reach useful positions and travel together while maintaining a requested shape and avoiding collisions. Each UAV has limited information about the others. We want to understand how much communication and memory are needed to do this reliably, including later obstacles and group changes.

The student's repository contains pieces of a recurrent multi-agent reinforcement learning system in PyBullet. The submission claims additional capabilities and experiments that are not fully traceable to that checkout. ALiGn will implement the retained system in OmniDrones, correct known inconsistencies, and create evidence for what actually works. The [claim register](../plans/scope-and-claims.md) distinguishes existing ideas, gaps, and later extensions.

## Simulator, environment, and controller

A simulator computes how bodies move and interact over time. PyBullet and the OmniDrones/Isaac Sim stack provide different simulation infrastructure and dynamics. Moving between them requires checking action meaning, units, control frequency, reset behavior, and observations.

The learning environment wraps that physics in a task. It provides observations, accepts actions, computes rewards, and decides when an episode ends. An episode is one attempt at a mission, from initialization to success, failure, or timeout.

A low-level controller converts higher-level commands into actuation. For example, a desired velocity is not the same as a motor thrust. If the old policy commands a velocity direction/speed but the new one commands body rates and thrust, the learning problem has changed. That difference must be explicit.

uv manages Python environments and packages. It does not simulate drones, supply GPU hardware, or replace the native simulator runtime.

## What one decision looks like

At a policy decision time:

1. The environment constructs a UAV's permitted observation: its motion, assigned target information, and available neighbor information.
2. The actor, also called the policy, combines that observation with its memory and chooses an action.
3. The controller and simulator advance the physical system.
4. The environment returns the next observation, reward components, and termination/truncation information.
5. Training stores this experience; evaluation records outcomes without changing the policy.

This loop runs for every UAV and may run across many independent environments in parallel. Many simulated worlds accelerate data collection; they are not all members of one enormous swarm.

## What MAPPO means here

Multi-agent PPO trains policies using repeated batches of experience. The actor chooses actions. The critic estimates expected future return and helps calculate whether an action's outcome was better or worse than expected. PPO limits the size of policy changes using its training objective; it does not guarantee collision-free flight.

Centralized training with decentralized execution means the critic may see global information during training while each executing actor uses only its declared local inputs. A shared actor means the same network weights are used by every UAV. Different observations, assignments, and memory states allow those UAVs to choose different actions.

The deployed policy must not secretly depend on the centralized critic or on a global neighbor-selection oracle. The information used before a network is called matters as much as the network's input list.

## Why recurrent memory matters

An LSTM keeps a hidden state and a cell state across decisions. A GRU keeps one recurrent state. These can help infer motion or retain information between intermittent observations. A memory-free MLP sees only its current encoded input unless history is supplied explicitly.

Using an LSTM layer does not automatically mean the training algorithm teaches temporal behavior correctly. Training must preserve ordered sequences, initial memory, and episode-reset masks. Memory from one episode must not leak into another. This is one reason the student's code needs careful correction rather than a simple simulator import change.

GRU is a reasonable comparison after the LSTM path is correct. Smaller recurrent state or parameter count may help deployment, but task performance and measured latency decide whether it is useful.

## Shapes and targets

A formation template describes desired offsets around a group reference. For a simple example, two target offsets might be (-1,0,0) and (1,0,0) metres. If the group reference moves from (0,0,2) to (5,0,2), the targets move from (-1,0,2)/(1,0,2) to (4,0,2)/(6,0,2).

Keeping the shape and reaching the group destination are different requirements. A perfect shape in the wrong location has low shape error but has not completed navigation. Conversely, drones can reach the destination region in a disordered arrangement.

Changing shape during flight means changing assigned offsets through a safe transition while maintaining control. Starting separate episodes in different shapes does not demonstrate this capability. Splitting into several groups additionally requires membership, separate references, and a reunion plan.

## Seeing a neighbor versus communicating

Suppose one UAV's policy uses its three nearest neighbors. This tells us how many neighbor features enter the policy. It does not tell us whether all seven other UAVs already transmitted messages, or whether a sensor supplied those measurements without radio messages.

Communication research therefore needs a model of discovery, sending, receiving, update rate, and message age. We will measure mission quality alongside that cost. The draft's rigidity argument can inform geometric checks, but it cannot by itself prove an optimal neighbor count for the policy.

## Rewards and results

Reward tells the learner what behavior to prefer. A reward can combine progress, formation, separation, smoothness, and communication terms. Each term needs a clear meaning and visible weight. A formation metric that is only logged, or multiplied by zero, does not directly reward formation learning.

A rising reward curve is encouraging only when independent outcomes improve too. We also need success rates, physical separation/collisions, formation error, travel time, and communication measurements. Changing reward weights can change the curve's scale without improving flight.

## What the laptop and lab are for

The laptop can support code understanding, many mathematical/model checks, report generation, and viewing exported videos. OmniDrones training and native rendering require the selected supported runtime and suitable GPU hardware. The exact lab GPU memory and software compatibility still need verification.

Training learns network parameters. Evaluation tests a frozen network. Rendering makes a visual record. These should be separate operations, so results and videos can be produced after training instead of requiring another full training run.

## Check your understanding

- If all UAVs share one actor, why can they choose different actions? Their observations, assignments, and memory states differ.
- If the swarm has the correct shape far from its destination, has it finished? No; shape and navigation completion are separate criteria.
- If a policy input has three neighbor slots, have we proved only three radio messages were needed? No; acquisition and transmission must be accounted for.
- If an LSTM layer exists in source code, have we verified recurrent learning? No; ordered sequence training and memory/reset behavior must be checked.

Continue with [training, results, and recovery](training-results-and-recovery.md) for the records needed to make these ideas measurable.
