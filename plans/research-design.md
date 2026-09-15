# Research design and mathematical decisions

## Policy, memory, and control

Use centralized training with decentralized execution: one shared actor acts for each UAV from its permitted local observation and recurrent state. A centralized critic may use global state during training. Global critic information must not enter the deployed actor through preprocessing or neighbor selection.

Start with the retained LSTM design, correcting temporal training. A local encoder combines self-motion, assigned target/mission information, and masked neighbor features. Keep obstacle encoding extensible. The target representation must distinguish the requested formation and its current transition; sampling different shapes only at reset does not teach commanded changes automatically.

Define the sensing assumptions before claiming decentralization: how a UAV gets its own state, relative positions/velocities, reference orientation, and assigned target. Computing relative features from simulator truth is a useful controlled experiment but does not establish a GPS-free sensing system.

Use a calibrated action/controller interface. The student's velocity-direction/speed action has different physics and limits from body-rate/thrust commands. Any change must be visible in configurations and comparisons. A bounded Gaussian mean does not bound samples; select a correct transformed distribution or explicitly account for clipping and executed actions.

### GRU comparison

GRU is a worthwhile extension. It uses three gates and one recurrent state, whereas LSTM uses four gates and separate hidden/cell states. For the same input/hidden widths, the gate-related parameter count is roughly one quarter smaller; total actor size and actual speed depend on the encoder, implementation, and hardware.

Compare MLP, GRU, and LSTM with the same input/output contracts, task distribution, training sample budget, and seeds. Include both equal-hidden-width and approximately matched-parameter comparisons. Measure task success, error, latency, working memory, and training throughput. Delayed, intermittent, or dropped observations provide a more informative memory test than assuming recurrence must improve a fully observed task.

## Neighbor counts and communication

Treat these as distinct quantities:

- Number of neighbors represented in the policy input.
- Number of candidate neighbors sensed or discovered.
- Number of messages/bytes transmitted and received.
- Information age, packet loss, and communication frequency.

Receiving all messages and discarding most of them at the actor does not demonstrate reduced radio traffic. Likewise, one broadcast reaching multiple UAVs is not automatically equivalent to multiple unicast transmissions. Start with an explicit simulated message-cost model and label its measurements as modeled communication, not measured RF energy or bandwidth.

Selection must use information available locally: sensing, previously received state, or explicitly accounted discovery messages. A global position sort performed invisibly at every step supplies free information. Define whether neighbor features are current sensed measurements or delayed received messages, and mask unavailable entries.

### Interpreting the draft's rigidity argument

For an appropriate three-dimensional distance framework, infinitesimal rigidity requires the rigidity matrix to have rank 3N-6 after accounting for rigid-body motion. Having at least 3N-6 undirected edges is necessary in this setting, but not sufficient. The associated average degree bound is 6-12/N, not a universal rule that every drone must see six neighbors.

For N=8, the count is at least 18 edges and average degree at least 4.5. A cube's 12 side edges do not satisfy that count. Geometry, degeneracy, and edge placement matter. Directed communication graphs and planar configurations need separate interpretation. The count/rank conditions do not prove how many neighbors a learned controller with targets, relative vectors, and memory needs. [Maxwell-count counterexamples](https://arxiv.org/abs/1308.3281) explain why edge counting alone is insufficient.

Use graph theory as a diagnostic and candidate design method. For suitable undirected distance graphs, record connectivity, rigidity rank, and a numerical conditioning margin with explicit tolerances. A range-feasible graph can be greedily pruned while checking those properties; that does not certify a globally minimal communication graph or a control guarantee.

### Empirical budget study

For eight drones, compare k=0 through k=7 where meaningful. k=0 is a diagnostic: target-conditioned agents may accomplish some missions without inter-agent messages under ideal sensing. Specify range, refresh rate, dropout/delay, discovery, and formation geometry for every condition.

Separate two questions: training a policy for each budget (or on a declared mixed-budget distribution), and reducing neighbors only at evaluation for a fixed policy. The latter measures robustness to an input change, not the optimum achievable after training at that budget.

Choose numerical success/error/safety thresholds before confirmatory experiments. Define the smallest acceptable budget as the smallest tested budget satisfying those thresholds with reported uncertainty, for the tested task family. Report a performance-cost curve rather than claiming a universal optimum. Keep collision sensing independent of the communication budget when the assumed platform has a separate safety sensor, and document its range.

Advanced scheduling can optimize communication subject to mission-quality and collision-risk constraints. Learned message compression/selection has prior art, including [IMAC](https://proceedings.mlr.press/v119/wang20i.html); novelty must come from a precise method and evidence.

## Formation changes and group missions

### Commanded single-group transitions

Represent a formation by target offsets around a moving group reference. A command selects a new template, scale, orientation, and transition duration. Assign UAVs to destination slots, normally keeping that assignment fixed during the transition to avoid target switching. Generate bounded target velocities/accelerations and validate minimum separation along the transition.

Interpolation between two safe endpoint templates is not a collision-free trajectory planner. Use assignment, trajectory generation, safety checks, and the policy/controller together. Record transition error, time to settle, collision/near-miss rates, and command failures.

Eight cube vertices already lie on a sphere. A meaningful cube-to-sphere example must use a distinct eight-point spherical arrangement, such as a chosen approximately uniform point set, and verify that its geometry differs from the cube. Eight UAVs form a discrete sample of a sphere, not a continuous spherical surface.

Commanded transitions belong in the retained core because the submission claims in-flight changes. Autonomous decisions about when/why to change shape are a separate extension.

### Multiple groups and reunification

Support explicit membership, per-group reference/targets, and persistent UAV identities. A concrete later mission is eight UAVs in a cube, a distinct eight-point spherical arrangement, two groups of four in tetrahedral formations, then one cube again.

Evaluate formation error within each active group; evaluate collision safety across all UAVs, including different groups. Do not penalize intentional group separation as failure of a single global rigid formation. Specify route separation, membership changes, target reassignment, rendezvous conditions, and reunification completion.

If groups become disconnected, reunion needs prearranged timing/locations or a specified communication mechanism. A centralized mission coordinator issuing group commands is compatible with decentralized low-level execution if declared; it is not evidence of autonomous distributed group planning. Multi-group switching also has prior work, including [distributed splitting/merging research](https://repository.tudelft.nl/record/uuid:58f6730e-10cb-46fe-a7f1-caa80f79e617).

## Reward formulation

Maintain a transparent reward ledger. Each term has a sign, units, normalization, weight, aggregation rule, and a corresponding task metric. Log both unweighted values and weighted contributions. Reward is an optimization signal; success, collisions, formation error, communication, and timing remain independent outcomes.

| Component | Intended behavior | Safeguards |
|---|---|---|
| Goal progress | Advance toward the mission goal/waypoint | Avoid rewarding repeated waypoint switching or oscillation |
| Formation | Match assigned offsets or desired pairwise geometry | Normalize for group size and formation scale; do not remove error by dropping communication edges |
| Separation | Maintain a safety margin and penalize actual collisions | Distinguish soft distance violations from simulator contacts and distinct collision events |
| Smoothness | Reduce abrupt commands/physical actuation changes | State action units and timestep; record executed action/rotor behavior as well as policy output |
| Flight stability | Limit unsafe tilt, speed, altitude, or saturation | Calibrate against controller/dynamics rather than arbitrary penalty magnitudes |
| Completion/time | Finish promptly and settle | Award completion once after dwell; prevent infinite hovering from becoming attractive |
| Communication | Reduce a defined message/byte cost | Account for discovery and update rate; do not confuse input sparsity with radio savings |

For pairwise distance objectives, use errors such as (distance(i,j)-desired_distance(i,j)) squared. Minimizing distance(i,j) squared alone encourages collapse. The draft's MPC expression needs this distinction; MAPPO does not become MPC because an MPC-style objective is written beside it.

Normalize geometric errors by an explicit length scale and average over the declared agents/pairs. A sum over all agents changes with swarm size even when individual performance is identical. Use a fixed evaluation relation/template independent of the selected communication graph, so dropping an edge cannot hide formation error.

Potential-based progress shaping can use F=gamma*Phi(next_state)-Phi(state). Its policy-invariance conditions require the appropriate state and terminal treatment; mission mode, group membership, and changing goals must be represented. Do not assume an arbitrary distance bonus inherits this guarantee. See [Ng, Harada, and Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

Begin with interpretable normalized terms and modest, recorded weight sweeps. Later compare constrained formulations for safety/communication; [constrained policy optimization](https://proceedings.mlr.press/v70/achiam17a.html) is relevant prior work, not an automatic safety guarantee for a different implementation.

## Obstacles and eventual benchmark comparison

Environmental obstacles require explicit static geometry, dynamic motion, observation/sensing assumptions, collision handling, and scene seeds. Add simple fixed obstacles before combining moving obstacles, shape changes, and splitting. Keep task difficulty and training exposure documented.

When paper-03 comparison begins, simulator choice alone is insufficient for fairness. Match or explicitly account for dynamics/controller, number of drones, obstacle distribution, horizon, success definition, training/evaluation budgets, and information available to each actor. Its Laplacian formation metric and successful-episode filtering differ from the student's Procrustes metric; retain both if needed and disclose the populations being averaged.
