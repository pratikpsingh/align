# Vectorized task state and resets

## Why an environment needs memory

A reinforcement-learning environment is more than a physics simulator. It turns an action into four outputs:

~~~text
next observation, reward, episode boundary, diagnostic information
~~~

Some calculations depend on the previous step. Progress compares old and new target distance. Smoothness compares old and new velocity commands. Formation success requires several consecutive settled steps. These values form environment memory.

When many worlds run together on one GPU, each world needs its own memory. Resetting world 0 must not erase world 3's progress or carry world 0's previous command into a new episode.

## One drone group per cloned world

The baseline batch has four worlds and four drones per world. A position tensor has shape:

~~~text
[environment, drone, XYZ] = [4, 4, 3]
~~~

The first index selects an independent copy of the scene. The second identifies a stable drone within that group. Operations on these tensors run together on the GPU, which reduces Python and simulator overhead compared with stepping each world separately.

Vectorization does not mean the worlds share an episode. Their counters, targets, rewards, terminal masks, and reset memory remain separate.

The ground is one global collision object shared by every cloned world. ALiGn creates it from Isaac Sim's local GroundPlane primitive. OmniDrones' convenience scene function instead points to a grid file on a remote asset server; an offline run cannot load that file, so it never creates the collision geometry the helper expects. Using the local primitive also matches the ground used in ALiGn's accepted one-world controller probes.

## One transition in order

For a transition at step k:

1. Read the target phase and target positions for k.
2. Decode each bounded policy action to a world-velocity command.
3. Let the Lee controller produce rotor commands and step physics.
4. Measure position, velocity, and contact.
5. Compute reward against the target used during the action.
6. Advance that environment's counter.
7. Decide true termination or time-limit truncation.
8. Build the next observation using the target for step k + 1.

The order matters at the takeoff boundary. Reward should judge the action against the target it was asked to follow. The next observation should show the next target so the following action can respond to it.

## Termination and truncation example

Consider two worlds whose step limit is 100.

- World 0 reaches step 100 safely.
- World 1 crosses the horizontal safety boundary on step 100.

Their masks are:

| World | terminated | truncated | done | reason |
|---|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | time limit |
| 1 | 1 | 0 | 1 | safety envelope |

The distinction matters to a future critic. World 0 might have continued normally, so its final value can be used for bootstrapping. World 1 reached a true task boundary, so future value must be zero for that episode.

## Shaping margin versus terminal distance

The reward safety margin is 0.55 m, while true separation termination occurs below 0.25 m.

At 0.50 m, the drones receive a small separation penalty and can move apart. At 0.20 m, the task declares a true failure. Using one threshold for both would either terminate too early or provide too little warning.

These distances are explicit baseline choices backed by the one-metre formation spacing. They are not claims about airframe collision geometry in every configuration.

## Partial reset example

Suppose counters are:

~~~text
world 0: 40
world 1: 40
world 2: 40
world 3: 40
~~~

Reset only world 0. The result must be:

~~~text
world 0: 0
world 1: 40
world 2: 40
world 3: 40
~~~

World 0 also clears its previous distance, previous command, phase marker, dwell counter, rotor state, velocity, and joints. The other worlds retain all of these. The batch GPU probe compares their full drone states immediately before and after the partial reset to detect leakage.

## CPU reference and GPU implementation

The CPU reference in align.tasks.environment joins the existing pure observation, reward, formation, and schedule functions. It is easy to test with hand-written states and provides the intended semantics.

The simulator implementation in align.simulation.vector_task performs the same calculations with PyTorch tensors on the GPU. The acceptance probe feeds one physical sample into both versions and measures the maximum absolute disagreement for:

- per-agent reward;
- local actor observation;
- centralized critic state.

This parity check catches a fast GPU implementation that has a reversed relative vector, different aggregation, missing phase reset, or wrong normalization.

## What the deterministic probe establishes

A passing probe establishes that the pinned runtime can:

- create one and four cloned Hummingbird groups;
- reset and step them with the velocity controller;
- return finite tensors with the declared shapes;
- preserve independent episode state;
- distinguish true termination from truncation;
- match the CPU observation and reward reference;
- save raw trajectories and resource measurements before shutdown.

The actions are generated by a proportional target follower. They are an environment test, not a learned policy.

## What comes next

The vector task passed GPU acceptance. The recurrent rollout contract stores ordered observations, actions, rewards, masks, value estimates, and LSTM states; its tests preserve sequence order, reset state at episode boundaries, and bootstrap time limits from final observations. Later accepted checks connect the shared recurrent actor and centralized critic to live collection and validate isolated MAPPO optimizer updates. Recoverable task-connected training remains pending.

## Why views must preserve cloned physics

A physics view holds handles to bodies that PhysX has already created. Imagine four drone groups copied from one scene: changing the source object's transform definitions or contact-report schema after those handles exist can cause inherited objects to be rebuilt. A handle may then point to an object that no longer exists.

The failed batch in run 20260916T154136.608722IST-14b3d22a reported exactly that kind of invalidation before its first action. The one-environment scenario in the same run completed 160 steps and passed its parity and reset checks.

The first retry made scene setup explicit: spawn source drones, prepare contact-report schemas, create the ground, clone, start physics, and initialize views while preserving transforms and existing contact schemas. A later run proved that this works for one world, but four worlds still invalidated the tensor view while constructing the base-link view.

A second retry kept the configured stabilization setting and failed at the same boundary, which ruled out that hypothesis. Reading the exact vendor source exposed a subtle rule: asking RigidPrimView to track contacts forces rigid-body preparation even when prepare_contact_sensors is False. That preparation writes a zero sleep threshold to every clone after physics has started.

The corrected order prepares the contact-report API, threshold, PhysX rigid-body API, and sleep threshold on the source bodies. Cloning carries those settings into every world. The ordinary base-link view then handles poses, masses, and forces without contact tracking. A separate RigidContactView reads the already-prepared contact reports without editing the scene. Think of this as wiring the sensors before switching on the machine, then connecting a read-only meter afterward. Only the next native run can confirm that the pinned simulator accepts this order. Every cloned world must also report ground contact, since a sensor returning zeros everywhere would miss collisions even if tensor reads succeeded.
