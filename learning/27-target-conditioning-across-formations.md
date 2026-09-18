# Target conditioning across formation templates

## The learning problem

A policy that controls only one plane can associate one observation pattern
with one destination layout. A policy for cube, sphere, pyramid, and plane must
know which destination applies now. Otherwise identical drone states could
require different actions with no input that explains the difference.

ALiGn supplies that missing information as each drone's relative assigned
target:

```text
target error = assigned target position - current drone position
```

This is target conditioning. The actor does not need the word `cube`; it needs
the local displacement that tells its drone where to go.

## A small example

Consider a drone at `(0, 0, 1)` m.

- If its plane slot is `(0.5, 0.5, 1.5)`, its target error is
  `(0.5, 0.5, 0.5)` m.
- If its pyramid slot is `(0.125, -0.125, 2.25)`, its target error is
  `(0.125, -0.125, 1.25)` m.

The same shared actor receives different local inputs and can produce different
velocity commands. Other drones receive their own assigned displacements. The
neighbor block remains bounded by range and capacity, so template conditioning
does not give the actor undeclared global swarm state.

## Why assignment stays fixed

Each template starts as an unordered set of slots. ALiGn uses the Hungarian
algorithm at the episode boundary to assign one slot to each stable drone
identity. That assignment remains fixed while the drone tracks its target.
Changing assignment every control step could swap destinations and make the
error appear smaller without demonstrating stable control.

## How the schedule works

Four cloned environments allow one environment per template at the start of
each rollout or evaluation batch. A seed-specific deterministic order decides
which environment initially gets each template. The order rotates at the next
PPO update. Episode resets also rotate a reset environment through the declared
list, so template counts can temporarily differ after partial resets.

The completed-update counter is the schedule position. It is already part of
the learner checkpoint, and the checkpoint task-sampler state stores the next
position as an explicit cross-check. After recovery, physical and LSTM state
restart, while the target sequence continues from the committed update.

This differs from sampling one shape when a process starts. That older pattern
could accidentally train mostly one template and would not make coverage easy
to audit.

## Data flow

1. The host resolves the four-template schedule into `config.json`.
2. The simulator builds target banks for cube, sphere, pyramid, and plane.
3. Each environment selects one template using the deterministic schedule.
4. Ground and takeoff phases retain the validated construction targets.
5. The formation phase uses that environment's assigned template targets.
6. The actor receives relative assigned targets in its existing local vector.
7. The critic receives all assigned targets in its existing centralized state.
8. Raw rollout and evaluation rows record `formation_kind`.
9. Host aggregation produces per-template metrics across seeds and checkpoints.

## How we know the wiring is correct

CPU checks establish that every batch contains all four templates, rotation is
reproducible, target layouts are distinct, all target altitudes fit the safety
envelope, and local actor target features change across templates. Reporting
tests require four per-template rows for every update and evaluation milestone.
The actor and critic input dimensions remain unchanged.

The live acceptance must additionally show that all four templates reach the
formation phase in real OmniDrones rollouts, actions and metrics remain finite,
PPO updates and checkpoints pass, and a frozen actor is evaluated separately on
every template at updates 0, 2, and 4.

Run `20260917T232002.612950IST-6bc40009` met that integration contract. Across
eight optimizer updates, every training CSV contained 768 rows per template,
including 218 formation-phase rows. Across six fresh-process evaluations, every
CSV contained 800 rows per template, including 250 formation-phase rows. This
is direct evidence that each command reached observation construction, live
physics collection, recurrent PPO, checkpointing, and frozen-policy evaluation.

The measurements also show why coverage and learning are different claims.
Mean assigned-position RMSE across seeds improved from 1.4355 m before updates
to 1.3969 m after four updates, while pairwise-distance RMSE worsened from
0.2745 m to 0.2987 m. Every one of the 24 evaluated environment episodes ended
at the time limit. The pipeline can now measure learning per template, but this
short run did not produce a successful formation policy.

## Limits

Template coverage establishes that one actor was exposed to and evaluated on
all four commands. Four PPO updates are an integration probe. Reliable learning
requires a larger controlled training budget, successful episodes, favorable
formation and safety metrics, and comparisons against plane-only or
single-template baselines. A template name in a log is not evidence that the
policy followed that shape.
