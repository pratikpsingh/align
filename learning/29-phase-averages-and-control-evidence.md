# Why formation-phase measurements matter

## The problem

A flight can have a better whole-episode average while failing the behavior we
care about. ALiGn's current evaluation lasts 800 steps: 50 on the ground, 500
in takeoff, and 250 with the formation target active. Combining them hides
what happens after the target changes.

For example, if assigned-position error averages 1.28 m before formation and
1.61 m during formation, the 800-step average is about 1.38 m. A policy could
slightly improve the first part while barely moving toward its formation
slots. The phase-specific test asks whether the last 50 formation steps are
better than the first 50 **within each flight**.

## What the diagnostic computes

The [host analyzer](../src/align/runtime/formation_diagnosis.py) reads each
saved evaluation row, groups it by environment and episode, and uses the
`phase` column to separate ground, takeoff, and formation. It recomputes the
source CSV metrics before publishing an independent report. An environment
reset starts a new trajectory; data from two flights are never treated as one
continuous response.

For each formation episode it compares two 50-step windows:

```text
change in assigned error = mean(last 50 errors) - mean(first 50 errors)
```

Negative is movement toward assigned slots. The same calculation for pairwise
error asks whether the distances between drones improve. The analyzer also
counts separation below the reward's 0.55 m shaping margin and rows where any
logged action component is within 0.02 of its ±1 policy bound. It retains
phase team rewards and the maximum absolute action extremum. These are
observations about the logged policy and drone group, not direct measurements
of rotor saturation or controller tracking.

## What we learned

The accepted 12-update plane specialist had 1.6108 m assigned error in its
first 50 formation steps and 1.6083 m in its last 50: only about 2.6 mm of
improvement. Its pairwise error increased from 0.9155 m to 0.9667 m. The
four-template policy on plane improved assigned error by about 16 mm, while
pairwise error still increased. All 64 evaluated flights timed out. No
formation environment-step had an action extremum near ±1, although typical
maximum absolute actions were around 0.4–0.45. This does not look like simple
policy-bound clipping in the saved actions; it leaves open whether the
controller realizes the commands or the policy commands the right direction.

The data do not include reward-component values, individual positions,
velocities, or commanded and realized controller signals. The next bounded
measurement should add those fields to a frozen-policy flight and audit them
against the current aggregate metrics. Only then can a reward or controller
change be justified. The open questions are kept in the
[tracked evidence register](../plans/09-unresolved-evidence.md).
