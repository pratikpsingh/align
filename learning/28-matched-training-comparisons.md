# Comparing a specialist with a formation generalist

## The question

A shared actor can be given a different destination for each drone. The recent
four-template probe showed that cube, sphere, pyramid, and plane targets reached
training and evaluation. It did not show successful formation control. The next
useful measurement is whether more training helps and whether learning four
shapes changes plane performance compared with learning only plane.

## A concrete example

Imagine four training environments. In the specialist arm, all four assign
plane targets. In the generalist arm, one starts with plane and the other three
start with cube, sphere, and pyramid. If both arms get 12 PPO updates, they use
the same total update budget. They do **not** see the same number of plane
examples. The report therefore records how many actual training rows belonged
to plane rather than assuming a perfect one-quarter split after resets.

At checkpoint 8, the analysis pairs seed 41's plane-only evaluation with seed
41's generalist plane evaluation, then does the same for seed 73. The reported
difference is `generalist metric - baseline metric`. For position error, a
positive difference means the generalist had more error on plane. For minimum
separation, a positive difference means more distance between drones; the
metric's meaning matters before calling any difference better or worse.

## Data flow and checks

1. Both arms load the same learning-curve configuration and derived image.
2. The plane command selects only plane targets; the generalist command uses
   all four target templates.
3. Each seed commits updates 1–12 in three four-update simulator processes.
4. Separate frozen-policy processes evaluate checkpoints 0, 4, 8, and 12.
5. The host comparator checks equal seeds, budget, image, source, and every
   resolved setting except the declared schedule.
6. It checks that every expected template has positive formation rows and
   that template row and outcome counts equal the evaluation totals.
7. It writes per-seed plane comparisons, milestone means, actual plane exposure,
   and the generalist's complete template trends.

The comparison code is in
[`template_comparison.py`](../src/align/runtime/template_comparison.py). The
plane launcher uses the same learning-curve runtime as the four-template
launcher; only its default schedule differs. CPU tests reject unequal budgets,
source/image changes, missing template rows, and wrong checkpoint loads.

## Limits

Two seeds give descriptive variation, not a strong uncertainty estimate. The
policies see different targets and therefore different trajectories even when
seed labels match. The 12-update budget is longer than the integration probe,
but success is an observed flight outcome, not a guaranteed result of running
more updates. A full C05 result requires successful, safe frozen-policy
behavior on every shape; the specialist is a reference for plane behavior, not
a proof of generalist superiority.

## What the lab run showed

Both arms passed 24 updates and eight checkpoint evaluations. The comparison
program matched image, source, configuration, seeds, and milestones, then
recomputed metrics from 48 training CSVs and 16 evaluation CSVs. Every one of
the 64 completed evaluation episodes timed out. The plane specialist saw four
times as many plane training rows per seed as the generalist: 36,864 versus
9,216. At the final checkpoint, their plane assigned-position errors were
1.3760 m and 1.3806 m. The 4.6 mm difference is tiny next to either error and
does not mean that the generalist learned the task equally well: neither
policy met the 0.10 m formation tolerance.

For both arms, assigned error improved through update 8 but was worse again at
update 12. Pairwise error grew and drone separation fell during this budget.
This points to a useful next diagnostic: inspect formation-phase trajectories,
reward terms, and actual commanded motion. Spending more updates before
understanding that behavior would make the result harder to interpret.
