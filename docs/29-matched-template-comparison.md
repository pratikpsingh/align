# Matched formation-training comparison

## Question and protocol

The four-template integration run reached every shape but had no successful
evaluation episodes. The next experiment asks whether a longer run produces
formation success and how plane performance differs when the same actor must
also serve cube, sphere, and pyramid. A plane-only policy provides a baseline.

Both arms use the same derived simulator image, host source, policy seeds
`41, 73`, 12 recurrent PPO updates per seed, four-update process segments,
identical task/reward/controller/observation/normalization configuration, and
frozen-policy evaluations at checkpoints `0, 4, 8, 12` for 800 steps. Only
`formation_schedule` differs. The host comparator rejects unmatched budgets,
source, image, other resolved configuration, missing templates, inconsistent
row/outcome totals, or wrong checkpoint loads.

This is an **equal-total-update** comparison. With four cloned environments,
the generalist initially devotes one environment to each shape, while all four
plane-baseline environments use plane. Partial episode resets may alter exact
exposure. `training-exposure.csv` reports actual plane rows for both arms; do
not describe this as equal plane-specific training data. Matching policy seed
pairs the reporting rows, but the two arms do not share an identical trajectory.

## Run on the lab

Prepare a new context with the current checkout, then use its printed build ID
for both arms. Check that GPU 0 is allocated and idle before each launch.

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run runs/runtime-build/<new-build-id>

uv run --locked python scripts/run_plane_baseline_training.py \
  --build-report runs/runtime-build/<new-build-id>/report.json \
  --learning-curve-config configs/learning-curve-template-comparison.json \
  --critic-normalization-config configs/critic-normalization-active-groups-floor.json \
  --accept-eula --gpu 0

uv run --locked python scripts/run_multi_template_training.py \
  --build-report runs/runtime-build/<new-build-id>/report.json \
  --learning-curve-config configs/learning-curve-template-comparison.json \
  --critic-normalization-config configs/critic-normalization-active-groups-floor.json \
  --accept-eula --gpu 0

uv run --locked python scripts/compare_template_training.py \
  --plane-run runs/plane-baseline-training/<plane-run-id> \
  --generalist-run runs/multi-template-training/<generalist-run-id>
```

Use the exact run IDs printed by each launcher. Do not rebuild between arms.
Each arm should produce six training containers and eight evaluation containers;
expect 24 committed updates per arm. A failed or interrupted arm remains
separate evidence and must be completed or recovered before comparison. The
comparator reads both run trees and writes a new immutable
`runs/template-comparison/<run-id>/` directory. Its `report.json` records input
hashes, image and source identity, actual training exposure, per-seed paired
plane metrics, milestone summaries, and the generalist's four-template trends.
`paired-plane.csv`, `plane-summary.csv`, and `training-exposure.csv` are the
small raw-derived tables. The input runs and all `runs/` artifacts are ignored
by Git and need separate backup.

## Reading the result

Terminal reason `1` is success; reason `6` is time limit. Success counts are
among **completed** evaluated episodes. Also inspect assigned-position RMSE,
pairwise-distance RMSE, minimum separation, other failures, and each shape's
full checkpoint trend. A lower plane error alone cannot establish that one
policy reliably handles all four templates. Two seeds and this 12-update
budget are a diagnostic comparison, not a significance or superiority claim.

## Accepted lab result, 2026-09-18

The paired 12-update arms passed on GPU 0 with the same derived image
`sha256:1037c868b329605cd104c1f2c1e3d0da6942de896fc5eb3f99dc3497a7e05750`
and source package hash. The [plane run](../runs/plane-baseline-training/20260918T105406.233167IST-d74f8b47/report.json)
and [four-template run](../runs/multi-template-training/20260918T111606.650830IST-7fb24d49/report.json)
each completed 24 optimizer updates and eight exact-checkpoint evaluations.
The final [comparison report](../runs/template-comparison/20260918T115133.727696IST-e9186486/report.json)
passed equal-config, source/image, milestone, coverage, and raw-data audits. It
recomputed 48 training CSVs and 16 evaluation CSVs with no discrepancies.

Per seed, the plane specialist saw 36,864 plane training rows. The generalist
saw 9,216 plane rows out of 36,864 total rows, with 9,216 for each other
shape. Across both arms and all four checkpoints, all 64 completed evaluation
episodes ended at the time limit. There were no successes or other terminal
failures.

At checkpoint 12, the cross-seed plane comparison was:

| Metric | Plane specialist | Generalist on plane | Paired generalist minus specialist |
|---|---:|---:|---:|
| Assigned-position RMSE | 1.3760 m | 1.3806 m | +0.0046 m |
| Pairwise-distance RMSE | 0.3652 m | 0.3609 m | -0.0043 m |
| Minimum separation, mean of seed minima | 0.6013 m | 0.6124 m | +0.0111 m |
| Successes / completed episodes | 0 / 8 | 0 / 2 | — |

The generalist's other final assigned-position RMSE means were cube 1.4025 m,
pyramid 1.3890 m, and sphere 1.3940 m. The task requires 0.10 m formation
RMSE, 0.08 m pairwise RMSE, and a speed dwell; these policies remain far from
success. Their assigned-position errors improved up to checkpoint 8 and then
worsened at checkpoint 12, while pairwise error generally rose and minimum
separation fell. The result supports the comparison pipeline and identifies a
learning/control gap. It does not support superiority, generalization, or
claim C05's successful behavior.

The next investigation should inspect formation-phase trajectories, reward
contributions, target tracking, action saturation, and controller response for
these frozen policies before spending a larger training budget. Report both
training exposure and completed-episode denominators in any later comparison.
