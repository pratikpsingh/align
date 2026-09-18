# Formation-phase learning diagnosis

## Purpose and command

Whole-episode reward or error averages mix 50 ground, 500 takeoff, and 250
formation steps in the current 800-step evaluation. This CPU-only command
separates those phases and measures whether error improves after formation
targets become active. It reads the accepted frozen-policy CSVs; no simulator,
Docker, or GPU is required.

```sh
uv run --locked --extra reporting python scripts/diagnose_formation_learning.py \
  --plane-run runs/plane-baseline-training/20260918T105406.233167IST-d74f8b47 \
  --generalist-run runs/multi-template-training/20260918T111606.650830IST-7fb24d49 \
  --plot
```

The command first applies the strict [matched comparison](29-matched-template-comparison.md)
and recomputes every source CSV. It then writes a unique
`runs/formation-diagnosis/<run-id>/` with `report.json`,
`episode-diagnostics.csv`, `phase-summary.csv`, `phase-trends.pdf`,
`phase-trends.png`, IST log, and JSONL events. `--extra reporting` installs the
pinned plotting dependencies from the project's uv lock; omit `--plot` and the
extra for a table-only run.
Inputs remain unchanged. Keep the ignored run directory with its source runs.

For each environment episode, the analyzer compares the mean of its first and
last 50 formation rows. It reports assigned-position and pairwise-distance
RMSE, phase team reward means, minimum formation separation, fraction of
formation environment-steps below the configured 0.55 m shaping margin, and
fraction with any logged action extremum within 0.02 of the ±1 policy bound.
It also records mean and maximum absolute action extremum. These are
**environment-step extrema across agents/action components**, not per-motor
saturation or realized velocity. Episode splits use each environment's
`episode_step` reset, so separate flights are never joined into one trend.

## Accepted result

Run `20260918T120338.089896IST-b257083e` passed on the two accepted
12-update arms. It audited 48 training CSVs and 16 evaluation CSVs and
summarized 64 frozen-policy episodes without modifying the training runs.
At checkpoint 12:

| Formation episode group | Episodes | Assigned RMSE, first→last 50 | Pairwise RMSE, first→last 50 | Below 0.55 m margin | Near ±1 action bound |
|---|---:|---:|---:|---:|---:|
| Plane specialist, plane | 8 | 1.6108→1.6083 m | 0.9155→0.9667 m | 35.4% | 0% |
| Four-template policy, plane | 2 | 1.6214→1.6054 m | 0.9131→0.9450 m | 6.0% | 0% |
| Four-template policy, cube | 2 | 1.6900→1.6770 m | 0.7357→0.7751 m | 11.6% | 0% |
| Four-template policy, pyramid | 2 | 1.6481→1.6346 m | 0.9512→0.9780 m | 10.0% | 0% |
| Four-template policy, sphere | 2 | 1.6613→1.6543 m | 1.0048→1.0358 m | 4.0% | 0% |

The specialist's mean environment-step maximum absolute policy action during
formation was 0.415 (generalist plane: 0.447). Logged actions are nonzero but
not near the policy bound. Assigned error changes little during formation and
pairwise error grows; this is more specific than the mixed whole-episode
1.38 m assigned RMSE. No terminal separation failures occurred, but the
shaping margin was crossed in many formation steps. Neither arm achieved the
0.10 m formation or 0.08 m pairwise success tolerances.

## What remains unobserved

The saved evaluation CSV has total team reward, not weighted reward components.
It has assigned error and action extrema, not per-agent world position, target,
velocity, commanded velocity, or realized controller output. Thus this report
shows the symptom but cannot identify whether reward balance, policy action,
controller tracking, or a target/frame error causes it. Do not infer a fix from
zero near-bound actions alone. The tracked next measurements and other open
capabilities are in [unresolved evidence](../plans/09-unresolved-evidence.md).
