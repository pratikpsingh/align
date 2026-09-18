# Communication accounting from saved trajectories

ALiGn's local observation contains selected neighbors. A selected neighbor is an observation relationship, not a recorded radio transmission. This CPU-only audit reconstructs the radius-filtered topology from an immutable four-drone trajectory and prices two declared packet schemes. It does not replay the policy.

## Run

From `align/`:

```sh
uv run --locked python scripts/audit_communication.py \
  runs/multi-drone/20260916T121136.980227IST-48963b76
```

The default input contract is `configs/local-observation-baseline.json`; packet assumptions and budgets are in `configs/communication-accounting-baseline.json`. The command writes a new ignored `runs/communication-accounting/<IST-run-id>/` with `report.json`, `topology.csv`, `budget-summary.csv`, and `communication.log`. `report.json` records the source trajectory SHA-256 and both configurations. `topology.csv` lists each selected directed sender-to-receiver edge at each saved snapshot. Copy ignored runs separately when transferring the project.

The current baseline charges six 32-bit position/velocity components plus 16 bytes of declared packet overhead, or **40 bytes per packet**. The unicast proxy charges one packet for each selected directed edge. The ideal broadcast proxy charges one packet per unique selected sender at a snapshot. Both exclude discovery, retries, acknowledgements, contention, routing, and unspecified physical-layer overhead. A snapshot is a saved controller observation, not evidence that a packet was sent at that instant. Neither proxy is measured network traffic.

## Accepted CPU audit

Run `20260918T165720.534716IST-80f3f426` passed on the accepted deterministic four-drone trajectory. It audited 2,210 snapshots and 8,840 agent steps. At budgets 1, 2, 3, and 7, selected directed edges were 8,840, 14,824, 17,380, and 17,380. Under the declared unicast model, these correspond to 353,600, 592,960, 695,200, and 695,200 proxy bytes. The radius stayed at 1.5 m; the budget-3 and budget-7 results coincide because no saved snapshot had more than three other drones. These are **counterfactual costs on one unchanged flight**. They do not measure task quality under a policy trained or evaluated with a different budget.

## Next validation

To complete C06/C07, save topology and message age during actual policy flight, define how state is acquired and how packets are scheduled, then run matched policy evaluations across budgets and ranges. Report observed topology, modeled bytes, and measured traffic separately. The draft's `3N-6` edge count does not prove rigidity or an optimal neighbor count.
