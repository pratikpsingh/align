# Unresolved evidence and capability questions

Updated 2026-09-18 IST. This register tracks concrete gaps after accepted ALiGn runs. A
passing integration check and a supported scientific claim are different states.
Keep old run IDs and negative results when resolving an item. The
[roadmap](02-implementation-roadmap.md) tracks implementation dependencies;
this register names the **next observation that would close each gap**.

| ID | Status | Evidence now | Required resolution |
|---|---|---|---|
| F01 Formation target convergence | Open, highest priority | Matched training comparison: 0/64 completed episodes succeeded. Telemetry replay `20260918T130125.401428IST-bee09d21`: 1.21–1.31 m altitude deficit at formation entry and only 0.25 m commanded travel before the 200-step dwell deadline. Matched timing run `20260918T132216.245829IST-f1dbf4b4`: first 8,800 drone rows identical, extended entry height improved, but formation RMSE worsened 1.582→1.779 m and both arms had 0/4 successes. | Run the prepared target-directed reference-controller comparison under the same action limit and safety thresholds; distinguish learned action selection from task/controller feasibility. Require independent-seed frozen-policy success before calling C03/C05 learned. |
| F02 Shape and separation quality | Open | Final plane-specialist pairwise error 0.9155→0.9667 m during formation; 35.4% of its formation environment-steps have minimum separation below the 0.55 m shaping margin. No terminal separation failure occurred. The longer frozen-policy timing arm had formation minimum separation 0.470 m and mean pairwise RMSE 1.504 m, both worse than its baseline. | Diagnose spatial trajectories and reward/safety trade-off; improve pairwise error without increasing collisions or near misses. |
| F03 Reward contribution visibility | Measured; causal test open | Accepted telemetry has raw and weighted component rows with exact reward closure. Formation-phase mean weighted tracking `−0.02502` versus formation `−0.004925` per drone step. | Run a controlled formation-weight ablation after the target-directed reference-controller feasibility test; check error and safety, not reward alone. |
| F04 Controller response visibility | Measured; broader control test open | Formation mean commanded speed `0.0953 m/s`, realized speed `0.0861 m/s`; only 45.775% of commands point toward assigned targets. Rotor clamp audit passed; no raw rotor clipping in this replay. | Compare commanded versus realized axes and transient response under deterministic interventions; do not infer full controller calibration from one frozen policy. |
| F05 Matched-learning quality | Open | Equal-total-update 12-update plane and four-template arms passed exact-source comparison, but all episodes timed out. Plane specialist saw 36,864 plane rows/seed versus 9,216 for the generalist. | After F01–F04, rerun frozen-policy evaluation with more seeds and explicit per-template success, completion, error, and safety; distinguish equal-compute from equal-plane-exposure questions. |
| C04 Travel maintenance | Pending implementation | Current target task is ground-to-fixed-formation, without route travel. | Command a moving group goal and measure error/safety throughout travel. |
| C06–C07 Communication | Offline proxy implemented; live study pending | Local masks and hard range pass. CPU audit `20260918T165720.534716IST-80f3f426` reconstructed 2,210 snapshots and a declared 40-byte unicast/broadcast packet proxy across budgets 1/2/3/7. The source flight was unchanged; these are not radio measurements. | Record topology and message age during live policy evaluation, define acquisition/scheduling, and run matched budget/range sweeps with flight quality, proxy cost, and any measured traffic labeled separately. |
| C09 Waypoints | Pending implementation | No complete advancing waypoint mission. | Implement waypoint progression and compare direct/waypoint runs over declared distances and seeds. |
| C10 Swarm-size generalization | Pending evaluation | Four-drone policy/task acceptance; no 8/16/32/64/128 evidence. | Establish feasible templates and fixed actor input contract, then log success, failures, memory and throughput for each size. |
| C11 In-flight shape changes | Pending implementation | Reset/update-time template selection is implemented; no commanded mid-episode transition. | Command a shape change in one flight, measure transition time, error, and safety. |
| C12 Policy export | Pending implementation | No standalone actor artifact or measured latency/memory. | Export actor, normalizer and recurrent-state contract; measure bytes, parameters, latency and working memory. |
| R01 Reproducible research release | Open | Raw runs are ignored by Git; rendering/video and independently usable frozen example are not yet validated. | Back up source runs and build context, regenerate each plot/table from raw data, validate fresh setup and selected video path. |

Extensions—obstacles, alternative memory architectures, adaptive communication,
multiple groups, reward optimization, and paper-03 comparison—remain separate
from these core closures. Do not count a negative result as a software failure;
keep it as evidence and update the next test.
