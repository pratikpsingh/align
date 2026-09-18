# Unresolved evidence and capability questions

Updated 2026-09-18 IST. This register tracks concrete gaps after accepted ALiGn runs. A
passing integration check and a supported scientific claim are different states.
Keep old run IDs and negative results when resolving an item. The
[roadmap](02-implementation-roadmap.md) tracks implementation dependencies;
this register names the **next observation that would close each gap**.

| ID | Status | Evidence now | Required resolution |
|---|---|---|---|
| F01 Formation target convergence | Open, highest priority | Matched comparison: 0/64 completed episodes succeeded. Accepted telemetry replay `20260918T130125.401428IST-bee09d21` logged 12,800 drone steps; all four timed out. At formation step 551, drones remained 1.21–1.31 m below target. The 50 command steps before the latest 200-step dwell start permit only 0.25 m at the declared 0.5 m/s command ceiling; altitude-only assigned-RMSE command-budget floor is 1.013–1.014 m versus 0.10 m tolerance. | Test unchanged-checkpoint timing/phase alternatives first, then controlled policy or controller changes. Require frozen-policy success over independent seeds before calling C03/C05 learned. |
| F02 Shape and separation quality | Open | Final plane-specialist pairwise error 0.9155→0.9667 m during formation; 35.4% of its formation environment-steps have minimum separation below the 0.55 m shaping margin. No terminal separation failure occurred. | Diagnose spatial trajectories and reward/safety trade-off; improve pairwise error without increasing collisions or near misses. |
| F03 Reward contribution visibility | Measured; causal test open | Accepted telemetry has raw and weighted component rows with exact reward closure. Formation-phase mean weighted tracking `−0.02502` versus formation `−0.004925` per drone step. | Run a controlled formation-weight ablation only after the timing feasibility test; check error and safety, not reward alone. |
| F04 Controller response visibility | Measured; broader control test open | Formation mean commanded speed `0.0953 m/s`, realized speed `0.0861 m/s`; only 45.775% of commands point toward assigned targets. Rotor clamp audit passed; no raw rotor clipping in this replay. | Compare commanded versus realized axes and transient response under deterministic interventions; do not infer full controller calibration from one frozen policy. |
| F05 Matched-learning quality | Open | Equal-total-update 12-update plane and four-template arms passed exact-source comparison, but all episodes timed out. Plane specialist saw 36,864 plane rows/seed versus 9,216 for the generalist. | After F01–F04, rerun frozen-policy evaluation with more seeds and explicit per-template success, completion, error, and safety; distinguish equal-compute from equal-plane-exposure questions. |
| C04 Travel maintenance | Pending implementation | Current target task is ground-to-fixed-formation, without route travel. | Command a moving group goal and measure error/safety throughout travel. |
| C06–C07 Communication | Pending study | Local neighbor masks exist; observed neighbors are not radio traffic. | Implement topology/freshness records and a traffic accounting model; run budget/range sweeps with clear proxy-versus-radio labels. |
| C09 Waypoints | Pending implementation | No complete advancing waypoint mission. | Implement waypoint progression and compare direct/waypoint runs over declared distances and seeds. |
| C10 Swarm-size generalization | Pending evaluation | Four-drone policy/task acceptance; no 8/16/32/64/128 evidence. | Establish feasible templates and fixed actor input contract, then log success, failures, memory and throughput for each size. |
| C11 In-flight shape changes | Pending implementation | Reset/update-time template selection is implemented; no commanded mid-episode transition. | Command a shape change in one flight, measure transition time, error, and safety. |
| C12 Policy export | Pending implementation | No standalone actor artifact or measured latency/memory. | Export actor, normalizer and recurrent-state contract; measure bytes, parameters, latency and working memory. |
| R01 Reproducible research release | Open | Raw runs are ignored by Git; rendering/video and independently usable frozen example are not yet validated. | Back up source runs and build context, regenerate each plot/table from raw data, validate fresh setup and selected video path. |

Extensions—obstacles, alternative memory architectures, adaptive communication,
multiple groups, reward optimization, and paper-03 comparison—remain separate
from these core closures. Do not count a negative result as a software failure;
keep it as evidence and update the next test.
