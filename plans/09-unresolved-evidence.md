# Unresolved evidence and capability questions

Updated 2026-09-18 IST. This register tracks concrete gaps after accepted ALiGn runs. A
passing integration check and a supported scientific claim are different states.
Keep old run IDs and negative results when resolving an item. The
[roadmap](02-implementation-roadmap.md) tracks implementation dependencies;
this register names the **next observation that would close each gap**.

| ID | Status | Evidence now | Required resolution |
|---|---|---|---|
| F01 Formation target convergence | Open, highest priority | Matched run comparison `20260918T115133.727696IST-e9186486`: 0/64 completed evaluation episodes succeeded. Phase diagnosis `20260918T120338.089896IST-b257083e`: final plane-specialist formation error 1.6108→1.6083 m over the first/last 50 formation steps, far above 0.10 m tolerance. | Record per-agent target, world pose, velocity, and command/realized motion through the target switch; identify and test a corrected control, reward, or task mechanism. Require frozen-policy success over independent seeds before calling C03/C05 learned. |
| F02 Shape and separation quality | Open | Final plane-specialist pairwise error 0.9155→0.9667 m during formation; 35.4% of its formation environment-steps have minimum separation below the 0.55 m shaping margin. No terminal separation failure occurred. | Diagnose spatial trajectories and reward/safety trade-off; improve pairwise error without increasing collisions or near misses. |
| F03 Reward contribution visibility | Open | Evaluation CSV has total team reward but lacks per-component formation, tracking, progress, separation, and smoothness contributions. Source computes these components, but their influence on these frozen flights cannot be audited from saved evaluations. | Save per-component weighted and raw reward values by phase and template; recompute totals from raw rows; run a controlled formation-weight ablation for C02. |
| F04 Controller response visibility | Open | Final plane-specialist environment-step maximum absolute policy action averages 0.415, with no action extrema within 0.02 of the ±1 bounds. Saved evaluation lacks commanded and realized velocity/rotor/control state. | Record policy action, commanded velocity, realized velocity, controller saturation and reset state per drone; test whether commands move the correct target-error axis. |
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
