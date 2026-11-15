# LP Fallback Validation Report

Experimental validation only. Production gate/fallback/Twin/RL/eval artifacts were not modified.

## 1. Scope

Five previously inconclusive-then-proven-feasible UsCarrier cases at scale 0.25:

- `seed=101|topo=UsCarrier|split=val|mid=26|li=6|scale=0.25|policy=improved_stable`
- `seed=202|topo=UsCarrier|split=val|mid=24|li=4|scale=0.25|policy=improved_stable`
- `seed=303|topo=UsCarrier|split=val|mid=21|li=1|scale=0.25|policy=improved_stable`
- `seed=404|topo=UsCarrier|split=val|mid=23|li=3|scale=0.25|policy=improved_stable`
- `seed=505|topo=UsCarrier|split=val|mid=27|li=7|scale=0.25|policy=improved_stable`

Traffic matrices identical across cases: **True**.
Cross-case LP result consistency: **True**.

## 2. Formulation verification

- Statement: `min U_max s.t. sum_{p in P_d} f_p = demand_d; sum_p A_{e,p} f_p <= U_max * cap_e; f_p >= 0; U_max >= 0`
- Demand vector aligns with `od_demands`: **True**
- Capacities align with Twin topology tensors: **True**
- Incidence aligns with bipartite tensors: **True**
- Path count aligns: **True**
- All checks passed: **True**

Independent load recomputation uses both incidence aggregation of LP flows and Twin `action_link_load` on the derived path fractions.

## 3. Results

| Method | MLU | Overloaded links | Total overload | Feasible |
|---|---:|---:|---:|---|
| Production fallback (shortest) | 1.870000 | 2 | — | False |
| Shortest | 1.870000 | 2 | 159.750 | False |
| Equal | 2.452500 | 20 | 696.354 | False |
| Path-rebalancing | 1.353427 | 11 | 99.830 | False |
| **LP catalog fallback** | **0.785000** | **0** | **0.000000** | **True** |

- LP solver status: **optimal** (gap=0.0)
- LP runtime (shared solve, reported per case): **6.007 s**
- Mean reported LP runtime across five rows: **6.007 s**
- Twin recomputed MLU from fractions: **0.785000**
- Max demand residual: **0.0**

All five case rows share the same LP MLU / feasibility / baseline comparisons (5/5 feasible, 5/5 beat all baselines).

## 4. Direct answers

1. **Does LP fallback consistently achieve MLU below 1?** **Yes — 5/5** (MLU=0.785000).
2. **Does it outperform all existing fallback candidates?** **Yes — 5/5** (vs shortest 1.870, equal 2.453, rebalance 1.353).
3. **Computational cost?** About **6.01 s** per UsCarrier catalog LP solve on this machine (HiGHS, ~98k paths, 24.8k demands, 378 links). Far above a constant-time shortest/equal evaluation.
4. **Technically justified as an additional fallback mechanism?** **Yes, for this narrow slice:** when a catalog-feasible path-split exists (forced-link bound ≤ 1), LP recovers safe routing where shortest/equal fail. It is **not** justified as a universal replacement — 35/40 prior failures remain catalog-infeasible and LP cannot clear them.
5. **Limitations?** (a) Helps only catalog-feasible TMs; (b) ~seconds of solve time vs microseconds for heuristics; (c) requires a reliable LP stack and catalog incidence; (d) continuous splits may need discretisation for some routers; (e) does not expand the path catalog.
6. **Smallest next experiment?** Offline A/B on the full 80-case paired set: run experimental LP fallback **only when** forced-link LB ≤ 1 (or production fallback is infeasible), else keep shortest/equal — report applied violation rate and p95 latency. Still no production wiring.

## 5. Artifacts

- `results.csv`
- `routing_allocations.json`
- `comparison_plot.png`

## 6. Preservation

No changes to `safety/gate.py`, `fallback.py`, `gate_logic.py`, checkpoints, datasets, traffic matrices, or A–F evaluation results.
