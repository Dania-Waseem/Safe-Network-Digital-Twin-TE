# Final Five-Case LP Feasibility Report

Diagnostic only. Production fallback/gate/catalog/eval artifacts were not modified.

## 1. Cases

Exact inconclusive case_ids from `fallback_investigation/failure_cases.json`:

- `seed=101|topo=UsCarrier|split=val|mid=26|li=6|scale=0.25|policy=improved_stable`
- `seed=202|topo=UsCarrier|split=val|mid=24|li=4|scale=0.25|policy=improved_stable`
- `seed=303|topo=UsCarrier|split=val|mid=21|li=1|scale=0.25|policy=improved_stable`
- `seed=404|topo=UsCarrier|split=val|mid=23|li=3|scale=0.25|policy=improved_stable`
- `seed=505|topo=UsCarrier|split=val|mid=27|li=7|scale=0.25|policy=improved_stable`

- Topology: **UsCarrier** (directed), 378 links, catalog yen4 paths.
- Traffic scale: **0.25**.
- Total demand (scaled): **6201.5000**.
- Five scaled traffic matrices are **identical**: `True` (val toy slice has a single unique matrix; local indices differ only in case_id metadata).
- Forced common-link MLU lower bound (catalog): **0.785000** (infeasible_proven=False).

## 2. Method

Independent sparse path-flow LP (HiGHS via `scipy.optimize.linprog`):

- Decision variables: continuous path flows `f_p ≥ 0` and `U_max ≥ 0`.
- Demand satisfaction: for each OD, sum of path flows equals demand.
- Link constraints: sum of flows on paths using link `e` ≤ `U_max * capacity_e`.
- Objective: minimise `U_max`.
- Directed edges match the existing UsCarrier topology JSON / bipartite tensors.

Path sets:

- **Catalog:** existing yen4 (`97974` paths) — production candidate table.
- **Expanded:** independently generated yen8 (`194182` paths) — diagnostic only; catalog files were not overwritten.

Link loads were recomputed from the returned path flows via incidence aggregation.

## 3. Results (shared across all five case_ids)

| Method | MLU | Feasible (MLU≤1) | Notes |
|---|---:|---|---|
| Shortest fallback | 1.870000 | no | production candidate |
| Equal fallback | 2.452500 | no | production candidate |
| Rebalance heuristic | 1.353427 | no | prior investigation |
| Forced-link LB | 0.785000 | bound only | catalog common links |
| **LP catalog yen4** | **0.785** | **True** | status=optimal, runtime=4.56s, gap=0.0 |
| **LP expanded yen8** | **0.785** | **True** | status=optimal, runtime=19.84s, gap=0.0 |

### Validation

- Catalog recomputed MLU from flows: **0.785** (overloaded links=0).
- Expanded recomputed MLU from flows: **0.7850000000000001** (overloaded links=0).

## 4. Direct answers

1. **How many of the five cases are feasible using existing catalog paths?** **5/5** (optimal catalog MLU=0.785000 ≤ 1).
2. **How many become feasible using additional topology paths?** **0/5 become newly feasible**; **5/5** are feasible under yen8 (expanded MLU=0.785000). Because catalog already achieves feasibility, expanded paths are not required for feasibility.
3. **Optimal maximum utilization for each case?** All five share catalog optimum **0.785000** and expanded optimum **0.785000** (identical TMs).
4. **Candidate generation vs insufficient capacity?** **Candidate / fallback limitation (verified).** Network capacity is sufficient under continuous catalog path-splitting (U*=0.785 ≤ 1, matching the forced-link lower bound). Shortest/equal (and the prior rebalance heuristic) are too restrictive.
5. **Does this justify developing a better fallback algorithm?** **Yes, for these five cases.** A catalog path-fraction optimiser (LP / column generation) recovers feasibility where shortest/equal fail. This does **not** overturn the prior finding that 35/40 failures are catalog-infeasible; a better fallback helps the inconclusive slice, not Cogentco / UsCarrier≥0.5.
6. **Smallest justified next experiment?** Offline evaluation of a **catalog path-fraction LP fallback** restricted to cases where the forced-link bound is ≤ 1 (starting with these five), measuring applied MLU and violation rate vs current least-violating shortest/equal — still not wired into production. Separately, revisit demand/scale or path-catalog policy for the 35 proven-infeasible failures.

## 5. Artifacts

- `five_case_results.csv`
- `lp_routing_allocations.json`
- `solver_diagnostics.json`
- `comparison_plot.png`

## 6. Preservation

No modifications to gate/fallback/Twin/RL checkpoints, eval results, splits, traffic matrices, or the yen4 candidate-path catalog.
