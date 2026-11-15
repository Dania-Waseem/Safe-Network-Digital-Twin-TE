# Fallback Failure Investigation Report

**Source evaluation:** `eval_runs/af_eval_k1_calibrated_devtopos_v1/`

Inspection only. Fallback/gate/Twin/RL/eval artifacts were not modified.

## 1. How many fallback failures?

**Verified count (variant E definition):** **40** cases

Definition used: proposed rejected ∧ fallback used ∧ `capacity_violations_applied > 0`.

Counts under the same definition for other gated variants:

- B_original_twin_mean: **40**
- C_original_twin_uncertainty: **40**
- D_improved_twin_mean: **40**
- E_improved_twin_calibrated_uncertainty: **40**
- F_physics_oracle: **40**

Paired gated variants report the counts above under the same failure definition.

Topologies: ['Cogentco', 'UsCarrier'] (B4 failures: **0**).

## 2. Case-level failure table

| case_id | topo | scale | prop MLU | short MLU | equal MLU | selected | applied MLU | overload | n_over | cause |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| seed=101|topo=UsCarrier|split=val|mid=26|li=6|scale=0.25|policy=improved_stable | UsCarrier | 0.25 | 3.010 | 1.870 | 2.453 | shortest | 1.870 | 159.750 | 2 | inconclusive |
| seed=101|topo=UsCarrier|split=val|mid=26|li=6|scale=0.5|policy=improved_stable | UsCarrier | 0.50 | 6.020 | 3.740 | 4.905 | shortest | 3.740 | 818.000 | 12 | physically_infeasible_on_catalog_paths |
| seed=101|topo=UsCarrier|split=val|mid=26|li=6|scale=0.75|policy=improved_stable | UsCarrier | 0.75 | 9.030 | 5.610 | 7.357 | shortest | 5.610 | 2090.500 | 33 | physically_infeasible_on_catalog_paths |
| seed=101|topo=UsCarrier|split=val|mid=26|li=6|scale=1.0|policy=improved_stable | UsCarrier | 1.00 | 12.040 | 7.480 | 9.810 | shortest | 7.480 | 10124.000 | 110 | physically_infeasible_on_catalog_paths |
| seed=202|topo=UsCarrier|split=val|mid=24|li=4|scale=0.25|policy=improved_stable | UsCarrier | 0.25 | 3.010 | 1.870 | 2.453 | shortest | 1.870 | 159.750 | 2 | inconclusive |
| seed=202|topo=UsCarrier|split=val|mid=24|li=4|scale=0.5|policy=improved_stable | UsCarrier | 0.50 | 6.020 | 3.740 | 4.905 | shortest | 3.740 | 818.000 | 12 | physically_infeasible_on_catalog_paths |
| seed=202|topo=UsCarrier|split=val|mid=24|li=4|scale=0.75|policy=improved_stable | UsCarrier | 0.75 | 9.030 | 5.610 | 7.357 | shortest | 5.610 | 2090.500 | 33 | physically_infeasible_on_catalog_paths |
| seed=202|topo=UsCarrier|split=val|mid=24|li=4|scale=1.0|policy=improved_stable | UsCarrier | 1.00 | 12.040 | 7.480 | 9.810 | shortest | 7.480 | 10124.000 | 110 | physically_infeasible_on_catalog_paths |
| seed=303|topo=UsCarrier|split=val|mid=21|li=1|scale=0.25|policy=improved_stable | UsCarrier | 0.25 | 3.010 | 1.870 | 2.453 | shortest | 1.870 | 159.750 | 2 | inconclusive |
| seed=303|topo=UsCarrier|split=val|mid=21|li=1|scale=0.5|policy=improved_stable | UsCarrier | 0.50 | 6.020 | 3.740 | 4.905 | shortest | 3.740 | 818.000 | 12 | physically_infeasible_on_catalog_paths |
| seed=303|topo=UsCarrier|split=val|mid=21|li=1|scale=0.75|policy=improved_stable | UsCarrier | 0.75 | 9.030 | 5.610 | 7.357 | shortest | 5.610 | 2090.500 | 33 | physically_infeasible_on_catalog_paths |
| seed=303|topo=UsCarrier|split=val|mid=21|li=1|scale=1.0|policy=improved_stable | UsCarrier | 1.00 | 12.040 | 7.480 | 9.810 | shortest | 7.480 | 10124.000 | 110 | physically_infeasible_on_catalog_paths |
| seed=404|topo=UsCarrier|split=val|mid=23|li=3|scale=0.25|policy=improved_stable | UsCarrier | 0.25 | 3.010 | 1.870 | 2.453 | shortest | 1.870 | 159.750 | 2 | inconclusive |
| seed=404|topo=UsCarrier|split=val|mid=23|li=3|scale=0.5|policy=improved_stable | UsCarrier | 0.50 | 6.020 | 3.740 | 4.905 | shortest | 3.740 | 818.000 | 12 | physically_infeasible_on_catalog_paths |
| seed=404|topo=UsCarrier|split=val|mid=23|li=3|scale=0.75|policy=improved_stable | UsCarrier | 0.75 | 9.030 | 5.610 | 7.357 | shortest | 5.610 | 2090.500 | 33 | physically_infeasible_on_catalog_paths |
| seed=404|topo=UsCarrier|split=val|mid=23|li=3|scale=1.0|policy=improved_stable | UsCarrier | 1.00 | 12.040 | 7.480 | 9.810 | shortest | 7.480 | 10124.000 | 110 | physically_infeasible_on_catalog_paths |
| seed=505|topo=UsCarrier|split=val|mid=27|li=7|scale=0.25|policy=improved_stable | UsCarrier | 0.25 | 3.010 | 1.870 | 2.453 | shortest | 1.870 | 159.750 | 2 | inconclusive |
| seed=505|topo=UsCarrier|split=val|mid=27|li=7|scale=0.5|policy=improved_stable | UsCarrier | 0.50 | 6.020 | 3.740 | 4.905 | shortest | 3.740 | 818.000 | 12 | physically_infeasible_on_catalog_paths |
| seed=505|topo=UsCarrier|split=val|mid=27|li=7|scale=0.75|policy=improved_stable | UsCarrier | 0.75 | 9.030 | 5.610 | 7.357 | shortest | 5.610 | 2090.500 | 33 | physically_infeasible_on_catalog_paths |
| seed=505|topo=UsCarrier|split=val|mid=27|li=7|scale=1.0|policy=improved_stable | UsCarrier | 1.00 | 12.040 | 7.480 | 9.810 | shortest | 7.480 | 10124.000 | 110 | physically_infeasible_on_catalog_paths |
| seed=101|topo=Cogentco|split=val|mid=23|li=3|scale=0.25|policy=improved_stable | Cogentco | 0.25 | 16.942 | 16.687 | 16.211 | shortest | 16.687 | 8367915.556 | 274 | physically_infeasible_on_catalog_paths |
| seed=101|topo=Cogentco|split=val|mid=23|li=3|scale=0.5|policy=improved_stable | Cogentco | 0.50 | 33.884 | 33.375 | 32.422 | shortest | 33.375 | 19924526.050 | 362 | physically_infeasible_on_catalog_paths |
| seed=101|topo=Cogentco|split=val|mid=23|li=3|scale=0.75|policy=improved_stable | Cogentco | 0.75 | 50.826 | 50.062 | 48.633 | shortest | 50.062 | 31802293.546 | 394 | physically_infeasible_on_catalog_paths |
| seed=101|topo=Cogentco|split=val|mid=23|li=3|scale=1.0|policy=improved_stable | Cogentco | 1.00 | 67.772 | 66.750 | 64.843 | shortest | 66.750 | 43750592.139 | 450 | physically_infeasible_on_catalog_paths |
| seed=202|topo=Cogentco|split=val|mid=26|li=6|scale=0.25|policy=improved_stable | Cogentco | 0.25 | 16.935 | 16.685 | 16.215 | shortest | 16.685 | 8352848.380 | 274 | physically_infeasible_on_catalog_paths |
| seed=202|topo=Cogentco|split=val|mid=26|li=6|scale=0.5|policy=improved_stable | Cogentco | 0.50 | 33.870 | 33.371 | 32.430 | shortest | 33.371 | 19892363.294 | 362 | physically_infeasible_on_catalog_paths |
| seed=202|topo=Cogentco|split=val|mid=26|li=6|scale=0.75|policy=improved_stable | Cogentco | 0.75 | 50.806 | 50.056 | 48.646 | shortest | 50.056 | 31752885.468 | 394 | physically_infeasible_on_catalog_paths |
| seed=202|topo=Cogentco|split=val|mid=26|li=6|scale=1.0|policy=improved_stable | Cogentco | 1.00 | 67.746 | 66.741 | 64.861 | shortest | 66.741 | 43682940.369 | 447 | physically_infeasible_on_catalog_paths |
| seed=303|topo=Cogentco|split=val|mid=22|li=2|scale=0.25|policy=improved_stable | Cogentco | 0.25 | 16.950 | 16.658 | 16.185 | shortest | 16.658 | 8359184.023 | 275 | physically_infeasible_on_catalog_paths |
| seed=303|topo=Cogentco|split=val|mid=22|li=2|scale=0.5|policy=improved_stable | Cogentco | 0.50 | 33.900 | 33.316 | 32.371 | shortest | 33.316 | 19904989.753 | 362 | physically_infeasible_on_catalog_paths |
| seed=303|topo=Cogentco|split=val|mid=22|li=2|scale=0.75|policy=improved_stable | Cogentco | 0.75 | 50.850 | 49.974 | 48.556 | shortest | 49.974 | 31772549.557 | 394 | physically_infeasible_on_catalog_paths |
| seed=303|topo=Cogentco|split=val|mid=22|li=2|scale=1.0|policy=improved_stable | Cogentco | 1.00 | 67.804 | 66.632 | 64.742 | shortest | 66.632 | 43711476.116 | 448 | physically_infeasible_on_catalog_paths |
| seed=404|topo=Cogentco|split=val|mid=20|li=0|scale=0.25|policy=improved_stable | Cogentco | 0.25 | 16.861 | 16.674 | 16.173 | shortest | 16.674 | 8350921.695 | 277 | physically_infeasible_on_catalog_paths |
| seed=404|topo=Cogentco|split=val|mid=20|li=0|scale=0.5|policy=improved_stable | Cogentco | 0.50 | 33.721 | 33.347 | 32.347 | shortest | 33.347 | 19887760.676 | 362 | physically_infeasible_on_catalog_paths |
| seed=404|topo=Cogentco|split=val|mid=20|li=0|scale=0.75|policy=improved_stable | Cogentco | 0.75 | 50.582 | 50.021 | 48.520 | shortest | 50.021 | 31747626.029 | 394 | physically_infeasible_on_catalog_paths |
| seed=404|topo=Cogentco|split=val|mid=20|li=0|scale=1.0|policy=improved_stable | Cogentco | 1.00 | 67.447 | 66.695 | 64.693 | shortest | 66.695 | 43676427.048 | 446 | physically_infeasible_on_catalog_paths |
| seed=505|topo=Cogentco|split=val|mid=22|li=2|scale=0.25|policy=improved_stable | Cogentco | 0.25 | 16.950 | 16.658 | 16.185 | shortest | 16.658 | 8359184.023 | 275 | physically_infeasible_on_catalog_paths |
| seed=505|topo=Cogentco|split=val|mid=22|li=2|scale=0.5|policy=improved_stable | Cogentco | 0.50 | 33.900 | 33.316 | 32.371 | shortest | 33.316 | 19904989.753 | 362 | physically_infeasible_on_catalog_paths |
| seed=505|topo=Cogentco|split=val|mid=22|li=2|scale=0.75|policy=improved_stable | Cogentco | 0.75 | 50.850 | 49.974 | 48.556 | shortest | 49.974 | 31772549.557 | 394 | physically_infeasible_on_catalog_paths |
| seed=505|topo=Cogentco|split=val|mid=22|li=2|scale=1.0|policy=improved_stable | Cogentco | 1.00 | 67.804 | 66.632 | 64.742 | shortest | 66.632 | 43711476.116 | 448 | physically_infeasible_on_catalog_paths |

## 3. Why fallback fails

### Verified findings

- In all 40 failures, **both** `shortest` and `equal` overload at least one link (`verified.all_failures_have_both_candidates_infeasible=True`).
- Fallback logging matches recomputed selection in **40/40** cases.
- Least-violating selection rule matches recomputation in **40/40** cases.
- The gate correctly rejects unsafe proposals on these cases; proposed vs applied MLUs are logged separately and align with recomputed physics.

### Feasibility diagnostic findings (catalog candidate paths)

Methods (inspection-only, not production fallback):

1. **Forced common-link lower bound:** if every catalog path of a demand uses a link, that demand's volume is unavoidable on the link. Forced load > capacity proves catalog-path infeasibility without an LP.
2. **Heuristic path-split search** (equal/shortest baselines, iterative rebalance, Dirichlet samples): a feasible witness proves candidate-set feasibility; absence of a witness is **not** proof of infeasibility.
3. **Exact sparse min-MLU LP** only when `n_paths ≤ 5000` (none of the failure topologies qualify: UsCarrier≈98k, Cogentco≈154k paths).

- Verified feasible on catalog paths (witness/LP) while shortest/equal fail: **0**
- Verified infeasible on catalog paths (forced-link bound and/or LP): **35**
- Inconclusive (no forced-link proof, no feasible witness, no exact LP): **5**

Physical class counts: inconclusive_feasibility=5, infeasible_under_path_set=35

Pattern (solver/bound-derived):

- **All 20 Cogentco failures** are proven infeasible on the catalog path set (forced-link MLU lower bound ≫ 1 at every evaluated scale, including 0.25).
- **15/20 UsCarrier failures** (scales 0.5, 0.75, 1.0) are proven infeasible (forced-link MLU lower bound > 1).
- **5/20 UsCarrier failures** (all at scale 0.25) are inconclusive: forced-link MLU lower bound ≈ 0.785 (< 1) while the best heuristic path-split still has MLU ≈ 1.35; true catalog feasibility is unknown.

### Hypotheses (not fully verified)

- Failures concentrate on UsCarrier/Cogentco (zero B4 failures) because those topologies/demands under the robustness scales exceed catalog-path capacity — **hypothesis** now strongly supported for Cogentco and UsCarrier≥0.5 by the forced-link bound.
- The 5 UsCarrier@0.25 inconclusive cases may still be catalog-feasible under an optimal path-split (heuristic closed gap 1.87→1.35 but not ≤1) — **hypothesis**.
- Paths outside the Phase-3 candidate table might help some infeasible-on-catalog cases — **hypothesis** (not tested).

## 4. Topology and scale breakdown

- **Cogentco**: n=20, mean proposed MLU=42.320, mean applied MLU=41.681, mean diagnostic MLU=27.796, feasible@1=0, infeas@1=20, inconclusive=0
- **UsCarrier**: n=20, mean proposed MLU=7.525, mean applied MLU=4.675, mean diagnostic MLU=2.105, feasible@1=0, infeas@1=15, inconclusive=5

- scale=0.25: n=10, mean applied MLU=9.271, mean diagnostic MLU=6.236, feasible@1=0, inconclusive=5
- scale=0.5: n=10, mean applied MLU=18.542, mean diagnostic MLU=11.903, feasible@1=0, inconclusive=0
- scale=0.75: n=10, mean applied MLU=27.814, mean diagnostic MLU=17.855, feasible@1=0, inconclusive=0
- scale=1.0: n=10, mean applied MLU=37.085, mean diagnostic MLU=23.806, feasible@1=0, inconclusive=0

## 5. Direct answers

1. **How many fallback failures?** **40** (not assumed; counted from JSONL under the definition above).
2. **How many are physically infeasible (on catalog paths)?** **35** verified (forced common-link lower bound > 1); **5** remain inconclusive (all UsCarrier @ scale 0.25). No infeasibility inferred from absent LP solves.
3. **How many are caused by limited fallback candidates (shortest/equal only)?** **0** verified (no case yet has a feasible catalog path-split witness while shortest/equal overload).
4. **Does the existing fallback implementation behave correctly?** **Yes (verified on these cases):** it detects that both candidates are infeasible, selects the least-violating candidate (`shortest` on all 40), and logs proposed vs applied separately.
5. **Is a better fallback strategy technically feasible?** **Not as a fix for most failures:** 35/40 are proven capacity-infeasible on the existing catalog path set, so no routing-only fallback can clear them. A richer path-split optimiser remains **possibly** useful only for the 5 UsCarrier@0.25 inconclusive cases (and for reducing residual overload elsewhere), but that is unproven.
6. **Smallest justified next experiment?** Offline scalable **min-MLU LP / column generation on the 5 UsCarrier@0.25 inconclusive cases only**—to settle whether any catalog path-split is feasible. If still infeasible (or if the goal is Cogentco), the next lever is **demand/scale or path-catalog policy**, not fallback heuristics. No gate/Twin retuning.

## 6. Artifacts

- `failure_cases.json`
- `candidate_comparison.csv`
- `feasibility_analysis.json`
- `topology_breakdown.csv`, `scale_breakdown.csv`
- `plots/overload_by_topology.png`, `overload_by_scale.png`,
  `proposed_vs_fallback_mlu.png`, `fallback_vs_lp_mlu.png`

## 7. Preservation

No modifications to `safety/gate.py`, `fallback.py`, `gate_logic.py`, Twin/RL checkpoints, evaluation results, or splits.
