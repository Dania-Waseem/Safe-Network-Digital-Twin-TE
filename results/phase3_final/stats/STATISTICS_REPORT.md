# Statistics, overhead and failure analysis

Sources: `main_grid_rows.jsonl` (1744 paired cases: 5 topologies × TMs × 8 load levels), `../ablations_v2/sequence_rows.jsonl`, `../e4_twin_error/paired_rows.csv.gz`, `../e2_holdout_af/case_rows.jsonl`, `../calibrated_traffic/calibration_rows.jsonl`. UsCarrier and Kdl traffic matrices are synthetic all-ones toy templates; every index is an identical copy, so additional TM indices or seeds add no traffic diversity on these topologies. All LPs use SciPy HiGHS (scipy.optimize.linprog, method='highs'). No Gurobi licence was available, so Teal/FIGRET-style Gurobi LPs are replaced by HiGHS.

## 1. Friedman + Nemenyi across (topology, load level) blocks

Each block is the mean over the TMs of one (topology, load mode, level) cell. Mean rank 1 = best (highest satisfaction / lowest MLU). `framework` = stable policy + improved twin + gate E + fallback.

### without_LP_all_topologies|sat  (blocks = 40, Friedman χ² = 74.85, p = 1e-14)

| Method | Mean rank |
|---|---:|
| framework | 1.98 |
| shortest | 2.67 |
| equal | 3.33 |
| frozen_p1 | 3.45 |
| actor_v2 | 4.75 |
| actor_stable | 4.83 |

Nemenyi p-values (pairs with p < 0.05):

shortest–actor_stable (4e-06), shortest–actor_v2 (1e-05), equal–actor_stable (0.00454), equal–actor_v2 (0.00863), equal–framework (0.0158), frozen_p1–actor_stable (0.013), frozen_p1–actor_v2 (0.0232), frozen_p1–framework (0.00564), actor_stable–framework (0), actor_v2–framework (0)

### without_LP_all_topologies|mlu  (blocks = 40, Friedman χ² = 78.89, p = 1.44e-15)

| Method | Mean rank |
|---|---:|
| equal | 2.10 |
| framework | 2.38 |
| shortest | 3.09 |
| frozen_p1 | 4.03 |
| actor_v2 | 4.46 |
| actor_stable | 4.95 |

Nemenyi p-values (pairs with p < 0.05):

shortest–actor_stable (0.000124), shortest–actor_v2 (0.013), equal–frozen_p1 (6.1e-05), equal–actor_stable (0), equal–actor_v2 (0), frozen_p1–framework (0.00113), actor_stable–framework (0), actor_v2–framework (9e-06)

### with_LP_excluding_Kdl|sat  (blocks = 32, Friedman χ² = 60.59, p = 3.42e-11)

| Method | Mean rank |
|---|---:|
| framework | 2.48 |
| lp | 2.75 |
| shortest | 3.45 |
| equal | 4.17 |
| frozen_p1 | 4.22 |
| actor_v2 | 5.25 |
| actor_stable | 5.67 |

Nemenyi p-values (pairs with p < 0.05):

lp–actor_stable (1e-06), lp–actor_v2 (7.5e-05), shortest–actor_stable (0.000788), shortest–actor_v2 (0.0153), equal–framework (0.0295), frozen_p1–framework (0.0224), actor_stable–framework (0), actor_v2–framework (6e-06)

### with_LP_excluding_Kdl|mlu  (blocks = 32, Friedman χ² = 110.35, p = 1.72e-21)

| Method | Mean rank |
|---|---:|
| lp | 1.00 |
| framework | 3.33 |
| equal | 3.34 |
| shortest | 4.22 |
| frozen_p1 | 5.06 |
| actor_v2 | 5.22 |
| actor_stable | 5.83 |

Nemenyi p-values (pairs with p < 0.05):

lp–shortest (0), lp–equal (0.000287), lp–frozen_p1 (0), lp–actor_stable (0), lp–actor_v2 (0), lp–framework (0.000327), shortest–actor_stable (0.0456), equal–frozen_p1 (0.0246), equal–actor_stable (8.6e-05), equal–actor_v2 (0.00933), frozen_p1–framework (0.0224), actor_stable–framework (7.5e-05), actor_v2–framework (0.00843)

## 2. Paired Wilcoxon signed-rank (Holm-corrected) with Cliff's δ

| Comparison (left vs right) | n | mean diff | left better | right better | p | p (Holm) | Cliff's δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| framework vs policy-only · dev · applied sat | 136 | +0.0206 | 0.49 | 0.04 | 7.83e-12 | 7.04e-11 * | +0.044 |
| framework vs policy-only · dev · applied mlu | 136 | -0.2721 | 0.47 | 0.07 | 7.03e-10 | 5.62e-09 * | -0.024 |
| v2 vs frozen_p1 · dev · actor sat | 136 | +0.0007 | 0.18 | 0.82 | 8.21e-05 | 0.000493 * | -0.102 |
| v2 vs frozen_p1 · dev · actor mlu | 136 | +0.2247 | 0.47 | 0.53 | 0.372 | 0.744 | -0.063 |
| v2 vs shortest · dev · actor sat | 136 | -0.0188 | 0.48 | 0.52 | 0.000624 | 0.002 * | +0.116 |
| v2 vs shortest · dev · actor mlu | 136 | +0.2158 | 0.47 | 0.53 | 0.243 | 0.728 | -0.039 |
| v2 vs actor_stable · dev · actor sat | 136 | +0.0003 | 0.75 | 0.25 | 2.95e-09 | 2.06e-08 * | +0.035 |
| v2 vs actor_stable · dev · actor mlu | 136 | -0.0171 | 0.94 | 0.06 | 1.56e-16 | 1.56e-15 * | -0.025 |
| stable vs frozen_p1 · dev · actor sat | 136 | +0.0004 | 0.18 | 0.82 | 8.44e-05 | 0.000493 * | -0.101 |
| stable vs frozen_p1 · dev · actor mlu | 136 | +0.2418 | 0.47 | 0.53 | 0.399 | 0.744 | -0.063 |
| framework vs policy-only · holdout · applied sat | 1608 | +0.0962 | 0.86 | 0.00 | 4.15e-229 | 7.06e-228 * | +0.575 |
| framework vs policy-only · holdout · applied mlu | 1608 | -0.6629 | 0.76 | 0.10 | 6.27e-187 | 1e-185 * | -0.363 |
| v2 vs frozen_p1 · holdout · actor sat | 1608 | -0.0728 | 0.07 | 0.93 | 3.28e-242 | 6.56e-241 * | -0.434 |
| v2 vs frozen_p1 · holdout · actor mlu | 1608 | +0.3355 | 0.31 | 0.69 | 2.92e-120 | 3.8e-119 * | +0.158 |
| v2 vs shortest · holdout · actor sat | 1608 | -0.0945 | 0.12 | 0.88 | 4.34e-239 | 7.81e-238 * | -0.554 |
| v2 vs shortest · holdout · actor mlu | 1608 | +0.6251 | 0.21 | 0.79 | 2.34e-164 | 3.5e-163 * | +0.329 |
| v2 vs actor_stable · holdout · actor sat | 1608 | +0.0007 | 0.69 | 0.31 | 7.52e-91 | 9.03e-90 * | -0.000 |
| v2 vs actor_stable · holdout · actor mlu | 1608 | -0.0002 | 0.52 | 0.34 | 1.63e-63 | 1.79e-62 * | -0.000 |
| stable vs frozen_p1 · holdout · actor sat | 1608 | -0.0735 | 0.07 | 0.93 | 4.77e-242 | 9.07e-241 * | -0.433 |
| stable vs frozen_p1 · holdout · actor mlu | 1608 | +0.3357 | 0.31 | 0.69 | 2.6e-120 | 3.63e-119 * | +0.158 |

\* significant at 0.05 after Holm correction. Satisfaction: higher is better; MLU: lower is better.

## 3. Exact McNemar tests on paired gate outcomes (Holm-corrected)

| Comparison (left vs right) | pairs | left-only | right-only | p | p (Holm) |
|---|---:|---:|---:|---:|---:|
| unsafe accept D vs E1 · dev · sigma=0.05 | 408 | 0 | 0 | 1.000 | 1.000 |
| false reject D vs E1 · dev · sigma=0.05 | 408 | 0 | 14 | 0.000122 | 0.004 * |
| unsafe accept D vs E2 · dev · sigma=0.05 | 408 | 0 | 0 | 1.000 | 1.000 |
| false reject D vs E2 · dev · sigma=0.05 | 408 | 0 | 23 | 2.38e-07 | 9.3e-06 * |
| unsafe accept D vs E1r · dev · sigma=0.05 | 408 | 0 | 0 | 1.000 | 1.000 |
| false reject D vs E1r · dev · sigma=0.05 | 408 | 0 | 12 | 0.000488 | 0.015 * |
| unsafe accept D vs E1 · dev · sigma=0.1 | 408 | 1 | 0 | 1.000 | 1.000 |
| false reject D vs E1 · dev · sigma=0.1 | 408 | 0 | 19 | 3.81e-06 | 0.000137 * |
| unsafe accept D vs E2 · dev · sigma=0.1 | 408 | 1 | 0 | 1.000 | 1.000 |
| false reject D vs E2 · dev · sigma=0.1 | 408 | 0 | 31 | 9.31e-10 | 3.91e-08 * |
| unsafe accept D vs E1r · dev · sigma=0.1 | 408 | 1 | 0 | 1.000 | 1.000 |
| false reject D vs E1r · dev · sigma=0.1 | 408 | 0 | 16 | 3.05e-05 | 0.001 * |
| unsafe accept D vs E1 · dev · sigma=0.2 | 408 | 3 | 0 | 0.250 | 1.000 |
| false reject D vs E1 · dev · sigma=0.2 | 408 | 0 | 27 | 1.49e-08 | 6.11e-07 * |
| unsafe accept D vs E2 · dev · sigma=0.2 | 408 | 3 | 0 | 0.250 | 1.000 |
| false reject D vs E2 · dev · sigma=0.2 | 408 | 0 | 71 | 8.47e-22 | 4.57e-20 * |
| unsafe accept D vs E1r · dev · sigma=0.2 | 408 | 3 | 0 | 0.250 | 1.000 |
| false reject D vs E1r · dev · sigma=0.2 | 408 | 0 | 21 | 9.54e-07 | 3.53e-05 * |
| unsafe accept D vs E1 · dev · sigma=0.4 | 408 | 1 | 0 | 1.000 | 1.000 |
| false reject D vs E1 · dev · sigma=0.4 | 408 | 0 | 67 | 1.36e-20 | 7.18e-19 * |
| unsafe accept D vs E2 · dev · sigma=0.4 | 408 | 1 | 0 | 1.000 | 1.000 |
| false reject D vs E2 · dev · sigma=0.4 | 408 | 0 | 95 | 5.05e-29 | 2.83e-27 * |
| unsafe accept D vs E1r · dev · sigma=0.4 | 408 | 1 | 0 | 1.000 | 1.000 |
| false reject D vs E1r · dev · sigma=0.4 | 408 | 0 | 57 | 1.39e-17 | 7.08e-16 * |
| unsafe accept D vs E1 · holdout · sigma=0.05 | 4824 | 6 | 0 | 0.031 | 0.844 |
| false reject D vs E1 · holdout · sigma=0.05 | 4824 | 0 | 48 | 7.11e-15 | 3.41e-13 * |
| unsafe accept D vs E2 · holdout · sigma=0.05 | 4824 | 7 | 0 | 0.016 | 0.439 |
| false reject D vs E2 · holdout · sigma=0.05 | 4824 | 0 | 103 | 1.97e-31 | 1.12e-29 * |
| unsafe accept D vs E1r · holdout · sigma=0.05 | 4824 | 5 | 0 | 0.062 | 1.000 |
| false reject D vs E1r · holdout · sigma=0.05 | 4824 | 0 | 39 | 3.64e-12 | 1.67e-10 * |
| unsafe accept D vs E1 · holdout · sigma=0.1 | 4824 | 23 | 0 | 2.38e-07 | 9.3e-06 * |
| false reject D vs E1 · holdout · sigma=0.1 | 4824 | 0 | 90 | 1.62e-27 | 8.89e-26 * |
| unsafe accept D vs E2 · holdout · sigma=0.1 | 4824 | 24 | 0 | 1.19e-07 | 4.77e-06 * |
| false reject D vs E2 · holdout · sigma=0.1 | 4824 | 0 | 232 | 2.9e-70 | 1.74e-68 * |
| unsafe accept D vs E1r · holdout · sigma=0.1 | 4824 | 19 | 0 | 3.81e-06 | 0.000137 * |
| false reject D vs E1r · holdout · sigma=0.1 | 4824 | 0 | 63 | 2.17e-19 | 1.13e-17 * |
| unsafe accept D vs E1 · holdout · sigma=0.2 | 4824 | 34 | 0 | 1.16e-10 | 5.12e-09 * |
| false reject D vs E1 · holdout · sigma=0.2 | 4824 | 0 | 225 | 3.71e-68 | 2.19e-66 * |
| unsafe accept D vs E2 · holdout · sigma=0.2 | 4824 | 36 | 0 | 2.91e-11 | 1.31e-09 * |
| false reject D vs E2 · holdout · sigma=0.2 | 4824 | 0 | 429 | 1.44e-129 | 9.09e-128 * |
| unsafe accept D vs E1r · holdout · sigma=0.2 | 4824 | 33 | 0 | 2.33e-10 | 1e-08 * |
| false reject D vs E1r · holdout · sigma=0.2 | 4824 | 0 | 169 | 2.67e-51 | 1.55e-49 * |
| unsafe accept D vs E1 · holdout · sigma=0.4 | 4824 | 50 | 0 | 1.78e-15 | 8.7e-14 * |
| false reject D vs E1 · holdout · sigma=0.4 | 4824 | 0 | 339 | 1.79e-102 | 1.11e-100 * |
| unsafe accept D vs E2 · holdout · sigma=0.4 | 4824 | 52 | 0 | 4.44e-16 | 2.22e-14 * |
| false reject D vs E2 · holdout · sigma=0.4 | 4824 | 0 | 461 | 3.36e-139 | 2.15e-137 * |
| unsafe accept D vs E1r · holdout · sigma=0.4 | 4824 | 47 | 0 | 1.42e-14 | 6.68e-13 * |
| false reject D vs E1r · holdout · sigma=0.4 | 4824 | 0 | 284 | 6.43e-86 | 3.93e-84 * |
| unsafe accept Dm vs E1 · dev · sigma=0.05 | 408 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E1 · dev · sigma=0.05 | 408 | 1 | 0 | 1.000 | 1.000 |
| unsafe accept Dm vs E1 · dev · sigma=0.1 | 408 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E1 · dev · sigma=0.1 | 408 | 2 | 2 | 1.000 | 1.000 |
| unsafe accept Dm vs E1 · dev · sigma=0.2 | 408 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E1 · dev · sigma=0.2 | 408 | 3 | 3 | 1.000 | 1.000 |
| unsafe accept Dm vs E1 · dev · sigma=0.4 | 408 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E1 · dev · sigma=0.4 | 408 | 2 | 2 | 1.000 | 1.000 |
| unsafe accept Dm vs E1 · holdout · sigma=0.05 | 4824 | 0 | 1 | 1.000 | 1.000 |
| false reject Dm vs E1 · holdout · sigma=0.05 | 4824 | 17 | 2 | 0.000729 | 0.022 * |
| unsafe accept Dm vs E1 · holdout · sigma=0.1 | 4824 | 1 | 0 | 1.000 | 1.000 |
| false reject Dm vs E1 · holdout · sigma=0.1 | 4824 | 17 | 28 | 0.135 | 1.000 |
| unsafe accept Dm vs E1 · holdout · sigma=0.2 | 4824 | 0 | 2 | 0.500 | 1.000 |
| false reject Dm vs E1 · holdout · sigma=0.2 | 4824 | 41 | 21 | 0.015 | 0.439 |
| unsafe accept Dm vs E1 · holdout · sigma=0.4 | 4824 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E1 · holdout · sigma=0.4 | 4824 | 52 | 18 | 5.85e-05 | 0.002 * |

## 4. Unified overhead (median milliseconds per decision)

| Topology | Policy (GNN) | Twin (5-member) | Gate + fallback | Fallback selection | LP (min-MLU) | Frozen P1 actor |
|---|---:|---:|---:|---:|---:|---:|
| B4 | 11.5 | 3.6 | – | 0.8 | 3.9 | 12.4 |
| UsCarrier | 971.4 | 171.2 | – | 60.3 | 2610.8 | 1037.5 |
| Cogentco | 1183.7 | 234.5 | – | 90.7 | 6713.4 | 1212.7 |
| GEANT | 17.4 | 1.7 | 0.4 | 1.0 | 11.5 | 18.1 |
| Kdl | 17412.5 | 3327.6 | 2646.3 | 1895.1 | – | 17631.4 |

Gate + fallback time is only measured where the A–F runner logged it (GEANT, Kdl). LP time is per TM (calibrated-load study); LP is not tractable on Kdl. Timings are single-process CPU wall clock on the laptop and include Python overhead.

## 5. Failure taxonomy

### No twin error (σ = 0): every overloaded applied decision of the full framework

| Scope | Cause | Count |
|---|---|---:|
| dev | no feasible k=4 routing exists (capacity/catalog infeasible) | 53 |
| dev |   of which proposal unsafe (policy out of distribution / overloaded) | 73 |
| dev | fallback infeasible, but a feasible k=4 routing exists (fallback limitation) | 20 |
| holdout | fallback infeasible, but a feasible k=4 routing exists (fallback limitation) | 547 |
| holdout |   of which proposal unsafe (policy out of distribution / overloaded) | 1095 |
| holdout | no feasible k=4 routing exists (capacity/catalog infeasible) | 543 |
| holdout | fallback infeasible (LP not available: Kdl) | 5 |

### Oracle-twin counterfactual (σ = 0): same proposals, perfect twin

| Scope | Cases | Framework overloaded | Oracle-twin overloaded |
|---|---:|---:|---:|
| dev | 136 | 73 | 73 |
| holdout | 1608 | 1095 | 1095 |

### Under member-wise twin error: overloaded applied decisions of gate E1

| Scope / σ | Cause | Count |
|---|---|---:|
| dev|sigma=0.0 | n | 136 |
| dev|sigma=0.0 | oracle (F) overloaded applied | 73 |
| dev|sigma=0.0 | rejected unsafe proposal -> fallback infeasible | 73 |
| dev|sigma=0.0 | E1 overloaded applied | 73 |
| dev|sigma=0.05 | n | 408 |
| dev|sigma=0.05 | oracle (F) overloaded applied | 219 |
| dev|sigma=0.05 | false reject -> overloaded fallback | 18 |
| dev|sigma=0.05 | E1 overloaded applied | 237 |
| dev|sigma=0.05 | rejected unsafe proposal -> fallback infeasible | 219 |
| dev|sigma=0.1 | n | 408 |
| dev|sigma=0.1 | oracle (F) overloaded applied | 219 |
| dev|sigma=0.1 | false reject -> overloaded fallback | 18 |
| dev|sigma=0.1 | E1 overloaded applied | 237 |
| dev|sigma=0.1 | rejected unsafe proposal -> fallback infeasible | 219 |
| dev|sigma=0.2 | n | 408 |
| dev|sigma=0.2 | oracle (F) overloaded applied | 219 |
| dev|sigma=0.2 | false reject -> overloaded fallback | 18 |
| dev|sigma=0.2 | E1 overloaded applied | 237 |
| dev|sigma=0.2 | rejected unsafe proposal -> fallback infeasible | 219 |
| dev|sigma=0.4 | n | 408 |
| dev|sigma=0.4 | oracle (F) overloaded applied | 219 |
| dev|sigma=0.4 | false reject -> overloaded fallback | 18 |
| dev|sigma=0.4 | E1 overloaded applied | 237 |
| dev|sigma=0.4 | rejected unsafe proposal -> fallback infeasible | 219 |
| holdout|sigma=0.0 | n | 1608 |
| holdout|sigma=0.0 | oracle (F) overloaded applied | 1095 |
| holdout|sigma=0.0 | rejected unsafe proposal -> fallback infeasible | 1095 |
| holdout|sigma=0.0 | E1 overloaded applied | 1095 |
| holdout|sigma=0.05 | n | 4824 |
| holdout|sigma=0.05 | oracle (F) overloaded applied | 3285 |
| holdout|sigma=0.05 | rejected unsafe proposal -> fallback infeasible | 3285 |
| holdout|sigma=0.05 | E1 overloaded applied | 3286 |
| holdout|sigma=0.05 | unsafe accept: twin under-prediction (coverage OK) | 1 |
| holdout|sigma=0.1 | n | 4824 |
| holdout|sigma=0.1 | oracle (F) overloaded applied | 3285 |
| holdout|sigma=0.1 | rejected unsafe proposal -> fallback infeasible | 3285 |
| holdout|sigma=0.1 | E1 overloaded applied | 3288 |
| holdout|sigma=0.1 | unsafe accept: twin under-prediction (coverage OK) | 1 |
| holdout|sigma=0.1 | false reject -> overloaded fallback | 2 |
| holdout|sigma=0.2 | n | 4824 |
| holdout|sigma=0.2 | oracle (F) overloaded applied | 3285 |
| holdout|sigma=0.2 | rejected unsafe proposal -> fallback infeasible | 3285 |
| holdout|sigma=0.2 | E1 overloaded applied | 3291 |
| holdout|sigma=0.2 | unsafe accept: twin under-prediction (coverage OK) | 3 |
| holdout|sigma=0.2 | false reject -> overloaded fallback | 3 |
| holdout|sigma=0.4 | n | 4824 |
| holdout|sigma=0.4 | oracle (F) overloaded applied | 3285 |
| holdout|sigma=0.4 | rejected unsafe proposal -> fallback infeasible | 3284 |
| holdout|sigma=0.4 | E1 overloaded applied | 3289 |
| holdout|sigma=0.4 | false reject -> overloaded fallback | 3 |
| holdout|sigma=0.4 | unsafe accept: twin under-prediction (coverage OK) | 2 |

## Artifacts

- `friedman_nemenyi.json`, `paired_wilcoxon_holm.json`, `mcnemar_holm.json`, `overhead.json`, `failure_taxonomy.json`, `main_grid_rows.jsonl`, `config.json`
