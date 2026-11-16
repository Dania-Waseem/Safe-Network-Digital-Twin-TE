# Holdout A–F evaluation on SNDlib Abilene and Germany50

Frozen stack of the GEANT holdout evaluation (`results/phase3_final/e2_holdout_af/`): policy `recommended_stable`, original and improved twin ensembles, conformal std scale 0.955, gate k = 1 with utilisation limit 1.0, shortest/ECMP least-violating fallback. Nothing was trained, recalibrated or tuned on these topologies.

- Abilene (12 nodes / 30 directed links): unseen topology with related traffic (same measurement campaign as the B4 training traffic). Germany50 (50 / 176): unseen topology, unrelated traffic (DFN 2005).
- Cases: 200 seeded real traffic matrices per topology × calibrated targets 0.5 / 0.7 / 0.9 / 1.1 (alpha from the k = 4 min-MLU LP) = 800 cases per topology.
- *Unsafe accept* = accepted proposal that overloads at least one link (physics ground truth). *False reject* = rejected proposal that was safe. Satisfaction = P1 definition (eqs. 23–28 adjustment).
- All LPs use SciPy HiGHS (scipy.optimize.linprog, method='highs'). No Gurobi licence was available, so Teal/FIGRET-style Gurobi LPs are replaced by HiGHS.

Data: `data/raw/sndlib/QUALITY_REPORT.md`; calibration: `data/processed/calibrated_load/sndlib_scaled_tm_index.json`.

## Abilene — all calibrated levels (n = 800 cases, 780 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible fallback | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.975 | 1.000 | 0.000 | 0.000 | 0.000 | 0.975 | 2.446 | 0.6551 | 0.1 |
| B orig. twin, mean | 0.025 | 0.000 | 0.000 | 0.000 | 0.975 | 0.896 | 0.896 | 1.900 | 0.9105 | 2.4 |
| C orig. twin, μ+σ | 0.025 | 0.000 | 0.000 | 0.000 | 0.975 | 0.896 | 0.896 | 1.900 | 0.9105 | 2.1 |
| D improved twin, mean | 0.025 | 0.000 | 0.000 | 0.000 | 0.975 | 0.896 | 0.896 | 1.900 | 0.9105 | 2.1 |
| E improved twin, calibrated μ+σ (k=1) | 0.025 | 0.000 | 0.000 | 0.000 | 0.975 | 0.896 | 0.896 | 1.900 | 0.9105 | 2.1 |
| F physics oracle | 0.025 | 0.000 | 0.000 | 0.000 | 0.975 | 0.896 | 0.896 | 1.900 | 0.9105 | 0.3 |

Baselines on the same 800 scaled matrices:

| Method | Satisfaction | MLU | Overload rate | median ms |
|---|---:|---:|---:|---:|
| shortest | 0.8961 | 2.017 | 0.921 | 0.0 |
| equal | 0.7106 | 3.320 | 1.000 | 0.0 |
| lp | 0.9820 | 0.800 | 0.250 | 8.4 |
| frozen_p1 | 0.7454 | 2.600 | 0.969 | 2.9 |
| actor_stable | 0.6551 | 2.446 | 0.975 | 3.2 |
| actor_v2 | 0.6576 | 2.357 | 0.974 | 2.9 |

Per load level:

| Target | A sat. | E sat. | LP sat. | Shortest sat. | ECMP sat. | Frozen P1 sat. | A unsafe acc. | E unsafe acc. | E false rej. | E fallback | E overload | LP overload | Policy MLU | LP MLU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.9058 | 0.9866 | 1.0000 | 0.9627 | 0.8593 | 0.9234 | 0.940 | 0.000 | 0.000 | 0.940 | 0.645 | 0.000 | 1.519 | 0.500 |
| 0.7 | 0.7010 | 0.9571 | 1.0000 | 0.9321 | 0.7837 | 0.8186 | 0.960 | 0.000 | 0.000 | 0.960 | 0.945 | 0.000 | 2.140 | 0.700 |
| 0.9 | 0.5567 | 0.8849 | 1.0000 | 0.8760 | 0.6523 | 0.6773 | 1.000 | 0.000 | 0.000 | 1.000 | 0.995 | 0.000 | 2.757 | 0.900 |
| 1.1 | 0.4570 | 0.8135 | 0.9280 | 0.8135 | 0.5471 | 0.5622 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 3.367 | 1.100 |

## Germany50 — all calibrated levels (n = 800 cases, 592 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible fallback | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.740 | 1.000 | 0.000 | 0.000 | 0.000 | 0.740 | 1.257 | 0.9305 | 0.6 |
| B orig. twin, mean | 0.260 | 0.000 | 0.000 | 0.000 | 0.740 | 0.646 | 0.646 | 1.168 | 0.9491 | 6.3 |
| C orig. twin, μ+σ | 0.260 | 0.000 | 0.000 | 0.000 | 0.740 | 0.646 | 0.646 | 1.168 | 0.9491 | 5.9 |
| D improved twin, mean | 0.260 | 0.000 | 0.000 | 0.000 | 0.740 | 0.646 | 0.646 | 1.168 | 0.9491 | 5.9 |
| E improved twin, calibrated μ+σ (k=1) | 0.260 | 0.000 | 0.000 | 0.000 | 0.740 | 0.646 | 0.646 | 1.168 | 0.9491 | 5.8 |
| F physics oracle | 0.260 | 0.000 | 0.000 | 0.000 | 0.740 | 0.646 | 0.646 | 1.168 | 0.9491 | 1.0 |

Baselines on the same 800 scaled matrices:

| Method | Satisfaction | MLU | Overload rate | median ms |
|---|---:|---:|---:|---:|
| shortest | 0.9178 | 1.463 | 0.757 | 0.0 |
| equal | 0.9493 | 1.155 | 0.646 | 0.0 |
| lp | 0.9891 | 0.800 | 0.250 | 59.5 |
| frozen_p1 | 0.9346 | 1.232 | 0.689 | 17.8 |
| actor_stable | 0.9305 | 1.257 | 0.740 | 18.4 |
| actor_v2 | 0.9354 | 1.227 | 0.690 | 17.7 |

Per load level:

| Target | A sat. | E sat. | LP sat. | Shortest sat. | ECMP sat. | Frozen P1 sat. | A unsafe acc. | E unsafe acc. | E false rej. | E fallback | E overload | LP overload | Policy MLU | LP MLU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.9990 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 0.9992 | 0.085 | 0.000 | 0.000 | 0.085 | 0.000 | 0.000 | 0.785 | 0.500 |
| 0.7 | 0.9786 | 0.9955 | 1.0000 | 0.9477 | 0.9955 | 0.9834 | 0.905 | 0.000 | 0.000 | 0.905 | 0.615 | 0.000 | 1.100 | 0.700 |
| 0.9 | 0.9033 | 0.9315 | 1.0000 | 0.8951 | 0.9314 | 0.9097 | 0.970 | 0.000 | 0.000 | 0.970 | 0.970 | 0.000 | 1.414 | 0.900 |
| 1.1 | 0.8412 | 0.8695 | 0.9563 | 0.8289 | 0.8704 | 0.8461 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.728 | 1.100 |

## D vs E and E vs oracle F

| Topology | n | D-only unsafe accepts | E-only unsafe accepts | D/E accept discordant | E/F accept discordant |
|---|---:|---:|---:|---:|---:|
| Abilene | 800 | 0 | 0 | 0 | 0 |
| Germany50 | 800 | 0 | 0 | 0 | 0 |

## Files

- `case_rows.jsonl`: one row per case (all six variants, all baselines, timings)
- `summary.json`: variant table (as for GEANT), D/E discordance, baseline table; `config.json`: frozen stack
- `figures/holdout_sat_unsafe_A_vs_E.{png,pdf}`
