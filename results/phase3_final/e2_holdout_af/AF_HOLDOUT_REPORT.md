# E2 holdout evaluation — gate variants A–F on unseen GEANT and Kdl

Frozen improved stack from `af_eval_k1_calibrated_devtopos_v1`: same policy (`recommended_stable`), same twins, same conformal scale, gate k=1 / utilisation limit 1.0, same shortest/equal least-violating fallback. **No recalibration, no threshold change, no training.** GEANT and Kdl were never used for training, tuning or calibration.

- GEANT (22 nodes / 72 links, FIGRET real TMs): 200 seeded test TMs × 8 load levels.
- Kdl (754/1790): 1 test TM × 8 load levels. UsCarrier and Kdl traffic matrices are synthetic all-ones toy templates; every index is an identical copy, so additional TM indices or seeds add no traffic diversity on these topologies.
- Load levels: fixed scales 0.25–1.0 and calibrated targets 0.5–1.1 (calibrated-load study; GEANT reference = LP MLU, Kdl reference = shortest MLU).
- *Unsafe accept* = accepted proposal that overloads at least one link (physics ground truth). Rates are per case; `| unsafe` / `| safe` columns condition on the proposal.
- Satisfaction = P1 definition (eqs. 23–28 adjustment, delivered / total demand).
- All LPs use SciPy HiGHS (scipy.optimize.linprog, method='highs'). No Gurobi licence was available, so Teal/FIGRET-style Gurobi LPs are replaced by HiGHS. LP is not run on Kdl.

## [improved_stable] GEANT — all load levels (n = 1600 cases, 1386 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.866 | 1.000 | 0.000 | 0.000 | 0.000 | 0.866 | 2.209 | 0.8802 | 0 |
| B orig. mean | 0.133 | 0.000 | 0.000 | 0.005 | 0.867 | 0.681 | 0.681 | 1.553 | 0.9763 | 2 |
| C orig. unc. | 0.133 | 0.000 | 0.000 | 0.009 | 0.868 | 0.681 | 0.681 | 1.553 | 0.9763 | 2 |
| D imp. mean | 0.134 | 0.000 | 0.000 | 0.000 | 0.866 | 0.681 | 0.681 | 1.553 | 0.9763 | 2 |
| E imp. cal. unc. (k=1) | 0.134 | 0.000 | 0.000 | 0.000 | 0.866 | 0.681 | 0.681 | 1.553 | 0.9763 | 2 |
| F physics oracle | 0.134 | 0.000 | 0.000 | 0.000 | 0.866 | 0.681 | 0.681 | 1.553 | 0.9763 | 0 |

## [improved_stable] GEANT — fixed load (n = 800 cases, 655 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.819 | 1.000 | 0.000 | 0.000 | 0.000 | 0.819 | 2.360 | 0.8641 | 0 |
| B orig. mean | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 3 |
| C orig. unc. | 0.180 | 0.000 | 0.000 | 0.007 | 0.820 | 0.665 | 0.665 | 1.767 | 0.9683 | 2 |
| D imp. mean | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 2 |
| E imp. cal. unc. (k=1) | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 2 |
| F physics oracle | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 0 |

## [improved_stable] GEANT — calibrated load (n = 800 cases, 731 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.914 | 1.000 | 0.000 | 0.000 | 0.000 | 0.914 | 2.059 | 0.8964 | 0 |
| B orig. mean | 0.085 | 0.000 | 0.000 | 0.014 | 0.915 | 0.698 | 0.698 | 1.339 | 0.9842 | 2 |
| C orig. unc. | 0.085 | 0.000 | 0.000 | 0.014 | 0.915 | 0.698 | 0.698 | 1.339 | 0.9842 | 2 |
| D imp. mean | 0.086 | 0.000 | 0.000 | 0.000 | 0.914 | 0.698 | 0.698 | 1.339 | 0.9842 | 2 |
| E imp. cal. unc. (k=1) | 0.086 | 0.000 | 0.000 | 0.000 | 0.914 | 0.698 | 0.698 | 1.339 | 0.9842 | 2 |
| F physics oracle | 0.086 | 0.000 | 0.000 | 0.000 | 0.914 | 0.698 | 0.698 | 1.339 | 0.9842 | 0 |

## [improved_stable] Kdl — all load levels (n = 8 cases, 7 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.875 | 1.000 | 0.000 | 0.000 | 0.000 | 0.875 | 5.978 | 0.8006 | 1198 |
| B orig. mean | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5073 |
| C orig. unc. | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5143 |
| D imp. mean | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5040 |
| E imp. cal. unc. (k=1) | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5525 |
| F physics oracle | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 1924 |

## [improved_stable] Kdl — fixed load (n = 4 cases, 4 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 1.000 | 1.000 | – | 0.000 | 0.000 | 1.000 | 10.724 | 0.6086 | 823 |
| B orig. mean | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 3787 |
| C orig. unc. | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 4050 |
| D imp. mean | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 4185 |
| E imp. cal. unc. (k=1) | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 4347 |
| F physics oracle | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 1449 |

## [improved_stable] Kdl — calibrated load (n = 4 cases, 3 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.750 | 1.000 | 0.000 | 0.000 | 0.000 | 0.750 | 1.232 | 0.9926 | 1574 |
| B orig. mean | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 6360 |
| C orig. unc. | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 6236 |
| D imp. mean | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 5896 |
| E imp. cal. unc. (k=1) | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 6703 |
| F physics oracle | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 2398 |

## [v2_calload_normscore] GEANT — all load levels (n = 1600 cases, 1386 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.866 | 1.000 | 0.000 | 0.000 | 0.000 | 0.866 | 2.209 | 0.8809 | 0 |
| B orig. mean | 0.133 | 0.000 | 0.000 | 0.005 | 0.867 | 0.681 | 0.681 | 1.553 | 0.9763 | 8 |
| C orig. unc. | 0.133 | 0.000 | 0.000 | 0.009 | 0.868 | 0.681 | 0.681 | 1.553 | 0.9763 | 7 |
| D imp. mean | 0.134 | 0.000 | 0.000 | 0.000 | 0.866 | 0.681 | 0.681 | 1.553 | 0.9763 | 7 |
| E imp. cal. unc. (k=1) | 0.134 | 0.000 | 0.000 | 0.000 | 0.866 | 0.681 | 0.681 | 1.553 | 0.9763 | 7 |
| F physics oracle | 0.134 | 0.000 | 0.000 | 0.000 | 0.866 | 0.681 | 0.681 | 1.553 | 0.9763 | 1 |

## [v2_calload_normscore] GEANT — fixed load (n = 800 cases, 655 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.819 | 1.000 | 0.000 | 0.000 | 0.000 | 0.819 | 2.359 | 0.8652 | 0 |
| B orig. mean | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 8 |
| C orig. unc. | 0.180 | 0.000 | 0.000 | 0.007 | 0.820 | 0.665 | 0.665 | 1.767 | 0.9683 | 7 |
| D imp. mean | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 7 |
| E imp. cal. unc. (k=1) | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 7 |
| F physics oracle | 0.181 | 0.000 | 0.000 | 0.000 | 0.819 | 0.665 | 0.665 | 1.768 | 0.9683 | 1 |

## [v2_calload_normscore] GEANT — calibrated load (n = 800 cases, 731 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.914 | 1.000 | 0.000 | 0.000 | 0.000 | 0.914 | 2.058 | 0.8966 | 0 |
| B orig. mean | 0.085 | 0.000 | 0.000 | 0.014 | 0.915 | 0.698 | 0.698 | 1.339 | 0.9842 | 8 |
| C orig. unc. | 0.085 | 0.000 | 0.000 | 0.014 | 0.915 | 0.698 | 0.698 | 1.339 | 0.9842 | 7 |
| D imp. mean | 0.086 | 0.000 | 0.000 | 0.000 | 0.914 | 0.698 | 0.698 | 1.339 | 0.9842 | 7 |
| E imp. cal. unc. (k=1) | 0.086 | 0.000 | 0.000 | 0.000 | 0.914 | 0.698 | 0.698 | 1.339 | 0.9842 | 8 |
| F physics oracle | 0.086 | 0.000 | 0.000 | 0.000 | 0.914 | 0.698 | 0.698 | 1.339 | 0.9842 | 1 |

## [v2_calload_normscore] Kdl — all load levels (n = 8 cases, 7 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.875 | 1.000 | 0.000 | 0.000 | 0.000 | 0.875 | 5.978 | 0.8004 | 1132 |
| B orig. mean | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5364 |
| C orig. unc. | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5577 |
| D imp. mean | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5664 |
| E imp. cal. unc. (k=1) | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 5484 |
| F physics oracle | 0.125 | 0.000 | 0.000 | 0.000 | 0.875 | 0.625 | 0.625 | 3.911 | 0.9294 | 2047 |

## [v2_calload_normscore] Kdl — fixed load (n = 4 cases, 4 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 1.000 | 1.000 | – | 0.000 | 0.000 | 1.000 | 10.724 | 0.6083 | 1187 |
| B orig. mean | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 5422 |
| C orig. unc. | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 5774 |
| D imp. mean | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 5891 |
| E imp. cal. unc. (k=1) | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 5460 |
| F physics oracle | 0.000 | 0.000 | 0.000 | – | 1.000 | 1.000 | 1.000 | 6.964 | 0.8591 | 2098 |

## [v2_calload_normscore] Kdl — calibrated load (n = 4 cases, 3 unsafe proposals)

| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A policy only | 1.000 | 0.750 | 1.000 | 0.000 | 0.000 | 0.000 | 0.750 | 1.232 | 0.9925 | 1076 |
| B orig. mean | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 5306 |
| C orig. unc. | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 5380 |
| D imp. mean | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 5438 |
| E imp. cal. unc. (k=1) | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 5508 |
| F physics oracle | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.250 | 0.859 | 0.9998 | 1996 |

## Per load level: E (full framework) vs references

| Policy | Topology | Mode | Level | n | Policy MLU | Shortest | Equal | LP | Policy/shortest | Policy feasible | E accept | E unsafe acc. | E false rej. | E applied MLU | E applied sat. | LP sat. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| improved_stable | GEANT | calibrated | 0.5 | 200 | 1.287 | 0.881 | 0.863 | 0.500 | 1.488 | 0.34 | 0.34 | 0.00 | 0.00 | 0.833 | 1.0000 | 1.0000 |
| improved_stable | GEANT | calibrated | 0.7 | 200 | 1.801 | 1.233 | 1.209 | 0.700 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.113 | 0.9962 | 1.0000 |
| improved_stable | GEANT | calibrated | 0.9 | 200 | 2.316 | 1.585 | 1.554 | 0.900 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.497 | 0.9819 | 1.0000 |
| improved_stable | GEANT | calibrated | 1.1 | 200 | 2.831 | 1.937 | 1.900 | 1.100 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.914 | 0.9587 | 0.9735 |
| improved_stable | GEANT | fixed | 0.25 | 200 | 0.944 | 0.709 | 0.665 | 0.398 | 1.488 | 0.72 | 0.72 | 0.00 | 0.00 | 0.795 | 1.0000 | 1.0000 |
| improved_stable | GEANT | fixed | 0.5 | 200 | 1.888 | 1.419 | 1.329 | 0.797 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.340 | 0.9960 | 0.9944 |
| improved_stable | GEANT | fixed | 0.75 | 200 | 2.832 | 2.128 | 1.994 | 1.195 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 2.098 | 0.9651 | 0.9552 |
| improved_stable | GEANT | fixed | 1.0 | 200 | 3.776 | 2.837 | 2.659 | 1.593 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 2.837 | 0.9123 | 0.8820 |
| improved_stable | Kdl | calibrated | 0.5 | 1 | 0.770 | 0.500 | 0.489 | – | 1.540 | 1.00 | 1.00 | 0.00 | 0.00 | 0.770 | 1.0000 | – |
| improved_stable | Kdl | calibrated | 0.7 | 1 | 1.078 | 0.700 | 0.685 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 0.685 | 1.0000 | – |
| improved_stable | Kdl | calibrated | 0.9 | 1 | 1.386 | 0.900 | 0.881 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 0.881 | 1.0000 | – |
| improved_stable | Kdl | calibrated | 1.1 | 1 | 1.694 | 1.100 | 1.076 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 1.100 | 0.9991 | – |
| improved_stable | Kdl | fixed | 0.25 | 1 | 4.290 | 2.785 | 2.725 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 2.785 | 0.9629 | – |
| improved_stable | Kdl | fixed | 0.5 | 1 | 8.579 | 5.571 | 5.451 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 5.571 | 0.9053 | – |
| improved_stable | Kdl | fixed | 0.75 | 1 | 12.869 | 8.357 | 8.177 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 8.357 | 0.8490 | – |
| improved_stable | Kdl | fixed | 1.0 | 1 | 17.158 | 11.142 | 10.902 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 11.142 | 0.7192 | – |
| v2_calload_normscore | GEANT | calibrated | 0.5 | 200 | 1.287 | 0.881 | 0.863 | 0.500 | 1.488 | 0.34 | 0.34 | 0.00 | 0.00 | 0.833 | 1.0000 | 1.0000 |
| v2_calload_normscore | GEANT | calibrated | 0.7 | 200 | 1.801 | 1.233 | 1.209 | 0.700 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.113 | 0.9962 | 1.0000 |
| v2_calload_normscore | GEANT | calibrated | 0.9 | 200 | 2.316 | 1.585 | 1.554 | 0.900 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.497 | 0.9819 | 1.0000 |
| v2_calload_normscore | GEANT | calibrated | 1.1 | 200 | 2.830 | 1.937 | 1.900 | 1.100 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.914 | 0.9587 | 0.9735 |
| v2_calload_normscore | GEANT | fixed | 0.25 | 200 | 0.944 | 0.709 | 0.665 | 0.398 | 1.488 | 0.72 | 0.72 | 0.00 | 0.00 | 0.795 | 1.0000 | 1.0000 |
| v2_calload_normscore | GEANT | fixed | 0.5 | 200 | 1.888 | 1.419 | 1.329 | 0.797 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 1.340 | 0.9960 | 0.9944 |
| v2_calload_normscore | GEANT | fixed | 0.75 | 200 | 2.831 | 2.128 | 1.994 | 1.195 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 2.098 | 0.9651 | 0.9552 |
| v2_calload_normscore | GEANT | fixed | 1.0 | 200 | 3.775 | 2.837 | 2.659 | 1.593 | 1.488 | 0.00 | 0.00 | 0.00 | 0.00 | 2.837 | 0.9123 | 0.8820 |
| v2_calload_normscore | Kdl | calibrated | 0.5 | 1 | 0.770 | 0.500 | 0.489 | – | 1.540 | 1.00 | 1.00 | 0.00 | 0.00 | 0.770 | 1.0000 | – |
| v2_calload_normscore | Kdl | calibrated | 0.7 | 1 | 1.078 | 0.700 | 0.685 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 0.685 | 1.0000 | – |
| v2_calload_normscore | Kdl | calibrated | 0.9 | 1 | 1.386 | 0.900 | 0.881 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 0.881 | 1.0000 | – |
| v2_calload_normscore | Kdl | calibrated | 1.1 | 1 | 1.694 | 1.100 | 1.076 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 1.100 | 0.9991 | – |
| v2_calload_normscore | Kdl | fixed | 0.25 | 1 | 4.290 | 2.785 | 2.725 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 2.785 | 0.9629 | – |
| v2_calload_normscore | Kdl | fixed | 0.5 | 1 | 8.579 | 5.571 | 5.451 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 5.571 | 0.9053 | – |
| v2_calload_normscore | Kdl | fixed | 0.75 | 1 | 12.869 | 8.357 | 8.177 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 8.357 | 0.8490 | – |
| v2_calload_normscore | Kdl | fixed | 1.0 | 1 | 17.158 | 11.142 | 10.902 | – | 1.540 | 0.00 | 0.00 | 0.00 | 0.00 | 11.142 | 0.7192 | – |

## Mean-only (D) vs uncertainty-aware (E) and E vs oracle (F)

| Policy | Topology | Mode | n | D-only unsafe accepts | E-only unsafe accepts | D/E accept discordant | E/F accept discordant |
|---|---|---|---:|---:|---:|---:|---:|
| improved_stable | GEANT | all | 1600 | 0 | 0 | 0 | 0 |
| improved_stable | GEANT | calibrated | 800 | 0 | 0 | 0 | 0 |
| improved_stable | GEANT | fixed | 800 | 0 | 0 | 0 | 0 |
| improved_stable | Kdl | all | 8 | 0 | 0 | 0 | 0 |
| improved_stable | Kdl | calibrated | 4 | 0 | 0 | 0 | 0 |
| improved_stable | Kdl | fixed | 4 | 0 | 0 | 0 | 0 |
| v2_calload_normscore | GEANT | all | 1600 | 0 | 0 | 0 | 0 |
| v2_calload_normscore | GEANT | calibrated | 800 | 0 | 0 | 0 | 0 |
| v2_calload_normscore | GEANT | fixed | 800 | 0 | 0 | 0 | 0 |
| v2_calload_normscore | Kdl | all | 8 | 0 | 0 | 0 | 0 |
| v2_calload_normscore | Kdl | calibrated | 4 | 0 | 0 | 0 | 0 |
| v2_calload_normscore | Kdl | fixed | 4 | 0 | 0 | 0 | 0 |

## Artifacts

- `case_rows.jsonl`: one compact row per case with all six variants and the references.
- `decisions.jsonl`: full decision records (same schema as the dev A–F run; large, not committed).
- `summary.json`, `config_<policy_id>.json` (frozen stack: `improved_stable`; policy v2 test pass: `v2_calload_normscore`).
