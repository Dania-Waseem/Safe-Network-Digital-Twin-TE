# P1 GRL-TE on the correct GEANT (FIGRET 22/72, real TMs)

Reproduction code unchanged; new runner `phase3_final/geant_figret.py`. Train: 1000 FIGRET train TMs; test: 200 seeded FIGRET test TMs (same subsample as Phase 3). Yen k=4 min-hop paths, TopoFlowNet L=6 + one-step A2C, 3 episodes, seeds 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000; ADMM rho=1, 100 iterations. h0: FIGRET units (/1e9).

Training uses a seeded subsample of 1000 of the 8,080 FIGRET train TMs (seed 2026; runtime deviation, fixed before any result).

LP = max-throughput path LP (P1 eqs. 1–4) solved with SciPy HiGHS (no Gurobi licence available).

## Demand satisfaction (%), mean ± SD over seeds (TM-mean per seed)

| Method | This reproduction | P1 Table 6 (GEANT) |
|---|---:|---:|
| GRL-TE + ADMM (Teal-order, headline) | 96.71 ± 0.03 (n=10) | 86.41 |
| GRL-TE + ADMM (P1-order, diagnostic) | failed on every TM (singular linear system in the F-update) | – |
| GRL-TE without ADMM | 83.23 ± 6.53 (n=10) | 71.32 (= 86.41 − 15.09) |
| LP (max throughput) | 97.51 | 88.40 |
| ECMP (equal split over 4 paths) | 88.29 | – |
| Shortest path | 91.23 | – |
| NCFlow | not run | 88.99 (reported in P1, not reproduced here) |

## ADMM ablation

Removing ADMM changes satisfaction by **-13.48 points** (no-ADMM − Teal-ADMM); P1 reports −15.09 points on GEANT.

## Per seed

| Seed | No ADMM | Teal ADMM | P1 ADMM | MLU (Teal) | Train s | Params finite |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 76.78 | 96.70 | – (200 of 200 TMs failed) | 1.000 | 186 | True |
| 200 | 88.34 | 96.72 | – (200 of 200 TMs failed) | 1.000 | 167 | True |
| 300 | 76.38 | 96.67 | – (200 of 200 TMs failed) | 1.000 | 133 | True |
| 400 | 77.81 | 96.72 | – (200 of 200 TMs failed) | 1.000 | 270 | True |
| 500 | 88.49 | 96.75 | – (200 of 200 TMs failed) | 1.000 | 241 | True |
| 600 | 88.58 | 96.74 | – (200 of 200 TMs failed) | 1.000 | 136 | True |
| 700 | 71.13 | 96.66 | – (200 of 200 TMs failed) | 1.000 | 118 | True |
| 800 | 89.69 | 96.74 | – (200 of 200 TMs failed) | 1.000 | 119 | True |
| 900 | 86.81 | 96.69 | – (200 of 200 TMs failed) | 1.000 | 95 | True |
| 1000 | 88.29 | 96.72 | – (200 of 200 TMs failed) | 1.000 | 72 | True |

P1-order ADMM failed (singular linear system in the F-update) on 2000 of 2000 seed×TM evaluations; its mean is over successful TMs only.

## Caveats

- P1 does not publish its GEANT TM split; we use FIGRET's published train/test histories.
- Teal-order ADMM is the frozen headline (stable); P1's printed order is reported separately and is known to be unstable at small iteration counts (BASELINE_FREEZE/ADMM_VARIANT_COMPARISON.md).
- Teal, FIGRET and NCFlow were not run (not part of this study); their numbers are quoted from P1 only.
