# P1 GRL-TE on the correct GEANT (FIGRET 22/72, real TMs)

Reproduction code unchanged; new runner `phase3_final/geant_figret.py`. Train: 1000 FIGRET train TMs; test: 200 seeded FIGRET test TMs (same subsample as Phase 3). Yen k=4 min-hop paths, TopoFlowNet L=6 + one-step A2C, 3 episodes, seeds 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000; ADMM rho=1, 100 iterations. h0: raw (P1 eq. 5).

Training uses a seeded subsample of 1000 of the 8,080 FIGRET train TMs (seed 2026; runtime deviation, fixed before any result).

LP = max-throughput path LP (P1 eqs. 1–4) solved with SciPy HiGHS (no Gurobi licence available).

## Demand satisfaction (%), mean ± SD over seeds (TM-mean per seed)

| Method | This reproduction | P1 Table 6 (GEANT) |
|---|---:|---:|
| GRL-TE + ADMM (Teal-order, headline) | 94.44 ± 1.81 (n=10) | 86.41 |
| GRL-TE + ADMM (P1-order, diagnostic) | failed on every TM (singular linear system in the F-update) | – |
| GRL-TE without ADMM | 81.77 ± 7.06 (n=10) | 71.32 (= 86.41 − 15.09) |
| LP (max throughput) | 97.51 | 88.40 |
| ECMP (equal split over 4 paths) | 88.29 | – |
| Shortest path | 91.23 | – |
| NCFlow | not run | 88.99 (reported in P1, not reproduced here) |

## ADMM ablation

Removing ADMM changes satisfaction by **-12.68 points** (no-ADMM − Teal-ADMM); P1 reports −15.09 points on GEANT.

## Per seed

| Seed | No ADMM | Teal ADMM | P1 ADMM | MLU (Teal) | Train s | Params finite |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 85.84 | 96.47 | – (200 of 200 TMs failed) | 1.000 | 194 | True |
| 200 | 71.29 | 95.84 | – (200 of 200 TMs failed) | 1.000 | 115 | True |
| 300 | 70.20 | 95.26 | – (200 of 200 TMs failed) | 1.000 | 148 | True |
| 400 | 78.25 | 93.29 | – (200 of 200 TMs failed) | 1.000 | 153 | True |
| 500 | 88.65 | 91.93 | – (200 of 200 TMs failed) | 1.000 | 199 | True |
| 600 | 79.88 | 90.85 | – (200 of 200 TMs failed) | 1.000 | 186 | True |
| 700 | 76.94 | 95.22 | – (200 of 200 TMs failed) | 1.000 | 128 | True |
| 800 | 89.06 | 94.39 | – (200 of 200 TMs failed) | 1.000 | 99 | True |
| 900 | 89.24 | 96.62 | – (200 of 200 TMs failed) | 1.000 | 140 | True |
| 1000 | 88.29 | 94.55 | – (200 of 200 TMs failed) | 1.000 | 184 | True |

P1-order ADMM failed (singular linear system in the F-update) on 2000 of 2000 seed×TM evaluations; its mean is over successful TMs only.

## Caveats

- P1 does not publish its GEANT TM split; we use FIGRET's published train/test histories.
- Teal-order ADMM is the frozen headline (stable); P1's printed order is reported separately and is known to be unstable at small iteration counts (BASELINE_FREEZE/ADMM_VARIANT_COMPARISON.md).
- Teal, FIGRET and NCFlow were not run (not part of this study); their numbers are quoted from P1 only.
