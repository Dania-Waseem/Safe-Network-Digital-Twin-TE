# B4 link failures — P1 §5.5-comparable table (frozen P1 baseline)

Frozen B4 protocol (`BASELINE_FREEZE`): TEAL real TMs, test TMs 28–35, Yen k=4, 10 archived checkpoints (seeds 100–1000), Teal-order ADMM (ρ=1, 2 iterations) + constraint adjustment. Failure cases from `data/scenarios/link_failures/B4.json` (P1 §5.5 sampler); failure seed s uses checkpoint seed s. Failed links get capacity 0; paths are not recomputed (P1-style). LP = max-throughput path LP, SciPy HiGHS (no Gurobi licence).

Satisfaction (%), mean ± SD over the 10 seeds (each seed = mean over the 8 test TMs):

| Failed links | GRL-TE + Teal ADMM | GRL-TE no ADMM | LP | ECMP | Shortest path | P1 reported GRL-TE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 97.44 ± 1.68 | 96.51 ± 2.75 | 99.74 ± 0.00 | 98.74 ± 0.00 | 97.76 ± 0.00 | 92.39 (Table 6, intact) |
| 1 | 91.29 ± 2.83 | 88.20 ± 4.27 | 99.74 ± 0.00 | 90.48 ± 3.03 | 91.33 ± 5.39 | – |
| 2 | 86.36 ± 2.67 | 81.87 ± 3.99 | 97.73 ± 3.26 | 84.02 ± 3.03 | 85.12 ± 4.95 | – |
| 3 | 80.61 ± 4.61 | 75.22 ± 5.76 | 96.42 ± 5.24 | 76.54 ± 5.21 | 82.27 ± 7.77 | 77.25 (§5.5) |

Change in GRL-TE satisfaction: 0 → 3 failed links **-16.84 points**, 1 → 3 failed links **-10.68 points**. P1 §5.5 (as quoted in the task brief) reports B4 dropping from 90.80% to 77.25% (−13.55 points) with 3 failed links; P1's intact B4 value (Table 6) is 92.39%, so the 90.80% starting point may correspond to a smaller number of failures — it is not assigned to a row here. Absolute levels differ because the frozen baseline already differs from P1 on intact B4 (97.44% vs 92.39%, see BASELINE_FREEZE).
