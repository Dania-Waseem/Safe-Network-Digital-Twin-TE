# Feasibility-calibrated traffic

**Setting label:** *calibrated load* (an additional evaluation setting; all fixed-scale results under `results/evaluation/` are kept unchanged).

## Why

Toy (UsCarrier, Kdl) and reconstructed (Cogentco gravity) matrices have arbitrary units. The fixed scales 0.25–1.0 leave these topologies above capacity for every routing, so no method can be safe and every method falls back. Calibration rescales each TM so that a method-independent reference MLU hits a chosen target, making the feasibility boundary visible.

## Definition

For each TM *D*: `alpha(t) = t / reference_mlu(D)`, calibrated TM = `alpha(t) * D`, targets t ∈ [0.5, 0.7, 0.9, 1.1].

| Topology | Reference MLU | Why |
|---|---|---|
| B4, GEANT, UsCarrier, Cogentco | k=4 path-LP optimum (min-MLU) | LP tractable (≤ 7 s per TM) |
| Kdl | shortest-path MLU | LP over ~2.3 M path variables not tractable on this laptop |

- The reference uses only the TM and the capacities, never any method's output.
- Shortest, equal and LP are linear in *D*, so one LP per TM is exact for every target.
- The identical scaled TM is given to every method (paired design).
- With the LP reference, a feasible k=4 routing exists **by construction** when t ≤ 1 (LP MLU = t) and none exists at t = 1.1. This is a design property, not a finding.
- With the shortest reference (Kdl), shortest itself is feasible when t ≤ 1.
- All LPs use SciPy HiGHS (scipy.optimize.linprog, method='highs'). No Gurobi licence was available, so Teal/FIGRET-style Gurobi LPs are replaced by HiGHS.
- UsCarrier and Kdl traffic matrices are synthetic all-ones toy templates; every index is an identical copy, so additional TM indices or seeds add no traffic diversity on these topologies. Identical TMs are detected by hash; their LP is solved once and reused.
- GEANT: fixed seeded subsample of 200 of the 2,693 FIGRET test TMs (seed 2026); 50 of these (seed 2027) are reserved for E3.

## Reference MLU per topology (native scale)

| Topology | Split | TMs | Unique TMs | Reference | Shortest MLU | Equal MLU | LP MLU | LP s/TM |
|---|---|---:|---:|---|---:|---:|---:|---:|
| B4 | train | 20 | 20 | lp | 1.084 | 0.993 | 0.941 | 0.00 |
| B4 | val | 8 | 8 | lp | 1.118 | 1.026 | 0.974 | 0.00 |
| UsCarrier | train | 20 | 1 | lp | 7.480 | 9.810 | 3.140 | 2.61 |
| UsCarrier | val | 8 | 1 | lp | 7.480 | 9.810 | 3.140 | n/a |
| Cogentco | train | 20 | 20 | lp | 66.711 | 64.701 | 44.485 | 6.69 |
| Cogentco | val | 8 | 8 | lp | 66.707 | 64.827 | 44.542 | 6.78 |
| GEANT | test | 200 | 200 | lp | 2.837 | 2.659 | 1.593 | 0.01 |
| Kdl | test | 8 | 1 | shortest | 11.142 | 10.902 | n/a | n/a |

## Fraction of feasible scenarios (MLU ≤ 1), before (fixed scale) and after (calibrated)

| Topology | Split | Mode | Level | Shortest | Equal | LP | Shortest or equal |
|---|---|---|---:|---:|---:|---:|---:|
| B4 | train | fixed | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | train | fixed | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | train | fixed | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | train | fixed | 1.0 | 0.00 | 0.50 | 1.00 | 0.50 |
| B4 | train | calibrated | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | train | calibrated | 0.7 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | train | calibrated | 0.9 | 0.10 | 1.00 | 1.00 | 1.00 |
| B4 | train | calibrated | 1.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| B4 | val | fixed | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | val | fixed | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | val | fixed | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | val | fixed | 1.0 | 0.00 | 0.12 | 0.88 | 0.12 |
| B4 | val | calibrated | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | val | calibrated | 0.7 | 1.00 | 1.00 | 1.00 | 1.00 |
| B4 | val | calibrated | 0.9 | 0.00 | 1.00 | 1.00 | 1.00 |
| B4 | val | calibrated | 1.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | train | fixed | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | train | fixed | 0.5 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | train | fixed | 0.75 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | train | fixed | 1.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | train | calibrated | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cogentco | train | calibrated | 0.7 | 0.00 | 0.00 | 1.00 | 0.00 |
| Cogentco | train | calibrated | 0.9 | 0.00 | 0.00 | 1.00 | 0.00 |
| Cogentco | train | calibrated | 1.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | val | fixed | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | val | fixed | 0.5 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | val | fixed | 0.75 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | val | fixed | 1.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cogentco | val | calibrated | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cogentco | val | calibrated | 0.7 | 0.00 | 0.00 | 1.00 | 0.00 |
| Cogentco | val | calibrated | 0.9 | 0.00 | 0.00 | 1.00 | 0.00 |
| Cogentco | val | calibrated | 1.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| GEANT | eval_test | fixed | 0.25 | 0.79 | 0.93 | 1.00 | 0.95 |
| GEANT | eval_test | fixed | 0.5 | 0.27 | 0.28 | 0.73 | 0.34 |
| GEANT | eval_test | fixed | 0.75 | 0.02 | 0.04 | 0.41 | 0.04 |
| GEANT | eval_test | fixed | 1.0 | 0.00 | 0.00 | 0.15 | 0.00 |
| GEANT | eval_test | calibrated | 0.5 | 1.00 | 0.85 | 1.00 | 1.00 |
| GEANT | eval_test | calibrated | 0.7 | 0.04 | 0.20 | 1.00 | 0.21 |
| GEANT | eval_test | calibrated | 0.9 | 0.00 | 0.00 | 1.00 | 0.00 |
| GEANT | eval_test | calibrated | 1.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| Kdl | eval_test | fixed | 0.25 | 0.00 | 0.00 | n/a | 0.00 |
| Kdl | eval_test | fixed | 0.5 | 0.00 | 0.00 | n/a | 0.00 |
| Kdl | eval_test | fixed | 0.75 | 0.00 | 0.00 | n/a | 0.00 |
| Kdl | eval_test | fixed | 1.0 | 0.00 | 0.00 | n/a | 0.00 |
| Kdl | eval_test | calibrated | 0.5 | 1.00 | 1.00 | n/a | 1.00 |
| Kdl | eval_test | calibrated | 0.7 | 1.00 | 1.00 | n/a | 1.00 |
| Kdl | eval_test | calibrated | 0.9 | 1.00 | 1.00 | n/a | 1.00 |
| Kdl | eval_test | calibrated | 1.1 | 0.00 | 0.00 | n/a | 0.00 |
| UsCarrier | train | fixed | 0.25 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | train | fixed | 0.5 | 0.00 | 0.00 | 0.00 | 0.00 |
| UsCarrier | train | fixed | 0.75 | 0.00 | 0.00 | 0.00 | 0.00 |
| UsCarrier | train | fixed | 1.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| UsCarrier | train | calibrated | 0.5 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | train | calibrated | 0.7 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | train | calibrated | 0.9 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | train | calibrated | 1.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| UsCarrier | val | fixed | 0.25 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | val | fixed | 0.5 | 0.00 | 0.00 | 0.00 | 0.00 |
| UsCarrier | val | fixed | 0.75 | 0.00 | 0.00 | 0.00 | 0.00 |
| UsCarrier | val | fixed | 1.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| UsCarrier | val | calibrated | 0.5 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | val | calibrated | 0.7 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | val | calibrated | 0.9 | 0.00 | 0.00 | 1.00 | 0.00 |
| UsCarrier | val | calibrated | 1.1 | 0.00 | 0.00 | 0.00 | 0.00 |

## Reading the table

- *Before*: rows with mode `fixed`. *After*: rows with mode `calibrated`.
- The `LP` column under calibration is 1.00 for t ≤ 0.9 and 0.00 for t = 1.1 by construction.
- Where LP is feasible but shortest/equal are not, a smarter routing (e.g. a learned policy) can add value; where all are infeasible, the best any method can do is reject and fall back.

## Artifacts

- `calibration_rows.jsonl`, `calibration_table.csv` — per-TM MLUs and alpha
- `feasibility_before_after.csv` — table above
- `scaled_tm_index.json` — alpha per (topology/split/position), GEANT subsample positions
- `config.json` — targets, scales, seeds, environment
