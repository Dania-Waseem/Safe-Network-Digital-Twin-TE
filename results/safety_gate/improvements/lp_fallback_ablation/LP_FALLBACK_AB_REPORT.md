# Conditional LP Fallback — Offline A/B Report

Experimental evaluation only. Production gate/fallback/Twin/RL/eval artifacts were not modified.

**Source decisions:** `af_eval_k1_calibrated_devtopos_v1` / variant `E_improved_twin_calibrated_uncertainty`.

## 1. Design

- **Variant A:** frozen gate decisions + existing shortest/equal least-violating fallback.
- **Variant B:** same gate decisions; if rejected and existing fallback infeasible, compute forced-link LB; invoke catalog min-MLU LP only if LB ≤ 1; apply LP only if independently verified feasible; otherwise keep A’s fallback.

Cases: **80** (unique ids: 80).
Gate decisions changed: **False**.
Validation: {'all_80_present': True, 'no_infeasible_lp_marked_feasible': True, 'gate_unchanged_all': True}.

## 2. Overall results

| Metric | A | B |
|---|---:|---:|
| Unsafe acceptance rate | 0.0000 | 0.0000 |
| Applied overload rate | 0.5000 | 0.4375 |
| Overloaded cases | 40 | 35 |
| Sum total overload | 518751460.399 | 518750661.649 |
| Feasible fallback rate (rejected) | 0.0000 | 0.1250 |
| Feasible fallback count (rejected) | 0 | 5 |
| Mean applied MLU | 11.8972 | 11.8294 |
| Mean overloaded links | 102.24 | 102.11 |

- Recovered by LP (A infeasible → B feasible): **5**
- Unrecovered rejected cases under B: **35**
- LP invocations: **5**; applied successes: **5**; success rate: **1.0**

## 3. Computational cost

- LP invocation count: **5**
- Mean / median / p95 / max runtime: **8.752s** / **8.807s** / **9.478s** / **9.482s**
- Sum of LP solve time: **43.758s**
- Total experiment wall time: **127.9s**

## 4. Paired comparison (descriptive)

- Mean MLU(B−A): {'mean': -0.0678125, 'ci95_low': -0.12572841349531522, 'ci95_high': -0.009896586504684793, 'n': 80} (negative means B lower MLU).
- Mean overload(B−A): {'mean': -9.984375, 'ci95_low': -18.51162585795079, 'ci95_high': -1.45712414204921, 'n': 80}
- Feasibility gain rate: {'mean': 0.0625, 'ci95_low': 0.009121277884502108, 'ci95_high': 0.11587872211549789, 'n': 80}
- Rejected feasibility discordance: {'A_feasible_B_infeasible': 0, 'A_infeasible_B_feasible': 5, 'note': 'Counts only; no p-value claimed without an explicit McNemar test decision rule.'}

No statistical significance claim is made beyond these paired descriptive summaries.

## 5. Topology breakdown

- **B4** (n=40): overload cases A/B=0/0, feasible FB rate A/B=nan/nan, LP applied=0, recovered=0
- **Cogentco** (n=20): overload cases A/B=20/20, feasible FB rate A/B=0.000/0.000, LP applied=0, recovered=0
- **UsCarrier** (n=20): overload cases A/B=20/15, feasible FB rate A/B=0.000/0.250, LP applied=5, recovered=5

## 6. Traffic-scale breakdown

- **scale=0.25** (n=20): overload cases A/B=10/5, LP applied=5, recovered=5
- **scale=0.5** (n=20): overload cases A/B=10/10, LP applied=0, recovered=0
- **scale=0.75** (n=20): overload cases A/B=10/10, LP applied=0, recovered=0
- **scale=1.0** (n=20): overload cases A/B=10/10, LP applied=0, recovered=0

## 7. Direct answers

1. **Does conditional LP fallback improve the feasible fallback rate?** **Yes** on rejected cases: 0.000 → 0.125 (0 → 5 feasible).
2. **How many additional cases become feasible?** **5** (all LP-applied recoveries).
3. **Does it reduce applied overload?** **Yes** where LP applies: overloaded cases 40 → 35; sum total overload 518751460.399 → 518750661.649.
4. **Does it change any Safety Gate decisions?** **No.** Acceptance/rejection is identical for A and B by design.
5. **Computational overhead?** 5 LP solves; mean 8.75s, p95 9.48s; experiment wall 127.9s (includes forced-LB screening on infeasible-fallback rejects).
6. **Is the improvement practically worthwhile?** **Narrowly yes for the catalog-feasible slice** (here the 5 UsCarrier@0.25 cases): perfect recovery when LP is eligible. **Not transformative globally:** most rejected failures remain capacity-infeasible under the catalog and are correctly skipped by the forced-link gate.
7. **Limitations remaining?** Catalog path set still insufficient for 35/40 prior overload failures; LP latency (~seconds) is high for online TE; continuous splits; no path expansion; toy/val matrices; conditional trigger depends on LB quality.
8. **Further fallback development vs final research evaluation?** Further fallback algorithm work is only justified as a **conditional, offline-validated catalog LP option for LB≤1 rejects**. The dominant residual risk is **demand/capacity/path-catalog infeasibility**, so the project should move to **final research evaluation / reporting** with this conditional LP documented as a limited optional recovery mechanism—not a production gate redesign.

## 8. Artifacts

- `paired_results.csv`
- `case_level_results.json`
- `lp_invocation_log.json`
- `topology_breakdown.csv`
- `traffic_scale_breakdown.csv`
- `plots/overload_A_vs_B.png`, `mlu_A_vs_B.png`, `feasible_fallback_rate.png`,
  `lp_runtime_distribution.png`, `lp_invocation_by_topo_scale.png`

## 9. Preservation

No modifications to production gate/fallback code, checkpoints, matrices, splits, or existing A–F evaluation artifacts.
