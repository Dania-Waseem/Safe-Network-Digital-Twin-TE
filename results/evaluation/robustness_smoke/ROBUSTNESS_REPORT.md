# Phase 3 multi-seed robustness evaluation

All methods saw the **same** seeded traffic matrices within each seed.
GEANT and Kdl were evaluated only on their **test** splits (unseen).
No training or hyperparameter tuning was performed in this step.

## Configuration

- Seeds: `[101]`
- Scales: `[0.25, 0.5, 0.75, 1.0]`
- Matrix draws: B4×2 (val), UsCarrier×1 (val), Cogentco×1 (val), GEANT×2 (test), Kdl×1 (test)
- Checkpoints: `{
  "frozen_p1": "C:\\Users\\hp\\Pictures\\CNET phase 3\\GRL_TE_Reproduction\\BASELINE_FREEZE\\models\\grl_te_b4_freeze_seed100.pt",
  "general_3epoch": "C:\\Users\\hp\\Pictures\\CNET phase 3\\phase3 implementation\\results\\policy\\general_grl.pt",
  "improved_best": "C:\\Users\\hp\\Pictures\\CNET phase 3\\phase3 implementation\\results\\policy\\improved\\best_checkpoint.pt",
  "improved_stable": "C:\\Users\\hp\\Pictures\\CNET phase 3\\phase3 implementation\\results\\policy\\improved\\recommended_stable.pt"
}`
- Feasible scenario: shortest **or** equal split has zero capacity violations.

## Per-topology actor MLU (mean ± std across seeds)

| Topology | Scale | Feasible rate | Frozen P1 | General 3-epoch | Improved best | Improved stable | Shortest | Equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 | 0.25 | 1.00 | 0.314±0 | 0.291±0 | 0.315±0 | 0.25±0 | 0.279±0 | 0.255±0 |
| B4 | 0.50 | 1.00 | 0.623±0 | 0.584±0 | 0.625±0 | 0.491±0 | 0.557±0 | 0.51±0 |
| B4 | 0.75 | 1.00 | 0.921±0 | 0.878±0 | 0.935±0 | 0.737±0 | 0.836±0 | 0.765±0 |
| B4 | 1.00 | 0.00 | 1.23±0 | 1.17±0 | 1.24±0 | 0.984±0 | 1.11±0 | 1.02±0 |
| UsCarrier | 0.25 | 0.00 | 2.83±0 | 3.24±0 | 3.38±0 | 3.01±0 | 1.87±0 | 2.45±0 |
| UsCarrier | 0.50 | 0.00 | 5.57±0 | 6.48±0 | 6.75±0 | 6.02±0 | 3.74±0 | 4.91±0 |
| UsCarrier | 0.75 | 0.00 | 7.99±0 | 9.71±0 | 10.1±0 | 9.03±0 | 5.61±0 | 7.36±0 |
| UsCarrier | 1.00 | 0.00 | 10.2±0 | 13±0 | 13.5±0 | 12±0 | 7.48±0 | 9.81±0 |
| Cogentco | 0.25 | 0.00 | 16.4±0 | 16.7±0 | 16.3±0 | 16.9±0 | 16.7±0 | 16.2±0 |
| Cogentco | 0.50 | 0.00 | 32.9±0 | 33.4±0 | 32.6±0 | 33.9±0 | 33.4±0 | 32.4±0 |
| Cogentco | 0.75 | 0.00 | 49.4±0 | 50±0 | 49±0 | 50.8±0 | 50.1±0 | 48.6±0 |
| Cogentco | 1.00 | 0.00 | 65.8±0 | 66.7±0 | 65.3±0 | 67.8±0 | 66.7±0 | 64.8±0 |
| GEANT | 0.25 | 1.00 | 0.662±0 | 0.889±0 | 0.9±0 | 0.886±0 | 0.535±0 | 0.69±0 |
| GEANT | 0.50 | 0.50 | 1.33±0 | 1.78±0 | 1.8±0 | 1.77±0 | 1.07±0 | 1.38±0 |
| GEANT | 0.75 | 0.00 | 2.01±0 | 2.66±0 | 2.7±0 | 2.66±0 | 1.61±0 | 2.07±0 |
| GEANT | 1.00 | 0.00 | 2.69±0 | 3.55±0 | 3.6±0 | 3.54±0 | 2.14±0 | 2.76±0 |
| Kdl | 0.25 | 0.00 | 4.28±0 | 4.29±0 | 4.32±0 | 4.29±0 | 2.79±0 | 2.73±0 |
| Kdl | 0.50 | 0.00 | 8.55±0 | 8.58±0 | 8.63±0 | 8.58±0 | 5.57±0 | 5.45±0 |
| Kdl | 0.75 | 0.00 | 12.8±0 | 12.9±0 | 13±0 | 12.9±0 | 8.36±0 | 8.18±0 |
| Kdl | 1.00 | 0.00 | 17.1±0 | 17.2±0 | 17.3±0 | 17.2±0 | 11.1±0 | 10.9±0 |

## Framework safety and applied routing

| Topology | Scale | Method | Accept | Unsafe | False reject | Fallback | Applied MLU | Actor viol. | Runtime s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 | 0.25 | frozen_p1 | 1±0 | 0±0 | 0±0 | 0±0 | 0.314±0 | 0±0 | 0.05±0 |
| B4 | 0.25 | general_3epoch | 1±0 | 0±0 | 0±0 | 0±0 | 0.291±0 | 0±0 | 0.029±0 |
| B4 | 0.25 | improved_best | 1±0 | 0±0 | 0±0 | 0±0 | 0.315±0 | 0±0 | 0.019±0 |
| B4 | 0.25 | improved_stable | 1±0 | 0±0 | 0±0 | 0±0 | 0.25±0 | 0±0 | 0.023±0 |
| B4 | 0.50 | frozen_p1 | 1±0 | 0±0 | 0±0 | 0±0 | 0.623±0 | 0±0 | 0.014±0 |
| B4 | 0.50 | general_3epoch | 1±0 | 0±0 | 0±0 | 0±0 | 0.584±0 | 0±0 | 0.012±0 |
| B4 | 0.50 | improved_best | 1±0 | 0±0 | 0±0 | 0±0 | 0.625±0 | 0±0 | 0.013±0 |
| B4 | 0.50 | improved_stable | 1±0 | 0±0 | 0±0 | 0±0 | 0.491±0 | 0±0 | 0.012±0 |
| B4 | 0.75 | frozen_p1 | 1±0 | 0±0 | 0±0 | 0±0 | 0.921±0 | 0±0 | 0.014±0 |
| B4 | 0.75 | general_3epoch | 1±0 | 0±0 | 0±0 | 0±0 | 0.878±0 | 0±0 | 0.014±0 |
| B4 | 0.75 | improved_best | 1±0 | 0±0 | 0±0 | 0±0 | 0.935±0 | 0±0 | 0.0098±0 |
| B4 | 0.75 | improved_stable | 1±0 | 0±0 | 0±0 | 0±0 | 0.737±0 | 0±0 | 0.012±0 |
| B4 | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 1.11±0 | 1.5±0 | 0.013±0 |
| B4 | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 1.08±0 | 4.5±0 | 0.013±0 |
| B4 | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 1.08±0 | 4±0 | 0.011±0 |
| B4 | 1.00 | improved_stable | 0.5±0 | 0±0 | 0.5±0 | 0.5±0 | 0.984±0 | 0±0 | 0.013±0 |
| UsCarrier | 0.25 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±0 | 27±0 | 0.98±0 |
| UsCarrier | 0.25 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±0 | 33±0 | 0.93±0 |
| UsCarrier | 0.25 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±0 | 33±0 | 1.3±0 |
| UsCarrier | 0.25 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±0 | 33±0 | 1.3±0 |
| UsCarrier | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±0 | 58±0 | 1±0 |
| UsCarrier | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±0 | 62±0 | 0.83±0 |
| UsCarrier | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±0 | 66±0 | 0.84±0 |
| UsCarrier | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±0 | 61±0 | 0.9±0 |
| UsCarrier | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.2e+02±0 | 1±0 |
| UsCarrier | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.3e+02±0 | 0.82±0 |
| UsCarrier | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.4e+02±0 | 0.83±0 |
| UsCarrier | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.3e+02±0 | 0.8±0 |
| UsCarrier | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±0 | 2.3e+02±0 | 1±0 |
| UsCarrier | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±0 | 2.4e+02±0 | 0.8±0 |
| UsCarrier | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±0 | 2.4e+02±0 | 0.83±0 |
| UsCarrier | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±0 | 2.3e+02±0 | 0.8±0 |
| Cogentco | 0.25 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0 | 3.4e+02±0 | 1.5±0 |
| Cogentco | 0.25 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0 | 3.5e+02±0 | 1.3±0 |
| Cogentco | 0.25 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0 | 3.6e+02±0 | 1.4±0 |
| Cogentco | 0.25 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0 | 3.5e+02±0 | 1.4±0 |
| Cogentco | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 33.4±0 | 4.3e+02±0 | 1.5±0 |
| Cogentco | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 33.4±0 | 4.3e+02±0 | 1.1±0 |
| Cogentco | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 33.4±0 | 4.3e+02±0 | 1.1±0 |
| Cogentco | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 33.4±0 | 4.3e+02±0 | 1.1±0 |
| Cogentco | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 50.1±0 | 4.4e+02±0 | 1.5±0 |
| Cogentco | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 50.1±0 | 4.4e+02±0 | 1.2±0 |
| Cogentco | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 50.1±0 | 4.4e+02±0 | 1.3±0 |
| Cogentco | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 50.1±0 | 4.4e+02±0 | 1.1±0 |
| Cogentco | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0 | 4.8e+02±0 | 1.5±0 |
| Cogentco | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0 | 4.8e+02±0 | 1.1±0 |
| Cogentco | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0 | 4.8e+02±0 | 1.2±0 |
| Cogentco | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0 | 4.8e+02±0 | 1.1±0 |
| GEANT | 0.25 | frozen_p1 | 1±0 | 0±0 | 0±0 | 0±0 | 0.662±0 | 0±0 | 0.021±0 |
| GEANT | 0.25 | general_3epoch | 1±0 | 0±0 | 0±0 | 0±0 | 0.889±0 | 0±0 | 0.018±0 |
| GEANT | 0.25 | improved_best | 0.5±0 | 0±0 | 0±0 | 0.5±0 | 0.733±0 | 0.5±0 | 0.02±0 |
| GEANT | 0.25 | improved_stable | 1±0 | 0±0 | 0±0 | 0±0 | 0.886±0 | 0±0 | 0.021±0 |
| GEANT | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 1.07±0 | 5.5±0 | 0.022±0 |
| GEANT | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 1.07±0 | 6.5±0 | 0.016±0 |
| GEANT | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 1.07±0 | 7±0 | 0.017±0 |
| GEANT | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 1.07±0 | 5.5±0 | 0.02±0 |
| GEANT | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 1.61±0 | 9±0 | 0.02±0 |
| GEANT | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 1.61±0 | 13±0 | 0.015±0 |
| GEANT | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 1.61±0 | 12±0 | 0.017±0 |
| GEANT | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 1.61±0 | 12±0 | 0.019±0 |
| GEANT | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 2.14±0 | 14±0 | 0.023±0 |
| GEANT | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 2.14±0 | 16±0 | 0.017±0 |
| GEANT | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 2.14±0 | 15±0 | 0.018±0 |
| GEANT | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 2.14±0 | 14±0 | 0.017±0 |
| Kdl | 0.25 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 63±0 | 42±0 |
| Kdl | 0.25 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 65±0 | 77±0 |
| Kdl | 0.25 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 67±0 | 1e+02±0 |
| Kdl | 0.25 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 65±0 | 90±0 |
| Kdl | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.7e+02±0 | 44±0 |
| Kdl | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.8e+02±0 | 1.2e+02±0 |
| Kdl | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.9e+02±0 | 95±0 |
| Kdl | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.8e+02±0 | 1.1e+02±0 |
| Kdl | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.3e+02±0 | 41±0 |
| Kdl | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.3e+02±0 | 1.1e+02±0 |
| Kdl | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.3e+02±0 | 97±0 |
| Kdl | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.2e+02±0 | 91±0 |
| Kdl | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8e+02±0 | 48±0 |
| Kdl | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8e+02±0 | 90±0 |
| Kdl | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8.1e+02±0 | 98±0 |
| Kdl | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8e+02±0 | 95±0 |

## Feasible vs infeasible scenarios

Feasibility is determined from capacity constraints under shortest/equal,
not from model predictions.

| Topology | Scenario | Improved stable applied MLU | Improved best applied MLU | Shortest MLU |
| --- | --- | ---: | ---: | ---: |
| B4 | feasible | 0.493±0 | 0.625±0 | 0.557±0 |
| B4 | infeasible | 0.984±0 | 1.08±0 | 1.11±0 |
| UsCarrier | infeasible | 4.68±0 | 4.68±0 | 4.68±0 |
| Cogentco | infeasible | 41.7±0 | 41.7±0 | 41.7±0 |
| GEANT | feasible | 0.859±0 | 0.756±0 | 0.625±0 |
| GEANT | infeasible | 1.77±0 | 1.77±0 | 1.77±0 |
| Kdl | infeasible | 6.96±0 | 6.96±0 | 6.96±0 |

## Does Cogentco distort aggregate performance?

| Slice | Frozen actor MLU | General 3-epoch | Improved best | Improved stable | Shortest |
| --- | ---: | ---: | ---: | ---: | ---: |
| All topologies | 9.05±0 | 9.49±0 | 9.44±0 | 9.47±0 | 8.2±0 |
| Without Cogentco | 3.7±0 | 4.12±0 | 4.21±0 | 3.99±0 | 2.62±0 |
| Cogentco only | 41.1±0 | 41.7±0 | 40.8±0 | 42.4±0 | 41.7±0 |
| Unseen GEANT+Kdl | 4.68±0 | 5.05±0 | 5.1±0 | 5.05±0 | 3.21±0 |
| Feasible cases only | 0.701±0 | 0.764±0 | 0.794±0 | 0.703±0 | 0.58±0 |
| Infeasible cases only | 13±0 | 13.6±0 | 13.5±0 | 13.6±0 | 11.8±0 |

On this run, improved_stable actor MLU is **9.47±0** across all topologies, **3.99±0** without Cogentco, and **42.4±0** on Cogentco alone. Cogentco therefore dominates any unweighted aggregate mean whenever it is included.

## Method notes

- `frozen_p1` uses the original Step-7 `Framework` fallback (last accepted → shortest).
- Generalized actors use `GeneralController` (least-violating among last/shortest/equal).
- Actor MLU columns are physics loads from the proposal before fallback.
- Applied MLU is after the twin+gate decision.
- Both `improved_best` (`best_checkpoint.pt`) and `improved_stable` (`recommended_stable.pt`) are evaluated; the validation-mean winner is not assumed best overall.

## Limitations / incomplete items

- ASN shards were not read or evaluated.
- Matrix budgets stay small on large graphs (1 UsCarrier / Cogentco / Kdl draw per seed) for runtime.
- Policies are deterministic; seed variation comes from traffic-matrix sampling, not stochastic actions.
- Frozen and generalized frameworks differ in fallback policy; compare actor-only MLU for a pure policy ranking.
- No retraining or gate/twin modification was performed.
