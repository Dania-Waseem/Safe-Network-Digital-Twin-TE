# Phase 3 multi-seed robustness evaluation

All methods saw the **same** seeded traffic matrices within each seed.
GEANT and Kdl were evaluated only on their **test** splits (unseen).
No training or hyperparameter tuning was performed in this step.

## Configuration

- Seeds: `[101, 202, 303, 404, 505]`
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
| B4 | 0.25 | 1.00 | 0.312±0.00354 | 0.291±0.00168 | 0.317±0.00173 | 0.25±0.0017 | 0.279±0.00479 | 0.255±0.00176 |
| B4 | 0.50 | 1.00 | 0.622±0.0069 | 0.585±0.00374 | 0.629±0.00901 | 0.492±0.00328 | 0.558±0.00957 | 0.51±0.00352 |
| B4 | 0.75 | 1.00 | 0.919±0.0101 | 0.881±0.00667 | 0.942±0.0131 | 0.738±0.00504 | 0.838±0.0144 | 0.766±0.00528 |
| B4 | 1.00 | 0.10 | 1.23±0.016 | 1.17±0.00779 | 1.24±0.00894 | 0.985±0.00681 | 1.12±0.0191 | 1.02±0.00704 |
| UsCarrier | 0.25 | 0.00 | 2.83±0 | 3.24±0 | 3.38±0 | 3.01±0 | 1.87±2.48e-16 | 2.45±0 |
| UsCarrier | 0.50 | 0.00 | 5.57±0 | 6.48±0 | 6.75±0 | 6.02±0 | 3.74±4.97e-16 | 4.91±0 |
| UsCarrier | 0.75 | 0.00 | 7.99±9.93e-16 | 9.71±0 | 10.1±0 | 9.03±0 | 5.61±0 | 7.36±0 |
| UsCarrier | 1.00 | 0.00 | 10.2±0 | 13±0 | 13.5±0 | 12±0 | 7.48±9.93e-16 | 9.81±0 |
| Cogentco | 0.25 | 0.00 | 16.4±0.0343 | 16.7±0.0374 | 16.3±0.038 | 16.9±0.0379 | 16.7±0.0142 | 16.2±0.0181 |
| Cogentco | 0.50 | 0.00 | 32.9±0.0667 | 33.3±0.0749 | 32.6±0.0755 | 33.9±0.0758 | 33.3±0.0285 | 32.4±0.0362 |
| Cogentco | 0.75 | 0.00 | 49.3±0.102 | 50±0.112 | 48.9±0.113 | 50.8±0.114 | 50±0.0427 | 48.6±0.0543 |
| Cogentco | 1.00 | 0.00 | 65.7±0.136 | 66.6±0.15 | 65.2±0.151 | 67.7±0.152 | 66.7±0.057 | 64.8±0.0724 |
| GEANT | 0.25 | 0.80 | 0.829±0.162 | 0.915±0.123 | 0.931±0.124 | 0.912±0.12 | 0.744±0.201 | 0.705±0.0673 |
| GEANT | 0.50 | 0.30 | 1.65±0.318 | 1.83±0.246 | 1.86±0.248 | 1.82±0.24 | 1.49±0.402 | 1.41±0.135 |
| GEANT | 0.75 | 0.00 | 2.47±0.47 | 2.74±0.368 | 2.79±0.372 | 2.74±0.361 | 2.23±0.603 | 2.11±0.202 |
| GEANT | 1.00 | 0.00 | 3.29±0.622 | 3.65±0.49 | 3.72±0.496 | 3.65±0.481 | 2.98±0.804 | 2.82±0.269 |
| Kdl | 0.25 | 0.00 | 4.28±0 | 4.29±0 | 4.32±0 | 4.29±0 | 2.79±0 | 2.73±0 |
| Kdl | 0.50 | 0.00 | 8.55±0 | 8.58±0 | 8.63±0 | 8.58±0 | 5.57±0 | 5.45±0 |
| Kdl | 0.75 | 0.00 | 12.8±0 | 12.9±1.99e-15 | 13±0 | 12.9±1.99e-15 | 8.36±0 | 8.18±0 |
| Kdl | 1.00 | 0.00 | 17.1±0 | 17.2±0 | 17.3±0 | 17.2±0 | 11.1±0 | 10.9±0 |

## Framework safety and applied routing

| Topology | Scale | Method | Accept | Unsafe | False reject | Fallback | Applied MLU | Actor viol. | Runtime s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 | 0.25 | frozen_p1 | 1±0 | 0±0 | 0±0 | 0±0 | 0.312±0.00354 | 0±0 | 0.046±0.034 |
| B4 | 0.25 | general_3epoch | 1±0 | 0±0 | 0±0 | 0±0 | 0.291±0.00168 | 0±0 | 0.027±0.0071 |
| B4 | 0.25 | improved_best | 1±0 | 0±0 | 0±0 | 0±0 | 0.317±0.00173 | 0±0 | 0.044±0.049 |
| B4 | 0.25 | improved_stable | 1±0 | 0±0 | 0±0 | 0±0 | 0.25±0.0017 | 0±0 | 0.026±0.008 |
| B4 | 0.50 | frozen_p1 | 1±0 | 0±0 | 0±0 | 0±0 | 0.622±0.0069 | 0±0 | 0.021±0.0063 |
| B4 | 0.50 | general_3epoch | 1±0 | 0±0 | 0±0 | 0±0 | 0.585±0.00374 | 0±0 | 0.011±0.0016 |
| B4 | 0.50 | improved_best | 1±0 | 0±0 | 0±0 | 0±0 | 0.629±0.00901 | 0±0 | 0.025±0.036 |
| B4 | 0.50 | improved_stable | 1±0 | 0±0 | 0±0 | 0±0 | 0.492±0.00328 | 0±0 | 0.011±0.0047 |
| B4 | 0.75 | frozen_p1 | 1±0 | 0±0 | 0±0 | 0±0 | 0.919±0.0101 | 0±0 | 0.021±0.0068 |
| B4 | 0.75 | general_3epoch | 1±0 | 0±0 | 0±0 | 0±0 | 0.881±0.00667 | 0±0 | 0.011±0.0036 |
| B4 | 0.75 | improved_best | 1±0 | 0±0 | 0±0 | 0±0 | 0.942±0.0131 | 0±0 | 0.013±0.0089 |
| B4 | 0.75 | improved_stable | 1±0 | 0±0 | 0±0 | 0±0 | 0.738±0.00504 | 0±0 | 0.012±0.0041 |
| B4 | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 1.12±0.0191 | 1.9±0.22 | 0.022±0.0086 |
| B4 | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 1.1±0.0371 | 4.5±0 | 0.011±0.0032 |
| B4 | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 1.1±0.0371 | 4.3±0.27 | 0.011±0.0057 |
| B4 | 1.00 | improved_stable | 0.9±0.22 | 0±0 | 0.1±0.22 | 0.1±0.22 | 0.985±0.00682 | 0±0 | 0.011±0.0043 |
| UsCarrier | 0.25 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±2.48e-16 | 27±0 | 1.4±0.25 |
| UsCarrier | 0.25 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±2.48e-16 | 33±0 | 0.92±0.13 |
| UsCarrier | 0.25 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±2.48e-16 | 33±0 | 1.3±0.37 |
| UsCarrier | 0.25 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 1.87±2.48e-16 | 33±0 | 1.4±0.15 |
| UsCarrier | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±4.97e-16 | 58±0 | 1.2±0.093 |
| UsCarrier | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±4.97e-16 | 62±0 | 0.75±0.072 |
| UsCarrier | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±4.97e-16 | 66±0 | 0.89±0.35 |
| UsCarrier | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 3.74±4.97e-16 | 61±0 | 0.8±0.078 |
| UsCarrier | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.2e+02±0 | 1.3±0.097 |
| UsCarrier | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.3e+02±0 | 0.75±0.071 |
| UsCarrier | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.4e+02±0 | 0.91±0.3 |
| UsCarrier | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 5.61±0 | 1.3e+02±0 | 0.79±0.095 |
| UsCarrier | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±9.93e-16 | 2.3e+02±0 | 1.3±0.15 |
| UsCarrier | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±9.93e-16 | 2.4e+02±0 | 0.76±0.091 |
| UsCarrier | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±9.93e-16 | 2.4e+02±0 | 0.84±0.21 |
| UsCarrier | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 7.48±9.93e-16 | 2.3e+02±0 | 0.83±0.084 |
| Cogentco | 0.25 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0.0142 | 3.4e+02±1 | 2±0.34 |
| Cogentco | 0.25 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0.0142 | 3.5e+02±1.3 | 1.4±0.22 |
| Cogentco | 0.25 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0.0142 | 3.5e+02±1.1 | 1.5±0.27 |
| Cogentco | 0.25 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 16.7±0.0142 | 3.5e+02±0.84 | 1.5±0.11 |
| Cogentco | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 33.3±0.0285 | 4.3e+02±0.45 | 1.7±0.21 |
| Cogentco | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 33.3±0.0285 | 4.3e+02±0 | 1.2±0.23 |
| Cogentco | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 33.3±0.0285 | 4.3e+02±0.45 | 1.1±0.19 |
| Cogentco | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 33.3±0.0285 | 4.3e+02±0.45 | 1.2±0.086 |
| Cogentco | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 50±0.0427 | 4.4e+02±0 | 1.7±0.21 |
| Cogentco | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 50±0.0427 | 4.4e+02±0 | 1.1±0.13 |
| Cogentco | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 50±0.0427 | 4.4e+02±0 | 1.1±0.058 |
| Cogentco | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 50±0.0427 | 4.4e+02±0.45 | 1.2±0.089 |
| Cogentco | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0.057 | 4.8e+02±1.5 | 1.7±0.16 |
| Cogentco | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0.057 | 4.8e+02±1.5 | 1.1±0.1 |
| Cogentco | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0.057 | 4.8e+02±1.5 | 1.1±0.069 |
| Cogentco | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 66.7±0.057 | 4.8e+02±1.5 | 1.2±0.057 |
| GEANT | 0.25 | frozen_p1 | 0.7±0.27 | 0±0 | 0±0 | 0.3±0.27 | 0.824±0.147 | 0.3±0.27 | 0.026±0.0045 |
| GEANT | 0.25 | general_3epoch | 0.8±0.27 | 0±0 | 0±0 | 0.2±0.27 | 0.847±0.0551 | 0.2±0.27 | 0.02±0.0051 |
| GEANT | 0.25 | improved_best | 0.6±0.22 | 0±0 | 0±0 | 0.4±0.22 | 0.824±0.0714 | 0.4±0.22 | 0.021±0.0039 |
| GEANT | 0.25 | improved_stable | 0.8±0.27 | 0±0 | 0±0 | 0.2±0.27 | 0.845±0.0544 | 0.2±0.27 | 0.022±0.0054 |
| GEANT | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 1.49±0.402 | 4.3±1.6 | 0.027±0.0066 |
| GEANT | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 1.43±0.357 | 6.9±1.9 | 0.016±0.0026 |
| GEANT | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 1.43±0.357 | 6.7±1.6 | 0.017±0.0028 |
| GEANT | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 1.43±0.357 | 5.6±1.9 | 0.019±0.002 |
| GEANT | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 2.23±0.603 | 8.1±3 | 0.028±0.0066 |
| GEANT | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 2.23±0.609 | 13±2.6 | 0.016±0.0029 |
| GEANT | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 2.23±0.609 | 13±2.5 | 0.017±0.0027 |
| GEANT | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 2.23±0.609 | 12±2.7 | 0.019±0.0036 |
| GEANT | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 2.98±0.804 | 14±2.8 | 0.027±0.005 |
| GEANT | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 2.98±0.804 | 17±2.3 | 0.017±0.0032 |
| GEANT | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 2.98±0.804 | 17±2.1 | 0.017±0.0025 |
| GEANT | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 2.98±0.804 | 16±2.1 | 0.02±0.0034 |
| Kdl | 0.25 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 63±0 | 62±19 |
| Kdl | 0.25 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 65±0 | 84±22 |
| Kdl | 0.25 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 67±0 | 98±17 |
| Kdl | 0.25 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 2.79±0 | 65±0 | 1e+02±12 |
| Kdl | 0.50 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.7e+02±0 | 53±16 |
| Kdl | 0.50 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.8e+02±0 | 1.2e+02±31 |
| Kdl | 0.50 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.9e+02±0 | 1e+02±15 |
| Kdl | 0.50 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 5.57±0 | 1.8e+02±0 | 1.2e+02±42 |
| Kdl | 0.75 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.3e+02±0 | 52±19 |
| Kdl | 0.75 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.3e+02±0 | 1e+02±25 |
| Kdl | 0.75 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.3e+02±0 | 98±10 |
| Kdl | 0.75 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 8.36±0 | 4.2e+02±0 | 1.1e+02±22 |
| Kdl | 1.00 | frozen_p1 | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8e+02±0 | 55±29 |
| Kdl | 1.00 | general_3epoch | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8e+02±0 | 1.1e+02±34 |
| Kdl | 1.00 | improved_best | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8.1e+02±0 | 1e+02±13 |
| Kdl | 1.00 | improved_stable | 0±0 | 0±0 | 0±0 | 1±0 | 11.1±0 | 8e+02±0 | 1.2e+02±28 |

## Feasible vs infeasible scenarios

Feasibility is determined from capacity constraints under shortest/equal,
not from model predictions.

| Topology | Scenario | Improved stable applied MLU | Improved best applied MLU | Shortest MLU |
| --- | --- | ---: | ---: | ---: |
| B4 | feasible | 0.507±0.0266 | 0.64±0.0204 | 0.573±0.0244 |
| B4 | infeasible | 0.989±0.0048 | 1.11±0.0229 | 1.12±0.00985 |
| UsCarrier | infeasible | 4.68±0 | 4.68±0 | 4.68±0 |
| Cogentco | infeasible | 41.7±0.0356 | 41.7±0.0356 | 41.7±0.0356 |
| GEANT | feasible | 0.806±0.0649 | 0.795±0.0566 | 0.689±0.122 |
| GEANT | infeasible | 2.22±0.476 | 2.22±0.477 | 2.26±0.502 |
| Kdl | infeasible | 6.96±0 | 6.96±0 | 6.96±0 |

## Does Cogentco distort aggregate performance?

| Slice | Frozen actor MLU | General 3-epoch | Improved best | Improved stable | Shortest |
| --- | ---: | ---: | ---: | ---: | ---: |
| All topologies | 9.15±0.107 | 9.5±0.089 | 9.46±0.0899 | 9.48±0.0876 | 8.35±0.138 |
| Without Cogentco | 3.83±0.129 | 4.14±0.101 | 4.24±0.101 | 4.01±0.099 | 2.79±0.166 |
| Cogentco only | 41.1±0.085 | 41.6±0.0936 | 40.7±0.0945 | 42.3±0.0947 | 41.7±0.0356 |
| Unseen GEANT+Kdl | 4.94±0.262 | 5.1±0.205 | 5.15±0.207 | 5.09±0.2 | 3.56±0.335 |
| Feasible cases only | 0.7±0.0498 | 0.724±0.0682 | 0.758±0.0631 | 0.652±0.0724 | 0.602±0.0447 |
| Infeasible cases only | 12.8±0.505 | 13.3±0.566 | 13.2±0.558 | 13.3±0.567 | 11.7±0.381 |

On this run, improved_stable actor MLU is **9.48±0.0876** across all topologies, **4.01±0.099** without Cogentco, and **42.3±0.0947** on Cogentco alone. Cogentco therefore dominates any unweighted aggregate mean whenever it is included.

## Findings

1. **The validation-mean winner is not the best policy overall.** `improved_best` (`best_checkpoint.pt`) is *worse* than `improved_stable` on B4 at every scale (e.g. 0.317±0.002 vs 0.250±0.002 at 0.25×) and worse on feasible-only actor MLU (0.758 vs 0.652). The Cogentco-heavy selection metric picked an early checkpoint that is not the best general-purpose policy here.
2. **`recommended_stable.pt` is the strongest learned actor on B4.** It undercuts frozen P1 and the 3-epoch general actor on B4 actor MLU, remains accepted through 0.75×, and at native scale is accepted on 90% of draws (applied MLU 0.985±0.007) with zero unsafe accepts.
3. **Gate safety held.** Across all seeds/methods/topologies in this run, unsafe-accept rates are 0. False rejects are essentially 0 except a small B4-native rate for `improved_stable` (0.1±0.22) where the actor was feasible but rejected.
4. **Unseen GEANT/Kdl.** On GEANT feasible low-scale traffic, learned actors are accepted most of the time but do not beat shortest/equal on actor MLU. On Kdl every scale is infeasible under shortest/equal; applied routing falls back and matches shortest MLU. No claim of Kdl zero-shot improvement is supported.
5. **UsCarrier / Cogentco remain capacity-bound.** Feasible rate 0 at all tested scales; framework always falls back; aggregate means without a per-topology view are misleading.
6. **Shortest-path still wins many infeasible large-graph cases** on actor MLU. The framework’s value there is safe rejection + fallback, not beating shortest with the actor proposal.

## Method notes

- `frozen_p1` uses the original Step-7 `Framework` fallback (last accepted → shortest).
- Generalized actors use `GeneralController` (least-violating among last/shortest/equal).
- Actor MLU columns are physics loads from the proposal before fallback.
- Applied MLU is after the twin+gate decision.
- Both `improved_best` (`best_checkpoint.pt`) and `improved_stable` (`recommended_stable.pt`) are evaluated; the validation-mean winner is not assumed best overall.

## Limitations / incomplete items

- ASN shards were not read or evaluated.
- Matrix budgets stay small on large graphs (1 UsCarrier / Cogentco / Kdl draw per seed) for runtime (~2.2 h for 5 seeds).
- Policies are deterministic; seed variation comes from traffic-matrix sampling, not stochastic actions.
- Frozen and generalized frameworks differ in fallback policy; compare actor-only MLU for a pure policy ranking.
- Kdl policy-graph construction allocates ~1.6 GiB; the harness caches that graph once per process to avoid repeated OOM on rebuild.
- A one-seed smoke tree also exists at `results/evaluation/robustness_smoke/` (development check only).
- No retraining or gate/twin modification was performed.
