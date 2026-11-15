# Ablation report (A–E)

Ablations share one policy proposal and one twin prediction per traffic case,
then apply decision rules offline. Cases are independent (no last-accepted carry).
GEANT/Kdl remain test/unseen only.

## Definitions

| ID | Configuration |
| --- | --- |
| A | Generalized RL policy only (always apply actor) |
| B | RL + Digital Twin prediction (apply actor; record twin error/calibration) |
| C | RL + oracle physics Safety Gate (accept iff true load fits capacity; no twin/uncertainty) |
| D | RL + Digital Twin + mean-only gate (`k=0`, no max-relative-std) |
| E | Complete RL + Digital Twin + uncertainty-aware Safety Gate (`k=2`, max_relative_std=0.5) |

Checkpoints: `{'improved_best': 'C:\\Users\\hp\\Pictures\\CNET phase 3\\phase3 implementation\\results\\policy\\improved\\best_checkpoint.pt', 'improved_stable': 'C:\\Users\\hp\\Pictures\\CNET phase 3\\phase3 implementation\\results\\policy\\improved\\recommended_stable.pt'}`

## Per-topology ablation table (improved_stable)

| Topology | Ablation | Applied MLU | Accept | Unsafe | False reject | Fallback | No feasible FB | Runtime s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 | A_policy_only | 0.616 | 1 | 0 | 0 | 0 | 0 | 0.018 |
| B4 | B_rl_twin | 0.616 | 1 | 0 | 0 | 0 | 0 | 0.022 |
| B4 | C_rl_oracle_gate | 0.616 | 1 | 0 | 0 | 0 | 0 | 0.018 |
| B4 | D_rl_twin_mean_gate | 0.616 | 1 | 0 | 0 | 0 | 0 | 0.022 |
| B4 | E_full_uncertainty_gate | 0.62 | 0.97 | 0 | 0.025 | 0.025 | 0.025 | 0.022 |
| UsCarrier | A_policy_only | 7.53 | 1 | 1 | 0 | 0 | 0 | 0.74 |
| UsCarrier | B_rl_twin | 7.53 | 1 | 1 | 0 | 0 | 0 | 0.95 |
| UsCarrier | C_rl_oracle_gate | 4.67 | 0 | 0 | 0 | 1 | 1 | 0.74 |
| UsCarrier | D_rl_twin_mean_gate | 4.67 | 0 | 0 | 0 | 1 | 1 | 0.95 |
| UsCarrier | E_full_uncertainty_gate | 4.67 | 0 | 0 | 0 | 1 | 1 | 0.95 |
| Cogentco | A_policy_only | 42.3 | 1 | 1 | 0 | 0 | 0 | 0.88 |
| Cogentco | B_rl_twin | 42.3 | 1 | 1 | 0 | 0 | 0 | 1.2 |
| Cogentco | C_rl_oracle_gate | 41.7 | 0 | 0 | 0 | 1 | 1 | 0.88 |
| Cogentco | D_rl_twin_mean_gate | 41.7 | 0 | 0 | 0 | 1 | 1 | 1.2 |
| Cogentco | E_full_uncertainty_gate | 41.7 | 0 | 0 | 0 | 1 | 1 | 1.2 |
| GEANT | A_policy_only | 2.28 | 1 | 0.8 | 0 | 0 | 0 | 0.012 |
| GEANT | B_rl_twin | 2.28 | 1 | 0.8 | 0 | 0 | 0 | 0.016 |
| GEANT | C_rl_oracle_gate | 1.87 | 0.2 | 0 | 0 | 0.8 | 0.7 | 0.012 |
| GEANT | D_rl_twin_mean_gate | 1.87 | 0.2 | 0 | 0 | 0.8 | 0.7 | 0.016 |
| GEANT | E_full_uncertainty_gate | 1.87 | 0.2 | 0 | 0 | 0.8 | 0.7 | 0.016 |
| Kdl | A_policy_only | 10.7 | 1 | 1 | 0 | 0 | 0 | 1e+02 |
| Kdl | B_rl_twin | 10.7 | 1 | 1 | 0 | 0 | 0 | 1.1e+02 |
| Kdl | C_rl_oracle_gate | 6.96 | 0 | 0 | 0 | 1 | 1 | 1e+02 |
| Kdl | D_rl_twin_mean_gate | 6.96 | 0 | 0 | 0 | 1 | 1 | 1.1e+02 |
| Kdl | E_full_uncertainty_gate | 6.96 | 0 | 0 | 0 | 1 | 1 | 1.1e+02 |

## Twin quality (from ablation B, improved_stable)

| Topology | Unseen | MAE | RMSE | Coverage@1σ | Coverage@2σ | corr(std,|err|) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B4 | False | 0.469 | 0.516 | 1 | 1 | 1 |
| UsCarrier | False | 0.174 | 0.248 | 1 | 1 | 1 |
| Cogentco | False | 21.6 | 31 | 1 | 1 | 1 |
| GEANT | True | 4.13e+05 | 6.05e+05 | 1 | 1 | 1 |
| Kdl | True | 1.86 | 3.68 | 0.91 | 0.98 | 0.68 |

## Paired statistical tests

| Comparison | n | mean diff | 95% CI | Wilcoxon p | t-test p | sig@0.05 (W) |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| E_vs_A applied MLU (stable, all cases) | 140 | -1.15 | [-1.43, -0.875] | 5.47e-15 | 1.27e-13 | True |
| E_vs_A unsafe accept rate proxy (stable, all cases) | 140 | -0.657 | [-0.737, -0.578] | 8.67e-22 | 4.07e-34 | True |
| E_vs_A applied MLU (stable, feasible only) | 42 | -0.0673 | [-0.133, -0.00134] | n/a | 0.0457 | False |
| E_vs_D false reject (uncertainty vs mean-only gate) | 140 | 0.00714 | [-0.00698, 0.0213] | n/a | 0.319 | False |
| E_vs_D unsafe accept (uncertainty vs mean-only gate) | 140 | 0 | [0, 0] | n/a | n/a | False |
| C_oracle_vs_E acceptance (oracle physics gate vs full gate) | 140 | 0.00714 | [-0.00698, 0.0213] | n/a | 0.319 | False |
| stable_actor_vs_frozen_actor MLU (all paired cases) | 140 | 0.328 | [0.225, 0.43] | 6.49e-06 | 3.36e-09 | True |
| stable_actor_vs_shortest MLU (all paired cases) | 140 | 1.13 | [0.853, 1.41] | 6.04e-13 | 4.08e-13 | True |
| stable_actor_vs_frozen_actor MLU (B4 only) | 40 | -0.153 | [-0.175, -0.131] | 3.54e-08 | 6.74e-17 | True |
| stable_actor_vs_shortest MLU (B4 only) | 40 | -0.0817 | [-0.0952, -0.0682] | 3.54e-08 | 6.14e-15 | True |

Differences are left−right as named (e.g. E_vs_A means E − A).
Significance flags require Wilcoxon p < 0.05 and are not claimed without that evidence.

## Incomplete / caveats

- Ablation fallback ignores last-accepted carry to keep cases independent and paired.
- Robustness_v1 generalized rows used last-accepted carry inside a topology loop; applied MLU can differ slightly from ablation E for that reason.
- ASN not evaluated.
