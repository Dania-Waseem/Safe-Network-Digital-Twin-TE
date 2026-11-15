# Digital Twin diagnostic report

## Findings

- Original twin multipliers stay within ~0.15% of 1.0 (near exact physics).
- GEANT max capacity / B4 max capacity = 2e+06; absolute load MAE scales with this factor.
- Utilisation MAE on GEANT remains ~1e-4, comparable to seen topologies — absolute MAE is a scale artifact.
- Ensemble std tracks load magnitude (near-perfect corr on seen graphs) → intervals are wide/conservative (coverage≈1).
- Hop feature hops/16 exceeds 1.0 on UsCarrier/Kdl; GEANT hop_norm stays ≤0.5 (in-distribution).
- No evidence of catastrophic GEANT feature failure; the ablation's huge absolute twin MAE was scale-dominated.

## Selected improvement strategy

- Accuracy: Physics-anchored training (MSE + L2(multiplier-1)) with weight decay and val early stopping to shrink systematic bias that dominates absolute GEANT error.
- Uncertainty: Validation-only conformal rescaling of ensemble std to target ~90% 2σ coverage with sharper intervals; no GEANT/Kdl labels used.

Not selected:
- MC dropout — ensemble already present.
- Heteroscedastic head — would change gate interfaces and risk overfitting tiny residuals.
- Heavy GNN redesign — labels are exact physics; residual bias is the main absolute-error driver.

## Feature / capacity notes

| Topology | denom | hop_norm max | demand_norm max |
| --- | ---: | ---: | ---: |
| B4 | 5e+03 | 0.375 | 0.2765 |
| UsCarrier | 5.85e+03 | 2.25 | 0.0001709 |
| Cogentco | 1e+04 | 1.812 | 0.1435 |
| GEANT | 1e+10 | 0.5 | 0.2979 |
| Kdl | 6.75e+04 | 3.688 | 1.481e-05 |
