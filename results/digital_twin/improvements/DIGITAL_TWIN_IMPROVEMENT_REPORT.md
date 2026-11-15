# Digital Twin improvement report

Original weights, Safety Gate, RL checkpoints, splits, and prior evals were not overwritten.
GEANT/Kdl were used only for final zero-shot scoring.

## Diagnosis summary

The previously reported “huge GEANT twin MAE” in the ablation study was **absolute load error**.
GEANT capacities are ~1e8–1e10 vs B4 5e3. Relative/utilisation error for the original twin on
GEANT is ~1e-4 — similar to seen topologies. Multipliers remain within ~0.15% of physics.

## Candidates

1. `original` — frozen Step-5 ensemble
2. `physics_anchored` — MSE + L2(mult−1), weight decay, early stopping on val util MAE; train traffic 0.5× augmentation
3. `physics_anchored_conformal` — same + validation-only conformal std scale

## Comparison (utilisation MAE / coverage)

| Model | Val MAE util | GEANT MAE util | Kdl MAE util | Val cov@2σ | GEANT cov@2σ | GEANT MAE abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.000754602 | 8.03749e-05 | 0.000198617 | 1 | 1 | 264828 |
| physics_anchored | 3.11354e-05 | 1.47235e-05 | 6.18032e-05 | 1 | 1 | 47685 |
| physics_anchored_conformal | 3.11354e-05 | 1.47235e-05 | 6.18032e-05 | 1 | 1 | 47685 |

## Did accuracy improve?

- Val util MAE: original 0.000754602 → best candidate 3.11354e-05.
- GEANT util MAE: original 8.03749e-05 → 1.47235e-05.
- GEANT abs MAE: original 264828 → 47685.

Improvements are meaningful when they shrink **bias-driven absolute error** and/or sharpen
calibrated intervals without hurting util MAE. Unseen util MAE was already strong for the
original twin; claims of “fixing GEANT” that rely only on absolute MAE are misleading.

## Did uncertainty calibration improve?

- Val coverage@2σ: 1 → 1
- Val interval width (util): 0.00570027 → 0.000653409

Conformal scaling is fit on seen validation only. Zero-shot GEANT/Kdl coverage is reported
but was not used for selection.

## Compatibility

- Same `EnsembleTwin` API and the same 3 path features as `twin.train._sample`.
- `twin.improvements.infer.predict_loads` matches Safety Gate array contracts
  (absolute mean/std loads).
- Original `models/twin/ensemble.pt` unchanged.

## Limitations / unsuccessful notes

- Labels are exact simulator loads, so a perfect twin is the physics model; learned
  residuals are small by construction.
- Absolute GEANT errors remain large whenever any nonzero relative bias meets 1e9-scale loads.
- ASN not evaluated.
- No architectural GNN rewrite (unjustified given physics-near multipliers).
- Ablation study absolute twin MAE is interpreted with capacity normalisation.

## Artifacts

- `models/ensemble_physics_anchored.pt`
- `models/ensemble_physics_anchored_conformal.pt`
- `BASELINE_REPORT.md`, `DIAGNOSTIC_REPORT.md`, this file
- `metrics_*.json`, `plots/`, `predictions/`
