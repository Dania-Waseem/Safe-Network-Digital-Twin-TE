# Improved generalized Graph RL policy training

This directory holds a **new** training campaign. It does not replace
`models/policy/general_grl.pt` or `GENERAL_POLICY_REPORT.md`.

Frozen P1 checkpoint, Digital Twin weights, Safety Gate, splits, and the
baseline generalized-policy artifacts were left unchanged.

## Setup

| Item | Value |
| --- | --- |
| Train topologies | B4, UsCarrier, Cogentco (train split only) |
| Val topologies | same three (val split only) |
| Held out | GEANT, Kdl (never read for train/tune) |
| Normalization | capacity and demand ÷ max(link capacity) |
| Path handling | 4 slots/demand; pad collapse onto real paths |
| Train scales | B4: 1.0, 0.5 · UsCarrier/Cogentco: 0.25, 0.5 |
| Matrices / epoch | B4: all 20 · large graphs: 4 train, 2 val |
| Selection metric | mean validation MLU across the three topologies |
| Early stopping | patience on that mean (per-run) |
| Output root | `results/policy/improved/` |

## Hyperparameter search

| Run | LR | Batch | Max epochs | Patience | Epochs ran | Best epoch | Best val mean MLU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lr1e4_bs1_e20 | 1e-4 | 1 | 20 | 5 | 9 | 3 | 6.9789 |
| lr3e4_bs2_e16 | 3e-4 | 2 | 16 | 4 | 5 | 0 | 7.0989 |
| lr5e5_bs1_e24 | 5e-5 | 1 | 24 | 6 | 23 | 16 | 6.9772 |
| **lr1e4_bs4_e16** (selected) | 1e-4 | 4 | 16 | 4 | 5 | 0 | **6.9695** |

Protocol-selected checkpoint: `best_checkpoint.pt` (copy of
`runs/lr1e4_bs4_e16/best.pt`) — lowest validation **mean** MLU.

Recommended for further work: `recommended_stable.pt` (copy of
`runs/lr5e5_bs1_e24/best.pt`) — best B4/UsCarrier learning curve under the
same held-out rules.

Baseline 3-epoch run (different val budget: 1 large-graph matrix) reported
val mean **7.0019**. On this campaign’s 2-matrix large-graph val set, the best
search mean is **6.9695**.

## Convergence

**B4 improves under longer low-LR training.** In `lr5e5_bs1_e24`, B4 val MLU
falls from ~1.19 (epoch 0) to ~0.99 by epoch 9 and stays there through the
best epoch (16). UsCarrier improves from ~3.28 to ~3.01. That is real learning
on the scaled, capacity-relative objective.

**Mean MLU is Cogentco-dominated.** Cogentco val MLU stays ~16.3–17.0 even at
0.25× because the toy/gravity traffic does not fit the capacities. A lucky
early Cogentco draw can win the mean even when B4 has not finished improving.
That is why `lr1e4_bs4_e16` wins selection at epoch 0 (Cogentco 16.29, B4 1.25)
while `lr5e5_bs1_e24` is the clearer learning curve (B4 0.99 at best mean).

**Higher LR / larger batches stopped early without useful progress.**
`lr3e4_bs2_e16` never beat its first epoch. `lr1e4_bs4_e16` also failed to
improve after epoch 0 on the mean metric.

**Train loss decreases only modestly** (about 1.025 → 1.008 on the best
longer run). Overload on Cogentco still dominates the differentiable loss.

## Artifacts

```
results/policy/improved/
  best_checkpoint.pt          # selected by val mean MLU
  recommended_stable.pt       # lr5e5 best; clearer B4/UsCarrier convergence
  search_summary.json
  search_config.json
  convergence_summary.json
  IMPROVED_TRAINING_REPORT.md
  runs/<run_id>/
    config.json
    training_curve.json
    val_metrics.json
    summary.json
    best.pt
```

Search grid defaults also live in `framework/policy_improved_config.json`.

## What was preserved

- Frozen B4 file under `baseline_p1_reproduction/BASELINE_FREEZE/` (hash unchanged)
- Baseline generalized checkpoint `models/policy/general_grl.pt` (hash unchanged)
- Twin models, Safety Gate, splits, and prior eval reports

## Limitations

- Objective is differentiable overflow + MLU − entropy, not P1 ADMM satisfaction.
- Only 4 of 20 train matrices are used on UsCarrier and Cogentco (compute budget).
- Mean val MLU is a weak selection score while Cogentco remains massively
  infeasible; prefer also inspecting per-topology curves (especially B4).
- GEANT / Kdl were not used for tuning and are not re-evaluated in this folder.
- ASN remains pending.
- Gains vs the 3-epoch baseline on the mean metric are small; B4-specific
  improvement under `lr5e5_bs1_e24` is the stronger positive signal.
