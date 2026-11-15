# Policy v2: calibrated-load training + per-topology normalised selection

One training run (`phase3_final/train_policy_v2.py`). Identical to the run behind `recommended_stable.pt` (GNN, loss, Adam lr 5e-5, batch 1, ≤24 epochs, patience 6, seed 2026, 4 large-graph train TMs) except for two changes: (1) training TMs at calibrated load (2 of the targets {0.5, 0.7, 0.9, 1.1} per TM per epoch, calibrated-load alpha), (2) selection on S = mean over B4/UsCarrier/Cogentco of (policy MLU / shortest MLU) on calibrated validation TMs. GEANT/Kdl and TM indices 28–35 were never read during training or selection.

UsCarrier and Kdl traffic matrices are synthetic all-ones toy templates; every index is an identical copy, so additional TM indices or seeds add no traffic diversity on these topologies.

## Validation selection (the only basis for choosing a checkpoint)

| Checkpoint | S (val) | B4 ratio | UsCarrier ratio | Cogentco ratio |
|---|---:|---:|---:|---:|
| untrained initialisation | 1.3469 | 1.2374 | 1.8449 | 0.9583 |
| `recommended_stable.pt` (reference) | 1.1700 | 0.8854 | 1.6096 | 1.0151 |
| v2 best (epoch 20 of 24) | 1.1736 | 0.8847 | 1.6230 | 1.0132 |

**Selected on validation: `recommended_stable.pt`.** v2 reaches S = 1.1736 versus 1.1700 for the reference (worse by 0.0036). The two changes did **not** produce a better policy on validation; this is a negative result. `recommended_stable.pt` remains the policy of the frozen framework.

The MLU ratio is almost constant across load targets for both checkpoints (v2: 0.5: 1.0290, 0.7: 1.0290, 0.9: 1.0292, 1.1: 1.0295), i.e. the actor's split barely depends on load magnitude even when trained across calibrated loads.

## Generalization gap (actor only; evaluated once on GEANT/Kdl test)

| Policy | Side | Topology | n | MLU / shortest | Satisfaction | Violation rate |
|---|---|---|---:|---:|---:|---:|
| improved_stable | seen | B4 | 64 | 0.8872 | 0.9946 | 0.141 |
| improved_stable | seen | UsCarrier | 8 | 1.6096 | 0.7005 | 1.000 |
| improved_stable | seen | Cogentco | 64 | 1.0157 | 0.4876 | 0.875 |
| improved_stable | unseen | GEANT | 1600 | 1.4881 | 0.8802 | 0.866 |
| improved_stable | unseen | Kdl | 8 | 1.5399 | 0.8006 | 0.875 |
| v2_calload_normscore | seen | B4 | 64 | 0.8865 | 0.9950 | 0.141 |
| v2_calload_normscore | seen | UsCarrier | 8 | 1.6230 | 0.6989 | 1.000 |
| v2_calload_normscore | seen | Cogentco | 64 | 1.0138 | 0.4880 | 0.875 |
| v2_calload_normscore | unseen | GEANT | 1600 | 1.4879 | 0.8809 | 0.866 |
| v2_calload_normscore | unseen | Kdl | 8 | 1.5399 | 0.8004 | 0.875 |

| Policy | MLU ratio seen | unseen | gap | Sat. seen | unseen | gap | Viol. seen | unseen | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| improved_stable | 1.1708 | 1.5140 | -0.3432 | 0.7275 | 0.8404 | -0.1129 | 0.672 | 0.871 | -0.199 |
| v2_calload_normscore | 1.1744 | 1.5139 | -0.3395 | 0.7273 | 0.8407 | -0.1134 | 0.672 | 0.871 | -0.199 |

Gap = seen − unseen (unweighted means over topologies). For the MLU ratio a negative gap means the policy is *worse* on unseen topologies; for satisfaction a positive gap means worse on unseen.

**Reading the gap.** The MLU ratio to shortest path is the cleanest generalization metric because it normalises away load difficulty: both policies are about 0.34 worse relative to shortest path on unseen GEANT/Kdl than on the seen topologies. Satisfaction and violation rate are confounded by load difficulty: the seen side includes UsCarrier and Cogentco, whose toy/gravity TMs are infeasible at most load levels (low satisfaction for every routing), so their 'gaps' have the opposite sign and should not be read as better generalization. v2 and `recommended_stable` are practically identical on the test topologies (MLU ratio 1.5139 vs 1.5140): calibrated-load training did not change generalization.

## Artifacts

- `runs/v2_calload_normscore/{config.json,training_curve.jsonl,summary.json}` (`best.pt`, `last.pt` not committed)
- `reference_recommended_stable_val.json`, `generalization_gap.json`
- Holdout rows for both policies: `../e2_holdout_af/case_rows.jsonl` (policy_id field)
