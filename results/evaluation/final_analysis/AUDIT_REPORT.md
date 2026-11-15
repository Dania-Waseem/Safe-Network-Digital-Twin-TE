# Robustness_v1 audit

- Rows: **840** (expected 840)
- Accounting: `5 seeds × (2+1+1+2+1 matrices) × 4 scales × 6 methods = 840`

## Findings

- **OK** `seeds`: all five seeds present: [101, 202, 303, 404, 505]
- **OK** `topologies`: all five topologies present: ['B4', 'Cogentco', 'GEANT', 'Kdl', 'UsCarrier']
- **OK** `scales`: all four scales present: [0.25, 0.5, 0.75, 1.0]
- **OK** `methods`: both improved checkpoints and baselines present
- **OK** `duplicates`: no duplicate (seed,topology,matrix,scale,method) keys
- **OK** `row_count`: 840 rows = 5 seeds × 7 matrices × 4 scales × 6 methods (140 traffic cases × 6 methods)
- **OK** `paired_matrices`: every traffic case has all six methods
- **OK** `catalog`: matrix ids match catalog.json for every seed/topology
- **OK** `held_out`: GEANT and Kdl appear only as unseen test splits
- **OK** `feasibility`: feasible_any_baseline matches zero violations under shortest or equal
- **OK** `report_recompute`: B4@0.25 improved_stable recomputed 0.250198±0.001697 matches report
- **INFO** `fallback_incompatibility`: frozen_p1 applied MLU uses Step-7 Framework fallback; generalized methods use least-violating among last/shortest/equal. Actor-only MLU remains comparable.
- **INFO** `baseline_null_gate`: shortest/equal rows have null acceptance/unsafe/false_reject/fallback fields by design

## Checkpoint inventory

- `frozen_p1`: exists=True bytes=332297 path=`C:\Users\hp\Pictures\CNET phase 3\GRL_TE_Reproduction\BASELINE_FREEZE\models\grl_te_b4_freeze_seed100.pt`
- `general_3epoch`: exists=True bytes=331579 path=`C:\Users\hp\Pictures\CNET phase 3\phase3 implementation\results\policy\general_grl.pt`
- `improved_best`: exists=True bytes=332499 path=`C:\Users\hp\Pictures\CNET phase 3\phase3 implementation\results\policy\improved\best_checkpoint.pt`
- `improved_stable`: exists=True bytes=329873 path=`C:\Users\hp\Pictures\CNET phase 3\phase3 implementation\results\policy\improved\recommended_stable.pt`
- `twin`: exists=True bytes=35885 path=`C:\Users\hp\Pictures\CNET phase 3\phase3 implementation\results\twin\models\ensemble.pt`
