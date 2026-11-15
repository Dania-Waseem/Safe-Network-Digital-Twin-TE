#!/usr/bin/env bash
# Re-runs every Phase 3 final experiment (proposed method). Outputs go to the repository-level results/phase3_final/; checkpoints to models/.
# Runtime: several hours on CPU. Baseline (P1) runs are in ../../baseline_p1_reproduction/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."   # implementation/
python -m pytest tests -q
python -m phase3_final.calibrate_load
python -m phase3_final.holdout_af --topologies GEANT
python -m phase3_final.holdout_af --topologies Kdl
python -m phase3_final.train_degraded_twin
python -m phase3_final.twin_error_sweep cache --topologies B4 UsCarrier Cogentco GEANT
python -m phase3_final.twin_error_sweep cache --topologies Kdl
python -m phase3_final.twin_error_sweep sweep
python -m phase3_final.train_policy_v2 score-reference
python -m phase3_final.train_policy_v2 train
python -m phase3_final.holdout_af --topologies GEANT --policy-id v2_calload_normscore --checkpoint ../models/policy_v2/v2_calload_normscore/best.pt
python -m phase3_final.holdout_af --topologies Kdl --policy-id v2_calload_normscore --checkpoint ../models/policy_v2/v2_calload_normscore/best.pt
python -m phase3_final.generalization_gap
python -m phase3_final.link_failure_eval --topologies B4 GEANT UsCarrier
python -m phase3_final.ablations_v2 --topologies B4 UsCarrier Cogentco GEANT
python -m phase3_final.ablations_v2 --topologies Kdl
python -m phase3_final.stats_final grid --topologies B4 UsCarrier Cogentco GEANT
python -m phase3_final.stats_final grid --topologies Kdl
python -m phase3_final.stats_final analyze
python -m phase3_final.figures
python -m phase3_final.consolidate
