# Re-runs every Phase 3 final experiment (proposed method). Outputs go to the repository-level results/phase3_final/; checkpoints to models/.
# Runtime: several hours on CPU. Baseline (P1) runs: see baseline_p1_reproduction/README.md.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')   # implementation/
python -m pytest tests -q; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.calibrate_load; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.holdout_af --topologies GEANT; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.holdout_af --topologies Kdl; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.train_degraded_twin; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.twin_error_sweep cache --topologies B4 UsCarrier Cogentco GEANT; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.twin_error_sweep cache --topologies Kdl; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.twin_error_sweep sweep; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.train_policy_v2 score-reference; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.train_policy_v2 train; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.holdout_af --topologies GEANT --policy-id v2_calload_normscore --checkpoint ../models/policy_v2/v2_calload_normscore/best.pt; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.holdout_af --topologies Kdl --policy-id v2_calload_normscore --checkpoint ../models/policy_v2/v2_calload_normscore/best.pt; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.generalization_gap; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.link_failure_eval --topologies B4 GEANT UsCarrier; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.ablations_v2 --topologies B4 UsCarrier Cogentco GEANT; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.ablations_v2 --topologies Kdl; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.stats_final grid --topologies B4 UsCarrier Cogentco GEANT; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.stats_final grid --topologies Kdl; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.stats_final analyze; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.figures; if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m phase3_final.consolidate; if ($LASTEXITCODE) { exit $LASTEXITCODE }
