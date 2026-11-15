# Policy and integrated framework

This package contains the graph-RL routing policy and the end-to-end framework (policy → twin → gate →
fallback).

| File | Contents |
|---|---|
| `general_policy.py` | The generalized GNN policy that proposes splits over k = 4 candidate paths on any topology |
| `train_policy.py` | Policy training with a hyperparameter search, a larger epoch budget and validation early stopping |
| `policy_adapter.py` | Adapter that runs the frozen single-topology P1 policy inside the framework |
| `pipeline.py`, `evaluate.py` | The original integrated pipeline with the frozen B4 actor; results in `results/framework/` |
| `general_controller.py`, `fallback_modes.py` | The controller and its fallback: a feasible candidate if one exists, otherwise the least-violating one among last-accepted, shortest and equal |
| `general_eval.py`, `scale_eval.py` | Evaluation of the generalized actor and of traffic scales; results in `results/policy/` and `results/framework/extended/` |
| `robustness_eval.py` | Multi-seed robustness of both improved checkpoints, the frozen P1 policy and the baselines; results in `results/evaluation/robustness_v1/` |
| `ablation_study.py` | The comparative analysis and ablations A–E; results in `results/evaluation/final_analysis/` |
| `config.json`, `policy_config.json`, `policy_improved_config.json` | Configurations of the framework and of the policy searches |

The trained policies are stored in `models/policy/`: `general_grl.pt` (3-epoch baseline),
`improved/recommended_stable.pt` (the policy used by the framework in all final experiments),
`improved/best_checkpoint.pt` and one checkpoint per search run. GEANT and Kdl are excluded from
training and tuning.
