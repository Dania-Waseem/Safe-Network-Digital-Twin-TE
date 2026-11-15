# Digital twin

This package implements the ensemble digital twin that predicts per-link load for a proposed routing.

| File | Contents |
|---|---|
| `physics.py` | The analytical flow model: path demands, link loads and utilisation |
| `model.py` | `EnsembleTwin`, a five-member ensemble whose mean and standard deviation give the predicted load and its uncertainty |
| `data.py` | Loading of topologies, graph tensors and split traffic matrices (`data/processed/`) |
| `train.py` | Training of the original twin |
| `improvements/` | The physics-anchored twin (`train_improved.py`), split-conformal calibration and evaluation (`run_pipeline.py`), inference helpers and prediction metrics |

The original twin is trained on the training split and stored in `models/twin/` (`ensemble.pt` and
`member_*.pt`); its metrics and predictions (`predictions/<split>_<topology>.npz` with `mean_load`,
`std_load` and `true_load`) are in `results/twin/`, described in `results/twin/TWIN_REPORT.md`.

The improved twin predicts a residual on top of the physics model and scales its ensemble standard
deviation by a conformal factor fitted on the validation split (c = 0.955 for 90 % coverage). Its
checkpoints are in `models/twin_improved/` and its evaluation in `results/digital_twin/improvements/`.
The twin never reads the ASN shards or modifies `baseline_p1_reproduction/`.
