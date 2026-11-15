# Phase 3 Step 5 — digital twin

The twin predicts the link load that follows a proposed routing action and
reports uncertainty as the standard deviation across an ensemble.
It trains on B4, UsCarrier and Cogentco. GEANT and Kdl are unseen test topologies.
ASN checkpoints are not used.

## Architecture

Each ensemble member is a two-layer MLP on three path features:
normalised demand, the routing fraction on that path, and hop count.
The network outputs a positive multiplier. Link load is the sum, over paths
that traverse the link, of `demand * routing fraction * multiplier`.
The initial multiplier is 1, which is the exact flow simulator.
Training lets the multiplier depend on the features; disagreement across
five independently bootstrapped members is the uncertainty.

The same scatter works for every graph size, so a model trained on the
smaller topologies can be applied to GEANT and Kdl without a new head.

## Training

- Loss: MSE of predicted versus simulated load, divided by the topology's maximum capacity.
- Actions: equal split, shortest-path split, and Dirichlet splits.
- Labels come from `twin.physics.link_load`, not from a measured counter.
- Optimiser: Adam, learning rate 0.001, 8 epochs, 5 members.

## Results

| Split | Topology | Cases | MAE utilisation | MLU abs error | 1-std coverage | Std/error corr | Finite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | B4 | 60 | 0.000122522 | 0.000305434 | 1.000 | 1.000 | yes |
| val | B4 | 24 | 0.000125421 | 0.000314777 | 1.000 | 1.000 | yes |
| train | UsCarrier | 60 | 0.000367106 | 0.00264317 | 1.000 | 1.000 | yes |
| val | UsCarrier | 24 | 0.000367067 | 0.00264133 | 1.000 | 1.000 | yes |
| train | Cogentco | 60 | 0.00312712 | 0.0188544 | 1.000 | 1.000 | yes |
| val | Cogentco | 24 | 0.00312896 | 0.0189055 | 1.000 | 1.000 | yes |
| test | GEANT | 810 | 0.000128099 | 0.000821368 | 1.000 | 1.000 | yes |
| test | Kdl | 24 | 0.000311715 | 0.00274578 | 0.930 | 0.766 | yes |

Coverage of a calibrated Gaussian interval is about 0.68 at one standard deviation.
On these runs the ensemble standard deviation is about 1.5–2 times the mean absolute error,
so the intervals are conservative: one-std coverage is 1.0 on the training topologies and
on GEANT, and 0.93 on Kdl. The standard deviation still tracks the error
(correlation 1.0 on seen graphs, 0.77 on Kdl). GEANT and Kdl predictions are finite
and have one value per link (72 and 1790).
GEANT test matrices are scored with stride 10 so the run stays within memory;
every selected matrix still uses the published FIGRET test histories only.

Saved artifacts:

- `models/twin/ensemble.pt`
- `models/twin/member_*.pt`
- `results/twin/metrics.json`
- `results/twin/predictions/<split>_<topology>.npz`
