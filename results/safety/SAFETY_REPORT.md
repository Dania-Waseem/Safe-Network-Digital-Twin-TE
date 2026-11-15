# Phase 3 Step 6 — safety gate

The gate reads the twin’s predicted mean load and ensemble standard deviation,
forms an upper bound `mean + k * std`, and accepts the routing action only if
that bound stays at or below capacity on every link. Excessively uncertain
predictions are also rejected. A rejected action is replaced by the last
accepted split if it is still safe, otherwise by shortest-path routing.

## Configuration

- k = 2.0
- utilisation limit = 1.0
- max relative std = 0.5
- last-resort shortest path = True

## Results

| Split | Topology | Cases | Accept | Fallback | Applied violations | Unsafe accept | False reject | Twin ms | Gate ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | B4 | 32 | 0.031 | 0.969 | 0.906 | 0.000 | 0.000 | 8.13 | 0.1601 |
| val_B4_feasible | B4 x0.25 | 24 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 9.56 | 0.1390 |
| val | UsCarrier | 24 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 306.78 | 0.2627 |
| val | Cogentco | 24 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 425.13 | 0.2657 |
| test | GEANT | 405 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 7.71 | 0.2179 |
| test | Kdl | 12 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 7329.24 | 0.5335 |

Overall: 521 decisions, acceptance 0.048, fallback 0.952, applied violation rate 0.948.

## How to read the rates

On the native TMs, maximum link utilisation is already above 1 even under
shortest-path routing (B4 ~1.1, GEANT ~2.8, UsCarrier/Kdl toy all-ones much higher).
The gate therefore rejects almost every proposed action (`unsafe_accept` stays 0)
and applies shortest-path as last resort, which still overloads. That is a
limitation of the fallback, not a missed reject.

B4 at 0.25× demand sits under capacity: the same gate accepts every action,
uses no fallback, and records zero applied violations. One native B4 validation
matrix (index 23, equal split) is also accepted because its upper-bound MLU is 0.992.

Unit tests cover the remaining cases independently of traffic scale:
accept when the upper bound is below capacity, reject when a link is overloaded,
reject when relative uncertainty is high even if the mean is low, and prefer
last-accepted then shortest-path as fallback. GEANT (72 links) and Kdl (1790 links)
produce finite upper bounds from the saved twin.

Gate time is well under 1 ms. Twin time dominates (milliseconds on B4/GEANT,
hundreds of ms on UsCarrier/Cogentco, seconds on Kdl).

## Assumptions

- Ground-truth load is the path-routing simulator, not a measured counter.
- Upper bound is mean + k * ensemble std, compared with utilisation_limit * capacity.
- Fallback order is last accepted action, then shortest-path routing.
- If every candidate fails the gate, shortest-path is still applied and marked last_resort.
- ASN is excluded; shards are not read.
- GEANT uses FIGRET test histories with a stride. Kdl uses a subset of the identical toy matrices.
- B4_feasible multiplies B4 validation demand by 0.25 so some native actions sit under capacity.

## Limitations

- The gate trusts the twin. An under-predicted load with a too-small std can be accepted unsafely.
- Shortest-path is not guaranteed to be feasible when demand exceeds cut capacity. On the native TMs in this corpus, MLU is already above 1, so last-resort shortest-path still overloads.
- Last-accepted is re-checked with the current traffic matrix before reuse.
- Kdl and GEANT were never used to train the twin; their std may be miscalibrated.
- Toy UsCarrier/Kdl matrices are identical all-ones placeholders, not Microsoft WAN traces.

Decision logs: `results/safety/decisions_*.jsonl`.
Metrics: `results/safety/metrics.json`.
Config: `safety/gate_config.json`.
