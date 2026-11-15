# Phase 3 Step 7 — integrated framework

Pipeline: frozen P1 actor → digital-twin ensemble → safety gate → applied routing.
The frozen repository and the twin weights were not modified.

Seen topologies completed: True.
Unseen topologies completed: True.

## Comparison

Policy-only applies the actor split with no twin and no gate.
Framework applies that split only when the gate accepts it; otherwise it uses the fallback.
Shortest is minimum-hop routing.

| Topology | Cases | Accept | Fallback | MLU framework | MLU policy-only | MLU shortest | Viol. rate framework | Viol. rate policy | MAE | Policy s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 | 4 | 0.000 | 1.000 | 1.114 | 1.227 | 1.114 | 1.000 | 1.000 | 0.541 | 0.01 |
| GEANT | 5 | 0.000 | 1.000 | 3.425 | 3.754 | 3.425 | 1.000 | 1.000 | 2.345e+05 | 0.01 |
| UsCarrier | 1 | 0.000 | 1.000 | 7.48 | 10.18 | 7.48 | 1.000 | 1.000 | 0.2293 | 0.95 |
| Kdl | 1 | 0.000 | 1.000 | 11.14 | 17.1 | 11.14 | 1.000 | 1.000 | 1.978 | 71.58 |

## How to read the table

Every native matrix in this run was infeasible: shortest-path MLU is already above 1,
so the gate rejected the actor (acceptance 0) and applied shortest-path as last resort.
The framework MLU therefore matches shortest-path and is lower than the ungated actor.
Decision-level violation rate stays 1 because the fallback itself still overloads.
The number of overloaded links does fall: B4 1.75 to 1.0, GEANT 11.4 to 5.6,
UsCarrier 229 to 110. Cogentco was not included in this run.

No jobs failed. Kdl, the large unseen topology, completed one test matrix
(policy forward about 72 s). GEANT completed five held-out FIGRET matrices.

## Limitations

- The actor is the frozen B4 checkpoint applied unchanged. It was not trained on UsCarrier, Cogentco, GEANT, or Kdl.
- ADMM and P1's capacity adjustment are not applied. Policy-only is the actor split itself.
- Full demand is routed. Capacity fit ratio is the share of carried flow that fits on the links, not P1's post-adjustment satisfaction.
- Native traffic matrices in this corpus often exceed capacity under every split, so the gate rejects them and shortest-path last resort can still violate.
- ASN remains incomplete and is excluded.
