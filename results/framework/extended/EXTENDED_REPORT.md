# Phase 3 extended evaluation

This file does not replace the Step 7 report. The frozen B4 checkpoint,
the twin weights, and `safety/gate.py` were not modified. ASN shards were not read.

## Actor mismatches at native scale

| Topology | h0 max | Mean entropy (max log 4) | Peaked splits | Demands with <4 paths | Uniform capacity |
| --- | ---: | ---: | ---: | ---: | --- |
| B4 | 5e+03 | 0.722 | 0.288 | 0 | yes |
| UsCarrier | 5.85e+03 | 0.605 | 0.236 | 602 | no |
| Cogentco | 1e+04 | 0.710 | 0.056 | 158 | yes |
| GEANT | 1e+10 | 0.806 | 0.095 | 0 | no |
| Kdl | 6.75e+04 | 0.636 | 0.158 | 404 | no |

The frozen actor was trained with raw B4 capacities and raw B4 demands.
A much larger h0 max means the same linear layers see a different numeric range.
Entropy near 0 and a high peaked fraction mean the softmax has collapsed onto one path.
Padding counts are demands whose missing paths are repeated only inside the adapter.

## Identical-condition comparison

| Topology | Scale | Cases | Shortest feasible | Accept | Unsafe accept | False reject | Fallback | MLU actor | MLU gate | MLU shortest | Feasible fallback | Unavoidable overload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 | 0.25 | 2 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.306 | 0.306 | 0.275 | 1.00 | 0.00 |
| B4 | 0.50 | 2 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.608 | 0.608 | 0.55 | 1.00 | 0.00 |
| B4 | 0.75 | 2 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.901 | 0.901 | 0.825 | 1.00 | 0.00 |
| B4 | 1.00 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.2 | 1.1 | 1.1 | 0.00 | 1.00 |
| Cogentco | 0.25 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 16.4 | 16.7 | 16.7 | 0.00 | 1.00 |
| Cogentco | 0.50 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 32.7 | 33.3 | 33.3 | 0.00 | 1.00 |
| Cogentco | 0.75 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 49.1 | 50 | 50 | 0.00 | 1.00 |
| Cogentco | 1.00 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 65.5 | 66.7 | 66.7 | 0.00 | 1.00 |
| GEANT | 0.25 | 2 | 0.50 | 0.50 | 0.00 | 0.00 | 0.50 | 1.13 | 1.13 | 1.16 | 0.50 | 0.50 |
| GEANT | 0.50 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 2.23 | 2.31 | 2.31 | 0.00 | 1.00 |
| GEANT | 0.75 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.35 | 3.47 | 3.47 | 0.00 | 1.00 |
| GEANT | 1.00 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 4.42 | 4.62 | 4.62 | 0.00 | 1.00 |
| Kdl | 0.25 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 4.28 | 2.79 | 2.79 | 0.00 | 1.00 |
| Kdl | 0.50 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 8.55 | 5.57 | 5.57 | 0.00 | 1.00 |
| Kdl | 0.75 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 12.8 | 8.36 | 8.36 | 0.00 | 1.00 |
| Kdl | 1.00 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 17.1 | 11.1 | 11.1 | 0.00 | 1.00 |
| UsCarrier | 0.25 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 2.83 | 1.87 | 1.87 | 0.00 | 1.00 |
| UsCarrier | 0.50 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 5.57 | 3.74 | 3.74 | 0.00 | 1.00 |
| UsCarrier | 0.75 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 7.99 | 5.61 | 5.61 | 0.00 | 1.00 |
| UsCarrier | 1.00 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 10.2 | 7.48 | 7.48 | 0.00 | 1.00 |

## Fallback reading

The gate itself is unchanged: it still falls back to the last accepted action,
then shortest-path, then shortest-path as last resort.
Separately, actor, shortest-path, and equal split are scored on the true load.
`feasible_fallback` means one of those three has no capacity violation.
`unavoidable_overload` means all three overload; the recorded least-violating
action is the one with the fewest overloaded links, then the lowest MLU.
That least-violating choice is not applied by the current gate.

## Findings from this run

Twenty-eight decisions were scored: B4 (2 matrices), GEANT (2 matrices), and one matrix each for UsCarrier, Cogentco, and Kdl, at scales 0.25, 0.5, 0.75, and 1.0.

**Feasible traffic.** On B4, scales 0.25, 0.5, and 0.75 are feasible under shortest-path. The gate accepted the actor every time (6/6), with unsafe-accept rate 0 and false-reject rate 0. Gate MLU equals actor MLU (0.306, 0.608, 0.901). At native scale both matrices are infeasible, the gate rejects the actor, and applied MLU falls from 1.20 to 1.11, matching shortest-path. On GEANT, one of the two matrices is feasible at 0.25× (actor MLU 0.764, accepted). The other is already infeasible at 0.25× (MLU about 1.50). UsCarrier, Cogentco, and Kdl are infeasible at every scale tested, including 0.25×, so those topologies do not provide a feasible-traffic acceptance result.

**Why the frozen actor is weak off B4.** The checkpoint was not modified and was not retrained. Measured at native scale:

- Feature scale: B4 h0 max is 5e3. GEANT is 1e10. Kdl is 6.75e4. The actor's first features are raw capacity and raw demand, the same choice as frozen `H0_SCALING = false`. GEANT is outside the numeric range the B4 weights saw.
- The softmax has not collapsed. Mean entropy is 0.61–0.81 on every topology (maximum is log 4 ≈ 1.39). Peaked splits are less common on GEANT (0.095) and Cogentco (0.056) than on B4 (0.288).
- Path padding is a mismatch only where a demand has fewer than four paths: UsCarrier 602, Cogentco 158, Kdl 404, B4 0, GEANT 0. GEANT still fails at native scale with no padding, so padding does not explain GEANT.
- UsCarrier capacities are the same order as B4 (h0 max 5.85e3), but the graph is larger, capacities are not uniform, 602 demands are padded, and the traffic is the toy all-ones matrix. At 0.25× its shortest-path MLU is already 1.87 and the actor's is 2.83.
- Cogentco uses a uniform placeholder capacity and gravity traffic. Even at 0.25× the shortest-path MLU is 16.7. The actor MLU is 16.4, so the dominant problem is that this traffic does not fit the capacities, not only the transferred weights.

**Gate behaviour.** Across all 28 decisions, unsafe accepts are 0 and false rejects are 0. Feasible actor splits were accepted. Infeasible actor splits were rejected. Rejection cannot remove an overload when every candidate overloads: 21 of 28 cases are unavoidable overload.

**Fallback.** The unchanged gate uses shortest-path as last resort. On B4, UsCarrier, and Kdl that lowers MLU relative to the actor. It is not always the least-violating candidate. On the infeasible GEANT matrix at 0.5×, equal split and shortest-path both overload 1 link, but equal-split MLU is 1.26 and shortest-path MLU is 1.62. The same pattern holds at 0.25× on GEANT matrix 600 (equal 1.16, shortest 1.50) and on B4 matrix 24 at native scale (equal 1.01, shortest 1.08). Those equal-split results are measured and are not what the current gate applies. On Cogentco the least-violating rule still selects shortest-path, which has fewer overloaded links even though its MLU is slightly higher than the actor's.

## What this run does not show

- It does not show that the frozen B4 actor generalises. The diagnostics and MLU columns are the evidence either way.
- It does not change ASN. ASN stays pending.
- Cogentco, UsCarrier, and Kdl use one validation or test matrix because a full sweep is too large for this machine. B4 uses two validation matrices and GEANT uses two test matrices, each at four scales.
