# Framework ablations

Same paired cases for every configuration: dev = B4/UsCarrier/Cogentco val, holdout = GEANT (200) + Kdl (1), 8 load levels each; TMs processed in time order within each (topology, level) so that (c) can keep the previous routing (initial routing = shortest path). Gate settings frozen (k=1, limit 1.0). UsCarrier and Kdl traffic matrices are synthetic all-ones toy templates; every index is an identical copy, so additional TM indices or seeds add no traffic diversity on these topologies.

† (d) is confounded: the frozen P1 actor differs from `recommended_stable` in training topology (B4 only) **and** in model/trainer (TopoFlowNet + A2C vs Phase-3 GNN with a differentiable loss). It is the only single-topology policy available without new training.

## dev (n = 136 cases)

| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full framework (E gate + fallback) | 10.627 | 0.7592 | 0.537 | 0.463 | 0.000 | 0.000 | 0.000 | 724.3 |
| (a) No gate | 10.899 | 0.7387 | 0.537 | 1.000 | 0.537 | 1.000 | 0.000 | 564.8 |
| (b) Gate without uncertainty | 10.627 | 0.7592 | 0.537 | 0.463 | 0.000 | 0.000 | 0.000 | 781.0 |
| (c) No fallback (keep previous routing) | 10.632 | 0.7594 | 0.537 | 0.463 | 0.000 | 0.000 | 0.000 | 724.3 |
| (d) B4-only policy (frozen P1)† | 10.672 | 0.7584 | 0.581 | 0.353 | 0.000 | 0.000 | 0.000 | 767.7 |
| (e) Twin without learned residual (physics) | 10.627 | 0.7592 | 0.537 | 0.463 | 0.000 | 0.000 | 0.000 | 553.0 |

Physics twin agrees with the oracle on 100.0% of cases.

## holdout (n = 1608 cases)

| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full framework (E gate + fallback) | 1.565 | 0.9760 | 0.681 | 0.134 | 0.000 | 0.000 | 0.000 | 15.8 |
| (a) No gate | 2.228 | 0.8798 | 0.866 | 1.000 | 0.866 | 1.000 | 0.000 | 10.8 |
| (b) Gate without uncertainty | 1.565 | 0.9760 | 0.681 | 0.134 | 0.000 | 0.000 | 0.000 | 16.0 |
| (c) No fallback (keep previous routing) | 1.682 | 0.9711 | 0.825 | 0.134 | 0.000 | 0.000 | 0.000 | 15.8 |
| (d) B4-only policy (frozen P1)† | 1.563 | 0.9760 | 0.681 | 0.174 | 0.000 | 0.000 | 0.000 | 14.6 |
| (e) Twin without learned residual (physics) | 1.565 | 0.9760 | 0.681 | 0.134 | 0.000 | 0.000 | 0.000 | 10.6 |

Physics twin agrees with the oracle on 100.0% of cases.

## B4 (n = 64 cases)

| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full framework (E gate + fallback) | 0.734 | 0.9939 | 0.141 | 0.859 | 0.000 | 0.000 | 0.000 | 17.5 |
| (a) No gate | 0.716 | 0.9946 | 0.141 | 1.000 | 0.141 | 1.000 | 0.000 | 11.5 |
| (b) Gate without uncertainty | 0.734 | 0.9939 | 0.141 | 0.859 | 0.000 | 0.000 | 0.000 | 17.4 |
| (c) No fallback (keep previous routing) | 0.734 | 0.9944 | 0.141 | 0.859 | 0.000 | 0.000 | 0.000 | 17.5 |
| (d) B4-only policy (frozen P1)† | 0.833 | 0.9920 | 0.234 | 0.625 | 0.000 | 0.000 | 0.000 | 16.7 |
| (e) Twin without learned residual (physics) | 0.734 | 0.9939 | 0.141 | 0.859 | 0.000 | 0.000 | 0.000 | 11.5 |

Physics twin agrees with the oracle on 100.0% of cases.

## UsCarrier (n = 8 cases)

| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full framework (E gate + fallback) | 3.290 | 0.9535 | 1.000 | 0.000 | 0.000 | 0.000 | – | 724.3 |
| (a) No gate | 5.296 | 0.7005 | 1.000 | 1.000 | 1.000 | 1.000 | – | 564.8 |
| (b) Gate without uncertainty | 3.290 | 0.9535 | 1.000 | 0.000 | 0.000 | 0.000 | – | 814.3 |
| (c) No fallback (keep previous routing) | 3.290 | 0.9535 | 1.000 | 0.000 | 0.000 | 0.000 | – | 724.3 |
| (d) B4-only policy (frozen P1)† | 3.290 | 0.9535 | 1.000 | 0.000 | 0.000 | 0.000 | – | 839.9 |
| (e) Twin without learned residual (physics) | 3.290 | 0.9535 | 1.000 | 0.000 | 0.000 | 0.000 | – | 553.0 |

Physics twin agrees with the oracle on 100.0% of cases.

## Cogentco (n = 64 cases)

| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full framework (E gate + fallback) | 21.437 | 0.5003 | 0.875 | 0.125 | 0.000 | 0.000 | 0.000 | 949.2 |
| (a) No gate | 21.782 | 0.4876 | 0.875 | 1.000 | 0.875 | 1.000 | 0.000 | 725.2 |
| (b) Gate without uncertainty | 21.437 | 0.5003 | 0.875 | 0.125 | 0.000 | 0.000 | 0.000 | 946.9 |
| (c) No fallback (keep previous routing) | 21.447 | 0.5001 | 0.875 | 0.125 | 0.000 | 0.000 | 0.000 | 949.2 |
| (d) B4-only policy (frozen P1)† | 21.434 | 0.5003 | 0.875 | 0.125 | 0.000 | 0.000 | 0.000 | 933.4 |
| (e) Twin without learned residual (physics) | 21.437 | 0.5003 | 0.875 | 0.125 | 0.000 | 0.000 | 0.000 | 706.9 |

Physics twin agrees with the oracle on 100.0% of cases.

## GEANT (n = 1600 cases)

| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full framework (E gate + fallback) | 1.553 | 0.9763 | 0.681 | 0.134 | 0.000 | 0.000 | 0.000 | 15.8 |
| (a) No gate | 2.209 | 0.8802 | 0.866 | 1.000 | 0.866 | 1.000 | 0.000 | 10.8 |
| (b) Gate without uncertainty | 1.553 | 0.9763 | 0.681 | 0.134 | 0.000 | 0.000 | 0.000 | 16.0 |
| (c) No fallback (keep previous routing) | 1.671 | 0.9713 | 0.826 | 0.134 | 0.000 | 0.000 | 0.000 | 15.8 |
| (d) B4-only policy (frozen P1)† | 1.551 | 0.9763 | 0.681 | 0.174 | 0.000 | 0.000 | 0.000 | 14.6 |
| (e) Twin without learned residual (physics) | 1.553 | 0.9763 | 0.681 | 0.134 | 0.000 | 0.000 | 0.000 | 10.6 |

Physics twin agrees with the oracle on 100.0% of cases.

## Kdl (n = 8 cases)

| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full framework (E gate + fallback) | 3.911 | 0.9294 | 0.625 | 0.125 | 0.000 | 0.000 | 0.000 | 40955.5 |
| (a) No gate | 5.978 | 0.8006 | 0.875 | 1.000 | 0.875 | 1.000 | 0.000 | 33702.1 |
| (b) Gate without uncertainty | 3.911 | 0.9294 | 0.625 | 0.125 | 0.000 | 0.000 | 0.000 | 40847.4 |
| (c) No fallback (keep previous routing) | 3.916 | 0.9294 | 0.625 | 0.125 | 0.000 | 0.000 | 0.000 | 40955.5 |
| (d) B4-only policy (frozen P1)† | 3.911 | 0.9294 | 0.625 | 0.125 | 0.000 | 0.000 | 0.000 | 39565.3 |
| (e) Twin without learned residual (physics) | 3.911 | 0.9294 | 0.625 | 0.125 | 0.000 | 0.000 | 0.000 | 32592.3 |

Physics twin agrees with the oracle on 100.0% of cases.

## Artifacts

- `sequence_rows.jsonl`, `summary.json`, `config.json`
