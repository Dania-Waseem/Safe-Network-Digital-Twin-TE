# Final P1 Experimental Results

## B4 (comparable *family*, not Table-6-identical)

P1 protocol: 3 episodes, seeds 100…1000, Teal-order ADMM, Yen paths, TEAL real TMs.

| Method | Mean sat | P1 Table 6 | Comparable? |
| --- | ---: | ---: | --- |
| LP | 99.74% | 99.83% | **Close** |
| GRL-TE + ADMM | 97.44% ± 1.68% | 92.39% | **No (protocol gaps)** |
| GRL-TE no ADMM | 96.51% ± 2.75% | −0.68 pt ablation | Directionally similar lift (+0.93 pt) |
| ECMP Yen | 98.74% | — | — |
| ECMP TEAL disjoint | 94.20% | — | Path A/B only |

Phase D (20 ep, not P1): no-ADMM 97.73% ± 1.02%; Teal ADMM 98.13% ± 0.80%.

## Other topologies

| Topology | Experiment | Result | vs Table 6 |
| --- | --- | --- | --- |
| GÉANT Zoo 40/122 | 1 seed, 3 ep, gravity | sat no-ADMM 4.97%; Teal ADMM 27.4% | **Incomparable** (wrong size + gravity) |
| UsCarrier | Paths built (99k path nodes); toy TM | No 10-seed GRL-TE | Toy TM ≠ P1 |
| Cogentco | Size 197/486 verified | No full train | Gravity reconstruction |
| Kdl / ASN | JSON size match | No train (scale) | Toy TM |

**Six-topology average 89.36%:** **not computed** (would be fabricated if averaged from incomparable runs).
