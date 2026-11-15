# Phase 3 dataset catalog

This file records what was collected for the Phase 3 digital-twin study.
The Phase 3 code is in `implementation/` and the data in `data/`. The frozen P1
implementation under `baseline_p1_reproduction` was not modified.

Real demand, TEAL toy placeholders, and the reconstructed Cogentco gravity
series live in three different folders. Nothing in `traffic_matrices/toy` or
`traffic_matrices/reconstructed` is measured operator traffic.

## Layout

```
data/raw/teal              TEAL repository (topologies, real B4 TMs, toy TMs)
data/raw/figret            FIGRET repository (GÉANT graph, histories, tunnels)
data/raw/topology_zoo      Internet Topology Zoo (Cogentco is the one we use)
data/processed/topologies
data/processed/traffic_matrices/real
data/processed/traffic_matrices/toy
data/processed/traffic_matrices/reconstructed
data/processed/candidate_paths
data/processed/graph_data
data/scenarios/link_failures
data/scenarios/twin_errors
```

## Topologies

Undirected average shortest-path length and diameter are computed on the
simple undirected skeleton. P1 Table 5 prints one decimal place; the flag
below is the node and directed-edge check.

| Topology | Nodes | Directed edges | Avg SP | Diameter | Table 5 size |
| --- | ---: | ---: | ---: | ---: | --- |
| B4 | 12 | 38 | 2.318 | 5 | yes |
| UsCarrier | 158 | 378 | 12.090 | 35 | yes |
| Kdl | 754 | 1790 | 22.727 | 58 | yes |
| ASN | 1739 | 8558 | 3.236 | 8 | yes |
| GEANT | 22 | 72 | 2.532 | 5 | yes |
| Cogentco | 197 | 486 | 10.510 | 28 | yes |

GÉANT is the FIGRET graph: 22 nodes and 72 directed links. That is the
P1 Table 5 topology. The earlier 40-node Zoo Geant2012 graph is not used.

### Topology Zoo GÉANT files left in raw only

| File | Nodes | Undirected edges | Decision |
| --- | ---: | ---: | --- |
| Geant.gml | 37 | 55 | not used |
| Geant2001.gml | 27 | 38 | not used |
| Geant2009.gml | 34 | 52 | not used |
| Geant2010.gml | 37 | 56 | not used |
| Geant2011.gml | 40 | 59 | not used |
| Geant2012.gml | 40 | 61 | not used |

Cogentco's Zoo file has no link speed. Processed capacities are the uniform
placeholder 10000 on each directed edge. That placeholder is the same number
the frozen loader used when a speed field was absent. It is not a measured
capacity. Zoo node ids are already 0..196 and those ids are the matrix
indices. A string sort of the same ids would permute the nodes; that
permutation is not used.

## Traffic

| Topology | Kind | What it is |
| --- | --- | --- |
| B4 | real | TEAL public real matrices, Yates/Abilene demands mapped onto B4 |
| GEANT | real | FIGRET published GÉANT demand histories |
| UsCarrier | toy | TEAL synthetic toy placeholder |
| Kdl | toy | TEAL synthetic toy placeholder |
| ASN | toy | TEAL synthetic toy placeholder |
| Cogentco | reconstructed_gravity | T_ij = A_i A_j (i != j), A_i = max(degree(i), 1), degree = in+out |

### Real

- **B4.** 36 TEAL `real_*_1.0` matrices, 12×12, diagonal already 0. TEAL documents these as Yates/Abilene demands mapped onto B4. Mean total demand over all 36 matrices is 26877.93. Mean over the protocol test slice (indices 28:36) is 27981.20, next to the 27980 figure cited for B4. The 20/8/8 cut is the frozen protocol assumption. Lu et al. do not publish it.
- **GÉANT.** FIGRET `train/` and `test/` histories, stored unscaled (float64). Train rows: 8080. Test rows: 2693. Mean total demand, train 1.048e+11, test 3.795e+10. P1's text cites about 43.94 billion for GÉANT; the stored matrices are not rescaled to force that number. FIGRET divides capacities and demands by 1e9 inside `normalize_size`. File `5.hist` has 16 rows and is kept. No validation split is added.

### Toy (not Microsoft WAN traffic)

UsCarrier, Kdl, and ASN have no public real TEAL traffic. Each of the 36 files per topology is an all-ones matrix, including the diagonal (float16). The processed rule, written in the sidecar JSON and not expanded into another matrix, is off-diagonal 1 and diagonal 0. These series cannot be used as P1 Table 6 traffic.

### Reconstructed (not a measured Cogentco trace)

Cogentco demands are a degree-gravity series: `T_ij = A_i A_j` with `A_i = max(degree(i), 1)`, base scaled to the sum of the placeholder capacities, then lognormal noise with seed 2025 and sigma 0.15. Thirty-six matrices. P1 does not publish the gravity constants, so this is an explicit reconstruction, not the authors' dump. The 20/8/8 index cut has no time meaning here.

## Candidate paths

Yen k-shortest paths, k=4, hop length, not edge-disjoint. If a pair has fewer than four simple paths, the shortest is repeated and the pair is counted as padded.

| Topology | Pairs | Padded | Missing | Seconds |
| --- | ---: | ---: | ---: | ---: |
| B4 | 132 | 0 | 0 | 0.031 |
| GEANT | 462 | 0 | 0 | 0.185 |
| UsCarrier | 24806 | 602 | 0 | 40.673 |
| Cogentco | 38612 | 158 | 0 | 69.231 |

Kdl and ASN path sets are not generated in this step (about 0.57 million and 3.0 million ordered pairs). `candidate_paths/Kdl_ASN_DEFERRED.json` records that. FIGRET's own GÉANT tunnels (usually three paths) are copied to `GEANT_figret_tunnels.txt` and are not rewritten as four paths.

## Generated scenarios

Link failures: for every topology and for seeds 100, 200, …, 1000, the cases with 0, 1, 2, and 3 failed directed edges are stored in `scenarios/link_failures`. Sampling follows the frozen Section 5.5 loop (`random.Random(seed)`, then draws of size 0, 1, 2, 3). Edge indices refer to the link order in the processed topology JSON.

Twin error: `scenarios/twin_errors/spec.json`. The stored tensor `z` is standard normal with shape `(n_seeds, n_levels, n_edges)`. Levels are sigma = 0, 0.05, 0.10, 0.20, 0.40. The evaluation formula is `load_hat = max(0, load_true * (1 + sigma * z))`. This perturbs a predicted link load. It is not a traffic matrix.

## Not decided here

Phase 2 trains on some topologies and holds others out, but it does not name the held-out set. This collection does not assign one.

## What was not invented

- No Microsoft WAN matrices for UsCarrier, Kdl, or ASN.
- No second GÉANT traffic series on a Zoo graph.
- No measured Cogentco link speeds and no author gravity constants.
- No validation split for the FIGRET histories.
