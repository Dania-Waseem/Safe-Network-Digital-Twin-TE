# Phase 3 preprocessing report

Raw datasets were read only. Normalisation uses P1's optional formula
`x / max(link capacity)`. The frozen B4 baseline does not apply that scale
(`H0_SCALING = False`); both the raw value and the scaled value are stored.
Paths are hop-shortest and are not repeated to force a count of four.

| Topology | Nodes | Directed links | Connected | Matrices | Invalid matrices | Paths | Fewer than 4 paths | Denominator |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| B4 | 12 | 38 | yes | 36 | 0 | 528 | 0 | 5000 |
| UsCarrier | 158 | 378 | yes | 36 | 0 | 97974 | 602 | 5850 |
| Kdl | 754 | 1790 | yes | 36 | 0 | 2270012 | 404 | 67500 |
| GEANT | 22 | 72 | yes | 10773 | 0 | 1848 | 0 | 1e+10 |
| Cogentco | 197 | 486 | yes | 36 | 0 | 154026 | 158 | 10000 |

## B4

Traffic: real_mapped. TEAL public real matrices: Yates/Abilene demands mapped onto B4. Not Microsoft WAN traffic.
Valid OD entries (finite and non-negative, including zeros): 4752. Of those, 0 are zero and 4752 are positive. Invalid OD entries: 0. Non-zero self-demand entries: 0.
Path backend: networkx.shortest_simple_paths. Demands with no path: 0. Path checks failed: 0.
Bipartite file: link nodes 38, path nodes 528, one-direction incidences 1696.

## UsCarrier

Traffic: synthetic_toy. TEAL toy placeholder. The public repository does not include Microsoft WAN traces for this topology. Not real traffic.
Valid OD entries (finite and non-negative, including zeros): 893016. Of those, 0 are zero and 893016 are positive. Invalid OD entries: 0. Non-zero self-demand entries: 5688.
Path backend: networkx.shortest_simple_paths. Demands with no path: 0. Path checks failed: 0.
Bipartite file: link nodes 378, path nodes 97974, one-direction incidences 1331330.

## Kdl

Traffic: synthetic_toy. TEAL toy placeholder. The public repository does not include Microsoft WAN traces for this topology. Not real traffic.
Valid OD entries (finite and non-negative, including zeros): 20439432. Of those, 0 are zero and 20439432 are positive. Invalid OD entries: 0. Non-zero self-demand entries: 27144.
Path backend: yen_bfs_unit_weight. Demands with no path: 0. Path checks failed: 0.
Bipartite file: link nodes 1790, path nodes 2270012, one-direction incidences 53899592.

## GEANT

Traffic: real_figret. FIGRET published GÉANT demand histories. Not synthesised in this project.
Valid OD entries (finite and non-negative, including zeros): 4977126. Of those, 485846 are zero and 4491280 are positive. Invalid OD entries: 0. Non-zero self-demand entries: 0.
Path backend: networkx.shortest_simple_paths. Demands with no path: 0. Path checks failed: 0.
Bipartite file: link nodes 72, path nodes 1848, one-direction incidences 6492.

## Cogentco

Traffic: reconstructed_gravity. Reconstructed degree-gravity traffic. Not a measured Cogentco trace. T_ij = A_i A_j, A_i = max(degree(i), 1), lognormal noise seed 2025, sigma 0.15. P1 does not publish the author constants.
Valid OD entries (finite and non-negative, including zeros): 1390032. Of those, 0 are zero and 1390032 are positive. Invalid OD entries: 0. Non-zero self-demand entries: 0.
Path backend: networkx.shortest_simple_paths. Demands with no path: 0. Path checks failed: 0.
Bipartite file: link nodes 486, path nodes 154026, one-direction incidences 1822212.
Warnings:
- Cogentco GML has no link speed. Each undirected Zoo edge is stored as two directed edges with placeholder capacity 10000. This is not a measured capacity.
- duplicate undirected edge 42-143 was not added again
- duplicate undirected edge 80-81 was not added again
Excluded link records: 2 (listed in the JSON report).

## Incomplete outputs

- **ASN — pending (not ready for experiments).** Path generation was paused before merge.
  - Checkpoint shards preserved: **1700 / 1739** under `candidate_paths/ASN_shards/` (`src_00000`–`src_01699`).
  - Missing source shards: **39** (indices 1700–1738).
  - Missing outputs: `ASN_yen4.npz`, `ASN_yen4.json`, `ASN_bipartite.pt`.
  - The existing shards are kept; the build is resumable with `--only ASN`.
  - Previous error class: `BrokenProcessPool` (worker terminated abruptly; consistent with memory pressure under high `--jobs`).

## Ready for experiments

| Topology | Paths + bipartite | Traffic | Role in Step 4 splits |
| --- | --- | --- | --- |
| B4 | yes | real_mapped (36) | train / val (time split) |
| UsCarrier | yes | synthetic_toy (36) | train / val (index convention) |
| Cogentco | yes | reconstructed_gravity (36) | train / val (index convention) |
| GEANT | yes | real_figret (8080+2693) | **test** (unseen topology; FIGRET test only) |
| Kdl | yes | synthetic_toy (36) | **test** (unseen topology) |
| ASN | no (shards only) | toy template only | **pending** |

Step 4 split artifacts: `data/processed/splits/` (`train/`, `val/`, `test/`, `held_aside/`, `SPLIT_MANIFEST.json`).

## Not processed

- FIGRET Facebook pod topologies are in the raw dump and are not part of the Phase 2 dataset list.
- Topology Zoo GÉANT graphs are not the 22-node FIGRET graph and are not processed.
- Kdl and ASN have no real traffic. Their toy matrices are labelled synthetic.

## Difference from P1 path padding

P1 `preprocessing/generate_paths.py` repeats the shortest path when a pair has fewer than four simple paths. This pipeline records the actual number and does not repeat a path.
