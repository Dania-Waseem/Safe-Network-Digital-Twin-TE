# Preprocessing

This package turns the raw datasets in `data/raw/` into the processed inputs used by the twin, the
policy and the experiments. `preprocess.py` is the entry point of the pipeline and
`split_datasets.py` builds the data splits; `collect_datasets.py` assembles the raw sources and writes
`data/DATASET_CATALOG.md`.

## Pipeline

1. **Topologies** (`topology.py`) — TEAL/FIGRET node-link JSON or Topology Zoo GML. Invalid links are
   recorded in `excluded_links`; no link is dropped silently to force connectivity.
2. **Traffic** (`traffic.py`) — four kinds: `real_mapped`, `real_figret`, `synthetic_toy` and
   `reconstructed_gravity`. Negative or non-finite demands are flagged, not clipped.
3. **Normalisation** (`normalize.py`) — the optional P1 formula `x / max(link capacity)` from
   `build_bipartite_graph.initial_features(scale=True)`. Frozen B4 training uses raw values
   (`H0_SCALING = False`). The denominator uses link capacities only, so no demand information leaks.
4. **Paths** (`paths.py`) — Yen k-shortest paths with hop cost and **no padding** when fewer than four
   simple paths exist. B4 uses NetworkX and **matches** the frozen P1 pickle; large graphs use a parallel
   BFS-Yen backend (tie order may differ from NetworkX; hop cost is identical).
5. **Bipartite graph** (`bipartite.py`) — the static link–path structure in
   `graph_data/{name}_bipartite.pt` (PyG `x`, `edge_index`, incidence matrix, OD vectors).

The outputs are `data/processed/{topologies,traffic_matrices,candidate_paths,graph_data,normalization}/`
and `data/processed/PREPROCESSING_REPORT.md`.

## Splits

`split_datasets.py` writes `data/processed/splits/{train,val,test,held_aside}/` and
`SPLIT_MANIFEST.json`. Training and validation use B4, UsCarrier and Cogentco with TEAL-style index
cuts; the test split contains the unseen topologies GEANT (FIGRET test histories only) and Kdl. ASN is
excluded because its path generation is incomplete; its partial shards are kept in
`candidate_paths/ASN_shards/`. The raw data and `baseline_p1_reproduction/` are read only.
