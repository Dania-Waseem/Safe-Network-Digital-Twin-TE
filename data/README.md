# Data

| Folder | Contents |
|---|---|
| `raw/teal/` | Snapshot of the TEAL repository: topologies (B4, UsCarrier, Kdl, ASN), real B4 traffic matrices, toy templates, path files and code |
| `raw/figret/` | FIGRET GEANT 22/72 topology and its real demand histories |
| `raw/topology_zoo/` | Topology Zoo graphs (including Cogentco) and the European NREN emulation archives |
| `processed/topologies/` | Validated node-link topologies with capacities |
| `processed/traffic_matrices/` | Traffic matrices by kind: `real/`, `toy/`, `reconstructed/` (Cogentco gravity model) |
| `processed/candidate_paths/` | k = 4 Yen candidate paths per topology, including the partial ASN shards |
| `processed/graph_data/` | Link–path bipartite graph tensors and their statistics |
| `processed/normalization/` | Normalisation statistics per topology |
| `processed/splits/` | Train, validation, test and held-aside slices with `SPLIT_MANIFEST.json` and `SPLIT_REPORT.md` |
| `scenarios/link_failures/` | Seeded 1–3 directed link-failure cases per topology (P1 §5.5 sampler) |
| `scenarios/twin_errors/` | Twin-error perturbation scenarios |

`DATASET_CATALOG.md` documents the source and properties of every dataset, `collection_report.json`
records the collection step, and `processed/PREPROCESSING_REPORT.md` records the preprocessing.
