# Phase 3 Step 4 dataset splits

ASN is pending and is not included in train, validation, or test.

## Topology roles

- **Train topologies:** B4, UsCarrier, Cogentco
- **Validation topologies (tuning scenarios):** B4, UsCarrier, Cogentco with later traffic indices
- **Test topologies (unseen):** GEANT, Kdl
- **Pending:** ASN

## Time / index cuts

- TEAL-style series: train `[0, 20]`, val `[20, 28]`, future held aside `[28, 36]`
- GEANT test uses FIGRET published test histories only; FIGRET train is held aside
- Kdl is test-only; early/mid toy indices are unused

## Matrix counts

- Train matrices: 60
- Validation matrices: 24
- Test matrices: 2701

## Output layout

```
data/processed/splits/train/
data/processed/splits/val/
data/processed/splits/test/
data/processed/splits/held_aside/
data/processed/splits/SPLIT_MANIFEST.json
```

Each topology folder entry stores a demand `.npz` slice plus a JSON pointer to
the shared topology, path, bipartite, and normalisation artifacts.

## ASN pending

- Shards present: 1700 / 1739
- Missing sources: 39
- Merged `ASN_yen4.npz` / `ASN_bipartite.pt`: not created
- Checkpoints under `candidate_paths/ASN_shards/` are preserved

