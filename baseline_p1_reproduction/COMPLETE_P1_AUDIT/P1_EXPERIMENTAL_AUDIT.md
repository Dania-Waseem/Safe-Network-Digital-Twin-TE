# P1 Experimental Audit

Verification of datasets, topologies, seeds, baselines, evaluation procedures, and reported metrics against Lu et al. (2025) §§5.1–5.5 and Tables 5–6.

---

## 1. Topologies (Table 5)

### 1.1 Method

For each WAN JSON/GML used by the repo, compute directed edge count and undirected connected-component average shortest-path length + diameter. Compare to Table 5. **Size alone is insufficient**; path statistics are required.

### 1.2 Results

| Topology | Source file | Nodes | Directed edges | Avg SPL | Diameter | Table 5 | Verdict |
|----------|-------------|------:|---------------:|--------:|---------:|---------|---------|
| B4 | TEAL `B4.json` | 12 | 38 | 2.318 | 5 | 12/38, 2.3, 5 | **Match** |
| UsCarrier | TEAL | 158 | 378 | 12.090 | 35 | 158/378, 12.1, 35 | **Match** |
| Cogentco | Zoo `Cogentco.gml` | 197 | 486 | 10.510 | 28 | 197/486, 10.51, 28 | **Match** |
| Kdl | TEAL | 754 | 1790 | 22.727 | 58 | 754/1790, 22.7, 58 | **Match** |
| ASN | TEAL `ASN2k.json` | 1739 | 8558 | 3.236 | 8 | 1739/8558, 3.2, 8 | **Match** |
| GÉANT | Zoo `GEANT.gml` | **40** | **122** | **3.528** | **8** | **22/72, 2.53, 5** | **Mismatch** |

**Assumption:** Table 5 averages are hop distances on the undirected simple graph. Directed capacities are retained for TE.

**GÉANT:** FIGRET/Uhlig 22-node pack is **not** in the workspace. Zoo Geant2012 is a different network. `experiments/geant.py` still refuses to claim a P1 GÉANT run.

---

## 2. Traffic matrices

### 2.1 Classification

| Topology | Class | Detail | Usable for Table 6? |
|----------|-------|--------|---------------------|
| B4 | **TEAL real (public)** | 36 files `B4.json_real_*_1.0_traffic-matrix.pkl`; mean total demand ≈ 2.69e4 | **Yes, with split caveat** |
| UsCarrier | **TEAL toy** | 36 matrices, entries originally unit float16; mean total = \(158\times157=24806\) | **No** |
| Kdl | **TEAL toy** | unit demand; mean total = \(754\times753\) | **No** |
| ASN | **TEAL toy** | unit demand; mean total = \(1739\times1738\) | **No** |
| Cogentco | **Reconstructed gravity** | `preprocessing/gravity.py`, seed 2025; not author dump | **No** |
| GÉANT | **Gravity on wrong graph** | 40×40 matrices | **No** |

P1 §5.1: B4/UsCarrier/Kdl/ASN from Teal; GÉANT from Figret; Cogentco gravity.  
**Public Teal does not ship Microsoft WAN TMs** for the large topologies — only toy placeholders. P1’s reported UsCarrier/Kdl/ASN numbers **cannot** be reproduced from the public Teal toy dump.

### 2.2 Split

| Item | P1 | Ours |
|------|----|------|
| Publication of indices | **None** | TEAL-style 20/8/8 on 36 B4 TMs |
| Test set size (B4) | Unknown | 8 TMs `[28,36)` |

**Cannot claim** the same train/test distribution as P1.

### 2.3 Normalization

B4 TMs used in physical units; \(h^0\) unscaled. No train-set leakage into test metrics beyond shared topology/paths (standard for TE).

---

## 3. Path generation

| Item | P1 text | Implementation | Notes |
|------|---------|----------------|-------|
| Count | 4 per node pair | `NUM_PATHS=4` | OK |
| Algorithm | “4 shortest paths” | Yen via `nx.shortest_simple_paths`, min-hop | Reasonable reading of P1 |
| Edge-disjoint | Not stated | `EDGE_DISJOINT=False` | TEAL pickles differ |
| Padding | Unstated | Repeat shortest if &lt;4 | Unverified |
| Metric | Unstated (hop vs inv-cap) | min-hop default | Unverified |

### Path A/B (B4, same LP/ECMP code)

From `results/logs/phase_f/path_ab.json`:

| Path set | LP sat | ECMP sat |
|----------|--------|----------|
| Yen min-hop | 0.99737 | **0.98739** |
| TEAL edge-disjoint | 0.99737 | **0.94201** |

**Finding:** LP is insensitive; ECMP (and likely learned policies) are not. The default is kept as Yen, per P1 wording.

---

## 4. Hyperparameters & seeds

| Hyperparameter | P1 §5.3 | `config.py` / runs | Match |
|----------------|---------|--------------------|-------|
| Layers / out dim | 6 / 7 | 6 / 7 | Yes |
| Actor | 28→4 ×2 | Yes | Yes |
| Critic | 28→256→256→1 | Yes | Yes |
| Actor LR | 1e-4 | 1e-4 | Yes |
| Critic LR | 1e-3 | 1e-3 | Yes |
| Optimizer | Adam | Adam | Yes |
| Episodes | 3 | 3 (B4 10-seed) | Yes |
| Seeds | 100…1000 step 100 | Same | Yes |
| \(\rho\) | 1.0 | 1.0 | Yes |
| ADMM iters | B4 2; mid 25; GÉANT 100 | Same map | Yes |

Phase D 20-episode runs are **extensions**, not P1 protocol — keep separate (`PHASE_D_REPORT.md`).

---

## 5. Evaluation pipeline

### 5.1 Metric

Demand satisfaction = \(\sum\hat a / \sum d\) after constraint adjustment. Matches P1’s reported “demand satisfaction rate.”

MLU and capacity-violation counts are computed; Table 6 emphasizes satisfaction.

### 5.2 Training vs deployment

| Phase | ADMM? | Action | Adjust? |
|-------|-------|--------|---------|
| Train | No | Stochastic Gaussian | Yes → reward |
| Deploy | Yes | \(\mu\) + softmax | Yes after ADMM |

Matches P1 narrative.

### 5.3 ADMM variant actually used in archived B4 10-seed

| Artifact | Seed-100 GRL-TE sat | Matches |
|----------|---------------------|---------|
| `b4_seed100.json` `ours` | 0.967608 | |
| `admm_p1_vs_teal.json` `mean_sat_teal` | 0.967608 | **Teal ADMM** |
| `admm_p1_vs_teal.json` `mean_sat_p1` | 0.976657 | Different |

**Archived headline 97.44% ± 1.68% = Teal-order ADMM**, not P1-order.

**Current** `evaluate.py` default `admm_variant="p1"` ⇒ **re-running seeds without an explicit variant will change results** and will **not** regenerate 97.44%.

### 5.4 Baselines present vs Table 6

| Baseline | In Table 6 | In repo | Comparable? |
|----------|------------|---------|--------------|
| LP | Yes | Yes (`baselines.py`) | B4: 99.74% vs P1 99.83% — **near** |
| NCFlow | Yes | **No** | — |
| Teal (full) | Yes | **No** (ADMM only) | — |
| Figret | Yes | **No** | — |
| ECMP | Not in Table 6 | Yes (equal split) | Internal only |
| Shortest path | Not in Table 6 | Yes | Internal only |

---

## 6. Reported numerical results

### 6.1 B4 10-seed aggregate (archived)

Source: `results/logs/b4_all_seeds_summary.json`

| Method | Mean | Std | Notes |
|--------|------|-----|-------|
| GRL-TE + ADMM (Teal) | **97.44%** | **1.68%** | Not P1 92.39% |
| GRL-TE no ADMM | 96.51% | 2.75% | |
| LP | 99.74% | 0 | vs P1 99.83% |
| ECMP Yen | 98.74% | 0 | |

**Protocol caveats for any comparison to 92.39%:**

1. Teal ADMM ≠ P1 ADMM order/closed forms.  
2. TM split unpublished.  
3. Path set may differ (Yen vs edge-disjoint).  
4. Critic aggregation / clamps unstated in P1.

**Therefore 92.39% is not reproduced, and the results do not show an improvement over P1.**

### 6.2 LP gap (99.74 vs 99.83)

~0.09 pp. Plausible causes: path set, TM subset/split, floating-point / post-adjust, solver. **Does not prove wrong topology** (Table 5 stats match). Treat as **near match**, not exact.

### 6.3 ECMP Yen 98.74 vs edge-disjoint 94.20

Documented A/B only. P1 does not publish an ECMP number in Table 6.

### 6.4 Multi-topology (Phase F)

| Topology | GRL-TE Table-6 attempt | Outcome |
|----------|------------------------|---------|
| B4 | Full 10-seed | Done (qualified) |
| GÉANT Zoo | 1-seed gravity | Incomparable (wrong size; sat ≪ 86.41%) |
| UsCarrier | Paths only / toy TM | No faithful train |
| Cogentco | Gravity only | No Table-6 claim |
| Kdl / ASN | Scale blocker (~2.3M / ~12M path nodes) | Not trained |

**89.36% average: not reproduced.**

### 6.5 Independent reproduction in this audit turn

- **Did:** topology path-stat verification; log cross-checks; code–equation comparison; confirmed Teal identity of 97.44% pipeline.  
- **Did not:** retrain 10 seeds (methodology freeze; expensive). Reproducibility of 97.44% is **log-consistent**, not freshly re-executed here.

---

## 7. Missing experimental details in P1 (blockers)

1. Train/validation/test indices for each topology.  
2. Exact path algorithm and padding.  
3. ADMM closed forms / stopping (fixed iters only).  
4. Gravity model constants for Cogentco.  
5. Whether public Teal toy TMs were used (unlikely given Table 6 magnitudes).  
6. Hardware-sensitive timing claims (not audited as accuracy metrics).  
7. Per-seed raw Table 6 numbers (only means published).

---

## 8. Statistical reporting

| Practice | Status |
|----------|--------|
| Mean ± std over 10 seeds (B4) | Present in summary JSON |
| Same test TMs for all methods | Yes within a run |
| Averaging incomparable topologies | **Avoided** in Phase F (correct) |
| Table 6 equality | **Not supported** |

---

## 9. Experimental verdict

**PARTIALLY VERIFIED.**

- B4 topology + public Teal TMs + hyperparameter schedule + metric definition: **solid**.  
- B4 GRL-TE headline number: **internally reproducible under Teal ADMM + Yen + 20/8/8**, **not** P1 Table 6.  
- Six-topology Table 6: **not achievable** with current data.  
- Default evaluation ADMM drift: **experimental footgun**.
