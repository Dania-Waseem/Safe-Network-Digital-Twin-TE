# Phase F Report — Final P1 Reproduction & Multi-Topology Validation

**Paper:** Lu et al., 2025 (GRL-TE)  
**Companions:** `FINAL_P1_DATA_AUDIT.md`, `FINAL_P1_METHODOLOGY_COMPLIANCE.md`, `FINAL_P1_EXPERIMENTAL_RESULTS.md`, `FINAL_P1_DEVIATIONS.md`  
**Logs/plots:** `results/logs/phase_f/`, `results/graphs/phase_f/`  
Phases A–E artifacts were **not overwritten**.

---

## Overall verdict: **PARTIALLY VERIFIED**

| Layer | Status |
| --- | --- |
| Independently implemented methodology (TopoFlowNet, one-step A2C, (23)–(28), P1+Teal ADMM code) | **Mostly verified** |
| Mathematically checked components | **Mostly verified** (ADMM closed forms derived; P1-order unstable @ 2 iters) |
| Experimentally validated B4 | **Mostly verified** (10 seeds, 3 episodes) |
| Directly comparable to Table 6 / 89.36% | **No** |
| Six-topology Table 6 reproduction | **Failed / blocked** on public data (see below) |

We do **not** claim 97.44% beats P1’s 92.39%.

---

## A. B4 discrepancy investigation (executed)

### 1. Candidate paths — **resolved as non-bug**

P1 text: “precompute **4 shortest paths**.” No edge-disjoint constraint.

| | Yen (ours, default) | TEAL edge-disjoint pickle |
| --- | --- | --- |
| Unique paths / pair | **4.0** | **2.35** (padded to 4) |
| Identical 4-lists | 6 / 132 | — |
| First path identical | 118 / 132 | — |
| Test **LP** | 99.74% | 99.74% |
| Test **ECMP** | 98.74% | 94.20% |

**Decision:** Keep Yen. Switching to TEAL disjoint would follow Teal, not P1. LP identical ⇒ path set does **not** explain the LP 99.74 vs 99.83 gap. It **does** change ECMP/GRL-TE feasible splits.

### 2. Train/test split — **unresolved limitation**

P1 does not publish a split. Ours: TEAL-style **20/8/8**. Test-set mean demand **27981 ≈ P1 27.98k**. Not claimed as the authors’ cut.

### 3. ADMM — **no solver change**

Both `admm_refine_p1` (order F→z→ξ→λ, unstable @ B4 2 iters) and `admm_refine_teal` (stable) preserved. Deploy baseline remains Teal-order with ρ=1, 2 iters. Unpublished argmins remain a limitation.

---

## B. Faithful B4 baseline (P1 protocol: 3 episodes, 10 seeds)

From archived `b4_all_seeds_summary.json` (unchanged):

| Method | Ours | P1 Table 6 |
| --- | ---: | ---: |
| GRL-TE + Teal ADMM | **97.44% ± 1.68%** | 92.39% |
| GRL-TE no ADMM | 96.51% ± 2.75% | (ablation −0.68 pt) |
| LP | 99.74% | 99.83% |
| ECMP (Yen) | 98.74% | — |

Phase D 20-ep numbers are **validation only**, not this baseline.

---

## C. Six topologies (one at a time)

| Topology | Table 5 | Ours | Size match | Traffic used | GRL-TE run |
| --- | --- | --- | --- | --- | --- |
| B4 | 12 / 38 | 12 / 38 | **Yes** | TEAL **real** | **10 seeds, 3 ep** |
| GÉANT | 22 / 72 | Zoo Geant2012 **40 / 122** | **No** | reconstructed gravity | 1 seed (incomparable) |
| UsCarrier | 158 / 378 | 158 / 378 | **Yes** | TEAL **toy unit-demand** | paths built; no 10-seed train |
| Cogentco | 197 / 486 | 197 / 486 | **Yes** | reconstructed gravity | topology only |
| Kdl | 754 / 1790 | 754 / 1790 | **Yes** | TEAL toy unit-demand | size only (~2.3M path nodes) |
| ASN | 1739 / 8558 | 1739 / 8558 | **Yes** | TEAL toy unit-demand | size only (~12M path nodes) |

**Blockers (not fabricated):**

- Microsoft/Teal production TMs are **not** in the public repo (toy = all 1.0).
- FIGRET 22-node GÉANT + TMs were **not** in this workspace; Zoo Geant2012 ≠ Table 5.
- P1 Cogentco gravity constants unpublished.
- Kdl/ASN full GRL-TE exceeds practical CPU path-form scale here.

GÉANT 40-node 1-seed: train sat ~5%, test no-ADMM 5%, Teal ADMM 27% — overloaded gravity, **not** Table 6 (86.41%).

---

## D. What “complete reproduction” means after Phase F

**Have:** a defensible B4 GRL-TE implementation + public TEAL data + documented hyperparameters.  
**Do not have:** a six-topology Table 6 replica or 89.36% average.

The Digital Twin work builds on the **qualified B4 baseline**, not on a claimed 89.36% match.
