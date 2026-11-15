# P1 Baseline Protocol Card — B4_GRLTE_FREEZE_v1

**Status:** FROZEN (research baseline)  
**Date:** 2026-11-07  
**Config source:** `BASELINE_FREEZE/frozen_b4_config.py`  
**Runner:** `BASELINE_FREEZE/run_b4_baseline_freeze.py`  
**Machine-readable results:** `BASELINE_FREEZE/logs/b4_freeze_summary.json`

> Overall P1 reproduction remains **PARTIALLY VERIFIED**.  
> This card freezes a **qualified B4 research baseline**, not a claim of Table-6 equivalence.

---

| Category | Verified value |
|----------|----------------|
| **Paper** | Lu et al., “Graph-based reinforcement learning for software-defined networking traffic engineering,” *J. King Saud Univ. Comput. Inf. Sci.* **37:119 (2025)** |
| **Topology** | **B4** — 12 nodes, 38 directed edges; TEAL `data/topologies/B4.json`. Undirected avg SPL 2.318, diameter 5 (Table 5: 2.3 / 5) — **verified** |
| **Traffic** | **TEAL public real** matrices: `data/traffic_matrices/B4/B4.json_real_*_1.0_traffic-matrix.pkl` (**36** TMs). Not Microsoft private WAN. |
| **Path generation** | **Yen** \(k\)-shortest simple paths, **\(k=4\)**, metric **min-hop**, `EDGE_DISJOINT=False`, pad by repeating first path if fewer than 4. Cache: `data/candidate_paths/B4-4-paths_min-hop.pkl` |
| **Traffic split** | **20 / 8 / 8** → train `[0,20)`, val `[20,28)`, test `[28,36)`. **Source:** TEAL-style consecutive indexing. **vs P1 authors: UNKNOWN** (P1 does not publish the split) |
| **Model** | **TopoFlowNet** \(L=6\), path emb dim **7**, actor in-dim **28**; raw \(h^0\) (no capacity scaling) |
| **RL algorithm** | **One-step A2C**; \(r_t=F_{\mathrm{total}}\) after eqs. (23)–(28); no \(\gamma\); Adam actor \(10^{-4}\), critic \(10^{-3}\); train **without** ADMM |
| **Constraint handling** | Eqs. **(23)–(28)** via `optimization/constraint_adjustment.py` (train reward + deploy post-ADMM) |
| **ADMM (frozen reference)** | Variant **`teal`** — update order **`s → λ → z → x`** (`admm_refine_teal`). **Labeled deviation** from P1 printed order F→z→ξ→λ |
| **ADMM (diagnostic only)** | Variant **`p1`** — order **`F → z → ξ → λ`** (`admm_refine_p1`). **not mixed** into the frozen mean |
| **Iterations** | **2** (B4); \(\rho=1.0\) |
| **Seeds** | **100, 200, 300, 400, 500, 600, 700, 800, 900, 1000** |
| **Episodes** | **3** training episodes over the 20 train TMs (60 gradient steps / seed) |
| **Baselines** | Path-form **LP** (SciPy HiGHS); **ECMP** = equal \(1/4\) on Yen paths; shortest-path (path 0 only). Same topology, paths, and test TMs as GRL-TE |
| **Metrics** | Primary: **demand satisfaction** \(=\sum\hat a/\sum d\) after (23)–(28). Also MLU, capacity-violation count, **pre-adjust** ADMM `flow_over_demand` |
| **Results (frozen Teal channel)** | **97.44% ± 1.68%** (10 seeds). Log: `BASELINE_FREEZE/logs/b4_freeze_summary.json`. Archive bit-match: `results/logs/b4_all_seeds_summary.json` |
| **Results (diagnostic P1 channel)** | **98.57% ± 0.73%** post-adjust; pre-adjust flow/demand **≈ 3.07–3.46**. **Not** the frozen baseline |
| **LP / ECMP (same test set)** | LP **99.74%**; ECMP **98.74%**; SP **97.76%** |
| **Checkpoints** | Archive (unchanged): `models/baseline_p1/b4/grl_te_b4_seed{seed}.pt`. Freeze retrain example: `models/baseline_p1/b4_freeze/grl_te_b4_freeze_seed100.pt` |
| **Deviations from P1** | See § Deviations below |
| **Limitations** | See § Limitations below |

---

## Deviations (every known difference from P1)

| # | Deviation | Status |
|---|-----------|--------|
| D1 | Frozen ADMM is **Teal-order**, not P1 eqs. (49)–(52) order | **Documented; intentional for baseline** |
| D2 | TM split **unpublished** in P1; we use TEAL 20/8/8 | UNKNOWN vs authors |
| D3 | Path set: Yen vs possible TEAL edge-disjoint — P1 says “4 shortest”; we use Yen | Reasonable; not author-confirmed |
| D4 | Critic aggregates per-demand values by **sum** | Eq. (31) ambiguous |
| D5 | `rsample`, log-std clamp \([-10,10]\), raw-action clamp \(\pm 10\), grad clip 1.0 | Not in P1 |
| D6 | Closed forms of ADMM (49)–(51) derived, not copied from paper (paper omits them) | Both variants |
| D7 | Public Teal B4 TMs ≠ private Microsoft WAN | Dataset ceiling |
| D8 | Repo `training/evaluate.py` default remains `admm_variant="p1"` — **the freeze runner does not rely on that default** | Footgun outside freeze |

---

## Limitations

- **Does not** reproduce P1 Table 6 B4 GRL-TE **92.39%**.  
- **Does not** reproduce six-topology average **89.36%**.  
- NCFlow / Figret / full Teal MARL baselines: **not in this baseline**.  
- Multi-topology traffic (UsCarrier/Kdl/ASN toy; GÉANT wrong size): **not covered** by this card.  
- Author ADMM numerics: **UNKNOWN**.

---

## Scripts

The frozen numbers are produced by `BASELINE_FREEZE/run_b4_baseline_freeze.py`, which re-evaluates the ten archived checkpoints; with `--retrain-seed 100` it retrains seed 100 into `models/baseline_p1/b4_freeze/`.

The reference number is the `frozen_b4_config.FROZEN_ADMM_VARIANT` (`teal`) channel. The `p1` diagnostic channel is reported separately and is not averaged with it.
