# Baseline Freeze Decision

**Protocol ID:** `B4_GRLTE_FREEZE_v1`  
**Date:** 2026-11-07  
**Final status:** **BASELINE FROZEN WITH DOCUMENTED LIMITATIONS**

---

## 1. Which configuration is frozen

| Field | Frozen value |
|-------|--------------|
| Topology | B4 (TEAL, 12/38) |
| Traffic | TEAL public real TMs (36), split 20/8/8 |
| Paths | Yen 4-shortest, min-hop |
| Model | TopoFlowNet \(L=6\) + one-step A2C (3 episodes, seeds 100…1000) |
| Constraint adjust | Eqs. (23)–(28) |
| **ADMM** | **`teal` — order s → λ → z → x**, \(\rho=1\), **2** iters |
| Metric | Demand satisfaction post-adjust |
| Reference result | **97.44% ± 1.68%** |
| Config file | `BASELINE_FREEZE/frozen_b4_config.py` |
| Runner | `BASELINE_FREEZE/run_b4_baseline_freeze.py` |

**Parallel diagnostic (not frozen):** P1-order ADMM → **98.57% ± 0.73%** post-adjust with pre-adjust flow/demand ≈ 3×.

---

## 2. Why this configuration

1. **Reproducible:** Independent re-eval of all 10 archive checkpoints with explicit Teal ADMM matches the archive to machine precision (Δ = 0).  
2. **Numerically coherent:** Pre-adjust flow/demand ≈ 0.97–1.00; ADMM is a real refinement, not an overflow repaired only by (23)–(28).  
3. **Honest labeling:** Teal-order is **not** claimed to be P1 eqs. (49)–(52).  
4. **P1-order rejected as freeze target:** Algebraically consistent with derived Lagrangian, but **unusable** as a 2-iter deploy procedure under published \(\rho\) (structural small-\(d\) explosion). Freezing it would bake in a misleading “ADMM improved sat” story.

---

## 3. What it faithfully reproduces

- B4 topology structure consistent with P1 Table 5.  
- Public TEAL B4 traffic cited by P1.  
- TopoFlowNet + one-step A2C + (23)–(28) methodology on B4.  
- P1 hyperparameter schedule (layers, dims, LRs, episodes, seeds, \(\rho\), iter count).  
- Yen reading of “4 shortest paths.”  
- A **stable** ADMM-style post-process in the Teal family (known deviation).

---

## 4. What it does **not** reproduce

- P1 Table 6 B4 GRL-TE **92.39%**.  
- Six-topology average **89.36%**.  
- P1’s printed ADMM update order as the deploy solver.  
- Author-unpublished TM split (we use TEAL 20/8/8).  
- NCFlow / Figret / full Teal MARL columns.  
- GÉANT / UsCarrier / Kdl / ASN Table-6 experiments.

---

## 5. Remaining limitations (accepted)

| Limitation | Impact on Phase G |
|------------|-------------------|
| Teal ADMM ≠ P1 text order | DT comparisons cite the Protocol Card, not “exact P1 ADMM” |
| Split UNKNOWN vs authors | Absolute sat may differ from paper; relative DT gains still meaningful on this fixed protocol |
| Critic sum / clamps | Fixed as part of the baseline; not ablated |
| `evaluate.py` default still `p1` | Use freeze runner / explicit kwargs only |
| Multi-topo data gaps | Out of scope until separate workstream |

---

## 6. Preservation

- Phase A–F reports, graphs, and `results/logs/b4_all_seeds_summary.json` **untouched**.  
- Archive checkpoints **untouched**.  
- Neither ADMM implementation’s equations **modified**.  
- All freeze artifacts live under `BASELINE_FREEZE/`.

---

## 7. Decision statement

We freeze **Teal-order ADMM + Yen-4 + TEAL B4 real TMs + 20/8/8 + 3-ep/10-seed GRL-TE** as the **qualified B4 research baseline** for Digital Twin work.

We preserve P1-order ADMM as a **documented diagnostic**, not as the baseline.

We do **not** claim a complete P1 experimental reproduction.
