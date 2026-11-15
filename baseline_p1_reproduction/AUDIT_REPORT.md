# GRL-TE Reproduction Audit Report (Phase A — Independent Re-audit)

**Paper:** Lu et al., 2025. *Graph-based reinforcement learning for software-defined networking traffic engineering.*  
*Journal of King Saud University Computer and Information Sciences* **37**:119.  
DOI: https://doi.org/10.1007/s44443-025-00133-z

**Codebase:** `baseline_p1_reproduction/`  
**Audit date:** Updated through Phase E (experimental fidelity)  
**Phase reports:** `PHASE_B_REPORT.md` … `PHASE_E_REPORT.md`, `P1_DATA_AUDIT.md`, `P1_METHODOLOGY_COMPLIANCE.md`  
**Prior correction log:** `CORRECTIONS.md`  
**Pre-correction archive:** `results/pre_correction_archive/`  
**B4 10-seed source:** `results/logs/b4_all_seeds_summary.json`

**Status vocabulary used below**

| Label | Meaning |
| --- | --- |
| **Verified** | Matches P1 and was checked in code / runs |
| **Mostly verified** | Matches intent; minor unspecified engineering choices |
| **Implemented but unverified** | Present; behaviour / fidelity not fully proven |
| **Not implemented** | Required by P1 evaluation; absent here |
| **Deviates from P1** | Intentionally or accidentally different from the paper |

---

## Section 1: Overall verdict

### Mostly faithful for B4 architecture; experimental Table-6 match PARTIALLY VERIFIED

| Checkpoint | Verdict |
| --- | --- |
| Pre Phase-2 audit | Partial reproduction (critical ADMM / reward / \(h^0\) bugs) |
| Post Phase-2 / A–D | Mostly faithful B4 model + training; critic mostly verified; ADMM order/numerics qualified |
| **Phase E** | TEAL B4 data/source **verified**; **97–98% not directly comparable** to P1 **92.39%** (paths / split / ADMM) — see `PHASE_E_REPORT.md` |

**Defendable claim**

> TopoFlowNet and one-step A2C follow Lu et al. (2025) on public TEAL B4 data, with raw \(h^0\), \(r_t=F_{\text{total}}\), constraint adjustment (23)–(28), and a **Teal-derived** ADMM post-process (\(\rho=1\), 2 iters). Ten seeds (3 eps): **97.44% ± 1.68%** (P1 B4: **92.39%**). Phase E: **not** claimed as a Table-6-comparable improvement (paths/split/ADMM differ).

**Not defendable**

> Exact GRL-TE reproduction / recovered Table 6 **92.39%** / six-topology average **89.36%**.

---

## Section 2: Repository map (Phase A)

```
baseline_p1_reproduction/
├── config.py                          # LRs, ρ, ADMM iters, seeds, P1_PUBLISHED
├── preprocessing/
│   ├── load_topology.py               # TEAL JSON DiGraph
│   ├── load_traffic.py                # TM pickles; Teal 20/8/8 split
│   ├── generate_paths.py              # Yen 4-shortest (min-hop)
│   └── build_bipartite_graph.py       # link–path graph; raw h⁰
├── models/
│   ├── topoflownet.py                 # eqs. (5)–(12)
│   ├── actor.py / critic.py / a2c.py
├── optimization/
│   ├── admm.py                        # Teal order s→λ→z→x  ← DEVIATES
│   └── constraint_adjustment.py      # eqs. (23)–(28)
├── training/train.py, evaluate.py
├── experiments/
│   ├── b4.py, run_b4_seeds.py         # B4 done
│   ├── baselines.py                   # LP, ECMP, SP only
│   ├── geant.py, unseen_topologies.py, link_failures.py  # stubs
├── data/                              # B4 only
└── results/                           # 10-seed logs + archive
```

**External (not wired):** `reference/teal/` holds UsCarrier, Kdl, ASN2k topologies + TMs. Cogentco and GÉANT are **missing** from the workspace.

---

## Section 3: Master reproduction audit table

| Component | Paper requirement | Code | Status |
| --- | --- | --- | --- |
| B4 topology / capacities | 12 nodes, 38 edges (Table 5) | `load_topology.py` | **Verified** |
| Four candidate paths | 4 shortest / pair | Yen min-hop (`generate_paths.py`) | **Mostly verified** (Yen not named in P1; TEAL uses edge-disjoint) |
| Bipartite link–path graph | §4.2.1 | `build_bipartite_graph.py` | **Verified** |
| Raw \(h^0\) (eq. 5) | \(c(e)\), \(d\) unscaled | `scale=False` | **Verified** |
| TopoFlowNet (5)–(12) | 6 GIN layers, path dim 7, skip-concat | `topoflownet.py` | **Verified** |
| Actor (14)–(19) | 28→4 Gaussian → softmax | `actor.py`, `a2c.py` | **Mostly verified** (clamps not in paper) |
| Critic | 28→256→256→1; \(V(s)\) | sum of per-demand heads | **Mostly verified** (aggregation unspecified) |
| Critic numerical stability | — | Diagnosed seed 100: learns; large MSE = \(F_{\text{total}}\) scale; no NaN | **Mostly verified** — see `PHASE_C_REPORT.md` |
| Reward (29)–(30) | \(r_t=F_{\text{total}}\) | `feasible.sum()` | **Verified** |
| One-step A2C (36)–(39) | \(A=r-V\); no \(\gamma\) | `train.py` | **Mostly verified** |
| Shared TopoFlowNet + Adam | actor 1e-4 / critic 1e-3 | joint Adam groups | **Verified** |
| Constraint adj. (23)–(28) | demand + capacity repair | `constraint_adjustment.py` | **Verified** |
| 10 seeds, 3 episodes | 100…1000 | `run_b4_seeds.py` | **Verified** |
| ADMM (40)–(52) | order **F→z→ξ→λ**; \(\rho=1\); B4: 2 iters | **P1 order implemented** (`admm_refine_p1`); Teal kept for A/B. Derived argmins + ReLU; B4 2-iter **numerically unstable** (flow/d≈3) | **Mostly verified** (order); **Deviates** (stable numerics / unpublished argmins) — see `PHASE_B_REPORT.md` |
| UsCarrier / Cogentco / Kdl / ASN / GÉANT | §5.1 six WANs | B4 only in `data/` | **Not implemented** |
| Baselines LP | Table 6 | path LP in `baselines.py` | **Verified** (B4 99.74% vs 99.83%) |
| Baselines Teal / FIGRET / NCFlow | Table 6 | absent (ECMP/SP only extras) | **Not implemented** |
| Six-topology average 89.36% | Table 6 | — | **Not implemented** |

---

## Section 4: What Phase-2 fixed (still stands)

| # | Original bug | After fix |
| --- | --- | --- |
| 1 | ADMM flow / demand ≈ **3.55×** | ≈ **0.99×**; g3 residual 21.8 → 1.73 over 2 iters |
| 2 | Reward ≠ eq. (30) | \(r_t=F_{\text{total}}\) |
| 3 | \(h^0\) scaled | raw features |
| 4 | Single seed | seeds 100…1000 |
| 5 | Weak shared TFN training | one Adam, two LRs, joint backward |

ADMM **stability** is fixed. ADMM **P1 mathematical fidelity** is **not** fixed (Phase B).

---

## Section 5: Remaining gaps (Phase A ranking)

### Priority 1 — ADMM (**Deviates from P1**)

| Aspect | P1 §4.4.5 | This repo |
| --- | --- | --- |
| Update order | F → z → ξ → λ | s → λ → z → x (Teal) |
| Primary vars | fractions \(F_d(p)\in[0,1]\) | path flows \(x=F\cdot d\) |
| Closed forms of (49)–(51) | **Not published** | Copied from `harvard-cns/teal` `lib/ADMM.py` |
| \(\rho\), B4 iters | 1, 2 | 1, 2 |

**Phase B goal:** P1-order solver with derived stationarity of (48); keep Teal solver for A/B; residual + demand tests. Do **not** “fix” by changing the formulation.

### Priority 2 — Critic (**Mostly verified** after Phase C)

- Seed-100 diagnosis: relative error 5.79 → 0.79 over 3 episodes; \(\mathrm{MSE}/r^2\approx(\mathrm{rel\ err})^2\) → **large MSE is scale**
- No NaN/Inf; critic architecture matches P1; target = immediate \(r\) (no \(\gamma\))
- Grad clip 1.0 always active (engineering, not in P1) — measured, not removed
- No reward normalization or other adaptations applied

### Priority 3 — Five topologies (**Not implemented**)

| Topology | P1 source | Local availability |
| --- | --- | --- |
| B4 | TEAL | **In `data/`** |
| UsCarrier | TEAL | `reference/teal/` — not wired |
| Kdl | TEAL | `reference/teal/` — not wired |
| ASN | TEAL | `reference/teal/ASN2k.json` — not wired |
| Cogentco | Topology Zoo + gravity TMs | **Missing** |
| GÉANT | FIGRET | **Missing** |

### Priority 4 — Baselines (**Not implemented**)

Wrap / call official Teal, FIGRET, NCFlow on **identical** TMs and capacity constraints. Toy clones are not treated as exact baselines. The LP is already a fair path-formulation baseline.

### Priority 5 — Full evaluation (**Not implemented**)

Six topologies × P1 seeds; honest comparison to Table 6, without tuning toward 89.36%.

---

## Section 6: Experimental evidence (B4)

### Protocol vs P1

| Setting | P1 | This repo | Match? |
| --- | --- | --- | --- |
| B4 size | 12 / 38 | 12 / 38 | Yes |
| Episodes / LRs / \(\rho\) | 3 / 1e-4 / 1e-3 / 1 | Same | Yes |
| Seeds | 10 (100…1000) | Same | Yes |
| ADMM B4 iters | 2 | 2 | Count yes; solver Teal |
| Topologies | 6 | 1 | No |

### Aggregate test demand satisfaction (seeds 100…1000)

| Method | Mean ± std | P1 Table 6 (B4) |
| --- | ---: | ---: |
| GRL-TE + ADMM | **97.44% ± 1.68%** | 92.39% |
| GRL-TE no ADMM | **96.51% ± 2.75%** | (ablation −0.68 pt) |
| ADMM lift | **+0.93 pt** | +0.68 pt |
| LP | 99.74% | 99.83% |
| ECMP | 98.74% | — |

### Per-seed GRL-TE + ADMM

| Seed | Sat. % |
| ---: | ---: |
| 100 | 96.76 |
| 200 | 96.74 |
| 300 | 93.49 |
| 400 | 97.07 |
| 500 | 99.68 |
| 600 | 97.86 |
| 700 | 96.47 |
| 800 | 98.69 |
| 900 | 98.83 |
| 1000 | 98.86 |

Gap to 92.39% is **reported honestly**; causes under investigation (TM load, path set, ADMM order, critic scale) — not forced by retuning.

---

## Section 7: Missing / approximated / unverifiable (paper silence)

| Item | Classification |
| --- | --- |
| Numerical argmin of (49)–(51) | **Not in paper** → Teal substitute (documented) |
| ADMM block order | **Different** from P1 list |
| Critic \(V(s)\) aggregation | **Not specified** → sum of heads |
| TM train/val/test cut | **Not specified** → Teal 20/8/8 |
| Path algorithm | **Not specified** → Yen min-hop |
| Grad clip / action clamps | **Engineering extras** |
| Cogentco / GÉANT data | **Missing** |
| Teal / FIGRET / NCFlow | **Missing** |

---

## Section 8: Prioritized plan (next phases)

| Phase | Task | Status |
| --- | --- | --- |
| **A** | Repo + paper audit (this document) | **Done** |
| **B** | ADMM F→z→ξ→λ fidelity + residual tests | **Done** — see `PHASE_B_REPORT.md` |
| **C** | Critic stability diagnosis (document any non-P1 fix) | **Done** — `PHASE_C_REPORT.md`; no methodology change |
| **D** | Extended A2C / critic multi-seed validation (B4) | **Done** — `PHASE_D_REPORT.md`; verdict **MOSTLY VERIFIED** |
| **E** | P1 experimental fidelity vs Table 6 / 92.39% | **Done** — `PHASE_E_REPORT.md`; verdict **PARTIALLY VERIFIED** |
| **F** | Multi-topology + B4 discrepancy close-out | **Done** — `PHASE_F_REPORT.md`; overall **PARTIALLY VERIFIED** |
| **G** | Official Teal / FIGRET / NCFlow comparisons | Pending |
| **H** | Six-topology eval vs Table 6 (honest) | Pending |

**First concrete implementation task:** Phase B — implement P1-order ADMM alongside Teal baseline; compare residuals and B4 seed-100 satisfaction without retuning the network.

---

## Section 9: Baseline / thesis readiness

| Question | Answer |
| --- | --- |
| Use as Phase 3 **B4** GRL-TE baseline? | **YES, with qualifications** |
| Claim exact P1 / Table 6 / 89.36%? | **NO** |
| Digital twin / safety gate on B4 now? | **YES** (reasonable) |
| Multi-WAN claims? | **NOT YET** |

---

## Section 10: Final checklist

**Verified**

- [x] B4 topology / capacities  
- [x] Bipartite graph  
- [x] TopoFlowNet (5)–(12)  
- [x] Raw \(h^0\)  
- [x] \(r_t = F_{\text{total}}\)  
- [x] Shared TopoFlowNet + Adam LRs  
- [x] Constraint adjustment (23)–(28)  
- [x] 10-seed B4 protocol  
- [x] ADMM non-exploding + eq. (28) feasibility  

**Mostly verified**

- [x] Four paths (Yen assumption)  
- [x] Actor / one-step A2C (clamps / critic aggregation)  

**Deviates / incomplete**

- [ ] ADMM update order & closed forms = P1 (**Deviates** — Teal)  
- [x] Critic numerical behaviour (**Mostly verified** — Phase C scale MSE; Phase D rel-err → ~0.06 at 20 eps)  
- [x] Extended multi-seed A2C (20 eps, 10 seeds) — **MOSTLY VERIFIED** (`PHASE_D_REPORT.md`)  
- [ ] Five other topologies  
- [ ] Teal / FIGRET / NCFlow  
- [ ] Six-topology average 89.36%  
- [ ] Match P1 B4 92.39% (not a tuning target)
