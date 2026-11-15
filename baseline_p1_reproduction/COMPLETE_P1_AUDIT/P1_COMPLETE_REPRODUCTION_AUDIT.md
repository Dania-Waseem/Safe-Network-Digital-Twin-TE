# Complete P1 Reproduction & Implementation Audit

**Paper:** Lu et al., “Graph-based reinforcement learning for software-defined networking traffic engineering,” *J. King Saud Univ. Comput. Inf. Sci.* 37:119 (2025) — hereafter **P1 / GRL-TE**.  
**Codebase audited:** `baseline_p1_reproduction/`  
**Audit date:** 2026-11-07  
**Scope:** Independent evidence-based audit. **No code or methodology was modified** during this audit. Prior Phase A–F artifacts were read as secondary evidence only; primary evidence is the paper (`p1.pdf`) and source files.

**Overall classification: PARTIALLY VERIFIED**

---

## 1. Executive summary

The repository implements a **substantially complete algorithmic pipeline** for GRL-TE on **B4**: TopoFlowNet (6-layer GIN + path MLP + skip), one-step A2C with \(r_t=F_{\mathrm{total}}\), constraint adjustment (23)–(28), ADMM warm-start refinement, LP/ECMP/shortest-path baselines, Yen 4-shortest paths, and the published hyperparameter schedule (3 episodes, seeds 100…1000, \(\rho=1\), B4 ADMM iters = 2).

It is **not** a faithful full-paper reproduction:

| Area | Status |
|------|--------|
| TopoFlowNet architecture | **Mostly correct** (eqs. 5–12; dims match §5.3) |
| One-step A2C | **Mostly correct** (eqs. 36–39); minor unstated clamps / `rsample` / critic aggregation |
| Constraint adjustment | **Correctly implemented** (23)–(28) |
| ADMM (P1-order closed forms) | **Algebraically derived from (48)–(52)**; **numerically unstable** at P1’s B4 budget (2 iters); paper does not publish closed forms |
| ADMM (Teal-order) | **Not P1**; used for the archived B4 headline number **97.44% ± 1.68%** |
| Six topologies (Table 5) | **5/6 structure-matched**; **GÉANT wrong graph** (40/122 vs 22/72) |
| Traffic (Table 6 data) | **Only B4 TEAL real TMs** are usable; others are toy / gravity / wrong-size |
| Table 6 / 89.36% / 92.39% | **Not independently reproducible** under equivalent conditions |
| NCFlow / Figret / full Teal baselines | **Missing** |
| Phase G (Digital Twin) readiness | **No** — B4 is a qualified research scaffold, not a verified P1 baseline |

**Headline numbers (archived, Teal ADMM, Yen paths, TEAL 20/8/8 split):**

| Method | Ours | P1 Table 6 |
|--------|------|------------|
| GRL-TE (B4) | **97.44% ± 1.68%** | **92.39%** |
| LP (B4) | **99.74%** | **99.83%** |
| ECMP Yen | **98.74%** | (not in Table 6) |
| ECMP edge-disjoint (A/B) | **94.20%** | — |

The ~5 pp gap vs P1’s 92.39% is **not** evidence that our stack is “better”; experimental conditions are **not equivalent** (ADMM numerics, unpublished TM split, path-set ambiguity). Claiming Table-6 comparability would be incorrect.

---

## 2. Repository inspection

### 2.1 Layout (primary modules)

```
baseline_p1_reproduction/
  config.py
  models/{topoflownet,actor,critic,a2c}.py
  training/{train,evaluate}.py
  optimization/{admm,constraint_adjustment}.py
  preprocessing/{load_topology,load_traffic,generate_paths,build_bipartite_graph,gravity}.py
  experiments/{b4,run_b4_seeds,baselines,phase_*,geant,link_failures}.py
  data/{topologies,traffic_matrices,candidate_paths,raw_zoo}/
  results/{logs,tables,graphs,models}/
  PHASE_*.md, FINAL_P1_*.md, AUDIT_REPORT.md   # prior reports (not overwritten)
```

### 2.2 What exists and connects

- Training path: TM → `h0` → TopoFlowNet → Actor → softmax → path flows → **constraint adjust** → \(r=F_{\mathrm{total}}\) → A2C update. **ADMM is not used in training** (matches P1 §4 / §4.3).
- Deployment path: deterministic \(\mu\) → **ADMM** → constraint adjust → metrics (`training/evaluate.py`).
- Baselines: path-LP, equal-split ECMP, shortest-path (`experiments/baselines.py`).
- Dual ADMM solvers in `optimization/admm.py`: `admm_refine_p1` (default alias) and `admm_refine_teal`.

### 2.3 Critical wiring inconsistency (evaluation default)

- Archived 10-seed B4 result **97.44%** matches Teal ADMM (`admm_p1_vs_teal.json`: seed-100 Teal sat = 0.9676 = `b4_seed100.json` `ours`).
- Current `evaluate.allocate(..., admm_variant="p1")` **defaults to P1-order ADMM**.
- `experiments/b4.py` does **not** pass `admm_variant="teal"`, so a fresh `run_b4_seeds.py` would **not** reproduce the archived 97.44% aggregate without changing the call site.
- Stale notes in `b4.py` still claim scaled \(h^0\) and normalized reward; code uses **raw** \(h^0\) and \(r=\sum\hat a\) (correct per eqs. 5, 30).

**Evidence:** `training/evaluate.py:24`, `results/logs/b4_seed100.json`, `results/logs/admm_p1_vs_teal.json`, `results/logs/b4_all_seeds_summary.json`.

---

## 3. Mathematical verification (summary)

Full derivations: see `P1_MATHEMATICAL_AUDIT.md`.

### 3.1 TopoFlowNet (eqs. 5–12)

| Check | Verdict |
|-------|---------|
| \(h^0=[c(e); d_p]\) raw | Correct (`TEProblem.initial_features`, `scale=False`) |
| Bidirectional GIN (6)–(8) | Correct: \((1+\epsilon)h + \sum N\), then ReLU-MLP with \(W\in\mathbb{R}^{l\times l}\) |
| Path collab MLP (10) | Correct: reshape to \(|D|\times(kl)\), two Linear+ReLU |
| Skip concat with \(h^0\) (12) | Correct → final path dim \(L+1=7\) |
| \(L=6\), actor in-dim 28 | Correct (`config.py`, `a2c.py`) |

**Deviations:** none material for architecture. Message passing is a hand-rolled `index_add_` GIN (not PyG); math matches (6)–(7).

### 3.2 One-step A2C (eqs. 13–22, 36–39)

| Check | Verdict |
|-------|---------|
| Reshape path emb (13) | Correct |
| \(\mu,\log\sigma\) linear heads (14)–(15) | Correct |
| Softmax fractions (19)–(21) | Correct |
| Deploy = mean + softmax | Correct (`deterministic=True`) |
| \(A=r-V\), no \(\gamma\) (36)–(37) | Correct |
| Actor loss \(\propto -A\log\pi\) (38) | Correct |
| Critic via MSE \(\Leftrightarrow\) (39) under GD | Correct (see math audit) |
| \(r_t=F_{\mathrm{total}}\) (29)–(30) | Correct (`feasible.sum()`) |

**Deviations / gaps:**

1. `dist.rsample()` vs paper “sample” (18) — minor; score-function still used.
2. `LOG_STD` clamp \([-10,10]\) and `RAW_ACTION_CLAMP=10` — **not in P1**.
3. Critic outputs per-demand scalars then **`.sum()`** — P1 eq. (31) is ambiguous; sum is a reasonable but **unverified** choice so \(V\) matches \(F_{\mathrm{total}}\) scale.
4. No entropy term — consistent with P1 (P1 does not specify entropy).
5. Grad clip 1.0 — unstated in P1.

### 3.3 ADMM (eqs. 40–52)

P1 publishes **argmins of \(L_p\)** (49)–(51) and dual step (52), **not** closed forms.

**Independent derivation** (math audit) yields, for \(\rho=1\), unconstrained stationarity + ReLU:

- **F-update:** solve \((11^\top + d^2\mathrm{diag}(n))F = \mathrm{rhs}/\rho\), then ReLU.
- **z-update:** per-edge \((I+11^\top)z=b\), closed form \(z=b-S\cdot 1\), then ReLU.
- **ξ-update:** separable ReLU projections.
- **λ-update:** \(\lambda\leftarrow\lambda+\rho G\).

**`admm_refine_p1`** implements exactly this order **F→z→ξ→λ**.

**Empirical fact (not patched):** with B4’s 2 iterations, post-ADMM \(\sum F\cdot d / \sum d \approx 3.07\) before constraint adjust (`admm_p1_vs_teal.json`). Algebra matches LBFGS stationarity audits (Phase B); instability is **conditioning + short iteration budget**, not a typo in the linear solve.

**`admm_refine_teal`** uses Teal’s **s→λ→z→x** order and Teal-style closed forms. It is a **deliberate non-P1 baseline**. It keeps flow≈demand (ratio≈0.98) and produced the archived 97.44% number.

**Verdict:**

- P1-order solver: **mostly correct derivation of unpublished closed forms**; **not numerically faithful as a 2-iter B4 deployment procedure** under observed residuals.
- Teal solver: **incorrect as a P1 component**; **useful A/B control only**.

### 3.4 Constraint adjustment & TE LP (1)–(4), (23)–(28)

- Demand scale-down (23)–(25) and capacity path scale (27)–(28): **Correct** (`constraint_adjustment.py`).
- Path LP max throughput s.t. demand & capacity: **Correct formulation** (`baselines.lp_allocate`); post-adjust for numerical cleanliness.

---

## 4. Architecture verification

| Module | Paper §5.3 | Implementation | Match |
|--------|------------|----------------|-------|
| TopoFlowNet layers | 6 | `NUM_LAYERS=6` | Yes |
| Output dim | 7 | path emb dim 7 | Yes |
| Actor | (28,4)×2 heads | `Actor` | Yes |
| Critic | 28→256→256→1 + ReLU | `Critic` | Yes (aggregation extra) |
| Adam LRs | \(10^{-4}/10^{-3}\) | `ACTOR_LR`/`CRITIC_LR` | Yes |
| Episodes | 3 | `EPISODES=3` | Yes |
| Seeds | 100…1000 | `SEEDS` | Yes |
| \(\rho\) | 1.0 | `RHO=1` | Yes |
| ADMM iters | B4:2; mid:25; GÉANT:100 | `ADMM_ITERS` | Yes |

Shared TopoFlowNet for actor+critic with two Adam param groups: **matches** P1’s end-to-end design (§4.3.1).

---

## 5. Dataset and topology verification

### 5.1 Table 5 structure check (beyond node/edge counts)

Undirected largest-component average shortest-path length and diameter:

| Topology | P1 (n/e, avg, diam) | Ours | Structure match |
|----------|---------------------|------|-----------------|
| B4 | 12/38, 2.3, 5 | 12/38, 2.318, 5 | **Yes** (TEAL JSON) |
| UsCarrier | 158/378, 12.1, 35 | 158/378, 12.090, 35 | **Yes** (TEAL) |
| Cogentco | 197/486, 10.51, 28 | 197/486, 10.510, 28 | **Yes** (Zoo GML) |
| Kdl | 754/1790, 22.7, 58 | 754/1790, 22.727, 58 | **Yes** (TEAL) |
| ASN | 1739/8558, 3.2, 8 | 1739/8558, 3.236, 8 | **Yes** (TEAL ASN2k) |
| GÉANT | **22/72**, 2.53, 5 | **40/122**, 3.528, 8 | **No** (Zoo GEANT.gml ≠ FIGRET 22-node) |

### 5.2 Traffic fidelity

| Topology | P1 claim | What we have | Faithful? |
|----------|----------|--------------|-----------|
| B4 | Teal TMs + caps | 36 TEAL `*_real_*` matrices | **Yes (public Teal)** — not Microsoft private WAN |
| UsCarrier / Kdl / ASN | Teal TMs | Public Teal **toy unit-demand** float16→float64 | **No** |
| Cogentco | Gravity synthetic | Local gravity reconstruction (`gravity.py`) | **Partially** — method class matches; constants/seed **not** P1’s |
| GÉANT | Figret / Uhlig | Gravity on **wrong** 40-node graph | **No** |

### 5.3 Paths

- Default: Yen / `nx.shortest_simple_paths`, \(k=4\), min-hop, pad by repeating first path (`generate_paths.py`).
- P1 text: “4 shortest paths” — **aligned**.
- TEAL public pickles are **edge-disjoint** (often &lt;4 unique); A/B: LP nearly identical; ECMP **98.74% (Yen) vs 94.20% (edge-disjoint)** (`path_ab.json`).
- **Cannot verify** which path set P1 used.

### 5.4 Train/val/test split

P1 does **not** publish the split. We use TEAL-style **20/8/8** on 36 B4 matrices (`config.B4_SPLITS`). **Cannot verify** equivalence to P1.

---

## 6. Experimental protocol verification

| Protocol element | P1 | Ours | Status |
|------------------|----|------|--------|
| 3 episodes | Yes | Yes (B4 10-seed) | Match |
| 10 seeds 100…1000 | Yes | Yes (archived) | Match |
| Demand satisfaction metric | Yes | `allocated/total_demand` | Match |
| Mean ± std over seeds | Implied | Reported | Match |
| ADMM after policy (deploy) | Yes | Yes | Match |
| Same paths for LP/ECMP/GRL | Implied | Yes | Match |
| NCFlow / Figret / Teal numbers | Table 6 | **Not reimplemented** | Missing |
| RTX 4090 / Xeon setup | Stated | CPU runs in logs | Hardware differs (not claimed as issue for B4 sat) |
| Link-failure §5.5 | Paper section | Script exists; not full Table reproduction | Incomplete |

Toy-TM or wrong-GÉANT runs are not averaged with B4 into an “89.36%” claim; Phase F likewise kept them separate.

---

## 7. Numerical results (independent reading of logs)

### 7.1 Archived B4 10-seed (Teal ADMM)

From `results/logs/b4_all_seeds_summary.json`:

- GRL-TE + ADMM: **97.44% ± 1.68%**
- GRL-TE no ADMM: **96.51% ± 2.75%**
- LP: **99.74%**
- ECMP (Yen): **98.74%**
- P1 GRL-TE B4: **92.39%** (published only)

### 7.2 P1 vs Teal ADMM (same seed-100 checkpoint)

From `admm_p1_vs_teal.json` (2 iters, then our logs compare pre-adjust residuals):

| Variant | Mean sat (after full pipeline in that script) | Mean flow/demand (ADMM out) |
|---------|-----------------------------------------------|------------------------------|
| No ADMM | 95.20% | — |
| P1-order | 97.67% | **~3.07** |
| Teal-order | 96.76% | **~0.98** |

P1-order “improves” satisfaction only after **constraint adjustment absorbs large infeasibility**. That is **not** evidence of a healthy ADMM warm start.

### 7.3 Path A/B

`path_ab.json`: LP Yen ≈ LP edge-disjoint ≈ 99.74%; ECMP diverges (98.74% vs 94.20%).

### 7.4 What was **not** re-run in this audit

Full 10-seed retraining was **not** repeated (hours; methodology freeze). Reproducibility claim for 97.44% rests on **existing seed logs + Teal ADMM identity check**, not a fresh independent train. Fresh `b4.py` would currently evaluate with **P1 ADMM default** unless `admm_variant` is set.

---

## 8. Identified discrepancies (ordered by severity)

1. **Table 6 not reproducible** — missing faithful multi-topo TMs; GÉANT graph wrong; ADMM closed forms unpublished; split unpublished.
2. **Dual ADMM + default mismatch** — research narrative uses Teal for B4 headline; code default is now P1-order.
3. **P1 ADMM @ 2 iters overflows** demand fractions (flow/demand ~3×) before (23)–(28).
4. **Non-B4 traffic is not P1 traffic** (toy / reconstructed gravity).
5. **Baselines NCFlow, Figret, Teal (full)** absent — cannot rebuild Table 6 columns.
6. **Critic aggregation** and **action clamps** are implementation choices not stated in P1.
7. **Stale comments** in `b4.py` notes contradict current \(h^0\)/reward code.
8. **`experiments/geant.py`** still exits as placeholder despite Phase F wiring of Zoo GEANT.

---

## 9. Missing components

- FIGRET 22-node GÉANT topology + TMs  
- Authentic Teal WAN TMs for UsCarrier/Kdl/ASN (if they exist outside public toy dump)  
- P1 gravity hyperparameters for Cogentco  
- Published TM split  
- Closed-form / reference ADMM implementation from authors  
- NCFlow, Figret, Teal baseline runners matching Table 6  
- Complete §5.5 failure-robustness table  
- Author confirmation of Yen vs edge-disjoint paths  

---

## 10. Recommended corrections (propose only — **not implemented**)

1. Pin evaluation ADMM variant explicitly in every experiment script; archive which variant produced each result file.  
2. Treat Teal ADMM as **labeled deviation**; report P1-order results separately with residual diagnostics.  
3. Acquire FIGRET GÉANT 22/72 + TMs before any GÉANT claim.  
4. Toy unit TMs on UsCarrier/Kdl/ASN do not support Table-6 claims.  
5. Document critic `.sum()` aggregation as an assumption; ablate mean vs sum.  
6. Remove or rewrite stale `b4.py` notes.  
7. Contact authors for split, paths, and ADMM numerics if claiming ±1% of 92.39%.  
8. Implement or cite external NCFlow/Figret only if rebuilding Table 6.

---

## 11. Final verdict (12 questions)

1. **Is the complete P1 methodology implemented?**  
   **No.** Core GRL-TE loop on B4 largely is; multi-topo data, several baselines, and a numerically healthy P1 ADMM deployment path are not.

2. **Is TopoFlowNet correctly implemented?**  
   **Mostly yes** — eqs. 5–12 and §5.3 dims match.

3. **Is A2C correctly implemented?**  
   **Mostly yes** — one-step objective matches; clamps / `rsample` / critic reduction are deviations or ambiguities.

4. **Are both ADMM solvers mathematically correct?**  
   - **P1-order:** derived updates match Lagrangian stationarity + stated ReLU; **not** proven identical to authors’ unseen code; **fails practical 2-iter B4 conditioning**.  
   - **Teal-order:** correct as Teal-style code, **incorrect as P1**.

5. **Is the complete optimization and routing pipeline correct?**  
   **Mostly on B4** (paths → bipartite → policy → adjust → metrics; LP OK). ADMM choice and non-B4 data break “complete P1” fidelity.

6. **Are all six topologies correctly reproduced?**  
   **No.** Five match Table 5 structure; **GÉANT does not**.

7. **Are the datasets faithful to P1?**  
   **Only B4 public Teal TMs**, with caveats. Others: **No**.

8. **Is the experimental evaluation protocol faithful to P1?**  
   **Partially** — seeds/episodes/ρ/iters match; split/paths/ADMM/baselines/data do not fully.

9. **Are the published results independently reproducible?**  
   **No** for 92.39% / 89.36%. Our 97.44% is reproducible **as our Teal-ADMM protocol**, not as P1 Table 6.

10. **Is the current B4 implementation reliable enough as our research baseline?**  
    **Conditionally yes for Digital-Twin *engineering* on B4**, if deviations are frozen and labeled. **No as a claim of verified P1 92.39% fidelity.**

11. **What exactly remains incomplete?**  
    See §9.

12. **Can we confidently move to Phase G (Digital Twin)?**  
    **Not if Phase G assumes a verified P1 baseline.**  
    **Yes only as a clearly caveated B4 scaffold** (Yen paths, stated split, chosen ADMM variant, no Table-6 equality claims). 

---

## Overall classification

### PARTIALLY VERIFIED

**Why not MOSTLY VERIFIED:** Table 6 / six-topology reproduction is blocked; GÉANT wrong; non-B4 TMs unfaithful; ADMM story bifurcated; headline GRL-TE number is Teal-ADMM, not P1-ADMM.

**Why not NOT VERIFIED:** B4 TopoFlowNet + one-step A2C + (23)–(28) + LP + Table-5-matched B4 graph + real Teal TMs are evidenced and runnable; five WAN graphs match Table 5 path statistics.

---

*Companion documents in this directory: `P1_COMPONENT_VERIFICATION_MATRIX.md`, `P1_MATHEMATICAL_AUDIT.md`, `P1_EXPERIMENTAL_AUDIT.md`.*
