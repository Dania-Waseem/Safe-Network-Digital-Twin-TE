# ADMM Variant Comparison — P1-order vs Teal-order

**Scope:** Independent inspection of both solvers in `optimization/admm.py`.  
**Constraint:** Neither solver’s equations were modified during baseline freeze.  
**Evidence:** Phase B report, `ADMM_F_UPDATE_AUDIT.md`, freeze re-eval logs.

---

## 1. Side-by-side

| Aspect | P1-order (`admm_refine_p1`) | Teal-order (`admm_refine_teal`) |
|--------|-----------------------------|-------------------------------|
| Role in freeze | **Diagnostic channel only** | **Frozen reference baseline** |
| Paper grounding | P1 eqs. (48)–(52) order **F→z→ξ→λ** | Teal (Xu et al. 2023) public closed forms |
| Primal variables | Fractions \(F\), aux \(z\), slacks \(\xi_1,\xi_3\) | Flows \(x=F\cdot d\), aux \(z\), slacks \(s_1,s_3\) |
| Update order | **F → z → ξ → λ** | **s → λ → z → x** |
| \(\rho\), B4 iters | 1.0, **2** | 1.0, **2** |
| Warm start | Softmax fractions → \(z=F\cdot d\) | Softmax → \(x=F\cdot d\), \(z=x\) on incidences |
| Non-negativity | ReLU on \(F,z\); ReLU on \(\xi\) | ReLU on \(x,z,s\) |
| Pre-adjust flow/demand (B4, 10 ckpts) | **≈ 2.86–3.46** (mean ~3.1) | **≈ 0.97–1.00** |
| Post-adjust sat (10 seeds) | **98.57% ± 0.73%** | **97.44% ± 1.68%** |
| Stable as 2-iter deploy? | **No** (relies on (23)–(28) to undo overflow) | **Yes** (near-feasible before adjust) |

---

## 2. Mathematical equations

### 2.1 Shared P1 Lagrangian (used to derive P1-order)

\[
L=-\sum F_d(p)\,d+\lambda^\top G+\frac{\rho}{2}\|G\|_2^2
\]

\[
G_1=\sum_p F+\xi_1-1,\quad
G_3=\sum z+\xi_3-c,\quad
G_4=F\cdot d-z
\]

P1-order implements unconstrained \(\arg\min\) of each block + ReLU (see `COMPLETE_P1_AUDIT/P1_MATHEMATICAL_AUDIT.md`). Closed forms are **derived**, not printed in P1.

### 2.2 Teal-order

Operates on path **flows** \(x\) with residual form \(d-\sum x-s_1\) (sign convention differs but related by \(x=F d\)). Block order and Hessians follow public Teal ADMM, **not** P1’s printed sequence.

---

## 3. Which scripts use which solver

| Call site | Variant used | Notes |
|-----------|--------------|-------|
| `admm_refine(..., variant=)` default | **`p1`** | Ambiguous if callers omit kwarg |
| `training/evaluate.py` default | **`p1`** | Same footgun |
| `experiments/b4.py` | **omits kwarg → p1 today** | Archived 10-seed was Teal-era |
| `experiments/phase_d_a2c.py` | **`teal`** explicit | |
| `experiments/phase_f_eval_topology.py` | **`teal`** explicit | |
| `BASELINE_FREEZE/run_b4_baseline_freeze.py` | **both**, explicit | Never uses default |
| `frozen_b4_config.FROZEN_ADMM_VARIANT` | **`teal`** | Reference |

**Training never calls ADMM** (`training/train.py`) — matches P1 (ADMM is deployment-only).

---

## 4. Why P1-order is unstable at 2 iterations

Root cause (from `ADMM_F_UPDATE_AUDIT.md`, confirmed again in freeze logs):

1. After first **z** update, small-demand paths receive \(z=O(c)\) capacity shares, not \(O(d)\).  
2. \(G_4=F d-z\) large negative → \(\lambda_4\) grows in **flow** units.  
3. Second **F** step correctly minimizes \(L\): matching \(G_4\) wants \(F\sim z/d\gg 1\) for small \(d\).  
4. Hessian diagonal \(\propto d^2\) is too weak; \(G_1\) penalty with \(\rho=1\) cannot keep \(\sum F\approx 1\) in one step.  
5. **Not** caused by: wrong algebra (matches L-BFGS), ill-conditioned \(A\) (\(\kappa\sim 1.5\)), or ReLU.

| Hypothesis | Verdict |
|------------|---------|
| Implementation algebra error | **Rejected** |
| Bad init alone | **Partial** — damage after first z/λ |
| Scaling / mixed fraction–flow units | **Primary structural cause** |
| Update ordering | Contributes (P1 order exposes F to large \(\lambda_4\)) |
| Stopping at 2 iters | Amplifies; no residual stop in P1 |
| \(\rho=1\) | Insufficient for small-\(d\) G4 drive |

**No equation change was made** to “fix” this. Stabilizing adaptations (simplex on \(F\), adaptive \(\rho\), Teal order inside `admm_refine_p1`) would be **new methodology**, not P1 transcription.

---

## 5. Recommendation for B4 reference baseline

**Freeze Teal-order ADMM** as the research baseline deploy step.

| Criterion | Teal | P1-order |
|-----------|------|----------|
| Numerically usable at published 2 iters | Yes | No |
| Bit-reproduces archived 10-seed suite | Yes (Δ=0) | No (different channel) |
| Matches P1 printed update order | No | Yes |
| Mathematically derived from P1 \(L\) | No | Yes (closed forms) |
| Suitable as DT comparison anchor | **Yes (labeled)** | Only as ablation |

**Rationale:** A baseline must be **reproducible and numerically coherent**. Teal-order is a **documented deviation** from P1’s printed ADMM order, but it is the only deploy configuration that (a) stays near-feasible before (23)–(28) and (b) matches the verified archive. P1-order remains available as a **diagnostic** channel with mandatory residual logging.

Do **not** describe the frozen baseline as “P1 ADMM.”  
Do **not** average the two channels.
