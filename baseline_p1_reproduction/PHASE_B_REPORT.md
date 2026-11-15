# Phase B Report — P1 ADMM (F → z → ξ → λ)

**Paper:** Lu et al., 2025, DOI 10.1007/s44443-025-00133-z, §4.4 eqs. (40)–(52)  
**Date:** Phase B implementation  
**Scope:** ADMM only. Critic (Phase C) not started.

---

## 1. P1 equations and implementation

### Formulation (paper)

| Eq. | Content |
| --- | --- |
| (40)–(41) | Capacity via \(z_{dpe}\); consistency \(F_d(p)\cdot d = z_{dpe}\) |
| (42)–(43) | Slacks \(\xi_{1d},\xi_{3e}\ge 0\) → equalities |
| (45)–(47) | \(G_1,G_3,G_4\) |
| (48) | \(L=-\sum F\cdot d+\lambda^\top G+(\rho/2)\|G\|^2\) |
| (49)–(52) | Order **F → z → ξ → λ** |
| §5.3 | \(\rho=1\); B4 **2** fixed iterations; policy warm start; ReLU for non-negativity |

### Derived closed forms (not printed in P1)

Implemented in `admm_refine_p1` / `_P1ADMMState` from \(\nabla L=0\):

**F (49), per demand**  
\((11^\top + d^2\mathrm{diag}(n))F = \mathrm{rhs}/\rho\),  
\(\mathrm{rhs}_p = d-\lambda_1-\rho(\xi_1-1)-d\sum_{e\in p}\lambda_{4,pe}+\rho d\sum_{e\in p}z_{pe}\), then \(F\leftarrow\mathrm{ReLU}(F)\).  
If \(d=0\): \(F=0\) (\(11^\top\) alone is singular).

**z (50), per edge**  
\((I+11^\top)z=b\), \(b_i=F_id_i+(\lambda_{4i}-\lambda_3)/\rho-\xi_3+c\),  
\(z=b-S\cdot 1\) with \(S=(\mathbf{1}^\top b)/(n_e+1)\), then \(z\leftarrow\mathrm{ReLU}(z)\).

**ξ (51)**  
\(\xi_1=\max(0,1-\sum F-\lambda_1/\rho)\),  
\(\xi_3=\max(0,c-\sum z-\lambda_3/\rho)\).

**λ (52)**  
\(\lambda\leftarrow\lambda+\rho G(F,z,\xi)\).

### Ambiguities explicitly **not** invented

1. Closed forms of (49)–(51) are **absent** from P1 → we derive them; labeled as derived.
2. Exact Euclidean projection of the F/z quadratics onto \(\ge 0\) is unspecified → unconstrained stationarity + ReLU only (as named in §4.4.1).
3. No \(F\in[0,1]\) / simplex projection is stated for ADMM → **not** added.
4. No residual-based stopping → fixed iteration counts only.

---

## 2. Old (Teal) vs corrected (P1) ADMM

| Aspect | `admm_refine_teal` | `admm_refine_p1` |
| --- | --- | --- |
| Order | s → λ → z → x | **F → z → ξ → λ** |
| Variables | flows \(x=F\cdot d\) | fractions \(F\) + \(z,\xi,\lambda\) as in P1 |
| Closed forms | Teal public code | Derived from (48) |
| B4 @ 2 iters flow/demand | **≈ 0.99** | **≈ 3.07–3.21** (unstable) |
| After eqs. (23)–(28) | Feasible | Feasible |

Default dispatch: `admm_refine(..., variant="p1")`. Teal preserved for A/B.

---

## 3. Files modified / created

| File | Role |
| --- | --- |
| `optimization/admm.py` | Teal kept; P1 solver added; docs for closed forms + limitation |
| `optimization/__init__.py` | Export `admm_refine_p1`, `admm_refine_teal` |
| `training/evaluate.py` | `admm_variant` argument (default `"p1"`) |
| `results/logs/admm_p1_tests.py` | Unit tests |
| `results/logs/admm_p1_test_results.json` | Test output |
| `results/logs/admm_p1_vs_teal.py` | Identical-actor comparison |
| `results/logs/admm_p1_vs_teal.json` | Comparison output |
| `PHASE_B_REPORT.md` | This report |

**Unchanged:** TopoFlowNet, reward, bipartite graph, B4 data, constraint adjustment, training loop.

---

## 4. Tests performed

| Test | Result |
| --- | --- |
| Update order F → z → ξ → λ | **PASS** |
| Non-negativity F,z,ξ; finite residuals | **PASS** |
| ξ stationarity on \(\xi>0\) | **PASS** (err = 0) |
| Feasibility after (23)–(28) | **PASS** (0 capacity violations) |
| B4 2-iter numerical stability (flow/d < 1.5) | **EXPECTED FAIL** (flow/d ≈ 3.19) |
| Unexpected failures | **0** |

Suite: `6/7` pass + `1` documented expected fail.

---

## 5. Experimental comparison (identical actor)

Checkpoint: `grl_te_b4_seed100.pt` (already trained under Teal-era deploy; **not** retrained).  
Same deterministic actor fractions → both ADMMs → eq. (28). Test split (8 TMs).

| Method | Mean demand satisfaction |
| --- | ---: |
| Actor only (no ADMM) | 95.20% |
| Actor + **P1 ADMM** + (28) | **97.67%** |
| Actor + **Teal ADMM** + (28) | **96.76%** |
| P1 published B4 | 92.39% (not a tuning target) |

| Pre–(28) diagnostic | P1 | Teal |
| --- | ---: | ---: |
| Mean flow / demand | **3.07** | 0.985 |
| Mean \|G3\| | 58.0 | 4.62 |

**Interpretation:** After hard repair (23)–(28), P1-order ADMM can still look competitive on satisfaction, but the **ADMM state itself is not a well-behaved feasible refinement** under B4’s 2 iterations (ΣF ≫ 1). Teal remains numerically stable. We do **not** claim the P1 solver “beats” Teal as a faithful optimizer—only that both pipelines were compared honestly.

Sweep (uniform warm start): even at 100 P1 iterations, flow/demand stayed ≈ 2.28. More iters alone do not restore Teal-like behaviour with these derived argmins + ReLU.

---

## 6. Root cause of instability (diagnosis)

Focused re-audit (no solver changes): **`ADMM_F_UPDATE_AUDIT.md`** + `results/logs/admm_f_update_audit.json`.

**Verdict:** Not an algebra/sign/`d²` bug. Closed form = independent L-BFGS on \(L_d\); Hessian SPD with \(\kappa\approx 1.5\). ReLU irrelevant (\(F^\star>0\) already).  
**Exact source:** after 1st z/λ, true unconstrained \(\arg\min_F L\) for small \(d\) has \(\sum F\gg 1\) because capacity-scale \(z\) and flow-valued \(\lambda_4\) dominate a Hessian \(\propto d^2\). Scale ratio \(|S_{\lambda_4}|/(\rho n d)\sim 1/d\). Paper does not specify a stable numerical remedy.

**Stopped short of inventing:** simplex projection, adaptive \(\rho\), Teal order inside `admm_refine_p1`, or flow-only rewrites presented as “exact P1.”

---

## 7. Remaining deviations / audit status

| Item | Status |
| --- | --- |
| ADMM update **order** F→z→ξ→λ | **Verified** (implemented + tested) |
| ADMM variables / \(G\) / \(L\) (45)–(48) | **Mostly verified** (structure matches; argmins derived) |
| ADMM numerical behaviour @ B4 2 iters | **Deviates from usable TE practice**; paper-silent on argmins |
| Teal ADMM baseline preserved | **Verified** |
| Critic stability | **Not started** (Phase C) |
| Five topologies / baselines / 89.36% | **Not implemented** |

---

## 8. Recommendation before Phase C

- Keep **both** solvers.
- For production-style B4 numbers matching prior 10-seed archive, deploy with `variant="teal"` until a paper-faithful **and** stable P1 numerical scheme exists (requires author clarification or a clearly labeled adaptation).
- Default code path is `p1` for fidelity experiments; document the 2-iter instability whenever reporting P1-ADMM metrics.
- Exact ADMM reproduction is not claimed; Phase B delivers order-correct, equation-derived ADMM with an honest stability gap.

---

## 9. Scripts

The ADMM unit tests are in `results/logs/admm_p1_tests.py`; the P1-order versus Teal-order comparison is produced by `results/logs/admm_p1_vs_teal.py`.
