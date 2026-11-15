# Mathematical Audit — P1 ADMM F-update Instability

**Scope:** Diagnose why `admm_refine_p1` reaches flow/demand ≈ 3.19× at B4’s 2 iterations.  
**Constraint:** No changes to either ADMM solver; no undocumented stabilization.  
**Artifacts:** `results/logs/admm_f_update_audit.py`, `results/logs/admm_f_update_audit.json`

---

## 1. Independent re-derivation from Lagrangian (48)

Fix \(z,\xi,\lambda\) and one demand \(d>0\) with path fraction vector \(F\in\mathbb{R}^k\).  
Let \(n_p=|\{e\in p\}|\), \(S_z_p=\sum_{e\in p}z_{pe}\), \(S_{\lambda_4,p}=\sum_{e\in p}\lambda_{4,pe}\).

Terms in \(L\) that depend on this demand’s \(F\):

\[
\begin{aligned}
L_d(F)
&= -d\sum_p F_p
+ \lambda_1\Bigl(\sum_p F_p+\xi_1-1\Bigr)
+ \sum_p S_{\lambda_4,p}\,(F_p d)\\
&\quad + \frac{\rho}{2}\Bigl(\sum_p F_p+\xi_1-1\Bigr)^2
+ \frac{\rho}{2}\sum_p\sum_{e\in p}(F_p d-z_{pe})^2.
\end{aligned}
\]

Expand the last sum: \(\sum_{e\in p}(F_p d-z_{pe})^2 = n_p(F_p d)^2-2 F_p d\,S_{z,p}+\mathrm{const}\).

**Gradient**

\[
\frac{\partial L_d}{\partial F_p}
=
-d+\lambda_1+\rho G_1
+ d\,S_{\lambda_4,p}
+ \rho n_p d^2 F_p
- \rho d\,S_{z,p},
\qquad
G_1=\sum_q F_q+\xi_1-1.
\]

**Stationarity** \(\nabla L_d=0\):

\[
\rho\,(11^\top + d^2\mathrm{diag}(n))\,F
=
\underbrace{d-\lambda_1-\rho(\xi_1-1)-d\,S_{\lambda_4}+\rho d\,S_z}_{\mathrm{rhs}}.
\]

**Hessian** (SPD for \(\rho>0\), \(d\ge 0\), \(n_p\ge 1\)):

\[
H = \rho\,11^\top + \rho\,d^2\mathrm{diag}(n).
\]

This is **identical** to the linear system in `admm_refine_p1` (`A F = rhs/\rho` with \(A=11^\top+d^2\mathrm{diag}(n)\)).

### Dual / residual units

| Object | Units |
| --- | --- |
| \(F\) | dimensionless (fraction) |
| \(G_1\) | dimensionless |
| \(G_4=F\cdot d-z\) | **flow** |
| \(\lambda_1\) | dual of \(G_1\) |
| \(\lambda_4\) | dual of \(G_4\) (**flow**) |
| Objective \(-F\cdot d\) | flow |

Signs match standard ADMM on \(\min L\) with \(\lambda\leftarrow\lambda+\rho G\).

---

## 2. Verification against independent numerical optimization

On B4 test TM0, for both small and large demands, **before iter 2**:

| Check | Result |
| --- | --- |
| Closed form vs L-BFGS on same \(L_d\) | \(\|F_{\mathrm{cf}}-F_{\mathrm{LBFGS}}\|_2\sim 10^{-8}\)–\(10^{-16}\) |
| \(L(F_{\mathrm{cf}})=L(F_{\mathrm{LBFGS}})\) | Equal to machine precision |
| \(\nabla L_d(F_{\mathrm{cf}})\) | \(\approx 0\) |
| Hessian eigenvalues | All **> 0** (SPD) |
| \(\kappa(H)\) | ≈ **1.53** (small \(d\)), ≈ **1.00** (large \(d\)) |

**Conclusion:** The implemented F closed form is the correct unconstrained \(\arg\min_F L\). Instability is **not** a wrong \(d^2\) coefficient, sign error, or bad linear solve.

*(Forward finite-difference gradients are noisy for large \(d\) because \(L\sim 10^6\); L-BFGS agreement is the decisive check.)*

---

## 3. Conditioning is not the culprit

At the exploding demand (\(d\approx 1.65\)):

- \(\kappa(H)\approx 1.53\) (well-conditioned)
- \(\lambda_{\min}(H)\approx 10.9 > 0\)
- No singular \(A\) (that only occurs at \(d=0\), already special-cased)

Ill-conditioning / float error **does not** explain \(F_{\mathrm{sum}}\approx 257\).

---

## 4. ReLU after unconstrained solve

P1 §4.4.1: non-negativity via **ReLU projection**.

Theory:

- \(\mathrm{ReLU}(F^\star)\) is the Euclidean projection of the unconstrained minimizer onto \(F\ge 0\).
- It is **not** in general \(\arg\min_{F\ge 0} L_d\), unless \(F^\star\ge 0\) already (KKT / active-set otherwise).

Empirical (pre-iter-2 small \(d\)):

| Quantity | Value |
| --- | --- |
| Unconstrained \(F^\star\) | all components **> 0**, \(\sum F^\star\approx 257.6\) |
| \(\|\mathrm{ReLU}(F^\star)-F^\star\|_2\) | 0 |
| vs numerical nonneg-penalized argmin | \(\ell_2\) diff \(\sim 10^{-6}\) |

**ReLU is not the source of the 3.19× blow-up.** The unconstrained minimizer itself already has \(\sum F\gg 1\). ReLU cannot enforce \(\sum F\le 1\) (P1 never states that projection for ADMM).

---

## 5. Exact source of \(\mathrm{flow}/\mathrm{demand}\approx 3.19\)

### Timeline on B4 (uniform warm start, \(\rho=1\))

| Stage | Small \(d\approx 1.65\) | Large \(d\approx 1488\) |
| --- | ---: | ---: |
| Init / after 1st F | \(\sum F\approx 1.41\) | \(\sum F\approx 1.00\) |
| After z, ξ, λ (enter 2nd F) | \(\sum_e z\) on paths \(\sim 260\)–\(425\); \(S_{\lambda_4}\sim -257\)…\(-422\) | \(z\sim 1600\); \(\|S_{\lambda_4}\|\sim 100\) |
| 2nd F closed form \(\sum F\) | **257.6** | **1.16** |
| Scale ratio \(\|S_{\lambda_4}\|/(\rho n d)\) | **33–51** | **0.02** |

### Causal chain (structural, not a coding bug)

1. **z-update (50)** enforces capacity coupling. With \(\lambda\approx 0\), incidences get \(z\) on the order of capacity shares \(O(c/(n_e+1))\), **not** \(O(d)\). Tiny demands can still receive large \(z_{dpe}\).
2. Then \(G_4=F\cdot d-z\) is largely negative for small \(d\) (path flow \(O(d)\), \(z=O(10^2)\)).
3. **λ-update (52)** accumulates \(\lambda_4\leftarrow\lambda_4+\rho G_4\) in **flow** units.
4. **F-update (49)** is the correct minimizer of \(L_d\). Matching \(G_4\) wants \(F\cdot d\approx z\), i.e. \(F\sim z/d\gg 1\) when \(d\) is small.
5. The \(G_1\) quadratic \(\frac{\rho}{2}(\sum F+\xi_1-1)^2\) with \(\rho=1\) is too weak to keep \(\sum F\approx 1\) against that drive in a **single** F-step.
6. Hessian diagonal from \(G_4\) is \(\rho n d^2=O(d^2)\). Relative dual drive scales as
   \[
   \frac{|S_{\lambda_4}|}{\rho\,n\,d}\sim\frac{1}{d},
   \]
   which **diverges as \(d\to 0\)**. Large demands stay stable (\(\sum F\sim 1.16\)).

Aggregate flow/demand ≈ 3.19 after 2 iters is the network-wide average of this per-demand explosion (many small-\(d\) rows), not a single global scaling bug.

### What it is / is not

| Hypothesis | Verdict |
| --- | --- |
| Wrong derivation / \(d^2\) / signs in code | **Rejected** (matches L-BFGS) |
| Ill-conditioned Hessian | **Rejected** (\(\kappa\sim 1.5\)) |
| ReLU incorrectly applied | **Rejected** (\(F^\star>0\) already; ReLU ≠ simplex) |
| Bad initialization alone | **Partial** — init is fine; damage appears after 1st z/λ |
| Paper’s unspecified numerical solver | **Central** — no published argmin, no \(F\in[0,1]\), fixed \(\rho=1\), only 2 iters |

Teal stays near flow/demand ≈ 0.99 because it optimizes **flows** \(x=F\cdot d\) with path Hessian \(I\)-like \(\mathrm{diag}(n)+11^\top\) **independent of \(1/d\)**, and uses a different block order.

---

## 6. Does the paper provide enough information?

**No — not enough to uniquely determine a stable numerical F-step.**

P1 gives \(L\) and the block order, but:

1. Does **not** publish closed forms of (49)–(51).
2. Names only **ReLU** for non-negativity — not a simplex / \([0,1]\) projection on \(F\).
3. Fixes \(\rho=1\) and B4 **2** iterations with no residual safeguard.
4. Leaves the mixed **fraction / flow** scaling (weak \(d^2\) Hessian vs flow-valued \(\lambda_4\)) unaddressed.

Therefore:

- We can confirm our F-update is the correct unconstrained stationarity condition of (48).
- We **cannot** claim a unique “author-intended” stable discretization from the text alone.
- Inventing simplex projection, adaptive \(\rho\), or Teal order inside `admm_refine_p1` would be an **adaptation**, not a transcription.

---

## 7. Bottom line

**Exact source of instability:**  
After the first z/λ updates, the **true unconstrained minimizer** of the P1 F-subproblem (48)–(49) for **small demands** has \(\sum_p F_d(p)\gg 1\), because capacity-driven \(z\) and flow-valued \(\lambda_4\) dominate a Hessian that shrinks as \(d^2\). ReLU does not fix this. The 3.19× flow/demand ratio is that mathematics under P1’s published \(\rho=1\), 2-iter schedule — **not** an implementation algebra error.

**Solvers:** both `admm_refine_p1` and `admm_refine_teal` left unchanged; prior Phase B tests preserved.
