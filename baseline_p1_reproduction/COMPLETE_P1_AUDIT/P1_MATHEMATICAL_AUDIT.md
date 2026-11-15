# P1 Mathematical Audit

Independent derivations compared line-by-line to `baseline_p1_reproduction` code.  
Paper equations cited from Lu et al. (2025), `p1.pdf`.

---

## 1. TopoFlowNet

### 1.1 Feature init (eq. 5)

\[
h^0=\begin{bmatrix}h^0_{V_e}\\ h^0_{V_p}\end{bmatrix},\quad
h^0_{V_e}=[c(e_i)],\quad
h^0_{V_p}=[d_{p_j}].
\]

**Code:** `TEProblem.initial_features` concatenates `capacities` and `demands.repeat_interleave(k)`, unsqueezes to \(|V|\times 1\). Default `scale=False`.

**Match:** Yes.

### 1.2 GIN update (eqs. 6–8)

\[
h^l_{e_i}=\mathrm{MLP}_l\Big((1+\epsilon_l)h^{l-1}_{e_i}+\sum_{p_j\in N(e_i)}h^{l-1}_{p_j}\Big)
\]
(and symmetrically for path nodes).  
\(\mathrm{MLP}_l(x)=\mathrm{ReLU}(x W^l_1+b^l_1)W^l_2+b^l_2\) with \(W\in\mathbb{R}^{l\times l}\).

**Code** (`GINLayer.forward`):

```text
aggregated.index_add_(0, dst, h_prev[src])   # sum over neighbors
gin_out = gin_mlp((1+eps)*h_prev + aggregated)
gin_mlp = Linear(l,l) → ReLU → Linear(l,l)
```

Bipartite `edge_index` is duplicated both ways, so the same MLP applies to all nodes — equivalent to applying (6) and (7) with shared weights (paper’s “GINConv module” presentation).

**Match:** Yes for (6)–(8). PyG `GINConv` not required.

### 1.3 Path MLP + skip (eqs. 10–12)

Reshape path features to \(|D|\times(kl)\), MLP with \(W\in\mathbb{R}^{kl\times kl}\), reshape back; concat edge GIN feats with updated path feats; skip-concat \(h^0\) on feature dim → width \(l+1\).

**Code:** matches. After 6 layers, path embedding width = 7.

**Match:** Yes.

### 1.4 Architectural note

Paper phrasing “input of layer \(l\) has width \(l\)” is satisfied: layer index `layer_index=l` uses `dim=l`, then skip produces \(l+1\) for the next layer. **No early width inflation inside GIN** (GIN stays at width \(l\)).

---

## 2. One-step A2C

### 2.1 Policy (eqs. 13–21)

\[
h_{\mathrm{reshape}}\in\mathbb{R}^{|D|\times k(L+1)},\quad
\mu=h W_\mu+b_\mu,\quad
\log\sigma=h W_\sigma+b_\sigma,
\]
\[
a^{\mathrm{raw}}\sim\mathcal{N}(\mu,\sigma^2),\quad
F_{d}(p)=\mathrm{softmax}_p(a^{\mathrm{raw}}_{d,\cdot}),\quad
a_{d,p}=F_d(p)\,d.
\]

**Code:** `encode` → reshape; `Actor`; `Normal`; softmax on last dim; training multiplies by demands outside actor.

**Deviations:**

- `log_std.clamp(-10,10)` — not in paper.
- `raw.clamp(±10)` before softmax — not in paper.
- `rsample()` instead of `sample()` — reparameterized sample; log-prob still score-function. Acceptable but not literal.

### 2.2 Advantage and updates (eqs. 36–39)

\[
V(s)=\mathbb{E}[r\mid s],\quad A=r-V,
\]
\[
\Delta\theta=\alpha\, A\,\nabla_\theta\log\pi(a\mid s),\quad
\Delta w=\beta\, A\,\nabla_w V(s).
\]

**Actor code:** `actor_loss = -(advantage * log_prob)` with `advantage` detached → Adam descent implements (38).

**Critic code:** `critic_loss = MSE(V, r)`.

**Equivalence of MSE to (39):**  
Let \(A=(r-V)_{\mathrm{stopgrad}}\).  
\(\nabla_w \tfrac12(V-r)^2=(V-r)\nabla V=-A\nabla V\).  
Gradient descent on MSE steps \(+A\nabla V\), matching \(\Delta w=\beta A\nabla V\).

**Match:** Yes (MSE is the squared residual form of the same fixed point).

**Aggregation:** Paper (31) writes \(V=\mathrm{Critic}(h_{\mathrm{reshape}})\). Critic net maps each of \(|D|\) rows to a scalar; code sums them. If Critic were meant to see a flattened \(|D|\cdot 28\) vector, architecture would differ from §5.3’s “(28, 256)”. **Summing per-demand values is the only reading consistent with both (28,256) and scalar \(r=F_{\mathrm{total}}\).** Still an assumption.

### 2.3 Reward (eqs. 23–30)

Constraint adjust then \(r=F_{\mathrm{total}}=\sum_{m,p}\hat a_{m,p}\).

**Code:** `adjust_allocation` then `reward = feasible.sum()`.

**Match:** Yes. (Stale `b4.py` note claiming normalization is wrong relative to code.)

---

## 3. ADMM — independent derivation

### 3.1 Problem (eqs. 40–47)

Variables: \(F_d(p)\), \(z_{dpe}\), \(\xi_{1d}\ge 0\), \(\xi_{3e}\ge 0\), multipliers \(\lambda\).

\[
\begin{aligned}
G_{1d}&=\sum_p F_d(p)+\xi_{1d}-1,\\
G_{3e}&=\sum_{d,p:e\in p}z_{dpe}+\xi_{3e}-c(e),\\
G_{4dpe}&=F_d(p)\,d-z_{dpe}.
\end{aligned}
\]

Augmented Lagrangian (48):

\[
L_p=-\sum_{d,p}F_d(p)\,d+\lambda^\top G+\frac{\rho}{2}\|G\|_2^2.
\]

Iteration (49)–(52): \(F\to z\to\xi\to\lambda\).

### 3.2 F-update (49)

Fix \(z,\xi,\lambda\). Terms involving \(F\) (per demand \(d\), drop \(d\) index; paths \(p=1..k\); \(n_p=\#\{e\in p\}\)):

\[
L_F=-\,d\sum_p F_p
+\lambda_1\sum_p F_p
+\sum_{p,e\in p}\lambda_{4,pe}\, (F_p d)
+\frac{\rho}{2}\Big(\sum_p F_p+\xi_1-1\Big)^2
+\frac{\rho}{2}\sum_{p,e\in p}(F_p d-z_{pe})^2
+\mathrm{const}.
\]

Stationarity \(\partial L/\partial F_p=0\):

\[
-d+\lambda_1+\rho\Big(\sum_{p'}F_{p'}+\xi_1-1\Big)
+d\sum_{e\in p}\lambda_{4,pe}
+\rho d^2 n_p F_p
-\rho d\sum_{e\in p}z_{pe}=0.
\]

In matrix form (\(\mathbf{1}\mathbf{1}^\top F\) from the G1 quadratic):

\[
\big(\mathbf{1}\mathbf{1}^\top + d^2\mathrm{diag}(n)\big)F
=\frac{1}{\rho}\Big[
d\mathbf{1}-\lambda_1\mathbf{1}-\rho(\xi_1-1)\mathbf{1}
-d\,\mathrm{sum}_e\lambda_4+\rho d\,\mathrm{sum}_e z
\Big].
\]

Then project \(F\leftarrow\mathrm{ReLU}(F)\) as P1 states for path flows.

**Code `update_F`:** builds the same `rhs`, solves `A F = rhs/ρ` with `A = ones + diag_embed(hops * d²)`, then `.relu()`. Zero-demand rows forced to 0 (necessary: \(d=0\) makes \(A=\mathbf{1}\mathbf{1}^\top\) singular).

**Match to derived stationarity:** Yes.  
**Match to unpublished author code:** Cannot verify.  
**Exact NNQP onto \(F\ge 0\):** Paper only says ReLU; unconstrained+ReLU is the stated mechanism, not Euclidean projection of the QP.

### 3.3 z-update (50)

Per edge \(e\), incidences \(i\in I_e\):

\[
L_z \supset \sum_i\lambda_{4,i}(-z_i)+\lambda_3\sum_i z_i
+\frac{\rho}{2}\Big(\sum_i z_i+\xi_3-c\Big)^2
+\frac{\rho}{2}\sum_i(F_i d_i-z_i)^2.
\]

Stationarity:

\[
(I+\mathbf{1}\mathbf{1}^\top)z = b,\quad
b_i=F_i d_i+\frac{\lambda_{4,i}-\lambda_3}{\rho}-\xi_3+c.
\]

Closed form: \(S=(\mathbf{1}^\top b)/(|I_e|+1)\), \(z=b-S\mathbf{1}\), then ReLU.

**Code `update_z`:** identical (`scatter_mean` of \(b\) union a phantom 0).

**Match:** Yes to derivation.

### 3.4 ξ-update (51)

Separable:

\[
\xi_1=\max\Big(0,\,1-\sum_p F_p-\lambda_1/\rho\Big),\quad
\xi_3=\max\Big(0,\,c-\sum z-\lambda_3/\rho\Big).
\]

**Code `update_xi`:** matches.

### 3.5 Dual update (52)

\[
\lambda\leftarrow\lambda+\rho\, G(F,z,\xi).
\]

**Code `update_lambda`:** matches.

### 3.6 Numerical pathology (B4, \(\rho=1\), 2 iters)

Hessian for \(F\) scales with \(d^2\) on the hop-diagonal while G4 multipliers live in **flow** units. For small \(d\), the G1 coupling \(\mathbf{1}\mathbf{1}^\top\) dominates weakly relative to large \(\lambda_4\) feedback after one dual step → second F-update can yield \(\sum_p F_p\gg 1\).

**Observed:** mean `flow_over_demand ≈ 3.07` after 2 P1 iters (`admm_p1_vs_teal.json`).  
Eqs. (23)–(28) still restore feasibility afterward.

**Conclusion:** Implementation matches the **derived** argmins; it does **not** demonstrate that authors’ 2-iter B4 ADMM behaves this way. Treating post-adjust satisfaction as proof of ADMM correctness is invalid.

### 3.7 Teal ADMM variant

Order **s → λ → z → x** with Teal closed forms (`_TealADMMState`). Constraints written as \(d-\sum x-s_1=0\) etc. (sign convention differs from P1’s \(G_1=\sum F+\xi-1\) but is algebraically related via \(x=F\cdot d\)).

**Verdict:** Valid Teal-style solver; **not** P1 eqs. (49)–(52). Used for archived B4 97.44%.

---

## 4. Constraint adjustment (23)–(28)

Given path flows \(a_{m,p}\):

1. If \(\sum_p a_{m,p}>d_m+\epsilon\), set \(\tilde a=a/\mathrm{ratio}\) with \(\mathrm{ratio}=(\sum a)/d\).  
2. \(u_e=\mathrm{load}_e/c(e)\); \(\hat a_{m,p}=\tilde a_{m,p}/\max_{e\in p}\{u_e,1\}\).

**Code:** identical logic (`overflow` mask; `scatter_reduce` amax for path max util).

**Match:** Yes.

**Subtlety:** Paper applies adjust to ADMM output **and** to training actions. Code does both. Good.

---

## 5. LP baseline (1)–(4)

\[
\max_x \mathbf{1}^\top x
\quad\text{s.t.}\quad
\sum_{p\in P_d}x_{dp}\le d,\quad
Ax\le c,\quad x\ge 0.
\]

**Code:** `linprog` with `A_ub=[demand block; incidence]`, `b_ub=[demands; caps]`.

**Match:** Yes for path-form TE. (Node-arc form would be equivalent under complete path sets; with \(k=4\) both are restricted path LPs — same restriction as GRL-TE.)

---

## 6. Checklist: equation → code

| Eqs | Status |
|-----|--------|
| 1–4 | Match (LP / constraints) |
| 5–12 | Match |
| 13–22 | Match with clamp/`rsample` extras |
| 23–30 | Match |
| 31 | Ambiguous aggregation |
| 32–35 | N/A (standard A2C contrast only) |
| 36–39 | Match (MSE form) |
| 40–52 | Order + derived closed forms match; author identity unknown; 2-iter numerics poor |
| Softmax (19)–(20) | Match |

---

## 7. Proposed mathematical corrections (not implemented)

1. Residual traces accompany every ADMM claim, not only post-adjust satisfaction.  
2. P1-order ADMM at 2 iterations would need a simplex projection, a scaled dual or more iterations (author input); a Teal-order substitute is labelled as such, not as P1.  
3. Ablate critic sum vs mean vs learned pooling.  
4. Ablate removal of raw-action and log-std clamps.
