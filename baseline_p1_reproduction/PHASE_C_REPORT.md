# Phase C Report — Critic Numerical Stability Diagnosis

**Paper:** Lu et al., 2025, DOI 10.1007/s44443-025-00133-z, §§4.3.3 & 5.3  
**Setup:** B4, seed **100**, 3 episodes × 20 train TMs (60 steps)  
**Artifacts:**  
- `results/logs/critic_diagnostics.py`  
- `results/logs/critic_diagnosis_seed100.json`  
- `results/graphs/critic_diagnosis_seed100.png`  

**Constraint honored:** No reward normalization, architecture change, LR change, or new optimizer tricks. ADMM solvers untouched. Production `train.py` unchanged (diagnosis uses a mirrored instrumented loop).

---

## 1. Architecture vs P1 (§5.3)

| Spec | P1 | Implementation | Status |
| --- | --- | --- | --- |
| Critic MLP | 28→256→256→1, ReLU on first two layers | Same | **Verified** |
| Input | \(h_{\mathrm{reshape}}\) (28-d / demand) | Same | **Verified** |
| \(V(s_t)\) | \(\mathrm{Critic}(h_{\mathrm{reshape}};w)\) (31) | Sum of per-demand heads | **Mostly verified** (aggregation unspecified in P1) |
| Target / advantage | \(A=r-V\) (37); \(r=F_{\mathrm{total}}\) (30); no \(\gamma\) | Same; MSE\((V,r)\) ≡ (39) | **Verified** |
| LRs | actor \(10^{-4}\), critic \(10^{-3}\) | Same | **Verified** |
| Grad clip 1.0 | Not in P1 | Present (engineering) | **Documented extra** |

**Target check:** every step uses immediate `reward = feasible.sum()` only. Flags `target_is_immediate_r=true`, `uses_gamma=false` for all 60 steps. No \(V(s')\) bootstrap.

---

## 2. Instrumented quantities

Per step: \(r\), \(V\), MSE, \(|r-V|/|r|\), advantage \(A=r-V\), actor/critic/TopoFlowNet gradient norms **before and after** global `clip_grad_norm_(·, 1.0)`.

---

## 3. Results (seed 100)

### Per-episode means

| Episode | mean \(r\) | mean \(V\) | mean MSE | mean rel.\|r−V\|/\|r\| | corr\((r,V)\) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 25827 | 6924 | \(4.4\times 10^{10}\) | **5.79** | 0.18 |
| 1 | 25485 | 19616 | \(3.4\times 10^{8}\) | **0.60** | 0.23 |
| 2 | 25779 | 27261 | \(5.5\times 10^{8}\) | **0.79** | 0.31 |

### Learning signals

| Signal | Ep0 → Ep2 |
| --- | --- |
| Relative error | 5.79 → **0.79** (decreased) |
| corr\((r,V)\) | 0.18 → **0.31** (improved) |
| mean \(V\) vs mean \(r\) | gap shrinks (6924→27261 vs \(r\sim 2.6\times 10^4\)) |
| Within ep0 (1st→4th quartile) rel err | 11.29 → **0.97** |

### Gradients / numerical health

| Check | Result |
| --- | --- |
| NaN / Inf in \(r,V\), losses, grads | **None** (0 events) |
| Mean ‖g‖ pre-clip (all params) | \(\sim 2.9\times 10^{12}\) |
| Fraction of steps clipped | **100%** |
| Post-clip ‖g‖ | scaled to global norm ≤ 1 |

Clip is always active; pre-clip norms are huge because MSE \(\sim (r-V)^2\) with \(r\sim 2.6\times 10^4\) and early \(|V|\) can be \(10^5\).

---

## 4. Is large MSE scale — or failure to learn?

**Scale-consistency test**

\[
\frac{\mathrm{MSE}}{r^2} \approx \Bigl(\frac{|r-V|}{|r|}\Bigr)^2
\]

Observed mean \(\mathrm{MSE}/r^2\) = **23.40**, mean \((\mathrm{rel\ err})^2\) = **23.40** (match).  
So absolute MSE \(\sim 10^8\)–\(10^{10}\) is **exactly** what relative errors of order 1–10 produce at \(r\sim 2.6\times 10^4\):

\[
(2.6\times 10^4)^2 \approx 6.8\times 10^8.
\]

**Classification:** `LEARNING_PARTIAL_LARGE_MSE_MOSTLY_SCALE`

- Critic **is learning** (rel err ↓, correlation ↑, \(V\) moves toward \(r\) scale).
- Absolute MSE stays large because \(r=F_{\mathrm{total}}\) is large — **scale effect**, not a broken loss.
- Learning is only **partial** after 3 episodes: ep2 mean rel err ≈ 0.79 (V still noisy vs r). That matches short P1 training (3 episodes), not a methodology bug.

**No adaptation applied.** None proposed as mandatory. Optional future adaptations (not implemented): longer training; document clip=1.0 impact; critic output bias init near \(\mathbb{E}[r]\) — each would be labeled **engineering / adaptation**, not P1.

---

## 5. Answers to Phase C checklist

| # | Question | Answer |
| --- | --- | --- |
| 1–4 | Instrument / log / plot | Done (JSON + PNG) |
| 5 | Target = immediate \(r\), no γ/bootstrap | **Yes** |
| 6 | Architecture vs P1 | **Matches** (28→256→256→1) |
| 7 | Large MSE = scale or failure? | **Primarily scale**; critic still learns |
| 8 | NaN/Inf / exploding preds? | No NaN/Inf; pre-clip grads huge but clipped; early ep0 \(V\) wild then recovers |

---

## 6. Updated critic audit status

| Item | Status |
| --- | --- |
| Critic architecture | **Verified** |
| One-step target / no γ | **Verified** |
| Numerical behaviour | **Mostly verified** — learns; large MSE = scale; always-on grad clip is non-P1 engineering |
| Aggregation \(V=\sum_d \mathrm{critic}_d\) | **Mostly verified** (unspecified in P1) |

---

## 7. Stop point

Phase C complete. No code-path changes to training/ADMM. Ready for user review before Phase D (topologies).
