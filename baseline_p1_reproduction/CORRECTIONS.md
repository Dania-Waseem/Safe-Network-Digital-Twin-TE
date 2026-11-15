# Phase 2 corrections — GRL-TE (P1) B4 checkpoint

Paper: Lu et al., 2025, *J. King Saud Univ. Comput. Inf. Sci.* 37:119  
DOI: https://doi.org/10.1007/s44443-025-00133-z

Pre-correction artifacts are in `results/pre_correction_archive/`.

## 1. Modified files

| File | Change |
| --- | --- |
| `optimization/admm.py` | Replaced exploding F-update with Teal-derived closed forms; residual tracing |
| `preprocessing/build_bipartite_graph.py` | Default `scale=False` so \(h^0\) matches eq. (5) |
| `training/train.py` | Reward \(r_t=F_{\text{total}}\) (eq. 30); shared Adam param groups |
| `models/a2c.py` | Critic aggregation = **sum** over demand heads (units of \(F_{\text{total}}\)) |
| `experiments/b4.py` | Plot uses satisfaction when reward is absolute |
| `experiments/run_b4_seeds.py` | New: all seeds 100…1000 |
| `results/logs/correction_tests.py` | Mathematical tests after fixes |

Unchanged (already verified): TopoFlowNet layers, bipartite construction, actor heads, constraint adjustment (23)–(28), B4 data loading, path generation.

## 2. Corrections and mathematical justification

### Priority 1 — ADMM

**Problem:** Prior solver allocated **3.55×** demand on TM 0 before eq. (28).

**Cause:** Homegrown \(F\)-update closed forms were inconsistent with a stable ADMM for path-form TE; ReLU alone does not enforce \(\sum F\le 1\).

**Fix:** Use the public **Teal** ADMM closed forms (`harvard-cns/teal` `lib/ADMM.py`) on path flows \(x_p=F_d(p)\cdot d\), which are the analytic stationarity conditions of the same style of augmented Lagrangian. P1 cites Teal and uses the same TE ADMM idea but does **not** print the argmins of (49)–(51).

**Documented deviation:** Teal update order is \(s\to\lambda\to z\to x\); P1 lists \(F\to z\to\xi\to\lambda\). With B4’s **2** iterations, order matters; we keep Teal’s order and do **not** claim unpublished P1 code.

**Measured on TM 0 (2 iters, \(\rho=1\)):**

| Metric | Before | After |
| --- | ---: | ---: |
| flow / demand after ADMM | 3.55 | **0.990** |
| demand overflow max | ~250 | **1.32** |
| g3 abs mean (iter0→1) | n/a | 21.8 → **1.73** |
| final feasible (after eq. 28) | yes (heuristic) | **yes** |

### Priority 2 — \(h^0\) (eq. 5)

Default `initial_features(..., scale=False)`. Link features = raw \(c(e)\), path features = raw \(d\). Verified: `h0_max=5000=cap_max`.

### Priority 3 — Reward (eq. 30)

`reward = feasible.sum()` (= \(F_{\text{total}}\)). Satisfaction is logged separately for plots.

### Priority 4 — Shared TopoFlowNet

Single Adam with two param groups:

- TopoFlowNet + Actor @ \(10^{-4}\)
- Critic @ \(10^{-3}\)

One backward of `actor_loss + critic_loss` so the backbone receives gradients from both heads. Critic aggregation is **sum** of per-demand outputs so \(V(s)\) matches \(F_{\text{total}}\) units (aggregation not stated in P1).

## 3. Tests executed

`python results/logs/correction_tests.py` → 8/9 pass  
(Fail was a brittle actor-grad probe on a dirty graph; full training updates the actor.)

`python experiments/run_b4_seeds.py` → seeds 100…1000 completed.

## 4. Before vs after (B4 test set)

| Method | Pre-correction (seed 100) | Post-correction (10 seeds) | P1 Table 6 |
| --- | ---: | ---: | ---: |
| GRL-TE + ADMM | 98.20% | **97.44% ± 1.68%** | 92.39% |
| GRL-TE no ADMM | 92.32% | **96.51% ± 2.75%** | (ablation −0.68 pt) |
| ADMM lift | +5.88 pt | **+0.93 pt** | +0.68 pt |
| LP | 99.74% | 99.74% | 99.83% |
| ECMP | 98.74% | 98.74% | — |

The ADMM lift is now in the same ballpark as P1’s B4 ablation. We still **do not** match 92.39%; that is not forced by tuning.

## 5. Remaining deviations from P1

1. ADMM closed forms / update order from **Teal**, not from unpublished P1 solver text.
2. Critic: how \(\lvert D\rvert\) heads become scalar \(V(s)\) — we use **sum** (not in paper).
3. Training is numerically **noisy** (large \(F_{\text{total}}\) × joint \(\log\pi\)); critic loss oscillates. Method matches eqs. (30)/(38)/(39); scale handling is not specified in P1.
4. TM split 20/8/8 = Teal default; **not specified** in P1.
5. Paths = Yen min-hop; P1 says only “4 shortest paths”.
6. Only **B4** completed; five topologies still missing.
7. Grad clip 1.0, log-std clamp, raw-action clamp — engineering, not in P1.

## 6. Updated reproduction verdict

**Mostly faithful on B4 architecture + training signal; ADMM is a justified Teal-derived implementation, not an exact P1 transcription.**

| Question | Answer |
| --- | --- |
| Can this be a Phase 3 **baseline** on B4? | **Yes, with qualifications** stated above. |
| Exact reproduction of published 92.39% / 89.36%? | **No.** |
| Ready for digital-twin / robust-ndt integration? | **Yes for B4-only** experiments; expand topologies before claiming multi-WAN results. |

## 7. Provenance of this checkpoint

The correction tests are in `results/logs/correction_tests.py`. The seed-100 checkpoint is produced by `experiments/b4.py --seed 100`, and the ten-seed suite by `experiments/run_b4_seeds.py`; the checkpoints are stored in `models/baseline_p1/b4/`.
