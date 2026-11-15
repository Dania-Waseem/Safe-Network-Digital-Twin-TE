# P1 Component Verification Matrix

Every verdict is tied to paper evidence and a concrete code location.  
Verdict vocabulary: **Correctly implemented** | **Mostly correct** | **Partially implemented** | **Incorrect** | **Missing** | **Cannot verify due to unavailable information**.

| # | P1 component | Required methodology | Our implementation | Evidence (paper → code) | Verdict |
|---|--------------|----------------------|--------------------|-------------------------|---------|
| 1 | TE objective | Max throughput \(\sum F_d(p)\cdot d\) (eq. 1) | Path LP; RL reward \(F_{\mathrm{total}}\); ADMM Lagrangian uses \(-\sum F\cdot d\) | P1 §4.1 / (1); `baselines.lp_allocate`; `train.py` reward; `admm.py` L | **Correctly implemented** |
| 2 | Demand constraints | \(\sum_p F_d(p)\le 1\) (eq. 2) | Softmax; ADMM G1; adjust (23)–(26) | (2)(20)(42); `a2c.py`, `admm.py`, `constraint_adjustment.py` | **Correctly implemented** |
| 3 | Capacity constraints | \(\sum_{d,p:e\in p} F_d(p)d \le c(e)\) (eq. 3) | LP; ADMM G3; adjust (27)–(28) | (3)(40)(43); same modules | **Correctly implemented** |
| 4 | Non-negativity | \(F\ge 0\) (eq. 4) | Softmax / ReLU in ADMM | (4); actor softmax; ADMM `.relu()` | **Correctly implemented** |
| 5 | Bipartite graph \(G=(V_e\cup V_p,E)\) | Link↔path incidence undirected bipartite | `build_bipartite_graph.py` bidirectional `edge_index` | §4.2.1 Table 2; `build_bipartite_graph.py:92–97` | **Correctly implemented** |
| 6 | \(h^0\) features | Capacities + path demands (eq. 5) | Raw concat, default unscaled | (5); `TEProblem.initial_features` | **Correctly implemented** |
| 7 | GINConv bidirectional | Eqs. (6)–(9), MLP (8) \(W\in\mathbb{R}^{l\times l}\) | Hand GIN + `Linear(l,l)`×2 + ReLU | (6)–(8); `topoflownet.GINLayer` | **Correctly implemented** |
| 8 | Path collaboration MLP | Reshape + MLP (10) | `path_mlp` on \(|D|\times(kl)\) | (10); `topoflownet.py:48–50` | **Correctly implemented** |
| 9 | Skip concat | Eq. (12) → dim \(l+1\) | `cat([concat_nodes, h0], -1)` | (12); `topoflownet.py:52–53` | **Correctly implemented** |
| 10 | \(L=6\), out dim 7 | §5.3 | `NUM_LAYERS=6`, path emb 7 | §5.3; `config.py`, `TopoFlowNet` | **Correctly implemented** |
| 11 | Actor \(\mu,\log\sigma\) | (14)–(17), 28→4 | `Actor` two Linear heads | §5.3; `actor.py` | **Correctly implemented** |
| 12 | Gaussian sample + softmax | (18)–(19) | `Normal`; train: `rsample`; deploy: mean | (18)–(19); `a2c.py:54–64` | **Mostly correct** (`rsample` + clamps) |
| 13 | Deploy deterministic | Softmax(\(\mu\)) | `deterministic=True` | §4.3.1 deploy paragraph; `a2c.py` | **Correctly implemented** |
| 14 | Critic 28→256→256→1 | §5.3 | `Critic` | §5.3; `critic.py` | **Mostly correct** (reduction ambiguous) |
| 15 | \(V=\mathrm{Critic}(h_{\mathrm{reshape}})\) | Eq. (31) | Per-demand critic then `.sum()` | (31); `a2c.py:38–45` | **Cannot verify** aggregation; **Mostly correct** if sum intended |
| 16 | One-step value / advantage | (36)–(37), no \(\gamma\) | \(A=r-V\); no discount | §4.3.3; `train.py:60–63` | **Correctly implemented** |
| 17 | Policy update (38) | \(\Delta\theta \propto A\nabla\log\pi\) | `-(advantage*log_prob)` Adam | (38); `train.py:67–68` | **Correctly implemented** |
| 18 | Critic update (39) | \(\Delta w \propto A\nabla V\) | MSE(\(V,r\)) ≡ same GD direction | (39); `train.py:71` + math audit | **Mostly correct** |
| 19 | Reward \(r=F_{\mathrm{total}}\) | (29)–(30) | `feasible.sum()` after adjust | (29)–(30); `train.py:58–60` | **Correctly implemented** |
| 20 | Shared TopoFlowNet | End-to-end §4.3.1 | One backbone; two LR groups | §4.3.1; `train.py:32–39` | **Correctly implemented** |
| 21 | Training episodes / LRs / Adam | 3 ep; \(10^{-4}/10^{-3}\) | `config.py` / `train.py` | §5.3 | **Correctly implemented** |
| 22 | Seeds 100…1000 | 10 repeats | `SEEDS`; `run_b4_seeds.py` | §5.3 | **Correctly implemented** (B4 archived) |
| 23 | Constraint adjust demand | (23)–(26) | Ratio scale-down | (23)–(25); `adjust_allocation` | **Correctly implemented** |
| 24 | Constraint adjust capacity | (27)–(28) | Divide by \(\max_{e\in p}\{u_e,1\}\) | (28); `constraint_adjustment.py:45–48` | **Correctly implemented** |
| 25 | ADMM auxiliaries \(z,\xi,\lambda\) | (40)–(47) | `_P1ADMMState` | Table 4; `admm.py` | **Correctly implemented** (variable defs) |
| 26 | Augmented Lagrangian | Eq. (48) | Used to derive updates (not evaluated as scalar each step) | (48); docstring + F/z/ξ solves | **Mostly correct** |
| 27 | ADMM order F→z→ξ→λ | (49)–(52) | `admm_refine_p1` / `.step()` | (49)–(52); `admm.py:225–230` | **Correctly implemented** (order) |
| 28 | ADMM closed forms | Unpublished | Derived stationarity + ReLU | Paper only prints argmin; `update_F/z/xi` | **Cannot verify** vs authors; **Mostly correct** vs Lagrangian |
| 29 | ADMM \(\rho=1\), B4 2 iters | §5.3 | `RHO`, `ADMM_ITERS['B4']=2` | §5.3; `config.py` | **Correctly implemented** (hyperparams) |
| 30 | ADMM numerical behavior @2 iters | Should refine toward feasibility | \(\sum x/\sum d\sim 3\times\) before adjust | `admm_p1_vs_teal.json` | **Incorrect** as practical P1 deployment under our closed forms |
| 31 | Teal-order ADMM | Not in P1 | `admm_refine_teal` | Teal paper / `admm.py:315+` | **Incorrect** as P1 component (**intentional A/B**) |
| 32 | Warm start from policy | §4.4.5 | Init F from softmax fractions | §4.4.5; `init_from_fractions` | **Correctly implemented** |
| 33 | ADMM deploy-only | Not in training | `train.py` no ADMM; `evaluate.py` ADMM | §4 narrative | **Correctly implemented** |
| 34 | Candidate paths | “4 shortest paths” §5.1 | Yen k-shortest min-hop, pad | §5.1; `generate_paths.py` | **Mostly correct**; vs Teal edge-disjoint **Cannot verify** which P1 used |
| 35 | Path padding | Unstated | Repeat first path to \(k=4\) | `generate_paths._pad_paths` | **Cannot verify** |
| 36 | B4 topology Table 5 | 12/38, avg 2.3, diam 5 | 12/38, 2.318, 5 | Table 5; TEAL `B4.json` | **Correctly implemented** |
| 37 | UsCarrier Table 5 | 158/378, 12.1, 35 | Match | Table 5; TEAL JSON | **Correctly implemented** (graph) |
| 38 | Cogentco Table 5 | 197/486, 10.51, 28 | Match | Zoo GML | **Correctly implemented** (graph) |
| 39 | Kdl Table 5 | 754/1790, 22.7, 58 | Match | TEAL | **Correctly implemented** (graph) |
| 40 | ASN Table 5 | 1739/8558, 3.2, 8 | Match | TEAL ASN2k | **Correctly implemented** (graph) |
| 41 | GÉANT Table 5 | 22/72, 2.53, 5 | Zoo 40/122, 3.528, 8 | Table 5 vs `GEANT.json` | **Incorrect** |
| 42 | B4 traffic | Teal TMs | 36 real TEAL pickles | §5.1; `data/traffic_matrices/B4` | **Mostly correct** (public Teal) |
| 43 | UsCarrier/Kdl/ASN traffic | Teal TMs | Toy unit-demand | §5.1 vs Phase F `setup.json` | **Incorrect** for Table 6 |
| 44 | Cogentco traffic | Gravity model | Local gravity | §5.1; `gravity.py` | **Partially implemented** |
| 45 | GÉANT traffic | Figret | Gravity on wrong topo | §5.1 | **Incorrect** |
| 46 | TM split | Unpublished | TEAL 20/8/8 | `config.B4_SPLITS` | **Cannot verify** |
| 47 | LP baseline | Global LP §5.2 | SciPy HiGHS path LP | §5.2; `baselines.lp_allocate` | **Mostly correct** (path formulation OK) |
| 48 | ECMP baseline | Not detailed in P1 | Equal \(1/k\) on candidate paths | `baselines.ecmp_allocate` | **Partially implemented** / **Cannot verify** vs P1’s ECMP if any |
| 49 | NCFlow baseline | Table 6 | Absent | §5.2 | **Missing** |
| 50 | Teal baseline (full MARL) | Table 6 | Only ADMM subroutine borrowed | §5.2 | **Missing** (full method) |
| 51 | Figret baseline | Table 6 | Absent | §5.2 | **Missing** |
| 52 | Demand satisfaction metric | Allocated / demand | `evaluate_allocation` | §5.4; `constraint_adjustment.py:59–61` | **Correctly implemented** |
| 53 | Table 6 multi-topo protocol | 6 WANs, reported means | B4 only faithful; others blocked | Table 6; Phase F logs | **Partially implemented** |
| 54 | Link-failure §5.5 | Robustness eval | `link_failures.py` scaffold | §5.5 | **Partially implemented** |
| 55 | Entropy regularization | Not specified | Not used | — | **Correctly implemented** (N/A) |
| 56 | Random seed control | Seeds listed | `set_seed` in experiments | §5.3; `b4.py:43–46` | **Mostly correct** (CUDA determinism not forced) |
| 57 | Checkpointing | Implied | `grl_te_b4_seed*.pt` | `b4.py:154–155` | **Correctly implemented** |
| 58 | Eval script independence | Deploy path | `evaluate.py` | — | **Mostly correct** (default ADMM variant trap) |
| 59 | Silent methodology fallback | Must not swap paths/ADMM silently | Yen default OK; ADMM default flipped to p1 after Phase B | `config.EDGE_DISJOINT=False`; `admm_refine` default `"p1"` | **Partially implemented** risk |
| 60 | Digital Twin (Phase G) | Out of P1 scope | Not started | User plan | **Missing** (by design) |

## Summary counts

| Verdict | Count (approx.) |
|---------|-----------------|
| Correctly implemented | 28 |
| Mostly correct | 10 |
| Partially implemented | 6 |
| Incorrect | 6 |
| Missing | 5 |
| Cannot verify | 5 |

Interpretation: **algorithmic core on B4 is largely present**; **experimental fidelity to P1 Table 6 is not**.
