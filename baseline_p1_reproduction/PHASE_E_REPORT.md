# Phase E Report — P1 Experimental Fidelity & Performance Validation (B4)

**Goal:** Establish whether our B4 pipeline is a trustworthy reproduction baseline before multi-topology work.  
**Companion docs:** `P1_DATA_AUDIT.md`, `P1_METHODOLOGY_COMPLIANCE.md`  
**Artifacts:** `results/logs/phase_e/`, `results/graphs/phase_e/`

Phases C–D preserved (no overwrite of critic / phase_d logs). **No methodology retuning** toward 92.39%.

---

## Verdict: **PARTIALLY VERIFIED**

| Claim | Result |
| --- | --- |
| Same TEAL B4 topology + capacity source as P1 cites | **Yes** (JSON bit-identical to `reference/teal`) |
| Same TM family (TEAL real, scale 1.0) | **Yes** |
| Directly comparable to Table 6 GRL-TE **92.39%** | **No** |
| Our 97.73% / 98.13% as “better than P1” | **Not justified** |

**Reason:** Hyperparameters, graph size, TEAL data source, and satisfaction metric intent match closely; **path set**, **TM split**, and **ADMM numerics** remain non-identical / underspecified. LP agrees with P1 (−0.09 pp); GRL-TE does not (+5.05 pp at 3-ep Teal-ADMM).

---

## 1. P1 experimental setup (extracted)

From §5.1–5.3 and Table 5–6:

- B4: 12 nodes, 38 edges; TMs & capacities from **Teal**; 4 shortest paths / pair.
- Train: **3 episodes**, Adam, actor \(10^{-4}\), critic \(10^{-3}\), seeds 100…1000.
- ADMM: \(\rho=1\), B4 **2** iterations.
- Metric: average **demand satisfaction** (Table 6); B4 GRL-TE **92.39%**, LP **99.83%**, Teal **84.77%**.
- Cited B4 mean demand **27.98k**.
- Split / path algorithm / ADMM closed forms: **not published**.

---

## 2. Implementation audit (evidence)

| Check | Evidence |
| --- | --- |
| Topology | 12/38; SP 2.32; diam 5; **identical** to TEAL `B4.json` |
| Capacities | Uniform 5000 |
| TMs | 36× (12×12) `real_*_1.0`; **measured TEAL**, not gravity |
| Demand mean | All-36: 26878; **test-8: 27981 ≈ 27.98k** |
| Paths | Yen 4-shortest; vs TEAL edge-disjoint: **6/132** identical |
| LP (test) | **99.74%** vs P1 99.83% |
| GRL-TE 3-ep + Teal ADMM | **97.44% ± 1.68%** vs P1 92.39% (**+5.05 pp**) |
| Phase D 20-ep + Teal ADMM | **98.13% ± 0.80%** (longer training; not P1 protocol) |

Machine dump: `results/logs/phase_e/phase_e_audit.json`.  
Plots: `results/graphs/phase_e/b4_vs_p1_table6.png`, `b4_tm_mean_vs_p1.png`.

---

## 3. Metric verification

| Metric | Ours | P1 | Compliance |
| --- | --- | --- | --- |
| Satisfaction | \(\sum \hat a/\sum d\) after (23)–(28) | “Demand satisfaction rate” (formula not printed) | **CLOSE** — same intent |
| Reward | \(F_{\mathrm{total}}\) | Eqs. (29)–(30) | **MATCH** |
| MLU / violations | Computed | Not Table 6 primary | Supporting only |

---

## 4. Why 97–98% ≠ a fair win over 92.39%

Likely non-exclusive causes (evidence-ranked):

1. **Path set (strong):** Yen vs TEAL edge-disjoint differ on 126/132 OD pairs → different feasible region for learning/LP path-form.
2. **ADMM (strong):** Deploy uses Teal-order solver; P1-order derived solver unstable at 2 iters (Phase B).
3. **TM split (medium):** 20/8/8 assumed; P1 silent. Test mean matches 27.98k; train is lighter.
4. **Eval / seed aggregation details (medium):** Unstated in P1 beyond “10 seeds.”
5. **Grad clip / clamps (weak):** Engineering extras.

**LP near-match** ⇒ we are not evaluating a wildly different network; the GRL-TE gap is **method/protocol**, not a wrong topology file.

---

## 5. Corrections

| Action | Done? |
| --- | --- |
| Fabricate/rescale TMs to force 92.39% | **No** |
| Silent path swap to edge-disjoint as “P1” | **No** (would be an adaptation; optional future A/B) |
| Change A2C / reward / LRs | **No** |
| Necessary bugfix in B4 data loading | **None found** |
| Document deviations | **Yes** (this report + compliance/data audits) |
| Preserve prior results | **Yes** |

Data integrity did not require a new run. Table-6-style numbers use **3-episode** training and state the ADMM variant.

---

## 6. Trustworthy B4 baseline statement

> We reproduce GRL-TE’s **TopoFlowNet + one-step A2C + constraint adjustment** on the **public TEAL B4** topology and real traffic matrices cited by Lu et al. (2025), with P1 hyperparameters (3 episodes, seeds 100…1000, \(\rho=1\), 2 ADMM iters). Under Yen 4-shortest paths, a TEAL-style 20/8/8 split, and Teal-derived ADMM, test demand satisfaction is **97.44% ± 1.68%** (20-ep validation: 98.13% ± 0.80%). LP is **99.74%** (P1: 99.83%). These figures are **not** claimed as directly comparable improvements over P1’s **92.39%** because path generation, TM split, and ADMM numerics are not uniquely fixed by the paper.

---

## 7. Overall compliance summary

| Area | Verdict |
| --- | --- |
| Topology / TEAL source data | **Verified** |
| Hyperparameters / A2C structure | **Verified** |
| Metrics (intent) | **Mostly verified** |
| Paths / split / ADMM numerics | **Deviates / ambiguous** |
| Table 6 number match | **Not verified** |
| **Phase E aggregate** | **PARTIALLY VERIFIED** |

Ready for multi-topology expansion only with the above qualifications carried forward.
