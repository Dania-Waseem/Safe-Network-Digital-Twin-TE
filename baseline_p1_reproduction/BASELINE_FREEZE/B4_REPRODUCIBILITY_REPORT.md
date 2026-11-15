# B4 Reproducibility Report

**Protocol:** `B4_GRLTE_FREEZE_v1`  
**Produced by:** `BASELINE_FREEZE/run_b4_baseline_freeze.py --retrain-seed 100`  
**Date:** 2026-11-07  
**Artifact:** `logs/b4_freeze_summary.json`

---

## 1. Setup verified

| Item | Check | Result |
|------|-------|--------|
| Topology B4 | 12 nodes / 38 edges | Pass |
| Capacities | TEAL JSON | Pass (unchanged archive topo) |
| Traffic | 36 TEAL real TMs | Pass |
| Split | 20/8/8 | Pass (`assert` vs frozen config) |
| Paths | Yen-4 min-hop | Pass (cached pickle) |
| Seeds | 100…1000 | Pass |
| Episodes | 3 (train path; ADMM not in train) | Pass |
| LP / ECMP / SP | Same problem + test TMs | Pass |
| Saturation | \(\sum\hat a/\sum d\) post (23)–(28) | Pass |
| Channels | teal / p1 / none **separate** | Pass |

---

## 2. Teal-order channel (frozen)

### 2.1 Re-evaluation of archived checkpoints

All 10 archive checkpoints re-evaluated with **explicit** `admm_refine_teal`:

| Seed | Teal sat | Archive sat | Δ (pp) | Pre-adj flow/demand |
|-----:|---------:|------------:|-------:|--------------------:|
| 100 | 0.967608 | 0.967608 | **0** | 0.985 |
| 200 | 0.967441 | 0.967441 | **0** | 0.980 |
| 300 | 0.934916 | 0.934916 | **0** | 0.966 |
| 400 | 0.970724 | 0.970724 | **0** | 0.988 |
| 500 | 0.996806 | 0.996806 | **0** | 0.999 |
| 600 | 0.978620 | 0.978620 | **0** | 0.991 |
| 700 | 0.964677 | 0.964677 | **0** | 0.986 |
| 800 | 0.986868 | 0.986868 | **0** | 0.995 |
| 900 | 0.988259 | 0.988259 | **0** | 0.995 |
| 1000 | 0.988560 | 0.988560 | **0** | 0.996 |

**Aggregate:** **97.44% ± 1.68%**  
**Archive match:** `reeval_teal_matches_archive_per_seed = true` (max |Δ| = 0).

**Verdict:** The archived **97.44% ± 1.68%** is **independently reproduced** under its original Teal-order configuration.

### 2.2 Fresh retrain (seed 100 only)

Trained into `models/baseline_p1/b4_freeze/grl_te_b4_freeze_seed100.pt` (archive models untouched).

| Channel | Retrain sat | Archive seed-100 |
|---------|-------------|------------------|
| Teal | 96.76% | 96.76% |
| P1 | 97.67% | (diag) |
| No ADMM | 95.20% | 95.20% |

Identical under `set_seed(100)` — confirms train+eval determinism for this protocol on CPU.

Full 10-seed **retrain** was not required for freeze once re-eval matched archive bit-for-bit; seed-100 retrain validates the training path under the Protocol Card.

---

## 3. P1-order channel (diagnostic — not frozen)

| Aggregate | Value |
|-----------|-------|
| Demand satisfaction | **98.57% ± 0.73%** |
| Pre-adjust flow/demand | **≈ 2.86–3.46** (per seed means) |

Post-adjust satisfaction is **higher** than Teal for most seeds, but ADMM itself leaves massively infeasible fractions. Feasibility is restored almost entirely by eqs. (23)–(28). **This channel is not the research baseline.**

These numbers are kept separate from the Teal-channel numbers; 98.57% is not a reproduction of P1.

---

## 4. No-ADMM ablation

**96.51% ± 2.75%** — matches archived `grl_te_no_admm_mean_std` exactly on re-eval.

---

## 5. Baselines (identical inputs)

| Method | Demand satisfaction |
|--------|---------------------|
| LP | **99.74%** (P1 Table 6: 99.83% — near; not exact) |
| ECMP (Yen equal-split) | **98.74%** |
| Shortest path | **97.76%** |

LP/ECMP share the same `TEProblem`, Yen paths, and test TMs as GRL-TE.

---

## 6. What is / is not claimed

| Claim | Allowed? |
|-------|----------|
| Teal-channel 97.44% ± 1.68% is reproducible under Protocol Card | **Yes** |
| Matches P1 published 92.39% | **No** |
| P1-order ADMM is a healthy 2-iter deploy solver | **No** |
| Six-topology / 89.36% | **No** |

---

## 7. Reproducibility verdict

**B4 frozen (Teal) configuration: REPRODUCIBLE.**  
**B4 P1-order diagnostic channel: REPRODUCIBLE as a separate measurement, numerically unstable pre-adjust.**
