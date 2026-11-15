# Phase D Report — Extended A2C Training & Critic Validation

**Paper:** Lu et al., 2025 (GRL-TE), §§4.3.3 & 5.3  
**Topology:** B4 (existing TEAL TMs; 20/8/8 split; no new data)  
**Seeds:** 100…1000 (10)  
**Episodes:** **20** (validation beyond P1’s published **3**)  
**Artifacts:** `results/logs/phase_d/`, `results/graphs/phase_d/`, `experiments/phase_d_a2c.py`

Phase C artifacts (`PHASE_C_REPORT.md`, `critic_diagnosis_seed100.json`, plot) were **not modified**.

---

## Verdict: **MOSTLY VERIFIED**

Extended multi-seed training shows consistent critic improvement and stable/high routing satisfaction without NaN/Inf or methodology changes. Qualifications: P1 only trains 3 episodes; grad clip 1.0 remains always-on; test-sat gains beyond ep3 are seed-dependent (60% of seeds improve no-ADMM sat).

---

## 1. Protocol (unchanged methodology)

| Item | Setting |
| --- | --- |
| Reward | \(r_t=F_{\mathrm{total}}\) (eq. 30) |
| Advantage | \(A=r-V\); no \(\gamma\); no bootstrap |
| Critic | 28→256→256→1 |
| LRs | actor \(10^{-4}\), critic \(10^{-3}\) |
| Grad clip | 1.0 (engineering; measured) |
| ADMM | **not modified**; test reported with no-ADMM and Teal ADMM |
| Longer horizon | 20 episodes = **validation only**, not a P1 hyperparameter change for Table 6 claims |

---

## 2. Comparison to Phase C (seed 100, first 3 episodes)

Phase D seed 100 reproduces Phase C’s early critic trajectory:

| Episode | Phase C rel err | Phase D rel err |
| ---: | ---: | ---: |
| 1 | 5.79 | 5.79 |
| 2 | 0.60 | 0.60 |
| 3 | 0.79 | 0.79 |

After ep20 (seed 100): train rel err **0.066**, test sat no-ADMM **96.5%**, Teal ADMM **97.3%**.

---

## 3. Multi-seed results (10 seeds × 20 episodes)

### Routing (test demand satisfaction)

| Checkpoint | no ADMM mean±std | Teal ADMM mean±std |
| --- | ---: | ---: |
| After ep 3 (P1 length) | **96.51% ± 2.75%** | **97.44% ± 1.68%** |
| After ep 20 | **97.73% ± 1.02%** | **98.13% ± 0.80%** |

- Fraction of seeds with final no-ADMM sat ≥ ep3: **60%**
- Mean sat rises; **std shrinks** (more stable across seeds)
- Matches prior 10-seed archive at ep3 (same 96.51% / 97.44%) — continuity check passed

### Critic

| Checkpoint | mean rel \|r−V\|/\|r\| | mean corr\((r,V)\) |
| --- | ---: | ---: |
| Ep 3 | 4.19 ± 5.14 | −0.03 ± 0.36 |
| Ep 20 | **0.062 ± 0.026** | **0.29 ± 0.22** |

- **100%** of seeds: final rel err ≤ ep3 rel err
- Absolute MSE remains large early (scale of \(F_{\mathrm{total}}\)); relative error collapses with longer training — confirms Phase C scale diagnosis

### Gradients / stability

| Check | Result |
| --- | --- |
| NaN / Inf | **None** (all seeds) |
| Clipping frequency | Remains high / often 100% per episode (pre-clip ‖g‖ still huge) |
| Policy collapse | Not observed; train/test sat stay high |

---

## 4. Does the actor learn a better routing policy?

**Evidence-based answer:** Yes, on average, with caveats.

- Aggregate test sat (no ADMM) **96.51% → 97.73%**; Teal path **97.44% → 98.13%**.
- Strongest gains on weaker ep3 seeds (e.g. seed 300: 89.6% → 98.0% no-ADMM).
- Some seeds are flat (already near-ceiling at ep3, e.g. 500, 1000) or slightly down (e.g. 800: 98.4% → 96.8%) — longer training is **not uniformly helpful**.
- Do **not** claim this recovers P1’s published 92.39%; our B4 numbers remain higher under the TEAL TM protocol.

---

## 5. Data limitations

- Same public TEAL B4 matrices (36 TMs). Mean demand scale differs slightly from P1’s cited 27.98k.
- Train/val/test cut is TEAL 20/8/8 (not specified in P1).
- No new dataset downloaded.

---

## 6. Files

| Path | Content |
| --- | --- |
| `experiments/phase_d_a2c.py` | Runner |
| `results/logs/phase_d/aggregate.json` | Multi-seed summary + verdict |
| `results/logs/phase_d/seed*.json` | Per-seed episode curves |
| `results/graphs/phase_d/phase_d_seed*.png` | Per-seed plots |
| `results/graphs/phase_d/phase_d_aggregate.png` | Mean±std curves |

---

## 7. Audit status (A2C / critic after Phase D)

| Component | Status |
| --- | --- |
| One-step A2C methodology | **Verified** |
| Critic architecture + immediate-\(r\) target | **Verified** |
| Critic learning under longer training | **Mostly verified** |
| Multi-seed routing stability (B4) | **Mostly verified** |
| Exact match to P1 3-ep / 92.39% protocol | **Not claimed** |

---

## 8. Stop

Phase D is complete.
