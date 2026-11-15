# Final Research Evaluation Audit

**Project:** Network Digital Twin + Safe Graph-RL (Phase 3)  
**Audit type:** Read-only inspection  
**Scope:** `results/**` + `baseline_p1_reproduction/` freeze/audit artifacts  
**Constraint honored:** No experiments run; no source/results modified; only this audit directory created.

---

## Executive summary

The project has a **substantial completed experimental core** for its *own* research claims (Safety Gate, Twin improvements, fallback diagnosis, conditional LP recovery) and a **partial, well-documented P1 (GRL-TE) reproduction**.

| Area | Verdict |
| --- | --- |
| P1 reproduction | **PARTIAL** (B4 freeze usable; Table-6 equivalence not established) |
| Twin original vs improved | **COMPLETE** (predictive metrics); decision-level Twin upgrade **PARTIAL/null on AF** |
| Safety Gate A–F (dev topos) | **COMPLETE** |
| Uncertainty incremental value (E≻D) | **COMPLETE as a null finding** on AF; **PARTIAL** for holdout generalization |
| Fallback + conditional LP | **COMPLETE** for the stated offline research questions |
| Unseen-topology generalization | **PARTIAL** (robustness/twin zero-shot exist; improved A–F on GEANT/Kdl missing) |
| Statistical packaging | **PARTIAL–GOOD** (strongest in robustness/final_analysis; uneven in early steps) |
| Thesis readiness | **YES with scoped claims** |

**Strongest demonstrated contribution:** decision-level Safety Gate that removes unsafe RL accepts (paired evidence), with honest negative findings that uncertainty did not change AF decisions and that many rejected cases have no feasible catalog fallback.

**Minimum remaining workload:** thesis synthesis of existing evidence; optionally one holdout A–F on GEANT/Kdl *if* unseen improved-gate claims are required.

---

## 1. What is complete

### 1.1 P1 / GRL-TE baseline (PARTIAL overall; B4 freeze COMPLETE)

**Implemented (evidence: `baseline_p1_reproduction/COMPLETE_P1_AUDIT/`):** TopoFlowNet-style actor, one-step A2C, constraint adjustment, ADMM channels, Yen paths, B4 multi-seed archive, LP/ECMP/shortest baselines.

**Not complete reproduction:** Table-6 equivalence; stable P1-order ADMM at paper budget; NCFlow/Figret; correct Zoo GEANT 22/72 in early P1 multi-topo attempts; proprietary-like TMs for several topos.

**Status labels:**

| Reproduction target | Status |
| --- | --- |
| TopoFlowNet + A2C on B4 | COMPLETE (with documented minor deviations) |
| B4 frozen baseline for Phase G | COMPLETE (`BASELINE_FREEZE/`) |
| ADMM fidelity to unpublished P1 closed forms | PARTIAL (Teal-order used for headline) |
| Table 6 numerical match under equivalent conditions | MISSING / NOT established |
| Full six-topology Table 5/6 campaign | PARTIAL / largely NOT APPLICABLE for Phase-3 claims |
| Documentation of deviations | COMPLETE |

### 1.2 Twin evaluation — COMPLETE for prediction; PARTIAL for gate-decision incremental value

Artifacts: `results/twin/`, `results/digital_twin/improvements/`, `uncertainty_diagnostics/`.

Measured:

- Original Twin trained on B4/UsCarrier/Cogentco; GEANT/Kdl zero-shot scored.
- Improved physics-anchored Twin reduces val util MAE (~7.5e-4 → ~3.1e-5).
- Conformal std scale ~0.955; coverage diagnostics with bootstrap CIs.
- Uncertainty–error correlation measured (Spearman ~0.69 overall on diagnostics).

**Twin measurable value:** **Yes for prediction error / sharper intervals.**  
**Twin measurable value inside AF gating:** **Not differentiated** — B≡D and C≡E on AF safety rates.

### 1.3 Safety Gate A–F — COMPLETE (development topologies)

Artifact: `eval_runs/af_eval_k1_calibrated_devtopos_v1/` (80 cases, 480 rows, 0 failures).

| Variant | Role |
| --- | --- |
| A | Policy only |
| B | Original Twin mean gate |
| C | Original Twin uncertainty gate |
| D | Improved Twin mean gate |
| E | Improved Twin calibrated uncertainty gate (k=1) |
| F | Physics oracle gate |

Paired identical proposals; GEANT/Kdl excluded by authorization. Key measured outcomes:

- A unsafe accept **0.50** → B–F **0.00**
- D ≡ E on all accept/reject metrics (null uncertainty incremental effect)
- E ≡ F on accept/reject
- Applied overload **0.50** for B–F (fallback still overloads on unsafe proposals)

Earlier broader suite: `evaluation/robustness_v1` (840 rows, includes GEANT/Kdl) + `final_analysis` with Wilcoxon tests (E vs A unsafe-accept reduction significant; E vs D not).

### 1.4 Fallback evaluation — COMPLETE for offline research questions

Verified against artifacts:

| Claim | Verified? |
| --- | --- |
| 40 fallback failures (E definition) | Yes |
| 35 forced-link catalog-infeasible | Yes (initial investigation) |
| 5 UsCarrier@0.25 catalog-feasible via LP (MLU 0.785) | Yes (`final_five_case_lp`) |
| Shortest 1.87 / equal ~2.45 / rebalance ~1.35 | Yes |
| LP validation 5/5 feasible, beats baselines | Yes |
| Conditional LP A/B: +5 feasible; overload cases 40→35 | Yes |
| Mean LP latency ~8.8 s in A/B | Yes (8.75 s mean; p95 ~9.48 s) |
| Gate decisions unchanged by LP | Yes |

**Further fallback experiments:** Not essential for the research claim. Production integration is engineering, not missing science.

### 1.5 Topology roles (multi-topology vs unseen)

| Topology | Train | Val | Seen eval | Unseen eval |
| --- | --- | --- | --- | --- |
| B4 | Yes | Yes | Yes | No |
| UsCarrier | Yes | Yes | Yes | No (toy TM caveat) |
| Cogentco | Yes | Yes | Yes | No |
| GEANT | No | No | — | Yes (robustness/twin; **not** in AF improved A–F) |
| Kdl | No | No | — | Yes (same) |
| ASN | Excluded | | | |

Evaluating UsCarrier/Cogentco does **not** establish unseen generalization.

---

## 2. Comparisons already performed

1. Policy-only vs Safety Gate (AF; final_analysis).
2. Mean-only vs uncertainty-aware gate (AF D vs E — null; final_analysis E vs D).
3. Original vs improved Twin predictors (digital_twin/improvements).
4. Original vs improved Twin *inside gate* on AF (B/D, C/E — tied).
5. Improved policy vs frozen P1 / baselines across scales (robustness_v1).
6. Seen vs unseen policy/twin scoring (robustness + twin reports).
7. Existing vs LP fallback (validation + 80-case conditional A/B).
8. Traffic scales 0.25–1.0 throughout major evals.
9. Paired statistical tests on robustness/final_analysis and AF D–E CIs.

---

## 3. Partially complete comparisons

| Item | What exists | Gap |
| --- | --- | --- |
| P1 full-paper fidelity | B4 freeze + audits | Table-6 equivalence, baselines, multi-topo real TMs |
| Uncertainty value | Diagnostics + AF null | No AF holdout where E uniquely helps |
| Improved Twin gate value | Predictive gains | No AF decision discord vs original Twin |
| Unseen generalization | Robustness/twin | Improved A–F stack not run on GEANT/Kdl |
| Overhead justification | Scattered runtimes | No single overhead–benefit table |
| Early Step-6/7 stats | Large GEANT counts in Step-6 | Small-n / infeasible-native matrices limit interpretability |

---

## 4. Experiments missing at the time of the audit

- A paired A–F evaluation on GEANT/Kdl under the frozen improved-gate stack, which is the evidence needed for any claim of improved-gate generalization to unseen topologies.
- A unified overhead–benefit table (twin, gate, fallback, conditional LP).

Another LP algorithm study, production LP wiring and a full P1 Table-6 redo are not needed for the core safety-gate claim. The E2 holdout evaluation and the statistical analysis in `results/phase3_final/` later supplied the first two items.

---

## 5. Statistical analysis audit

| Requirement | Status |
| --- | --- |
| Mean ± std multi-seed | COMPLETE in robustness_v1 |
| Paired tests + CIs | COMPLETE in final_analysis; AF D–E CIs |
| Significance only when justified | MOSTLY GOOD (final_analysis); LP A/B correctly avoids overclaiming |
| Per-topology / per-scale | COMPLETE in major reports |
| Runtime overhead | PARTIAL (present, not unified) |
| Memory | MISSING / sparse |

---

## 6. Research comparison matrix (summary)

Full CSV: `research_comparison_matrix.csv`.

| # | Question | Status |
| ---: | --- | --- |
| 1 | P1 reproduction | PARTIAL |
| 2 | Original vs improved Twin (prediction) | COMPLETE |
| 3 | Policy-only vs Safety Gate | COMPLETE |
| 4 | Mean-only vs uncertainty gate | PARTIAL (complete null on AF; holdout incomplete) |
| 5 | Original vs improved Twin in gate | PARTIAL |
| 6 | Existing vs LP fallback | COMPLETE |
| 7 | Seen vs unseen topologies | PARTIAL |
| 8 | Traffic scales | COMPLETE |
| 9 | Computational overhead | PARTIAL |
| 10 | Statistical reliability | PARTIAL–GOOD |

---

## 7. Status at the time of the audit

Demonstrated at audit time: measured Safety Gate gains, twin prediction improvements, a null incremental effect of uncertainty on the A–F catalogue, fallback infeasibility, and selective LP recovery. Not demonstrated at audit time: a complete P1 reproduction, uncertainty-driven decision gains, and improved-gate A–F generalization to GEANT/Kdl. The claims ledger in `results/phase3_final/RESULTS_SUMMARY.md` (§11) records the final status of each claim after the Phase 3 final experiments.

---

## 10. Audit outputs

- `FINAL_RESEARCH_AUDIT.md` (this file)
- `experiment_inventory.csv`
- `research_comparison_matrix.csv`
- `research_contribution_assessment.md`
- `artifact_verification.json`
