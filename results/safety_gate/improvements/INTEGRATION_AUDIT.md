# Safety Gate × Improved Digital Twin — Integration Audit

**Mode:** inspection only. No code, models, experiments, or prior artifacts were modified.

**Date context:** Phase 3 continuation after Digital Twin improvements under `results/digital_twin/improvements/`.

---

## 1. Files inspected

### Twin improvement reports
- `results/digital_twin/improvements/BASELINE_REPORT.md`
- `results/digital_twin/improvements/DIAGNOSTIC_REPORT.md`
- `results/digital_twin/improvements/DIGITAL_TWIN_IMPROVEMENT_REPORT.md`
- `results/digital_twin/improvements/tables/conformal.json`
- `results/digital_twin/improvements/config.json` (referenced via reports / prior pipeline)

### Original Safety Gate
- `safety/gate.py` — `GateConfig`, `inspect_prediction`, `select_fallback`
- `safety/gate_config.json` — production defaults (`k=2`, `utilisation_limit=1`, `max_relative_std=0.5`)
- `safety/evaluate.py` — `load_config`, `load_trained_ensemble`, `predict_action`

### Controllers / fallbacks
- `framework/general_controller.py` — RL + twin + `inspect_prediction` + least-violating fallback among last/shortest/equal
- `framework/pipeline.py` — frozen-P1 `Framework` using `select_fallback` (last → shortest → last_resort)
- `framework/fallback_modes.py` — `classify_candidates` (feasible then least-violating)

### Ablation / evaluation pipelines
- `framework/ablation_study.py` — offline A–E rules; oracle / mean-only / full uncertainty gates
- `framework/robustness_eval.py` — multi-seed catalog, paired matrices (referenced by ablation)
- `results/evaluation/final_analysis/ABLATION_REPORT.md` (prior A–E outcomes; not modified)

### Improved Twin inference & calibration
- `twin/improvements/infer.py` — `load_ensemble_checkpoint`, `predict_loads` (+ conformal scale)
- `twin/improvements/train_improved.py` — `fit_conformal_scale`, `attach_conformal`
- `twin/train.py` — `_sample`, `_predict_numpy` (shared feature/label pipeline)
- `twin/model.py` — `EnsembleTwin.predict` → mean, variance, member stack
- `twin/physics.py` — `link_load` (capacity-feasibility ground truth)

### Checkpoints (paths only; not overwritten)
- Original: `models/twin/ensemble.pt`
- Improved: `models/twin_improved/ensemble_physics_anchored.pt`
- Improved+calibrated: `models/twin_improved/ensemble_physics_anchored_conformal.pt`

---

## 2. Findings — answers to audit questions

### How does the original gate make its decisions?

`inspect_prediction(mean_load, std_load, capacity, config)` is **array-only and topology-agnostic**:

1. Validate shapes/finiteness; reject with `invalid_prediction` if bad.
2. Form an upper confidence load:  
   `upper(e) = mean(e) + k * max(std(e), 0)`.
3. Convert to utilisation: `util_mean = mean/cap`, `util_upper = upper/cap`, `rel_std = std/cap`.
4. Reject with `excessive_uncertainty` if  
   `max(rel_std) > max_relative_std` (default **0.5**), when that threshold is enabled.
5. Reject with `upper_bound_exceeds_capacity` if any  
   `util_upper > utilisation_limit` (default **1.0**).
6. Accept **only if** there are no reasons.

Default production config (`gate_config.json`): `k=2.0`, `utilisation_limit=1.0`, `max_relative_std=0.5`.

**Important:** The gate never sees the true simulator load. Safety metrics such as unsafe accept / false reject are computed **after the fact** by comparing the accept/reject decision to physics (`load > capacity`).

### How are predicted means and uncertainties represented?

| Quantity | Representation | Units |
| --- | --- | --- |
| Twin mean | `mean_load` length `n_links` | **Absolute** link load (after × `topo.denominator`) |
| Twin uncertainty | `std_load` from √ensemble variance (same denormalisation) | **Absolute** load std |
| Gate utilisation | `mean/capacity`, `upper/capacity` | Dimensionless |
| Relative uncertainty filter | `std/capacity` | Dimensionless |

Training/eval of the twin optimises normalised load (`load/denom`), then `_predict_numpy` multiplies mean/std back by `denominator` before the gate.

Ensemble disagreement is the sole uncertainty source in the original twin. The improved conformal checkpoint stores a scalar `conformal_std_scale` (~0.955) that **multiplies** `std_load` at inference when using `predict_loads(..., conformal_std_scale=...)`.

### Does the improved Twin expose the same interface expected by the original gate?

**Yes, at the array and model-API level; not automatically at the loader level.**

Compatible:
- Same `EnsembleTwin` class and `state_dict` layout.
- Same 3 path features via `twin.train._sample` / `_predict_numpy`.
- Gate still consumes `(mean_load, std_load, capacity)` absolute arrays — identical contract.
- `twin.improvements.infer.predict_loads` returns `(true_load, mean_load, std_load, twin_ms)`, matching the shape of `safety.evaluate.predict_action`’s return (true, mean, std, ms).

Not automatic:
- `safety.evaluate.load_trained_ensemble` **hardcodes** `models/twin/ensemble.pt`.
- `safety.evaluate.predict_action` does **not** read or apply `conformal_std_scale` from a checkpoint.
- Therefore plugging the conformal twin into the **unmodified** original evaluate path would load the old weights and/or omit calibration unless a **new** caller uses `infer.load_ensemble_checkpoint` + `predict_loads`.

### Is calibrated uncertainty applied correctly during inference?

**Correctly implemented in the improvement package; not wired into the original Safety Gate path.**

- Fit: `fit_conformal_scale` grid-searches a scale on **B4/UsCarrier/Cogentco validation only** to hit ≥0.90 empirical 2σ coverage while minimising utilisation interval width (`tables/conformal.json`: scale ≈ **0.9545**, val coverage still 1.0 at that point).
- Persist: `attach_conformal` writes `conformal_std_scale` into the conformal checkpoint.
- Apply: `infer.predict_loads` does `std_load *= conformal_std_scale`.

Risks / nuances:
- The checkpoint’s scale is passed explicitly (or read from the payload); without it the raw ensemble std is used.
- Calibration used a **budgeted** val subset (full B4 + 1 matrix each large topology), not every val matrix.
- Coverage remained 1.0 after mild shrink; intervals are sharper but still conservative on reported val aggregates.
- GEANT/Kdl were not used for calibration (correct for the research protocol).

### What does the independent capacity oracle evaluate?

In `framework/ablation_study.apply_ablation` config **C** (`C_rl_oracle_gate`):

- Accept iff physics true load of the **proposed actor action** has **zero** capacity violations:  
  `count(load > capacity + 1e-6) == 0`.
- No twin, no uncertainty.
- On reject: fallback via `classify_candidates` on shortest vs equal **true** loads.

This is the ground-truth safety referee used for unsafe-accept / false-reject labelling everywhere (accepted ∧ violations>0 ⇒ unsafe accept; rejected ∧ violations==0 ⇒ false reject).

Related feasibility labels in robustness eval: `feasible_any_baseline` = shortest **or** equal has zero violations (scenario split), independent of the actor.

### How is fallback routing selected?

There are **two** fallback policies in the codebase:

| Path | Mechanism | Order / rule |
| --- | --- | --- |
| Original Step-7 `Framework` + `select_fallback` | Twin-gated candidates | Proposed → last accepted (if its twin verdict accepted) → shortest (if twin accepts) → shortest last resort |
| `GeneralController` + ablation A–E | Physics `classify_candidates` | Prefer feasible (0 violations); else least violations then lowest MLU among **shortest** and **equal** (controller also considers last accepted) |

Ablation offline rules intentionally **omit last-accepted carry** so cases stay independently paired.

The audit's conclusion for the gate study: **one** documented fallback (ablation-style least-violating shortest/equal, for clean pairing), not mixed with `select_fallback` when the uncertainty benefit is compared.

### Which existing components can be reused without modification?

- `safety/gate.py` — keep as the baseline uncertainty/mean gate implementation; call it from a **new** module rather than editing it.
- `safety/gate_config.json` — baseline hyper-parameters.
- `framework/fallback_modes.classify_candidates` — oracle-side fallback and least-violating selection.
- `framework/ablation_study.apply_ablation` pattern — offline paired rules from one propose+twin forward.
- `framework/robustness_eval` catalog / seeds / scales — identical traffic matrices.
- `twin.train._sample` / `_predict_numpy` — feature compatibility.
- `twin.improvements.infer` — load improved checkpoint + apply conformal scale.
- Physics `link_load` — capacity oracle.
- RL checkpoints (`recommended_stable`, `best_checkpoint`) — load-only.

### What integration incompatibilities or risks exist?

1. **Loader lock-in:** `load_trained_ensemble` always loads the original twin path.
2. **Conformal is opt-in:** Original `predict_action` never scales std; calibrated uncertainty is easy to “lose.”
3. **Dual fallback semantics:** Framework vs GeneralController/ablation can change applied MLU even when accept/reject matches — confound for “gate quality” claims if applied MLU is the only metric.
4. **`max_relative_std=0.5`:** With utilisation-scale std already ≪ 0.5 for the improved twin, this filter may rarely fire; uncertainty benefit will mostly come from the `mean + k·std` bound, not the relative-std veto. Do not assume the veto is active without measuring reason codes.
5. **Near-physics twin:** Absolute GEANT twin MAE is still large in raw units; gate decisions use utilisation. Evaluate safety in util/oracle space, not absolute MAE.
6. **Ablation E ≈ D historically:** Prior final analysis found little E-vs-D difference with the **original** twin. At the time of the audit this had not been re-tested with the improved twin (the E2 and E4 experiments later did).
7. **Over-rejection risk:** Raising `k` or using wider std can look “safer” (lower unsafe accept) while increasing false rejects — success is judged on the **unsafe vs false-reject trade-off**, not reject rate alone.

---

## 3. Compatibility issues (summary)

| Issue | Severity | Mitigation for next stage |
| --- | --- | --- |
| Original evaluate loader ignores improved twin | High | New experiment module loads via `infer.load_ensemble_checkpoint` |
| Conformal scale not applied by original `predict_action` | High | The improved runner uses `predict_loads` with payload[`conformal_std_scale`] |
| Two fallback implementations | Medium | Fix one fallback policy in the new eval harness; log which |
| Editing `safety/gate.py` would violate preservation | Process | Implement improved gate in a **separate** package/path |
| Threshold tuning on GEANT/Kdl | Protocol | Tune only on B4/UsCarrier/Cogentco val; hold out GEANT/Kdl |

**Conclusion:** The improved Twin **can** integrate without changing original implementations, provided a **parallel** gate/eval stack consumes `(mean, calibrated_std)` arrays and does not alter `safety/gate.py`, original twin weights, or prior result trees.

---

## 4. Proposed implementation design (next stage — not started)

### Goals
Measure whether **calibrated** improved-twin uncertainty adds safety value over mean-only filtering, with oracle-validated metrics, under identical matrices.

### Separate package (originals unchanged)
Suggested layout:

```
safety/improvements/                 # or framework/gate_improvements/
  __init__.py
  twin_backend.py      # wrap original vs improved(+conformal) predict
  gate_variants.py     # mean-only, UCB util, optional soft risk score
  oracle.py            # thin wrappers around physics violations
  run_eval.py          # paired A–F configs, seeds, logging
  stats.py             # paired tests, risk-coverage curves
```

Outputs only under: `results/safety_gate/improvements/`  
(reports, raw jsonl, configs, plots — no overwrites of prior trees).

### Decision variants to implement/compare

| ID | Twin | Gate rule |
| --- | --- | --- |
| A | none | Always apply RL action |
| B | **original** | Mean-only: `k=0`, `max_relative_std=None` via `inspect_prediction` |
| C | **original** | Full original config (`k=2`, `max_relative_std=0.5`) |
| D | **improved** (+ optional raw std) | Mean-only |
| E | **improved conformal** | Calibrated UCB: `upper = mean + k_cal * std_cal`; utilise `inspect_prediction` or a thin wrapper that logs util headroom |
| F | none | Physics oracle (violations==0) |

Primary actor: `recommended_stable.pt`. Secondary (if compute allows): `best_checkpoint.pt`.

**Principled E rule (proposed):**  
Accept iff ∀ links  
`mean(e) + k * std_cal(e) ≤ utilisation_limit * capacity(e)`,  
with `std_cal = conformal_std_scale * ensemble_std`, without rejecting solely for large `std` when mean headroom is large (`max_relative_std` disabled or relaxed for the improved gate), justified by validation risk-coverage curves on seen topologies only.

### Validation protocol
- Matrices: reuse `robustness_v1` catalog (same seeds/scales) for pairing.
- Dev/tune: B4, UsCarrier, Cogentco val only (e.g. choose `k` on a val risk-coverage curve).
- Final unseen: GEANT, Kdl test only — no threshold selection.
- Oracle labels every accept/reject.
- Stats: paired Wilcoxon/t-tests on unsafe accept, false reject, applied MLU; report CI; no significance claims without tests.

### Metric denominators
- **Unsafe accept rate:** `# {accepted ∧ physics_violations(actor)>0} / #cases` (or `/ #accepted` as secondary, labelled as such).
- **False reject rate:** `# {rejected ∧ physics_violations(actor)==0} / #cases` (and `/ #feasible_actor` secondary).
- **Acceptance / fallback / no-feasible-fallback:** per case flags as in ablation.
- Coverage / width: link-level or case-aggregated; state which.

---

## 5. Planned file layout of the gate-improvement package

### Create (expected)
- `safety/improvements/` (or equivalent) modules listed above
- `tests/test_safety_gate_improvements.py`
- `results/safety_gate/improvements/` outputs:
  - `SAFETY_GATE_IMPROVEMENT_REPORT.md`
  - `config.json`, `reproducibility.json`
  - `raw/*.jsonl`
  - `tables/*.json`
  - `plots/*.png` (calibration, risk-coverage, runtime)

### Modify
- **None of the preserved originals** (`safety/gate.py`, `safety/evaluate.py`, `twin/`, RL checkpoints, prior `results/evaluation/*`, `models/twin/*`, `results/digital_twin/improvements/*` weights/reports).

### Preserved (unmodified) artefacts
- Original Digital Twin weights and code paths used in production loading
- Original Safety Gate implementation
- Improved Twin training artifacts (reuse read-only)
- RL checkpoints, prior evaluation results, splits, frozen P1, ASN shards

---

## 6. Audit verdict

| Question | Verdict |
| --- | --- |
| Can improved Twin integrate without changing originals? | **Yes**, via a parallel loader/infer + new eval harness |
| Is conformal applied in the original gate path today? | **No** |
| Is the array contract compatible? | **Yes** (`mean_load`, `std_load`, `capacity`) |
| Ready to implement improved gate experiments? | **Yes**, after this audit |

**Next stage may begin implementation** in a separate directory following the design above; this audit file is the prerequisite deliverable and does not itself change any model or gate behaviour.
