# Experimental Protocol Audit

**Mode:** inspection + limited synthetic tests only.  
**Not done:** full A–F experiment, threshold tuning, GEANT/Kdl selection.

| Item | Result |
| --- | --- |
| Protocol unit tests | `tests.test_experimental_protocol` — **9 OK** |
| Eval infra tests | `tests.test_safety_gate_eval_infra` — **9 OK** |
| Combined | **18 OK** |
| Full A–F launched | **No** |

---

## 1. Files inspected

- `safety/improvements/run_eval.py`
- `safety/improvements/decision.py`
- `safety/improvements/oracle.py`
- `safety/improvements/fallback.py`
- `safety/improvements/diagnostics.py`
- `safety/improvements/twin_backend.py`
- `safety/improvements/gate_logic.py`
- `safety/improvements/policy_backend.py`
- `safety/improvements/graph_cache.py`
- `safety/improvements/metrics.py` *(added this audit)*
- `safety/improvements/resume.py` *(added this audit)*
- `results/safety_gate/improvements/gate_improvements_config.json`
- `results/safety_gate/improvements/smoke/` (prior 12-row smoke)

---

## 2. Paired comparison integrity

### Verified

| Requirement | Status | Evidence |
| --- | --- | --- |
| B–E receive identical policy actions | **Pass** | One `policy_action` proposed per case; all variants share `policy_action_fingerprint`; test asserts single fingerprint |
| Identical traffic matrices | **Pass** | Same `matrix = demands[local_index] * scale` object/value passed into every variant in the case |
| Seed handling consistent | **Pass with note** | Seed is logged on every row. **Full** mode: seed drives deterministic matrix sampling RNG. **Smoke** mode: seed is metadata; matrices are the first `n_matrices` indices (not seed-resampled) |
| Candidate paths identical | **Pass** | Fallback always rebuilds `{shortest, equal}` from topology; selection is physics-only and deterministic for a fixed matrix |
| Proposed vs applied distinguishable | **Pass** | `proposed_action_source="policy"` vs `applied_action_source` / `selected_fallback`; reject tests confirm divergence on fallback |
| No ordering advantage | **Pass** | Accept/reject identical under reversed variant order; Twin preds cached per backend within a case (efficiency only) |

### Pairing mechanism (code)

`run_evaluation` → single `policy.propose` → `evaluate_paired_case(..., policy_action=...)` → loop variants with the **same** `matrix` and `policy_action`. Fairness check raises if fingerprints diverge.

---

## 3. Independent oracle verification

| Check | Status |
| --- | --- |
| Actual loads from physical routing (`link_load` / incidence) | **Pass** — `oracle.action_link_load` |
| Violations vs actual capacities | **Pass** — `load > capacity + 1e-6` |
| Feasibility not inferred from Twin utilisation | **Pass** — F and oracle labels never call Twin |
| Fallback ignores Twin uncertainty | **Pass** — `select_physics_fallback` uses only physics loads + `classify_candidates` |

Synthetic test poisons Twin-like means/stds; oracle violation counts still match true loads.

---

## 4. Diagnostic verification

| Check | Status |
| --- | --- |
| Mean and uncertainty share link axis `(n_links,)` | **Pass** |
| Raw vs calibrated uncertainty retained separately | **Pass** — `raw_uncertainty_utilisation` / `calibrated_uncertainty_utilisation` |
| Calibration applied exactly once | **Pass** — `twin_uncertainty_views` recovers `raw = cal/scale` when already scaled; test checks not `scale²` |
| Prediction errors vs actual simulated util/load | **Pass** — `true_load = action_link_load(...)`; util error uses actual util |
| Capacity proximity consistent | **Pass** — bins on `actual_mlu` from physics |
| Aggregations retain topology / scale | **Pass** — `summarize_diagnostics` groups by topology, scale, capacity_proximity; **seed retained on every row** (group-by-seed available via row filter; default summary facets omit seed — see risks) |

---

## 5. Metric definitions (exact denominators)

Implemented in `safety/improvements/metrics.py` (`compute_safety_metrics`).

Let:
- **N** = all logged lines for the slice (including failures)
- **C** = completed decisions (`decision_status=="ok"` and `acceptance` present)
- **F** = failed/incomplete (`n_failed`); **F is never dropped from N**
- **A** / **R** = accepted / rejected among C
- **U** / **S** = unsafe / safe **proposed** actions among oracle-labeled completed rows

| Metric | Formula | Denominator notes |
| --- | --- | --- |
| Acceptance rate | \|A\| / \|C\| | Failed rows excluded from denom; counted in `n_failed` |
| Unsafe acceptance rate | \|accepted ∧ unsafe\| / \|C\| | Primary. Secondary among accepts: / \|A\| |
| False rejection rate | \|rejected ∧ safe\| / \|C\| | Secondary among safe: / \|S\| |
| Fallback frequency | \|fallback_used\| / \|C\| | |
| No-feasible-fallback frequency | \|fallback_used ∧ no_feasible_fallback\| / \|C\| | Also / \|R\| secondary |
| Capacity violations | mean/sum of proposed & applied violation counts over C | |
| Error detection rate | \|rejected ∧ unsafe\| / \|U\| | Recall on unsafe proposals; NaN if U=0 |
| False alarm rate | \|rejected ∧ safe\| / \|S\| | = false-reject among safe; NaN if S=0 |
| Runtime overhead | mean `decision_runtime_ms` over C | Twin runtime mean separate; failed runtimes under `failed_runtime_ms_mean` |

**Missing/failed decisions:** retained in `n_logged` / `n_failed`; **not** treated as rejects; **not** silently omitted from the log when case execution is caught.

---

## 6. Evaluation runner reliability

| Failure mode | Behaviour |
| --- | --- |
| Failed policy inference | Case caught; `decision_status=failed` row appended; run continues |
| Invalid Twin predictions | Gate rejects (`invalid_prediction`); fallback may apply for B–E |
| Missing candidate paths | `KeyError` from fallback (unsupported name) → failed case row |
| Failed fallback selection | Surfaces as case failure if exception; classify_candidates always returns a selection if candidates exist |
| Missing traffic matrices | `IndexError` → failed case row |
| Interrupted runs | Per-case append to `decisions.jsonl`; completed A–F blocks skipped on resume |
| Duplicate records | `resume=True` (default) skips `case_id`s with ≥6 OK variant rows; `--no-resume` rewrites |
| Failed case retry | Failed-only rows do **not** mark `case_id` complete → resume retries |

### Resume semantics

- `case_id = seed|topo|split|mid|li|scale|policy`
- Complete iff ≥ `len(variants)` successful rows for that id
- New rows appended; summary recomputed from full log

---

## 7. Limited tests added

`tests/test_experimental_protocol.py`:
- Pairing fingerprint + seed/scale/matrix identity
- Variant-order invariance
- Proposed vs applied on reject
- Oracle physics independence
- Calibration once / link-aligned diagnostics
- Metric denominators with failed rows
- Resume skip complete / retry failed
- Fallback candidate identity

---

## 8. Remaining experimental risks

1. **Smoke seed does not resample matrices** — document clearly when interpreting smoke vs full catalog.
2. **Default diagnostic summary does not facet by seed** — rows carry `seed`; add seed grouping in experiment reporting if needed.
3. **Crash mid-`evaluate_paired_case`** before append loses that case (no partial A–F written) — acceptable; resume will retry. No torn half-blocks.
4. **Diagnostic UCB for mean-only rows** uses `gate_k=0` from the decision record — correct for decision UCB; calibrated uncertainty columns remain available for analysis.
5. **Catalog sampling in `--full`** uses local RNG in `run_eval`, not necessarily identical to `robustness_eval.sample_matrix_indices` — before claiming cross-study pairing with robustness_v1, align sampling helper.
6. **Thresholds untuned at the time of this audit** — `k=1.0` was a placeholder; the threshold was later selected on the seen validation topologies only (`uncertainty_diagnostics/THRESHOLD_SELECTION.md`).

---

## 9. Approval recommendation

**Conditional approval to proceed to the full paired A–F evaluation**, provided the experiment stage:

1. Uses `resume=True` and a dedicated `eval_runs/<tag>/` output directory.
2. Publishes metrics via `compute_safety_metrics` (explicit denominators).
3. Does **not** tune on GEANT/Kdl.
4. Aligns matrix sampling with the robustness catalog if cross-comparison is required.
5. Treats smoke numbers as plumbing only (already enforced).

Protocol integrity for pairing, oracle independence, diagnostics, failure accounting, and resume/dedup is **sufficient to start full evaluation**. No full A–F run was executed in this audit.
