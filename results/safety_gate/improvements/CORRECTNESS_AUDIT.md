# Safety Gate Improvements — Second-Stage Correctness Audit

**Mode:** integration and correctness verification only.  
**Not done:** full A–F experiment, threshold tuning, GEANT/Kdl selection.

**Live verification:** `python -m safety.improvements._audit_correctness`  
**Raw machine log:** `results/safety_gate/improvements/correctness_audit_raw.json`

| Metric | Value |
| --- | ---: |
| Checks | 61 |
| Passed | 61 |
| Failed | 0 |
| Warnings | 2 (missing eval runner; no dedicated policy_backend module) |
| Unit tests (`tests.test_safety_gate_improvements`) | 23 OK |

---

## 1. Twin backend integration

### Checkpoints exercised
| Backend | Path | Role |
| --- | --- | --- |
| `original` | `models/twin/ensemble.pt` | B / C |
| `improved` | `models/twin_improved/ensemble_physics_anchored.pt` | D |
| `improved_calibrated` | `…/ensemble_physics_anchored_conformal.pt` | E |

### Architecture / compatibility
- All three load through `EnsembleTwin` with **n_members=5**, **hidden=32**.
- Parameter counts match across improved vs calibrated payloads.
- **`improved` and `improved_calibrated` state_dicts are byte-identical** (same weights; conformal payload only adds `conformal_std_scale` ≈ **0.954545…**).
- Original twin SHA256 unchanged: `7b74fee96cdf3aa71c78aaa185adbc24c99f262f342689a4ec78d97b7db73685`.

### Output shapes and units
- On B4 val matrix 0 / equal-split: mean/std/true shapes = `(n_links,)` = `(38,)`, matching `topo.capacity`.
- Values are **absolute link loads** (max mean ≈ 5121 with denom 5000), not utilisation. The improved gate converts to utilisation internally via `/ max(capacity, min_capacity)`.

### Original inference unchanged
- `safety.evaluate.predict_action(load_trained_ensemble(...))` vs `TwinBackend("original")`:
  - mean maxabs diff = **0**
  - std maxabs diff = **0**
  - true load maxabs diff = **0**
- Conclusion: the new stack does not alter original Twin behaviour; it only wraps the same `_predict_numpy` path for the original weights when `apply_conformal=false`.

---

## 2. Conformal calibration (D vs E)

| Property | Result |
| --- | --- |
| Identical means (D backend vs E backend) | **Yes** (maxabs 0) |
| D uses uncertainty in the gate | **No** (`gate_k=0`, std zeroed for `inspect_utilisation_ucb`, `gate_used_uncertainty=false`) |
| E uses calibrated std | **Yes** (`gate_k>0`, conformal flag true) |
| Correct conformal checkpoint for E | `ensemble_physics_anchored_conformal.pt` |
| Correct anchored checkpoint for D | `ensemble_physics_anchored.pt` |
| Scale applied exactly once | **Yes**: `E.std == raw_std * scale`, and ≠ `raw_std * scale²` |
| Latent double-scale in backend | **No** (mean ratio of E.std/raw_std == scale) |

D still **logs** Twin predictive uncertainty for analysis, but the acceptance UCB is mean-only (`UCB = mean_util`).

E applies `payload["conformal_std_scale"]` only inside `predict_loads` via `TwinBackend.resolved_conformal_scale()`; there is no second multiply in `gate_logic` or `decision`.

---

## 3. Capacity / numerical safeguards

| Case | Behaviour |
| --- | --- |
| Positive capacity, safe UCB | Accept |
| Zero capacity + positive load | Reject (`ucb_exceeds_utilisation_limit` via `min_capacity` floor) |
| Very small capacity + load | Reject |
| NaN / Inf capacity | Reject (`invalid_prediction`) |
| NaN / Inf mean or std | Reject |
| Zero predicted load | Accept (if within limit) |
| Negative predicted mean | Reject (`invalid_prediction`) — **hardened during this audit** |
| Negative capacity | Reject (`invalid_prediction`) — **hardened during this audit** |
| Clear overload | Reject |

Safeguard intent: numerical edge cases must not silently accept. Invalid inputs fail closed (reject → physics fallback for B–E).

---

## 4. Fallback consistency (B–E)

- Candidate set fixed: **`("shortest", "equal")`** from config / `DEFAULT_CANDIDATES`.
- Selection reuses unmodified `framework.fallback_modes.classify_candidates`:
  - prefer **feasible** (0 physics violations);
  - else **least-violating** (violations, then MLU, then name).
- `feasible_fallback` XOR `no_feasible_fallback` holds.
- On rejecting high-scale B4 cases, every gated variant B–E selected the **same** fallback as a direct physics call.
- Logs distinguish:
  - `proposed_action_source="policy"`
  - `applied_action_source ∈ {policy, shortest, equal}`
  - `selected_fallback` only when fallback used
- Variant **A** never falls back (by design).

---

## 5. RL integration

### Checkpoint inspection
| Item | Detail |
| --- | --- |
| Default config policy | `models/policy/improved/recommended_stable.pt` (`checkpoint_id=improved_stable`) |
| Also present | `models/policy/improved/best_checkpoint.pt` |
| Loader | `framework.general_policy.GeneralPolicy.load` → `payload["model"]` into GRL actor |
| Robustness helper | `framework.robustness_eval._load_general` (eval mode) |

### Live propose → gate
- `GeneralPolicy.load(recommended_stable)` succeeds; payload contains `"model"`.
- `propose(topo, od_demands(matrix))` returns finite, non-negative action with shape `(n_path_nodes,)`.
- That action flows through variants **A** and **E** with correct `policy_checkpoint_id` and twin backend tags.

### Gap (warning, not a failure)
- At the time of this audit `safety/improvements/policy_backend.py` did not exist. The gate API accepts a precomputed `policy_action`; the policy backend with shared `PolicyGraph` caching (as in `robustness_eval`) was added afterwards (`POLICY_BACKEND_REPORT.md`), which avoids OOM and inconsistent loading in multi-topology runs.

---

## 6. Changes made during this audit

Only inside the **improved** stack (originals untouched):

1. `gate_logic.inspect_utilisation_ucb` now rejects **negative capacity** and **negative mean load** as `invalid_prediction`.
2. Added unit tests for those cases (+ NaN capacity).
3. Added reproducible auditor `safety/improvements/_audit_correctness.py` and `correctness_audit_raw.json`.

Original `safety/gate.py`, original Twin weights, improved Twin weights, RL checkpoints, and prior evaluation trees were not overwritten.

---

## 7. Verdict and remaining work before full A–F

**Verdict: ready for experiment harness wiring**, with the caveats below.

| Remaining item | Status |
| --- | --- |
| Twin backends + conformal math | Verified |
| Capacity fail-closed behaviour | Verified (+ hardened) |
| Fallback paired consistency | Verified |
| RL checkpoint load + propose | Verified |
| `run_eval.py` over robustness catalog | **Not started** (by design) |
| Dedicated `policy_backend` / graph cache | **Added after this audit** |
| Threshold tuning on seen val only | **Not started** |
| Claims about safety improvement | **Not yet supported at audit time** |

GEANT/Kdl are not used for tuning. The full A–F sweep followed once the evaluation runner and policy loader were in place.
