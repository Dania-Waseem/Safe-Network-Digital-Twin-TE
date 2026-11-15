# Evaluation Runner Report

**Stage:** infrastructure + limited smoke only.  
**Modules:** `safety/improvements/run_eval.py`, `diagnostics.py`  
**Config:** `results/safety_gate/improvements/gate_improvements_config.json`  
**Smoke artifacts:** `results/safety_gate/improvements/smoke/`

---

## Scope

This stage implements the paired A–F evaluation **harness**. It does **not**:
- execute the full multi-seed A–F experiment;
- tune `k` / utilisation thresholds;
- calibrate on GEANT/Kdl;
- train or modify RL / Twin / original Safety Gate artifacts.

Full catalog execution requires an explicit `--full` / `allow_full=True` flag and remains reserved for the experiment stage.

---

## Supported variants (paired)

| ID | Mode |
| --- | --- |
| A | Policy only |
| B | Original Twin mean-only |
| C | Original Twin uncertainty-aware |
| D | Improved Twin mean-only |
| E | Improved Twin calibrated uncertainty-aware |
| F | Independent physics oracle |

For each case, **one** RL `policy_action` is proposed, then all variants consume that identical action (`policy_action_fingerprint` must match within the case block).

Proposed vs applied actions remain distinct fields (`proposed_action_source` vs `applied_action_source` / `selected_fallback`).

---

## Diagnostics collected

Per decision row (where applicable):
- predicted mean utilisation
- raw uncertainty (utilisation)
- calibrated uncertainty (utilisation)
- actual utilisation / actual MLU
- absolute (and utilisation) prediction error
- UCB utilisation
- oracle feasibility + capacity violations
- gate acceptance / rejection reason
- fallback use + feasibility flags
- decision / twin runtime

Grouped summaries by **topology**, **scale**, and **capacity_proximity** (`far_below` / `mid` / `near_limit` / `over_limit`).

---

## Fairness / protocol guards

- Identical policy actions across A–F within a case.
- Separate Twin backends (`original`, `improved`, `improved_calibrated`).
- Config asserts `tune_on ∩ never_tune_on = ∅` (GEANT/Kdl never in `tune_on`).
- Conformal scale is **frozen** from the improved Twin package (seen-topo fit); runner does not refit.

---

## Smoke integration result

Command: `python -m safety.improvements.run_eval --smoke`

| Metric | Value |
| --- | ---: |
| Cases | 2 (B4 × scales 0.5, 1.0 × seed 101) |
| Rows | 12 (2 × 6 variants) |
| Acceptance rate (smoke) | 1.0 |
| Fallback rate (smoke) | 0.0 |
| Mean actual MLU | ≈ 0.741 |
| Outputs | `smoke/decisions.jsonl`, `smoke/summary.json`, `smoke/run_meta.json` |

Smoke numbers are **plumbing checks only**, not safety claims.

---

## Configuration updates

`gate_improvements_config.json` now includes:
- `policy.checkpoints` map
- `evaluation.smoke` slice
- `evaluation.topology_spec` (full catalog scaffold)
- `evaluation.diagnostics` field list / group-by
- explicit never-tune / calibration notes

---

## Tests

| Suite | Result |
| --- | --- |
| `tests.test_safety_gate_eval_infra` | OK (RL load, reproducibility, cache isolation, paired smoke, logging, full-blocked) |
| `tests.test_safety_gate_improvements` | OK |
| Combined | **32 passed** |

---

## Modes

`safety.improvements.run_eval` runs a two-case smoke evaluation by default (`--smoke`); `--full` evaluates the whole catalogue into `results/safety_gate/improvements/eval_runs/<tag>/`. Threshold selection uses the B4/UsCarrier/Cogentco validation split only.

---

## Preservation

Original Twin SHA `7b74fee9…`, original `safety/gate.py`, improved Twin weights, RL checkpoints, splits, and prior evaluation trees were not overwritten. New writes are confined to `results/safety_gate/improvements/` and `safety/improvements/`.
