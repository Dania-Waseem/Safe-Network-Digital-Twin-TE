# Paired A–F Safety Gate Evaluation Report

**Run tag:** `af_eval_k1_calibrated_devtopos_v1`  
**Directory:** `results/safety_gate/improvements/eval_runs/af_eval_k1_calibrated_devtopos_v1/`

## 1. Frozen configuration

| Item | Value |
| --- | --- |
| Policy | `improved_stable` (`recommended_stable.pt`) |
| Twin B/C | Original ensemble |
| Twin D | Physics-anchored (mean-only, k=0) |
| Twin E | Physics-anchored + conformal, **k=1.0**, utilisation_limit=**1.0** |
| Catalog | `results/evaluation/robustness_v1/catalog.json` |
| Seeds / scales | 101–505 / 0.25–1.0 |
| Cases | **80** (B4×2 + UsCarrier×1 + Cogentco×1 matrices per seed × 4 scales) |
| Decision rows | **480** (80 × A–F) |
| Failed cases | **0** |

No further threshold tuning or training was performed.

## 2. Topology scope (GEANT / Kdl)

**Excluded from this run.**

Reason: GEANT and Kdl are the held-out test topologies. This run covers the development topologies (B4/UsCarrier/Cogentco) used for calibration, tuning and diagnostics; the single holdout evaluation on GEANT/Kdl is reported separately (`results/phase3_final/e2_holdout_af/`).

## 3. Completeness and resumability

| Check | Result |
| --- | --- |
| Completed A–F blocks | 80 / 80 |
| Incomplete blocks | 0 |
| Fingerprint violations (paired policy action) | 0 |
| Failed case rows | 0 |
| Resume re-run | skipped **80** cases, appended **0** new rows, still **480** completed rows |

## 4. Metrics by variant

Denominators follow `compute_safety_metrics` (completed decisions C=80 per variant; failed rows separate).

| Variant | Accept | Unsafe accept | False reject | Fallback | Error detection | Applied overload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Policy only | 1.000 | **0.500** | 0.000 | 0.000 | 0.000 | 0.500 |
| B Orig. mean | 0.500 | **0.000** | 0.000 | 0.500 | 1.000 | 0.500 |
| C Orig. unc. | 0.500 | **0.000** | 0.000 | 0.500 | 1.000 | 0.500 |
| D Imp. mean | 0.500 | **0.000** | 0.000 | 0.500 | 1.000 | 0.500 |
| E Imp. cal. unc. | 0.500 | **0.000** | 0.000 | 0.500 | 1.000 | 0.500 |
| F Physics oracle | 0.500 | **0.000** | 0.000 | 0.500 | 1.000 | 0.500 |

Of 80 cases, **40** have an unsafe proposed RL action and **40** are safe (oracle on the proposal).

**Applied overload remains 0.5 for B–F:** whenever the proposal is unsafe, shortest/equal fallbacks are also infeasible on this catalog slice, so the least-violating fallback still overloads. Gates remove *unsafe accepts of the policy*, but cannot create a feasible route when none exists among candidates.

## 5. Primary comparison — D vs E

| Quantity | D (mean-only) | E (calibrated UCB) | Paired diff D−E [95% CI] |
| --- | ---: | ---: | --- |
| Unsafe acceptance | 0.000 | 0.000 | 0.000 [0.000, 0.000] |
| False rejection | 0.000 | 0.000 | 0.000 [0.000, 0.000] |
| Acceptance | 0.500 | 0.500 | 0.000 [0.000, 0.000] |
| Discordant unsafe accepts | — | — | D-only=0, E-only=0 |

### Does calibrated uncertainty add measurable incremental value?

**Measured answer: No (not demonstrated).**

On these 80 paired cases, D and E are **identical** on accept/reject, unsafe accept, and false reject. Mean-only improved-Twin gating already matches the physics oracle’s accept/reject pattern; adding calibrated UCB does not change any decision.

This is a **measured null incremental effect** on the development catalog under the frozen (k=1, limit=1) rule — not a claim that uncertainty can never help on other topologies, residual regimes, or thresholds.

## 6. Other prioritized comparisons

| Comparison | Result (measured) |
| --- | --- |
| **B vs D** (orig. vs improved mean) | Unsafe accept both 0; identical accept pattern (diff 0) |
| **C vs E** (orig. vs improved unc.) | Identical (diff 0) |
| **A vs D/E** (introduce gate) | Unsafe accept 0.50 → 0.00; paired diff 0.50 [0.39, 0.61] |
| **E vs F** (gate vs oracle) | Acceptance identical (0.50); E matches oracle accept/reject on all 80 cases |

**Measured finding:** introducing any Twin gate (B–E) eliminates unsafe policy accepts relative to A, with **zero false rejects** on this set. Original and improved Twins, mean-only and uncertainty-aware, all reproduce the oracle’s accept/reject decisions here.

## 7. Topology / scale

Breakdowns are in `tables/analysis.json` → `breakdowns`. Across B4/UsCarrier/Cogentco and scales 0.25–1.0, D and E remain matched (no topology where E uniquely rejects an unsafe accept that D accepts).

## 8. Measured findings vs hypotheses

| Statement | Status |
| --- | --- |
| Safety Gate (vs policy-only) cuts unsafe accepts on this catalog | **Measured** (0.50 → 0.00) |
| Improved Twin beats original Twin on safety rates here | **Not demonstrated** (tied at 0 unsafe accept / 0 false reject) |
| Calibrated uncertainty (E) beats mean-only (D) | **Not demonstrated** (identical decisions) |
| E closes the gap to the oracle | **Measured** (E ≡ F on accept/reject) |
| Uncertainty will help on GEANT/Kdl | **Hypothesis only** (excluded from this run) |

## 9. Preservation

- Original `safety/gate.py`, Twin checkpoints, conformal artifacts, RL checkpoints, splits, frozen P1, and prior baselines: **not modified**
- Outputs written only under this run directory (+ frozen eval config fields)
- No retraining; no threshold retuning after diagnostics

## 10. Artifacts

- `decisions.jsonl` — raw paired decisions  
- `run_meta.json`, `summary.json`  
- `tables/analysis.json` — metrics, paired tests, breakdowns  
- `plots/unsafe_vs_false_reject.png`, `acceptance_rate.png`, `fallback.png`, `d_vs_e_by_topology.png`  
- `AF_EVALUATION_REPORT.md` (this file)
