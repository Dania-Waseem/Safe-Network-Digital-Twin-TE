# Uncertainty Diagnostic Report

**Type:** Digital Twin uncertainty diagnostic (not the final A–F safety evaluation).

## Scope

| Item | Value |
| --- | --- |
| Topologies | B4, UsCarrier, Cogentco |
| Split / catalog | robustness_v1 **val** (`results/evaluation/robustness_v1/catalog.json`) |
| Scales | 0.25, 0.5, 0.75, 1.0 |
| Seeds | 101, 202, 303, 404, 505 (80 cases) |
| Policy actions | `improved_stable` (`recommended_stable.pt`) |
| Twin | physics-anchored + conformal scale **0.954545…** |
| Cases / link rows | **80** / **18800** |
| Held out of this study | **GEANT, Kdl** (no calibration, selection, or reporting dependence) |

### Measured vs hypothesis

- **Measured:** coverage, interval widths, MAE, case-level Spearman(std, error) with bootstrap CIs, capacity-bin descriptive stats, threshold-grid detection/FAR on selection seeds.
- **Hypothesis (not claimed):** that calibrated UCB gating will improve A–F safety metrics vs mean-only — deferred to the full paired evaluation.

Statistical note: bootstrap CIs treat **cases** as resampled units. Link-level summaries are descriptive and **not** independent experimental seeds.

---

## 1. Uncertainty quality (overall)

Case-level means with 95% bootstrap CIs (n=80):

| Metric | Estimate | 95% CI |
| --- | ---: | --- |
| MAE utilisation | 2.59e-05 | [2.13e-05, 3.05e-05] |
| MAE absolute load | 0.181 | [0.133, 0.233] |
| Spearman(cal std_util, util_err) | 0.686 | [0.621, 0.757] |
| Spearman(raw std_util, util_err) | 0.686 | [0.621, 0.757] |

Spearman(raw)=Spearman(cal) is expected: conformal scaling is a positive constant.

### Empirical coverage (utilisation space, rule `|err| ≤ k·std`)

| k | Raw coverage | Calibrated coverage | Notes |
| ---: | ---: | ---: | --- |
| 1.0 | 0.907 [0.889, 0.925] | 0.612 [0.525, 0.690] | Cal **undercovers** at 1σ |
| 1.5 | 1.000 | 1.000 | |
| 2.0 | 1.000 | 1.000 | Conformal target regime |
| 2.5 | 1.000 | 1.000 | |
| 3.0 | 1.000 | 1.000 | |

**Measured finding:** residuals often sit near the 1σ ensemble scale. Shrinking std by ~4.5% (conformal) therefore drops 1σ coverage sharply, while **k≥1.5 remains fully covering** on this set.

### Interval width @ k=2 (util)

| | Raw | Calibrated |
| --- | ---: | ---: |
| Mean width | 6.82e-04 | 6.51e-04 |
| Median width (case-mean of medians) | — | 4.12e-04 |

**Measured finding:** at the conformal operating point (≈2σ), calibration **preserves coverage (1.0)** and **slightly narrows** intervals (~4.5%), matching the design goal.

---

## 2. Reliability across traffic scales

Coverage @ k=2 (cal) remains **1.0** at every scale 0.25–1.0.  
MAE util and mean MLU increase with scale (see `tables/summary.json` → `by_scale` and `plots/coverage_by_scale.png`).

---

## 3. Topology-specific results

| Topology | n | MAE util | cov raw@2 | cov cal@2 | Spearman(cal) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4 | 40 | 1.42e-05 | 1.0 | 1.0 | 0.996 |
| UsCarrier | 20 | 1.28e-05 | 1.0 | 1.0 | 0.464 |
| Cogentco | 20 | 5.13e-05 | 1.0 | 1.0 | 0.289 |

**Measured finding:** uncertainty ranks errors almost perfectly on B4; association is weaker on UsCarrier/Cogentco (still positive overall when pooled at case level).

---

## 4. Capacity-proximity analysis

Bins use **actual** link utilisation. Higher mean cal std on over-capacity links is **observed**, but that alone does not prove decision usefulness.

| Bin | n_links | MAE util | mean cal std | Spearman(cal, err) | Large-err detection* | FAR* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| far_below | 3972 | 8.01e-06 | 2.31e-05 | 0.066 | 0.088 | 0.102 |
| mid | 3160 | 1.27e-05 | 5.37e-05 | -0.007 | 0.047 | 0.107 |
| near_limit | 873 | 1.46e-05 | 7.10e-05 | -0.101 | 0.169 | 0.097 |
| over_limit | 10795 | 5.01e-05 | 5.07e-04 | 0.322 | 0.214 | 0.087 |

\*Within-bin: high-uncertainty = top 10% `cal_std_util`; large error = top 10% `util_err`. Descriptive only.

**Measured finding:** mean uncertainty and MAE are largest in `over_limit`, with the strongest std–error Spearman there. In low/mid/near bins, high-uncertainty alarms are **weak** detectors of large local errors (detection ≈ chance to slightly above; FAR ≈ 0.1).  
**Hypothesis:** capacity-proximal uncertainty may still help *case-level* UCB overload gating even when link-local ranking is weak — test in A–F (D vs E).

---

## 5. Raw vs calibrated

| Property | Result |
| --- | --- |
| Mean ratio cal/raw std | **0.954545…** (exactly the frozen scale) |
| Applied once | Yes (calibrated backend only) |
| Coverage @ k=2 | Both 1.0 |
| Width @ k=2 | Cal ~4.5% narrower |
| Coverage @ k=1 | Cal worse (undercoverage) |

**Conclusion (measured):** calibration is appropriate for **multi-σ / UCB-style** use (≥1.5–2), not as a 1σ predictive interval on this residual distribution.

---

## 6. Threshold selection (post-diagnostic)

See `THRESHOLD_SELECTION.md`.

- Mechanical score winner: **k=0.5, utilisation_limit=1.0, calibrated** (tie-break among perfect detection/FAR configs).
- **Recommended for A–F:** **k=1.0, utilisation_limit=1.0, calibrated**, `max_relative_std=null`.
- Mean predictions already separate safe vs unsafe well here; **uncertainty’s incremental value is not established** until D vs E.

---

## 7. Artifacts

```
results/safety_gate/improvements/uncertainty_diagnostics/
  raw/case_records.jsonl
  raw/link_records.jsonl
  tables/summary.json
  tables/threshold_selection.json
  plots/coverage_vs_k.png
  plots/width_vs_k.png
  plots/reliability_calibrated.png
  plots/uncertainty_vs_error.png
  plots/capacity_proximity.png
  plots/coverage_by_scale.png
  config.json
  THRESHOLD_SELECTION.md
  UNCERTAINTY_DIAGNOSTIC_REPORT.md
```

Runner: `python -m safety.improvements.uncertainty_diagnostics`

---

## 8. Preservation

- Original / improved Twin checkpoints: **not modified**
- Original / improved Safety Gate: **not modified** (diagnostic module only added)
- RL checkpoints, splits, frozen P1, prior eval trees: **not overwritten**
- Full A–F experiments: **not launched**
