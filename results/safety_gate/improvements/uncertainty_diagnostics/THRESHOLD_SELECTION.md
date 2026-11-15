# Threshold selection report

Selection used **validation** matrices on B4/UsCarrier/Cogentco only.  
Selection seeds: `[404, 505]` (32 cases).  
Sensitivity seeds (not used for selection): `[101, 202, 303]` (48 cases).

**GEANT and Kdl were not used.**

## Objective

```
score = error_detection_rate - 0.5 * false_alarm_rate - 0.02 * k
```

on selection seeds, for gate-proxy alarms:

`reject if any link has util_mean + k * std_util > utilisation_limit`,

compared with physics `any(util_true > 1)`.

Grid: `k ∈ {0.5,1.0,1.5,2.0,2.5}`, `utilisation_limit ∈ {0.95,1.0,1.05}`, uncertainty ∈ {calibrated, raw}.

## Measured outcome

On this catalog the Twin is near-physics: **all calibrated (and raw) grid points with `utilisation_limit ≥ 1.0` achieved**

- error_detection_rate = **1.0**
- false_alarm_rate = **0.0**

on both selection and sensitivity seeds. The score therefore collapses to a **tie-break on smaller k** (`-0.02·k`).

### Score-maximising configuration (mechanical)

| Field | Value |
| --- | --- |
| Uncertainty | calibrated |
| k | **0.5** |
| utilisation_limit | **1.0** |
| Selection score | 0.99 |
| Selection detection / FAR | 1.0 / 0.0 |
| Sensitivity detection / FAR | 1.0 / 0.0 |

### Recommended configuration for upcoming A–F (judgment)

Because detection/FAR are tied across the informative region of the grid, prefer a configuration aligned with the existing improved-gate default and the conformal **2σ** operating regime:

| Field | Value | Rationale |
| --- | --- | --- |
| Uncertainty | calibrated | Matches variant E |
| k | **1.0** | Default in `gate_improvements_config.json`; not an aggressive under-bound |
| utilisation_limit | **1.0** | Capacity-respecting UCB |
| max_relative_std | `null` | Avoid blanket rejection of uncertain but low-mean actions |

**Hypothesis (not measured in A–F yet):** k=1.0 will behave similarly to k=0.5 on this near-physics twin for overload detection, while remaining more conservative if residuals grow on unseen topologies later.

Secondary sensitivity candidates to log in A–F (not selected here): `(k=0.5, limit=1.0)`, `(k=2.0, limit=1.0)`.

## Best raw-uncertainty comparator

Same grid winner: `k=0.5`, `limit=1.0` (tie-break). Raw vs calibrated did not change detection/FAR on this set.

## Caveats

- Thresholds were **not** fit on GEANT/Kdl.
- Selection seeds are still **val** matrices; they are a held-out *seed* split within development topologies, not a hidden test topology.
- Perfect detection here does **not** prove uncertainty adds value over mean-only gating — that requires the paired A–F comparison (especially D vs E).

Full candidate table: `tables/threshold_selection.json`.
