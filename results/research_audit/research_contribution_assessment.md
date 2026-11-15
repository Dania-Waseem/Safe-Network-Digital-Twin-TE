# Research Contribution Assessment

Evidence-based. Distinguishes **demonstrated findings** from **hypotheses**.

## 1. Strongest demonstrated contribution

**Decision-level Safety Gate that eliminates unsafe acceptance of RL proposals on the evaluated catalogs**, with paired evidence that:

- Policy-only unsafe acceptance is high (AF: 0.50 on 80 cases; final analysis: ~0.66 on broader robustness suite).
- Twin-gated variants reduce unsafe acceptance to **0** on those paired sets, with **zero or near-zero false rejects** in the AF development catalog.
- Applied overload can still remain when shortest/equal fallbacks are themselves infeasible — a measured limitation, not hidden.

This contribution is **decision-level safety**, not “better MLU than shortest on all topologies.”

## 2. Contributions with experimental support

| Proposed contribution | Support level | Evidence |
| --- | --- | --- |
| Safety Gate rejects unsafe RL proposals | **Demonstrated** | AF A vs B–F; final_analysis E vs A (Wilcoxon significant) |
| Twin enables gating without physics oracle at decision time | **Demonstrated** | E ≡ F accept/reject on AF 80 cases |
| Improved Twin reduces prediction util MAE | **Demonstrated** | digital_twin/improvements comparison |
| Uncertainty correlates with error (diagnostics) | **Demonstrated** | uncertainty_diagnostics Spearman≈0.69 |
| Calibrated uncertainty improves *gate decisions* vs mean-only | **Not demonstrated** (null) | AF D ≡ E; final_analysis E vs D unsafe identical / FR not significant |
| Improved Twin improves *gate decisions* vs original Twin | **Not demonstrated** on AF | B ≡ D, C ≡ E on safety rates |
| Generalized / improved RL beats frozen P1 on B4 | **Demonstrated (B4)** | robustness_v1; paired tests; NOT global across all topos |
| Zero-shot policy beats shortest on unseen Kdl/GEANT | **Not supported** | final_analysis / GENERAL_POLICY_REPORT |
| Conditional LP recovers catalog-feasible rejects | **Demonstrated** | 5/5 UsCarrier@0.25; A/B 40→35 overloaded cases |
| Most fallback failures are capacity/catalog infeasible | **Demonstrated** | 35/40 forced-link LB > 1 |

## 3. Unsupported or overstated claims

- Full faithful P1 Table-6 reproduction.
- Uncertainty-aware gating as a measured safety upgrade over mean-only on the AF catalog.
- Safety Gate / improved Twin A–F generalization to GEANT/Kdl under the *same* improved stack (not run).
- LP fallback as a general cure for overload (fails when catalog-infeasible).
- “Unseen topology success” for UsCarrier/Cogentco (they are training topologies).

## 4. Negative findings

1. **D ≡ E** on AF: calibrated UCB added no decision changes.
2. **Applied overload 0.5** for gated AF variants: rejecting unsafe proposals does not create feasible routes when candidates overload.
3. **35/40** fallback failures are physically infeasible on catalog paths.
4. **Policy does not globally beat shortest**; often worse on large/unseen graphs.
5. **P1 reproduction is PARTIAL**; Teal-order ADMM used for frozen B4 headline.
6. Toy/synthetic traffic on some topologies limits external validity.

## 5. Limitations

- Simulator-labeled Twin (near-physics multipliers).
- Yen-4 catalog path set.
- Continuous path fractions in LP.
- Multi-second LP latency.
- ASN excluded.
- The improved A–F run covers the development topologies only; GEANT/Kdl are evaluated separately.
- Seeds resample matrices; policies deterministic.

## 6. Scope of supported claims

The evidence supports: the problem (unsafe RL TE decisions); the method (twin + safety gate + fallback + conditional LP diagnostic); the measured safety gains and the null incremental effect of uncertainty; the fallback-infeasibility analysis and selective LP recovery; and the partial P1 reproduction with its limitations. It does not support a complete P1 reproduction, uncertainty-driven decision gains, or improved-gate A–F generalization to unseen topologies without a holdout run (later provided by the E2 holdout evaluation).
