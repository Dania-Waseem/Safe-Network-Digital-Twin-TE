# Final comparative analysis

This report synthesizes `robustness_v1` multi-seed evaluation with new A–E ablations
under identical traffic matrices. No training or tuning used GEANT/Kdl.

## Audit summary

- Robustness rows: 840 / expected 840 (5 seeds × (2+1+1+2+1 matrices) × 4 scales × 6 methods = 840).
- Audit errors: **0**.
- Full audit: `AUDIT_REPORT.md`.

## Experimental answers

### 1. Does generalized Graph RL improve upon the original policy?

**On B4: yes for `recommended_stable` — demonstrated.** Per-topology robustness tables
show lower actor MLU than frozen P1 and the 3-epoch actor at every B4 scale
(e.g. 0.250±0.002 vs 0.312±0.004 at 0.25×). Paired B4-only tests are in
`tables/paired_tests.json`. **Across all topologies pooled:** stable actor MLU is
*higher* than frozen on average (paired mean diff ≈ +0.33, Wilcoxon p ≪ 0.05) because
UsCarrier/Kdl/Cogentco dominate — so improvement is **not** global.
`improved_best` is weaker on B4 than `recommended_stable` and is not a better
general-purpose choice despite winning validation-mean selection.

### 2. Does it outperform traditional routing baselines?

**Only selectively — demonstrated with limits.** B4 feasible scales: competitive with
equal/shortest. Pooled over all cases, stable actor MLU is **worse** than shortest
(paired mean diff ≈ +1.13, significant). UsCarrier/Cogentco/GEANT/Kdl usually favor
shortest or equal on actor MLU. Cogentco-heavy aggregates are not a valid ranking.

### 3. Does it generalize to unseen topologies?

**Weak / not for Kdl — demonstrated.** GEANT low-scale proposals are sometimes accepted,
but actor MLU does not beat shortest/equal. Kdl is infeasible under shortest/equal at
all tested scales; actor MLU remains above shortest; the gate rejects and falls back.
Zero-shot Kdl routing improvement is **not** supported.

### 4. Does the Digital Twin improve decision quality?

**As a predictive input to gating: yes for safety; not as a direct MLU reducer.**
Ablation B applies the same action as A. Twin MAE is small on B4/UsCarrier but
**very large on GEANT** (order 1e5 load units in this eval) — twin transfer to GEANT
is unreliable even when coverage@2σ looks high (scale/std artifacts). Twin enables
D/E; mean-only gate (D) already matches E on unsafe accepts here.

### 5. Does uncertainty-aware filtering reduce unsafe decisions?

**Yes vs always-apply (A) — demonstrated.** On improved_stable, A unsafe-accept rate
≈ 0.657 vs E ≈ 0.0 (paired diff −0.657, Wilcoxon p ≪ 0.05). E also lowers applied MLU
vs A by falling back on overloads (mean diff ≈ −1.15, significant). **E vs D:** unsafe
accepts identical (0); false-reject difference ≈ 0.007 is **not** significant —
uncertainty terms add little beyond twin-mean gating in this suite. Zero unsafe
accept under E is measured against a **physics oracle**, not an online guarantee.

### 6. Does the complete framework justify its computational overhead?

**Context-dependent.** B4 overhead is tens of ms. Kdl decisions are ~100 s with twin.
Benefit is safety filtering + fallback, not beating shortest on large unseen graphs.
Strongest justification: avoid unsafe applies when proposals overload. Weakest:
massively infeasible Cogentco/Kdl where fallback is inevitable and twin is expensive.

### 7. Remaining research limitations

- ASN not evaluated.
- Small matrix budgets; seeds resample matrices, policies are deterministic.
- Training objective ≠ P1 ADMM satisfaction.
- Cogentco-distorted checkpoint selection (`best_checkpoint` vs `recommended_stable`).
- Twin labels are simulator loads; GEANT twin errors are severe.
- Ablation fallback omits last-accepted carry (cleaner pairing vs robustness_v1).

## Failure investigations

### Why shortest beats the policy on some unseen topologies

GEANT/Kdl were never trained. The actor optimizes soft overload/MLU/entropy on B4 and
scaled UsCarrier/Cogentco. Empirically actor MLU > shortest on GEANT/Kdl.

### Why Kdl zero-shot transfer is weak

Always infeasible under shortest/equal here; large padded graph; actor worse than
shortest; gate rejects. No feasible Kdl regime in this eval.

### Cogentco distortion

Aggregate actor MLU ~9.5 with Cogentco vs ~4.0 without. Validation-mean selection
preferred an early `best_checkpoint`; `recommended_stable` is better on B4.

### Is the Safety Gate overly conservative?

E false-reject ≈ 0.007 overall (B4 shows the nonzero mass). C (oracle) and D
(mean-only) accept slightly more often than E; the E−D gap is small and not
Wilcoxon-significant here. Conservatism exists but is limited in this suite.

### Is zero unsafe acceptance genuine?

Genuine vs physics ground truth on accepts. Not an oracle-free deployment proof.
A shows that without a gate, unsafe accepts are common (~66% of cases).

### Does the Digital Twin justify its cost?

Necessary for predictive gating (D/E). B alone does not change actions. GEANT twin
MAE is a major caveat. Kdl runtime dominates cost.

## Demonstrated vs hypothesis

| Claim | Status |
| --- | --- |
| `recommended_stable` improves B4 vs frozen/3-epoch | Demonstrated |
| Stable improves frozen on **pooled** actor MLU | **Rejected** (paired +0.33) |
| `best_checkpoint` is best general policy | **Rejected** by B4 evidence |
| Beats shortest on pooled actor MLU | **Rejected** (paired +1.13) |
| Strong Kdl zero-shot routing | Not supported |
| Full gate reduces unsafe accepts vs always-apply | Demonstrated (A vs E) |
| Uncertainty terms beat mean-only twin gate | Not supported here (E≈D) |
| Cogentco distorts aggregates and selection | Demonstrated |
| Twin alone improves applied routing vs policy-only | Not supported (B≡A actions) |
| Twin transfers accurately to GEANT | Not supported (huge MAE) |
| Framework always worth Kdl-scale compute | Hypothesis / context-dependent |

## Artifacts

- `AUDIT_REPORT.md`, `ABLATION_REPORT.md`, this file
- `tables/*.json`, `plots/*.png`, `raw/ablation_decisions.jsonl`
- `config.json`, `reproducibility.json`
