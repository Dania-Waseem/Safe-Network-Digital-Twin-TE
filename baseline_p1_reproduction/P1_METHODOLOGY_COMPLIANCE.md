# P1 Methodology Compliance — B4

Component-by-component compliance for experimental fidelity (Phase E).

Status labels: **MATCH** · **CLOSE** · **AMBIGUOUS** · **DEVIATES** · **EXTRA** · **UNSPECIFIED**

---

## Compliance table

| Component | P1 | Ours | Status |
| --- | --- | --- | --- |
| B4 size (Table 5) | 12 / 38 | 12 / 38 | **MATCH** |
| Avg SP / diameter | 2.3 / 5 | 2.32 / 5 | **MATCH** |
| Capacity / TM source | Teal | TEAL JSON + real TMs | **MATCH** |
| Mean demand citation | 27.98k | test mean 27.981k; all-36 mean 26.88k | **CLOSE** |
| 4 candidate paths | “4 shortest” | Yen min-hop (not edge-disjoint) | **AMBIGUOUS** |
| TopoFlowNet | 6 layers, dim 7 | Same | **MATCH** |
| Actor | 28→4 mean & log-std | Same (+ clamps) | **CLOSE** |
| Critic | 28→256→256→1 ReLU | Same | **MATCH** |
| Reward | \(r=F_{\mathrm{total}}\) | Same | **MATCH** |
| One-step A2C | \(A=r-V\), no \(\gamma\) | Same | **MATCH** |
| Episodes (Table 6 protocol) | 3 | 3 (Phase D 20 = validation only) | **MATCH** |
| Seeds | 100…1000 | Same | **MATCH** |
| Adam LRs | 1e-4 / 1e-3 | Same | **MATCH** |
| ADMM \(\rho\), B4 iters | 1.0, 2 | Same | **MATCH** |
| ADMM block order / argmins | F→z→ξ→λ; unpublished closed forms | Teal deploy (stable); P1-order code exists but unstable @ 2 iters | **DEVIATES** |
| Constraint adj. (23)–(28) | Demand then capacity | Same | **MATCH** |
| Demand satisfaction metric | Satisfied / offered (formula not printed) | \(\sum a / \sum d\) after (28) | **CLOSE** |
| TM train/test split | Unspecified | 20/8/8 | **UNSPECIFIED** |
| Grad clip 1.0 | Not stated | Used | **EXTRA** |
| Hardware | RTX 4090 | Local machine | **DEVIATES** (env) |

---

## Metric definitions

| Metric | Definition used | Notes |
| --- | --- | --- |
| Demand satisfaction | \(\sum_{m,p}\hat a_{m,p} / \sum_m d_m\) | After (23)–(28); aligns with P1 “satisfaction rate” intent |
| Reward \(F_{\mathrm{total}}\) | \(\sum \hat a_{m,p}\) | Eq. (29)–(30) |
| MLU | \(\max_e \mathrm{load}(e)/c(e)\) | Reported; not Table 6 primary |
| Capacity violations | Count edges with load > 1.001\(c\) | Engineering threshold |

P1 does **not** print the algebraic satisfaction formula; Fig. 6 title “Total satisfied demand” is consistent with throughput / offered demand.

---

## Claims not supported

- That **97.73% / 98.13%** is a fair “improvement” over P1 **92.39%**.
- That ADMM is an exact transcription of unpublished P1 argmins.
- That Yen paths are proven identical to the authors’ path set.

## What can be claimed

- Faithful **architecture + training signal** on the **public TEAL B4** instance P1 cites.
- LP sanity check ≈ P1 LP (99.74% vs 99.83%).
- Documented deviations: paths, split, ADMM numerics, clip, hardware.
