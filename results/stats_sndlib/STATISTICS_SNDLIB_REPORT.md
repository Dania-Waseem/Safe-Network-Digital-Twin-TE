# Statistics with the SNDlib topologies

- GEANT and Kdl rows are the frozen-stack rows of results/phase3_final/e2_holdout_af/case_rows.jsonl, restricted to the calibrated load levels so that every topology contributes the same four levels.
- Kdl (1 TM x 4 levels) is not tested on its own; it enters the pooled test.
- Holm correction is applied within each family (Wilcoxon + overload McNemar; twin-error McNemar).

## Paired Wilcoxon signed-rank, framework (E) vs policy only (A), with Cliff's δ; McNemar on applied overload

| Comparison (left vs right) | n | mean diff | left better | right better | p | p (Holm) | Cliff's δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| framework (E) vs policy only (A) · Abilene · applied sat | 800 | +0.2554 | 0.93 | 0.04 | 9e-113 | 8.1e-112 * | +0.722 |
| framework (E) vs policy only (A) · Abilene · applied mlu | 800 | -0.5456 | 0.87 | 0.10 | 6.4e-81 | 3.2e-80 * | -0.416 |
| applied overload framework (E) vs policy only (A) · Abilene (McNemar; left-only 0, right-only 63) | 800 | -0.0787 | – | – | 2.2e-19 | 2.2e-19 * | – |
| framework (E) vs policy only (A) · Germany50 · applied sat | 800 | +0.0186 | 0.72 | 0.01 | 1.2e-94 | 9.3e-94 * | +0.151 |
| framework (E) vs policy only (A) · Germany50 · applied mlu | 800 | -0.0891 | 0.72 | 0.02 | 5.4e-94 | 3.2e-93 * | -0.144 |
| applied overload framework (E) vs policy only (A) · Germany50 (McNemar; left-only 0, right-only 75) | 800 | -0.0938 | – | – | 5.3e-23 | 1.1e-22 * | – |
| framework (E) vs policy only (A) · GEANT (calibrated) · applied sat | 800 | +0.0879 | 0.91 | 0.01 | 4e-121 | 4.4e-120 * | +0.659 |
| framework (E) vs policy only (A) · GEANT (calibrated) · applied mlu | 800 | -0.7196 | 0.85 | 0.07 | 5.8e-113 | 5.8e-112 * | -0.484 |
| applied overload framework (E) vs policy only (A) · GEANT (calibrated) (McNemar; left-only 0, right-only 173) | 800 | -0.2162 | – | – | 1.7e-52 | 6.7e-52 * | – |
| framework (E) vs policy only (A) · SNDlib pooled (Abilene + Germany50) · applied sat | 1600 | +0.1370 | 0.83 | 0.03 | 1.3e-197 | 1.7e-196 * | +0.382 |
| framework (E) vs policy only (A) · SNDlib pooled (Abilene + Germany50) · applied mlu | 1600 | -0.3173 | 0.80 | 0.06 | 7e-159 | 8.4e-158 * | -0.228 |
| applied overload framework (E) vs policy only (A) · SNDlib pooled (Abilene + Germany50) (McNemar; left-only 0, right-only 138) | 1600 | -0.0863 | – | – | 5.7e-42 | 1.7e-41 * | – |
| framework (E) vs policy only (A) · all unseen pooled (GEANT + Kdl + Abilene + Germany50, calibrated) · applied sat | 2404 | +0.1204 | 0.85 | 0.02 | 0 | 0 * | +0.439 |
| framework (E) vs policy only (A) · all unseen pooled (GEANT + Kdl + Abilene + Germany50, calibrated) · applied mlu | 2404 | -0.4513 | 0.81 | 0.06 | 5.7e-267 | 8e-266 * | -0.307 |
| applied overload framework (E) vs policy only (A) · all unseen pooled (GEANT + Kdl + Abilene + Germany50, calibrated) (McNemar; left-only 0, right-only 313) | 2404 | -0.1302 | – | – | 1.2e-94 | 9.3e-94 * | – |

Satisfaction: higher is better; MLU: lower is better. \* significant at 0.05 after Holm correction.

## Friedman test + Nemenyi post-hoc

Blocks are (topology, load mode, level) cells; each block value is the mean over the cell's TMs. `framework` = recommended_stable + improved twin + gate E + fallback.

| Block set | Test | Blocks | χ² | p | Mean ranks (1 = best) |
|---|---|---:|---:|---:|---|
| all_blocks_with_sndlib | without_LP_all_topologies|sat | 48 | 93.0 | 1.5e-18 | framework 1.84, shortest 2.82, equal 3.29, frozen_p1 3.46, actor_v2 4.67, actor_stable 4.92 |
| all_blocks_with_sndlib | without_LP_all_topologies|mlu | 48 | 81.5 | 4e-16 | framework 2.24, equal 2.33, shortest 3.23, frozen_p1 4.10, actor_v2 4.22, actor_stable 4.88 |
| all_blocks_with_sndlib | with_LP_excluding_Kdl|sat | 40 | 88.2 | 7.3e-17 | framework 2.40, lp 2.44, shortest 3.67, equal 4.15, frozen_p1 4.28, actor_v2 5.25, actor_stable 5.81 |
| all_blocks_with_sndlib | with_LP_excluding_Kdl|mlu | 40 | 133.2 | 2.7e-26 | lp 1.00, framework 3.17, equal 3.58, shortest 4.36, actor_v2 4.97, frozen_p1 5.15, actor_stable 5.76 |
| calibrated_blocks_with_sndlib | without_LP_all_topologies|sat | 28 | 45.4 | 1.2e-08 | framework 1.95, shortest 3.09, equal 3.30, frozen_p1 3.32, actor_v2 4.39, actor_stable 4.95 |
| calibrated_blocks_with_sndlib | without_LP_all_topologies|mlu | 28 | 44.2 | 2.1e-08 | framework 2.05, equal 2.57, shortest 3.38, actor_v2 4.02, frozen_p1 4.18, actor_stable 4.80 |
| calibrated_blocks_with_sndlib | with_LP_excluding_Kdl|sat | 24 | 58.0 | 1.2e-10 | lp 1.85, framework 2.71, shortest 4.10, equal 4.21, frozen_p1 4.21, actor_v2 5.00, actor_stable 5.92 |
| calibrated_blocks_with_sndlib | with_LP_excluding_Kdl|mlu | 24 | 80.4 | 3e-15 | lp 1.00, framework 2.96, equal 3.79, shortest 4.50, actor_v2 4.79, frozen_p1 5.25, actor_stable 5.71 |
| unseen_calibrated_blocks | without_LP_all_topologies|sat | 16 | 44.3 | 2.1e-08 | framework 1.44, shortest 2.97, equal 3.00, frozen_p1 3.56, actor_v2 4.75, actor_stable 5.28 |
| unseen_calibrated_blocks | without_LP_all_topologies|mlu | 16 | 34.7 | 1.7e-06 | framework 1.75, equal 2.50, shortest 3.38, actor_v2 4.09, frozen_p1 4.19, actor_stable 5.09 |
| unseen_calibrated_blocks | with_LP_excluding_Kdl|sat | 12 | 53.7 | 8.5e-10 | lp 1.25, framework 2.04, equal 3.88, shortest 4.08, frozen_p1 4.67, actor_v2 5.50, actor_stable 6.58 |
| unseen_calibrated_blocks | with_LP_excluding_Kdl|mlu | 12 | 46.3 | 2.5e-08 | lp 1.00, framework 2.46, equal 3.92, shortest 4.62, actor_v2 4.67, frozen_p1 5.33, actor_stable 6.00 |
| previous_without_sndlib | without_LP_all_topologies|sat | 40 | 74.8 | 1e-14 | framework 1.98, shortest 2.67, equal 3.33, frozen_p1 3.45, actor_v2 4.75, actor_stable 4.83 |
| previous_without_sndlib | without_LP_all_topologies|mlu | 40 | 78.9 | 1.4e-15 | equal 2.10, framework 2.38, shortest 3.09, frozen_p1 4.03, actor_v2 4.46, actor_stable 4.95 |
| previous_without_sndlib | with_LP_excluding_Kdl|sat | 32 | 60.6 | 3.4e-11 | framework 2.48, lp 2.75, shortest 3.45, equal 4.17, frozen_p1 4.22, actor_v2 5.25, actor_stable 5.67 |
| previous_without_sndlib | with_LP_excluding_Kdl|mlu | 32 | 110.4 | 1.7e-21 | lp 1.00, framework 3.33, equal 3.34, shortest 4.22, frozen_p1 5.06, actor_v2 5.22, actor_stable 5.83 |

`previous_without_sndlib` repeats `results/phase3_final/stats/friedman_nemenyi.json` for comparison.

