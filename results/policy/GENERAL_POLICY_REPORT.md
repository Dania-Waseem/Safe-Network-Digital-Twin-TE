# Generalized policy and improved fallback

The new actor was trained on the B4, UsCarrier, and Cogentco train splits.
Validation matrices selected the checkpoint. GEANT and Kdl were not used until this evaluation.
The frozen B4 checkpoint was not loaded or modified.

| Topology | Scale | Accept | Unsafe accept | False reject | Fallback | Unavoidable | MLU new actor | MLU applied | MLU shortest | MLU frozen actor | Policy s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 | 0.25 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.288 | 0.288 | 0.275 | 0.306 | 0.01 |
| B4 | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.577 | 0.577 | 0.55 | 0.608 | 0.01 |
| B4 | 0.75 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.871 | 0.871 | 0.825 | 0.901 | 0.01 |
| B4 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.16 | 1.06 | 1.1 | 1.2 | 0.01 |
| Cogentco | 0.25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 16.6 | 16.7 | 16.7 | 16.4 | 1.31 |
| Cogentco | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 33.2 | 33.3 | 33.3 | 32.7 | 0.84 |
| Cogentco | 0.75 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 49.8 | 50 | 50 | 49.1 | 0.86 |
| Cogentco | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 66.4 | 66.7 | 66.7 | 65.5 | 0.80 |
| GEANT | 0.25 | 0.50 | 0.00 | 0.00 | 0.50 | 0.50 | 1.09 | 0.937 | 1.16 | 1.13 | 0.01 |
| GEANT | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 2.18 | 1.8 | 2.31 | 2.23 | 0.01 |
| GEANT | 0.75 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 3.28 | 3.2 | 3.47 | 3.35 | 0.01 |
| GEANT | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 4.37 | 4.62 | 4.62 | 4.42 | 0.01 |
| Kdl | 0.25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 4.29 | 2.79 | 2.79 | 4.28 | 110.97 |
| Kdl | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 8.58 | 5.57 | 5.57 | 8.55 | 125.90 |
| Kdl | 0.75 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 12.9 | 8.36 | 8.36 | 12.8 | 114.85 |
| Kdl | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 17.2 | 11.1 | 11.1 | 17.1 | 101.14 |
| UsCarrier | 0.25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 3.24 | 1.87 | 1.87 | 2.83 | 1.03 |
| UsCarrier | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 6.48 | 3.74 | 3.74 | 5.57 | 0.58 |
| UsCarrier | 0.75 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 9.71 | 5.61 | 5.61 | 7.99 | 0.62 |
| UsCarrier | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 13 | 7.48 | 7.48 | 10.2 | 0.60 |

Applied MLU is the new actor when the gate accepts it. Otherwise it is the
feasible fallback, or the least-violating candidate when every fallback overloads.
Frozen-actor MLU is the earlier identical-condition run, not a new frozen-file execution.

## Training

- Checkpoint: `C:\Users\hp\Pictures\CNET phase 3\phase3 implementation\results\policy\general_grl.pt`
- Selection metric: 7.001899778842926

## Measured comparison with Step 7 / frozen B4 actor

Evidence from the identical 28 decisions (same matrices and scales as `results/framework/extended/`):

- **Gate safety:** unsafe accepts 0, false rejects 0. Seven proposals accepted, all under capacity.
- **B4 (seen):** new actor MLU is lower than the frozen actor at every scale (for example 0.288 vs 0.306 at 0.25×). At 0.25–0.75× the gate accepts the new actor. At native scale both actors are infeasible; the improved fallback applied **equal** on matrix 24 (MLU 1.006) instead of shortest-path (1.081).
- **GEANT (unseen):** on matrix 0 the new actor beats the frozen actor at every scale and is accepted at 0.25× (0.710 vs frozen 0.764, vs shortest 0.810). When rejected, least-violating fallback chose **equal** on several GEANT cases (for example 0.5× applied 1.262 vs shortest 1.620). That is a real change from Step 7, which always applied shortest-path last resort.
- **UsCarrier / Kdl:** the new actor does **not** beat the frozen actor. Applied MLU still follows shortest-path after rejection. No generalization claim is made for those graphs.
- **Cogentco:** traffic remains far above capacity even at 0.25×. Actor MLU is comparable to the frozen run; overload is unavoidable under the candidate set.

Applied-action counts: `p1_general` 7, `shortest` 16, `equal` 5.

## Remaining limits

- Three epochs and two train matrices on UsCarrier and Cogentco are a short run, not a full P1 A2C schedule.
- The reward is differentiable overload and MLU, not P1's ADMM-adjusted satisfaction.
- Zero-shot gains are measured on GEANT matrix 0 at low scale. UsCarrier and Kdl still look like the frozen-actor failure mode.
- ASN was not trained or evaluated.
- The twin was not retrained.
