# E3 link failures: measured findings

Sources: `E3_REPORT.md`, `case_rows.jsonl` and `summary.json` in this folder (`python -m phase3_final.link_failure_eval`).
The P1 §5.5-comparable table is in `baseline_p1_reproduction/phase3_final/results/b4_link_failures/B4_P1_SECTION55_TABLE.md`
(`python -m phase3_final.b4_link_failures`).

Setup: 10 failure seeds × {0, 1, 2, 3} failed directed links (P1 §5.5 sampler). Paths are rebuilt per failure
case. B4 uses 8 val TMs, GEANT 50 test TMs, UsCarrier 1 toy TM; each at fixed {0.5, 1.0} and calibrated
{0.7, 0.9} load. Cases: B4 1,280, GEANT 8,000, UsCarrier 160.

## 1. The gate stays safe under failures

On every topology and failure count, the twin gates (D, E) and the oracle (F) have **zero unsafe accepts**, and
D, E and F agree on every case. Without a gate (A), the policy's unsafe-accept rate grows with failures:

| Topology | A unsafe accept (0 / 1 / 2 / 3 failed links) | D, E, F |
|---|---|---|
| B4 | 0.03 / 0.29 / 0.49 / 0.47 | 0 everywhere |
| GEANT | 1.00 / 0.99 / 0.97 / 0.97 | 0 everywhere |
| UsCarrier | 1.00 / 1.00 / 1.00 / 1.00 | 0 everywhere |

The twin is rebuilt on the failed graph (capacity 0 on failed links, re-routed candidate paths) and still
predicts loads accurately enough that no unsafe proposal is accepted.

## 2. Satisfaction under failures (P1 definition, %, mean over all cases)

| Topology | Policy only (A) 0 → 3 failed | Framework (E) 0 → 3 failed | Shortest 0 → 3 failed | LP (min-MLU) 0 → 3 failed |
|---|---|---|---|---|
| B4 | 99.96 → 97.37 | 99.90 → 97.29 | 99.28 → 96.97 | 99.98 → 98.05 |
| GEANT | 85.42 → 88.76 | 97.21 → 96.41 | 97.07 → 96.35 | 97.35 → 96.15 |
| UsCarrier | 65.72 → 66.70 | 93.70 → 91.20 | 93.70 → 91.20 | 87.41 → 86.39 |

- **B4.** The learned policy is already good on B4; the framework matches it (within 0.1 point) and stays safe.
- **GEANT and UsCarrier.** The framework improves satisfaction by about 8–12 points (GEANT) and 24.5–28 points
  (UsCarrier) over policy-only, because it rejects the policy's overloading proposals and falls back. Its
  satisfaction then equals or slightly exceeds shortest path. **The value comes from rejection and fallback, not
  from the policy.**
- The min-MLU LP minimises the worst link, not delivered traffic. Under heavy overload (UsCarrier) its P1
  satisfaction is therefore below shortest path; P1's LP is a max-throughput LP.

## 3. Comparison with P1 §5.5 (B4, frozen P1 baseline, own ADMM pipeline, fixed paths)

| Failed links | Frozen P1 GRL-TE (Teal ADMM) | LP (max throughput) | Shortest | P1 reported |
|---:|---:|---:|---:|---:|
| 0 | 97.44 ± 1.68 | 99.74 | 97.76 | 92.39 (Table 6, intact) |
| 3 | 80.61 ± 4.61 | 96.42 | 82.27 | 77.25 (§5.5) |

The reproduction's B4 GRL-TE loses 16.84 points from 0 to 3 failed links (10.68 from 1 to 3). P1 reports
90.80% → 77.25% (−13.55 points); P1's starting point is not assigned to a row because it does not match its
intact Table 6 value. The direction and order of magnitude match P1. The absolute level is higher because the
frozen baseline is already higher than P1 on intact B4 (97.44% vs 92.39%).

## 4. Caveats

- UsCarrier TMs are identical toy copies (1 unique TM). 11 of its 40 failure cases disconnect node pairs; up to
  1.2% of demand is unservable and counted as undelivered for every method.
- The experiment does not include Kdl or Cogentco link failures.
- The E3 framework uses fixed-scale and calibrated α from the *intact* topology, so failures add load on top of
  the nominal level.
