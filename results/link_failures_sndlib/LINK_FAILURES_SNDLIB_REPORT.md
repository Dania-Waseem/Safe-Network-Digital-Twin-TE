# Link failures on SNDlib Abilene and Germany50

Same protocol as the GEANT link-failure experiment (`results/phase3_final/e3_link_failures/`): failure cases from `data/scenarios/link_failures/<topology>.json` (P1 §5.5 sampler, seeds 100–1000, 0–3 failed directed links); candidate paths through a failed link are dropped and Yen k = 4 is recomputed for pairs left without a path; demand of pairs disconnected by the failure is unservable and counted as undelivered. 50 seeded TMs per topology × calibrated targets 0.5 / 0.7 / 0.9 / 1.1 (alpha from the intact topology) × 40 failure cases = 8,000 cases per topology. Frozen stack (policy `recommended_stable`, gate k = 1, frozen fallback).

GEANT was run at fixed scales 0.5 / 1.0 and calibrated 0.7 / 0.9; the `0.7+0.9` rows below are the levels shared with GEANT.

## Abilene — load levels: all (unservable demand by failed links — 0: 0.00%, 1: 0.02%, 2: 2.47%, 3: 6.28%)

| Method | Sat. 0 | Sat. 1 | Sat. 2 | Sat. 3 | MLU 0 | MLU 3 | Overload 0 | Overload 3 | Unsafe acc. 0/1/2/3 | False rej. given safe 0/1/2/3 | Fallback 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| LP (min-MLU) | 98.17 | 91.15 | 84.71 | 83.83 | 0.800 | 1.188 | 0.25 | 0.56 | –/–/–/– | –/–/–/– | –/–/–/– |
| shortest | 88.73 | 85.02 | 77.36 | 78.74 | 2.093 | 2.511 | 0.90 | 0.83 | –/–/–/– | –/–/–/– | –/–/–/– |
| ECMP (equal split) | 71.21 | 68.69 | 63.85 | 65.25 | 3.285 | 3.602 | 1.00 | 0.99 | –/–/–/– | –/–/–/– | –/–/–/– |
| frozen P1 actor | 75.40 | 67.18 | 64.55 | 68.17 | 2.522 | 3.130 | 0.95 | 0.96 | –/–/–/– | –/–/–/– | –/–/–/– |
| policy only (actor) | 66.29 | 66.05 | 66.80 | 71.42 | 2.339 | 2.805 | 0.96 | 0.90 | –/–/–/– | –/–/–/– | –/–/–/– |
| framework A (no gate) | 66.29 | 66.05 | 66.80 | 71.42 | 2.339 | 2.805 | 0.96 | 0.90 | 0.96/0.98/0.96/0.90 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 |
| framework D (mean-only gate) | 90.70 | 86.63 | 78.63 | 79.64 | 1.940 | 2.420 | 0.85 | 0.82 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.96/0.98/0.96/0.90 |
| framework E (gate k=1) | 90.70 | 86.63 | 78.63 | 79.64 | 1.940 | 2.420 | 0.85 | 0.82 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.96/0.98/0.96/0.90 |
| framework F (oracle) | 90.70 | 86.63 | 78.63 | 79.64 | 1.940 | 2.420 | 0.85 | 0.82 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.96/0.98/0.96/0.90 |

## Abilene — load levels: 0.7+0.9 (unservable demand by failed links — 0: 0.00%, 1: 0.02%, 2: 2.47%, 3: 6.28%)

| Method | Sat. 0 | Sat. 1 | Sat. 2 | Sat. 3 | MLU 0 | MLU 3 | Overload 0 | Overload 3 | Unsafe acc. 0/1/2/3 | False rej. given safe 0/1/2/3 | Fallback 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| LP (min-MLU) | 100.00 | 92.36 | 85.06 | 84.73 | 0.800 | 1.188 | 0.00 | 0.58 | –/–/–/– | –/–/–/– | –/–/–/– |
| shortest | 89.47 | 85.36 | 77.33 | 79.13 | 2.093 | 2.511 | 0.97 | 0.86 | –/–/–/– | –/–/–/– | –/–/–/– |
| ECMP (equal split) | 71.94 | 68.44 | 63.01 | 64.57 | 3.285 | 3.602 | 1.00 | 0.99 | –/–/–/– | –/–/–/– | –/–/–/– |
| frozen P1 actor | 75.83 | 66.14 | 63.28 | 68.07 | 2.534 | 3.141 | 0.95 | 0.98 | –/–/–/– | –/–/–/– | –/–/–/– |
| policy only (actor) | 63.79 | 64.06 | 65.68 | 71.25 | 2.341 | 2.804 | 0.97 | 0.94 | –/–/–/– | –/–/–/– | –/–/–/– |
| framework A (no gate) | 63.79 | 64.06 | 65.68 | 71.25 | 2.341 | 2.804 | 0.97 | 0.94 | 0.97/0.98/0.98/0.94 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 |
| framework D (mean-only gate) | 91.61 | 87.29 | 78.81 | 80.21 | 1.909 | 2.382 | 0.94 | 0.85 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.97/0.98/0.98/0.94 |
| framework E (gate k=1) | 91.61 | 87.29 | 78.81 | 80.21 | 1.909 | 2.382 | 0.94 | 0.85 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.97/0.98/0.98/0.94 |
| framework F (oracle) | 91.61 | 87.29 | 78.81 | 80.21 | 1.909 | 2.382 | 0.94 | 0.85 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.97/0.98/0.98/0.94 |

## Germany50 — load levels: all (unservable demand by failed links — 0: 0.00%, 1: 0.00%, 2: 0.00%, 3: 0.00%)

| Method | Sat. 0 | Sat. 1 | Sat. 2 | Sat. 3 | MLU 0 | MLU 3 | Overload 0 | Overload 3 | Unsafe acc. 0/1/2/3 | False rej. given safe 0/1/2/3 | Fallback 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| LP (min-MLU) | 98.91 | 98.79 | 98.79 | 97.37 | 0.800 | 0.900 | 0.25 | 0.37 | –/–/–/– | –/–/–/– | –/–/–/– |
| shortest | 91.80 | 91.75 | 90.79 | 89.83 | 1.474 | 1.607 | 0.77 | 0.81 | –/–/–/– | –/–/–/– | –/–/–/– |
| ECMP (equal split) | 94.87 | 94.80 | 94.59 | 93.62 | 1.166 | 1.249 | 0.65 | 0.68 | –/–/–/– | –/–/–/– | –/–/–/– |
| frozen P1 actor | 93.27 | 92.89 | 92.47 | 92.55 | 1.251 | 1.316 | 0.71 | 0.72 | –/–/–/– | –/–/–/– | –/–/–/– |
| policy only (actor) | 92.82 | 92.76 | 92.38 | 92.29 | 1.280 | 1.328 | 0.78 | 0.74 | –/–/–/– | –/–/–/– | –/–/–/– |
| framework A (no gate) | 92.82 | 92.76 | 92.38 | 92.29 | 1.280 | 1.328 | 0.78 | 0.74 | 0.78/0.74/0.76/0.74 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 |
| framework D (mean-only gate) | 94.88 | 94.80 | 94.57 | 93.47 | 1.176 | 1.287 | 0.65 | 0.66 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.78/0.74/0.76/0.74 |
| framework E (gate k=1) | 94.88 | 94.80 | 94.57 | 93.47 | 1.176 | 1.287 | 0.65 | 0.66 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.78/0.74/0.76/0.74 |
| framework F (oracle) | 94.88 | 94.80 | 94.57 | 93.47 | 1.176 | 1.287 | 0.65 | 0.66 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.78/0.74/0.76/0.74 |

## Germany50 — load levels: 0.7+0.9 (unservable demand by failed links — 0: 0.00%, 1: 0.00%, 2: 0.00%, 3: 0.00%)

| Method | Sat. 0 | Sat. 1 | Sat. 2 | Sat. 3 | MLU 0 | MLU 3 | Overload 0 | Overload 3 | Unsafe acc. 0/1/2/3 | False rej. given safe 0/1/2/3 | Fallback 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| LP (min-MLU) | 100.00 | 99.97 | 99.93 | 98.68 | 0.800 | 0.900 | 0.00 | 0.24 | –/–/–/– | –/–/–/– | –/–/–/– |
| shortest | 92.11 | 92.07 | 91.07 | 89.96 | 1.474 | 1.607 | 0.99 | 0.99 | –/–/–/– | –/–/–/– | –/–/–/– |
| ECMP (equal split) | 96.24 | 96.15 | 95.80 | 94.64 | 1.166 | 1.249 | 0.80 | 0.82 | –/–/–/– | –/–/–/– | –/–/–/– |
| frozen P1 actor | 94.41 | 93.86 | 93.36 | 93.36 | 1.247 | 1.315 | 0.86 | 0.89 | –/–/–/– | –/–/–/– | –/–/–/– |
| policy only (actor) | 93.80 | 93.70 | 93.26 | 93.07 | 1.280 | 1.328 | 0.98 | 0.91 | –/–/–/– | –/–/–/– | –/–/–/– |
| framework A (no gate) | 93.80 | 93.70 | 93.26 | 93.07 | 1.280 | 1.328 | 0.98 | 0.91 | 0.98/0.92/0.91/0.91 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 |
| framework D (mean-only gate) | 96.25 | 96.17 | 95.80 | 94.60 | 1.167 | 1.265 | 0.80 | 0.82 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.98/0.92/0.91/0.91 |
| framework E (gate k=1) | 96.25 | 96.17 | 95.80 | 94.60 | 1.167 | 1.265 | 0.80 | 0.82 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.98/0.92/0.91/0.91 |
| framework F (oracle) | 96.25 | 96.17 | 95.80 | 94.60 | 1.167 | 1.265 | 0.80 | 0.82 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.98/0.92/0.91/0.91 |

Satisfaction in %, P1 definition, counting unservable demand as undelivered. MLU capped at 1e6 for averaging. Overload = any link above capacity after the method's routing (for framework rows: after gate and fallback).

## Files

- `case_rows.jsonl` (one row per failure case × TM × load), `summary.json`, `config.json`
- `figures/link_failure_curves.{png,pdf}`
