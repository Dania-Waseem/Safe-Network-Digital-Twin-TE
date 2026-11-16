# Twin-error sweep on SNDlib Abilene and Germany50

Same error model as the GEANT sweep (`results/phase3_final/e4_twin_error/`): member-wise perturbation of the improved twin, `member_j(e) ← max(0, member_j(e)·(1 + σ z_j(e)))`, z ~ N(0, 1) with seed 4040 per (topology, case, replicate), the same z for every σ and every gate, 3 replicates for σ > 0. μ and std are recomputed from the perturbed members.

Gates on identical inputs: A no gate · D mean-only (k = 0) · Dm mean-only with the tighter utilisation limit L chosen on dev validation in the GEANT sweep (L = 1.0 at σ=0.0, 0.95 at σ=0.05, 0.935 at σ=0.1, 0.85 at σ=0.2, 0.745 at σ=0.4; not refitted) · E μ + 1·c·std with the frozen conformal scale c = 0.9545 · F physics oracle. Rejections use the frozen fallback.

Cases: 200 TMs × 4 calibrated targets = 800 per topology (× 3 replicates for σ > 0).

## Abilene

| σ | Variant | Accept | Unsafe acc. given unsafe | False rej. given safe | Applied MLU | Applied overload | Applied sat. |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0 | A | 1.000 | 1.000 | 0.000 | 2.446 | 0.975 | 0.6551 |
| 0.0 | D | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.0 | Dm | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.0 | E | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.0 | F | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.05 | A | 1.000 | 1.000 | 0.000 | 2.446 | 0.975 | 0.6551 |
| 0.05 | D | 0.028 | 0.003 | 0.000 | 1.898 | 0.896 | 0.9109 |
| 0.05 | Dm | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.05 | E | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.05 | F | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.1 | A | 1.000 | 1.000 | 0.000 | 2.446 | 0.975 | 0.6551 |
| 0.1 | D | 0.028 | 0.003 | 0.000 | 1.898 | 0.896 | 0.9108 |
| 0.1 | Dm | 0.025 | 0.000 | 0.017 | 1.900 | 0.897 | 0.9105 |
| 0.1 | E | 0.024 | 0.000 | 0.050 | 1.901 | 0.897 | 0.9104 |
| 0.1 | F | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.2 | A | 1.000 | 1.000 | 0.000 | 2.446 | 0.975 | 0.6551 |
| 0.2 | D | 0.028 | 0.004 | 0.017 | 1.898 | 0.897 | 0.9108 |
| 0.2 | Dm | 0.017 | 0.000 | 0.333 | 1.908 | 0.905 | 0.9096 |
| 0.2 | E | 0.017 | 0.000 | 0.317 | 1.908 | 0.904 | 0.9096 |
| 0.2 | F | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |
| 0.4 | A | 1.000 | 1.000 | 0.000 | 2.446 | 0.975 | 0.6551 |
| 0.4 | D | 0.024 | 0.002 | 0.133 | 1.902 | 0.900 | 0.9103 |
| 0.4 | Dm | 0.007 | 0.000 | 0.733 | 1.915 | 0.915 | 0.9089 |
| 0.4 | E | 0.010 | 0.000 | 0.600 | 1.913 | 0.911 | 0.9090 |
| 0.4 | F | 0.025 | 0.000 | 0.000 | 1.900 | 0.896 | 0.9105 |

## Germany50

| σ | Variant | Accept | Unsafe acc. given unsafe | False rej. given safe | Applied MLU | Applied overload | Applied sat. |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0 | A | 1.000 | 1.000 | 0.000 | 1.257 | 0.740 | 0.9305 |
| 0.0 | D | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |
| 0.0 | Dm | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |
| 0.0 | E | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |
| 0.0 | F | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |
| 0.05 | A | 1.000 | 1.000 | 0.000 | 1.257 | 0.740 | 0.9305 |
| 0.05 | D | 0.267 | 0.016 | 0.021 | 1.168 | 0.656 | 0.9491 |
| 0.05 | Dm | 0.239 | 0.000 | 0.082 | 1.166 | 0.646 | 0.9491 |
| 0.05 | E | 0.243 | 0.001 | 0.067 | 1.166 | 0.647 | 0.9491 |
| 0.05 | F | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |
| 0.1 | A | 1.000 | 1.000 | 0.000 | 1.257 | 0.740 | 0.9305 |
| 0.1 | D | 0.274 | 0.032 | 0.037 | 1.168 | 0.664 | 0.9490 |
| 0.1 | Dm | 0.239 | 0.002 | 0.085 | 1.166 | 0.647 | 0.9491 |
| 0.1 | E | 0.237 | 0.000 | 0.090 | 1.166 | 0.646 | 0.9491 |
| 0.1 | F | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |
| 0.2 | A | 1.000 | 1.000 | 0.000 | 1.257 | 0.740 | 0.9305 |
| 0.2 | D | 0.285 | 0.051 | 0.051 | 1.170 | 0.675 | 0.9489 |
| 0.2 | Dm | 0.190 | 0.001 | 0.269 | 1.164 | 0.647 | 0.9491 |
| 0.2 | E | 0.200 | 0.001 | 0.232 | 1.164 | 0.647 | 0.9491 |
| 0.2 | F | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |
| 0.4 | A | 1.000 | 1.000 | 0.000 | 1.257 | 0.740 | 0.9305 |
| 0.4 | D | 0.266 | 0.058 | 0.141 | 1.169 | 0.672 | 0.9488 |
| 0.4 | Dm | 0.068 | 0.001 | 0.740 | 1.159 | 0.647 | 0.9491 |
| 0.4 | E | 0.076 | 0.001 | 0.710 | 1.159 | 0.647 | 0.9491 |
| 0.4 | F | 0.260 | 0.000 | 0.000 | 1.168 | 0.646 | 0.9491 |

## sndlib_pooled

| σ | Variant | Accept | Unsafe acc. given unsafe | False rej. given safe | Applied MLU | Applied overload | Applied sat. |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0 | A | 1.000 | 1.000 | 0.000 | 1.851 | 0.858 | 0.7928 |
| 0.0 | D | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |
| 0.0 | Dm | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |
| 0.0 | E | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |
| 0.0 | F | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |
| 0.05 | A | 1.000 | 1.000 | 0.000 | 1.851 | 0.858 | 0.7928 |
| 0.05 | D | 0.147 | 0.009 | 0.019 | 1.533 | 0.776 | 0.9300 |
| 0.05 | Dm | 0.132 | 0.000 | 0.075 | 1.533 | 0.771 | 0.9298 |
| 0.05 | E | 0.134 | 0.000 | 0.061 | 1.533 | 0.772 | 0.9298 |
| 0.05 | F | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |
| 0.1 | A | 1.000 | 1.000 | 0.000 | 1.851 | 0.858 | 0.7928 |
| 0.1 | D | 0.151 | 0.015 | 0.034 | 1.533 | 0.780 | 0.9299 |
| 0.1 | Dm | 0.132 | 0.001 | 0.079 | 1.533 | 0.772 | 0.9298 |
| 0.1 | E | 0.130 | 0.000 | 0.086 | 1.534 | 0.772 | 0.9298 |
| 0.1 | F | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |
| 0.2 | A | 1.000 | 1.000 | 0.000 | 1.851 | 0.858 | 0.7928 |
| 0.2 | D | 0.156 | 0.024 | 0.048 | 1.534 | 0.786 | 0.9298 |
| 0.2 | Dm | 0.104 | 0.000 | 0.275 | 1.536 | 0.776 | 0.9294 |
| 0.2 | E | 0.109 | 0.000 | 0.240 | 1.536 | 0.775 | 0.9294 |
| 0.2 | F | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |
| 0.4 | A | 1.000 | 1.000 | 0.000 | 1.851 | 0.858 | 0.7928 |
| 0.4 | D | 0.145 | 0.026 | 0.140 | 1.536 | 0.786 | 0.9295 |
| 0.4 | Dm | 0.037 | 0.000 | 0.740 | 1.537 | 0.781 | 0.9290 |
| 0.4 | E | 0.043 | 0.000 | 0.700 | 1.536 | 0.779 | 0.9291 |
| 0.4 | F | 0.142 | 0.000 | 0.000 | 1.534 | 0.771 | 0.9298 |

## Exact McNemar tests (Holm-corrected within this family)

| Comparison (left vs right) | pairs | left-only | right-only | p | p (Holm) |
|---|---:|---:|---:|---:|---:|
| unsafe accept D vs E · Abilene · sigma=0.05 | 2400 | 6 | 0 | 0.031 | 0.562 |
| false reject D vs E · Abilene · sigma=0.05 | 2400 | 0 | 0 | 1.000 | 1.000 |
| unsafe accept Dm vs E · Abilene · sigma=0.05 | 2400 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E · Abilene · sigma=0.05 | 2400 | 0 | 0 | 1.000 | 1.000 |
| unsafe accept D vs E · Abilene · sigma=0.1 | 2400 | 6 | 0 | 0.031 | 0.562 |
| false reject D vs E · Abilene · sigma=0.1 | 2400 | 0 | 3 | 0.250 | 1.000 |
| unsafe accept Dm vs E · Abilene · sigma=0.1 | 2400 | 1 | 0 | 1.000 | 1.000 |
| false reject Dm vs E · Abilene · sigma=0.1 | 2400 | 0 | 2 | 0.500 | 1.000 |
| unsafe accept D vs E · Abilene · sigma=0.2 | 2400 | 9 | 0 | 0.004 | 0.082 |
| false reject D vs E · Abilene · sigma=0.2 | 2400 | 0 | 18 | 7.6e-06 | 0.00018 * |
| unsafe accept Dm vs E · Abilene · sigma=0.2 | 2400 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E · Abilene · sigma=0.2 | 2400 | 4 | 3 | 1.000 | 1.000 |
| unsafe accept D vs E · Abilene · sigma=0.4 | 2400 | 5 | 0 | 0.062 | 0.938 |
| false reject D vs E · Abilene · sigma=0.4 | 2400 | 0 | 28 | 7.5e-09 | 1.9e-07 * |
| unsafe accept Dm vs E · Abilene · sigma=0.4 | 2400 | 0 | 0 | 1.000 | 1.000 |
| false reject Dm vs E · Abilene · sigma=0.4 | 2400 | 8 | 0 | 0.008 | 0.156 |
| unsafe accept D vs E · Germany50 · sigma=0.05 | 2400 | 27 | 0 | 1.5e-08 | 3.6e-07 * |
| false reject D vs E · Germany50 · sigma=0.05 | 2400 | 0 | 29 | 3.7e-09 | 1e-07 * |
| unsafe accept Dm vs E · Germany50 · sigma=0.05 | 2400 | 0 | 2 | 0.500 | 1.000 |
| false reject Dm vs E · Germany50 · sigma=0.05 | 2400 | 10 | 1 | 0.012 | 0.223 |
| unsafe accept D vs E · Germany50 · sigma=0.1 | 2400 | 56 | 0 | 2.8e-17 | 8.6e-16 * |
| false reject D vs E · Germany50 · sigma=0.1 | 2400 | 0 | 33 | 2.3e-10 | 6.8e-09 * |
| unsafe accept Dm vs E · Germany50 · sigma=0.1 | 2400 | 3 | 0 | 0.250 | 1.000 |
| false reject Dm vs E · Germany50 · sigma=0.1 | 2400 | 2 | 5 | 0.453 | 1.000 |
| unsafe accept D vs E · Germany50 · sigma=0.2 | 2400 | 89 | 0 | 3.2e-27 | 1.1e-25 * |
| false reject D vs E · Germany50 · sigma=0.2 | 2400 | 0 | 113 | 1.9e-34 | 7.1e-33 * |
| unsafe accept Dm vs E · Germany50 · sigma=0.2 | 2400 | 1 | 2 | 1.000 | 1.000 |
| false reject Dm vs E · Germany50 · sigma=0.2 | 2400 | 40 | 17 | 0.003 | 0.070 |
| unsafe accept D vs E · Germany50 · sigma=0.4 | 2400 | 101 | 0 | 7.9e-31 | 2.8e-29 * |
| false reject D vs E · Germany50 · sigma=0.4 | 2400 | 0 | 355 | 2.7e-107 | 1.1e-105 * |
| unsafe accept Dm vs E · Germany50 · sigma=0.4 | 2400 | 1 | 1 | 1.000 | 1.000 |
| false reject Dm vs E · Germany50 · sigma=0.4 | 2400 | 55 | 36 | 0.059 | 0.937 |
| unsafe accept D vs E · SNDlib pooled · sigma=0.05 | 4800 | 33 | 0 | 2.3e-10 | 6.8e-09 * |
| false reject D vs E · SNDlib pooled · sigma=0.05 | 4800 | 0 | 29 | 3.7e-09 | 1e-07 * |
| unsafe accept D vs E · SNDlib pooled · sigma=0.1 | 4800 | 62 | 0 | 4.3e-19 | 1.4e-17 * |
| false reject D vs E · SNDlib pooled · sigma=0.1 | 4800 | 0 | 36 | 2.9e-11 | 8.7e-10 * |
| unsafe accept D vs E · SNDlib pooled · sigma=0.2 | 4800 | 98 | 0 | 6.3e-30 | 2.1e-28 * |
| false reject D vs E · SNDlib pooled · sigma=0.2 | 4800 | 0 | 131 | 7.3e-40 | 2.8e-38 * |
| unsafe accept D vs E · SNDlib pooled · sigma=0.4 | 4800 | 106 | 0 | 2.5e-32 | 8.9e-31 * |
| false reject D vs E · SNDlib pooled · sigma=0.4 | 4800 | 0 | 383 | 1e-115 | 4.1e-114 * |

## Proposal margin

| Topology | Cases | Unsafe share | True MLU in [0.9, 1.1] | median policy ms | median twin ms |
|---|---:|---:|---:|---:|---:|
| Abilene | 800 | 0.97 | 0.01 | 3.8 | 1.8 |
| Germany50 | 800 | 0.74 | 0.20 | 22.5 | 7.0 |

## Files

- `sweep.csv`, `paired_rows.csv.gz`, `summary.json`, `config.json`
- `figures/twin_error_curves.{png,pdf}`; `cache/` (member stacks, not committed)
