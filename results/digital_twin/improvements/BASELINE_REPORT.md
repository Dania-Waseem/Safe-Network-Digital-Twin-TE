# Baseline Digital Twin reproduction

Original checkpoint: `C:\Users\hp\Pictures\CNET phase 3\phase3 implementation\results\twin\models\ensemble.pt`
SHA256: `7b74fee96cdf3aa71c78aaa185adbc24c99f262f342689a4ec78d97b7db73685`

Architecture: 5-member path-multiplier MLP ensemble; link load via scatter;
uncertainty = ensemble std; features = demand/max_capacity, action, hops/16.

## Per-topology snapshot (equal split, first matrix)

| Topology | Denom | MAE abs | MAE util | MAE rel load | max|mult-1| |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4 | 5e+03 | 0.685593 | 0.000137119 | 0.000295571 | 0.00144303 |
| UsCarrier | 5.85e+03 | 0.272331 | 0.000429832 | 0.000300831 | 0.00137055 |
| Cogentco | 1e+04 | 32.8316 | 0.00328316 | 0.000292722 | 0.00137413 |
| GEANT | 1e+10 | 374345 | 0.000126256 | 0.000296921 | 0.00144625 |
| Kdl | 6.75e+04 | 3.55359 | 0.000376101 | 0.000339411 | 0.00136137 |

## Full evaluation summaries

### Val seen (B4/UsCarrier/Cogentco, scales 0.25–1)

- MAE abs=6.70294, RMSE=10.1297, R²=1, max|err|=40.0116
- MAE util=0.000754602, coverage@2σ=1.000, std–|err| corr=1.000

### Test GEANT (unseen)

- MAE abs=264828, RMSE=375966, R²=1, max|err|=1.22663e+06
- MAE util=8.03749e-05, coverage@2σ=1.000, std–|err| corr=1.000

### Test Kdl (unseen)

- MAE abs=1.6084, RMSE=3.17967, R²=1, max|err|=29.9712
- MAE util=0.000198617, coverage@2σ=0.977, std–|err| corr=0.782
