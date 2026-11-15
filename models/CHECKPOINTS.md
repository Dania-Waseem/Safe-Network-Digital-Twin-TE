# Model checkpoints

Every `.pt` checkpoint of the project is in this folder and stored with **Git LFS**.
Paths in code come from `implementation/repo_paths.py` (`MODELS`). The processed graph tensors (`data/processed/graph_data/*.pt`) are data, not checkpoints, and stay in `data/`.

| Folder | Contents | Written by | Read by |
|---|---|---|---|
| `models/twin/` | Original 5-member ensemble twin | `implementation/twin/train.py` | safety gates B/C, framework, tests |
| `models/twin_improved/` | Physics-anchored ensemble twin, with and without conformal scale c = 0.955 | `implementation/twin/improvements/run_pipeline.py` | gates D/E, all Phase 3 final experiments |
| `models/twin_degraded/` | Degraded twins trained on 10/25/50 % of the data (E4) | `implementation/phase3_final/train_degraded_twin.py` | `twin_error_sweep.py` |
| `models/policy/` | GNN routing policies: `general_grl.pt` (3-epoch baseline), `improved/recommended_stable.pt` (frozen framework policy), `improved/best_checkpoint.pt`, and one `best.pt` per search run | `implementation/framework/train_policy.py` | safety backend, E2–E4, ablations, statistics |
| `models/policy_v2/` | Policy v2: calibrated-load training + normalised validation score (not selected) | `implementation/phase3_final/train_policy_v2.py` | `holdout_af.py`, `generalization_gap.py`, `link_failure_eval.py` |
| `models/baseline_p1/b4/` | P1 GRL-TE on B4, 10 seeds (archived reproduction) | `baseline_p1_reproduction/experiments/run_b4_seeds.py` | `BASELINE_FREEZE/run_b4_baseline_freeze.py`, `phase3_final/b4_link_failures.py` |
| `models/baseline_p1/b4_freeze/` | Frozen P1 B4 checkpoint (seed 100) used as the single-topology policy ablation | `BASELINE_FREEZE/run_b4_baseline_freeze.py` | `implementation/framework/policy_adapter.py`, ablations (d), E3 |
| `models/baseline_p1/geant_figret/` | P1 GRL-TE on FIGRET GEANT 22/72, 10 seeds, raw h⁰ (headline channel) | `baseline_p1_reproduction/phase3_final/geant_figret.py` | same script (`--report-only`) |
| `models/baseline_p1/geant_figret_units/` | P1 GRL-TE on FIGRET GEANT 22/72, 10 seeds, h⁰ in FIGRET units (sensitivity) | `geant_figret.py --figret-units` | same script |

## Files

| Path | Size (KB) | SHA256 |
|---|---:|---|
| `models/baseline_p1/b4/grl_te_b4_seed100.pt` | 324 | `e171f176301744c84bdd9857126be1b489df66f8e6c75cc901425e621a26b2bc` |
| `models/baseline_p1/b4/grl_te_b4_seed1000.pt` | 324 | `cef178b3c196685243627b670e74d2c6db6f53886ceaed42480f8f5a44568154` |
| `models/baseline_p1/b4/grl_te_b4_seed200.pt` | 324 | `10cfa3dba0a23a57565f07d9c5681db022a7e80ba8b76a60302027183612a8bd` |
| `models/baseline_p1/b4/grl_te_b4_seed300.pt` | 324 | `f9778edd67cf57437f71e1d8d6e91845566181f723b35aca55f9c5601c8e989e` |
| `models/baseline_p1/b4/grl_te_b4_seed400.pt` | 324 | `11df141da87687ea079ac321690085dcbb265edde9a7c3fe1b125fc449911d8f` |
| `models/baseline_p1/b4/grl_te_b4_seed500.pt` | 324 | `3d43d201f15548991005b66c49fddc5b9282f08ebe2a05131a471556f8c068d4` |
| `models/baseline_p1/b4/grl_te_b4_seed600.pt` | 324 | `6aa5637f4518d2c6ce587beefaed1bc06e8b6ba30b5485e324c1c63c21cfa1fa` |
| `models/baseline_p1/b4/grl_te_b4_seed700.pt` | 324 | `97f4de1e4541c5a61bf2b3538fe7894cf9afedc99c2091c371e9d90024a23827` |
| `models/baseline_p1/b4/grl_te_b4_seed800.pt` | 324 | `9455bec09f52e4f5155b8ae1508997a1dc14125dff08bd2ebb8d39288856297b` |
| `models/baseline_p1/b4/grl_te_b4_seed900.pt` | 324 | `27874d064f237c9766d24b029a3b2514816b87ec8d3344cc418bc8bfd407f31b` |
| `models/baseline_p1/b4_freeze/grl_te_b4_freeze_seed100.pt` | 325 | `11fdb7019a8b5138a6f065d6bb1e3a430b7205ded4f091deafb243efd56d1896` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed100.pt` | 325 | `6137586e7784ef6a5e52b28d8fd20165717432be8ed2f755d62a4ff0b258158d` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed1000.pt` | 325 | `18d34b571c5e90b56a411404fb91c92bd7f33c21166f2db6578f4f7034426e6f` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed200.pt` | 325 | `0e21550848ec70e17b1420574bb6d1004e78be4d80268fee2464576660e79e82` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed300.pt` | 325 | `2a20c07781b0a63c0f7258ecaab2d9138caab3c277a79e18f3c11b69137d0073` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed400.pt` | 325 | `364835436f88b2377d9d6a051c811f11a6cdd839a5d3544b3c0a101c8570d926` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed500.pt` | 325 | `ffc942448a83b3083e71ad91b35c5cfa994bdefbec923dd6ca0973c6c1660669` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed600.pt` | 325 | `7835105f4a338605a89a0e860710412bcd5386d128ee60198876cf34f63311cf` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed700.pt` | 325 | `10e7eb3addf39c34e7699c86afd8757bb21a4b7ca4e49ce6f235612b176f89e2` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed800.pt` | 325 | `5a841a612be1083312f1dd0b67173657b9ddcc3091b73d29508a38cc0ef4bf4d` |
| `models/baseline_p1/geant_figret/grl_te_geant_figret_seed900.pt` | 325 | `26d18269275a9cd3329e7a1c40f4dcf739edddf6b9324cf1f26be9f0d5d6992d` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed100.pt` | 325 | `2f4ff20e761006898a64e2c59c3dec2099485611302b7a45d086e5dc415463dd` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed1000.pt` | 325 | `5539f8f35e964a8f5eb6cb743c9cba4bbb570acc37dc4883cd63c13704d356de` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed200.pt` | 325 | `e6e9c7648e23cec0b027418fa1a7d8335903f546d94a73fd36e75366721564c8` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed300.pt` | 325 | `ee2ce9991a26aa459a5ba54b1b0e18748ba99faed06d07d5c997a868add63ac4` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed400.pt` | 325 | `5692996dbd916f82babf1d8295023515bc70e94f80718675e71f8de3fcf500e6` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed500.pt` | 325 | `5e7034b410e06d71f2974e2a1bb51b5eb2cfc2f89cda793ad5a45dfde621bb21` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed600.pt` | 325 | `2d02fd8c3a7917b3061b01693620d467e95427f275be9d99d3a410a5b1a4f1b2` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed700.pt` | 325 | `9daa105df3779be8402341562e06ac2ecc30b27ff0df6c825de9c980f93758e7` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed800.pt` | 325 | `1d0ae5c43726356cd7e9910e5ab14c24edc3bc8cc1abba17e5d0dc9035c982df` |
| `models/baseline_p1/geant_figret_units/grl_te_geant_figret_seed900.pt` | 325 | `243b26380b7aa832619e1970702a5e04ba12da9043a78dc71a913498de4c9131` |
| `models/policy/general_grl.pt` | 324 | `fd4561f5ab1c85ba12d7dffed313362f651dda6318317b5dee084cf9306fcb22` |
| `models/policy/improved/best_checkpoint.pt` | 325 | `4130f64fa2639d55c5907b8196ce05b943ec7f7fef0c6d37483f45c0077cc1e0` |
| `models/policy/improved/recommended_stable.pt` | 322 | `ec4f6ae67eabcd5f64cfd4998e49616a713469bc0ddc8c2c7232929f680380a8` |
| `models/policy/improved/runs/lr1e4_bs1_e20/best.pt` | 322 | `878506be8f8a7540482f412adff22dbf02acbc51a45ca629acee15dc7ab9411a` |
| `models/policy/improved/runs/lr1e4_bs4_e16/best.pt` | 322 | `dbef1957cce5f72e7adaa83c83a782dbbe2f3cf3f71672661a9ac53e316f13d0` |
| `models/policy/improved/runs/lr3e4_bs2_e16/best.pt` | 322 | `3310fd777fcc82229f8b8f4dc2d1906849c1021e2860a38b18d653eabd45798f` |
| `models/policy/improved/runs/lr5e5_bs1_e24/best.pt` | 322 | `ec4f6ae67eabcd5f64cfd4998e49616a713469bc0ddc8c2c7232929f680380a8` |
| `models/policy/improved/runs/timing_smoke/best.pt` | 322 | `fea1b6177961e17bbd763f1f046687724314710d562710b3251a992bb09eac92` |
| `models/policy_v2/v2_calload_normscore/best.pt` | 322 | `ef4bcbe39c638f77d1969a52ab88334f7c991aa9c3c07cf29ef7677d5cbf28a4` |
| `models/policy_v2/v2_calload_normscore/last.pt` | 388 | `6cbcc16a3e36f4098dfbaf62ad17fe18f1858fb357f1f3c4e8d231fa4d038120` |
| `models/twin/ensemble.pt` | 35 | `7b74fee96cdf3aa71c78aaa185adbc24c99f262f342689a4ec78d97b7db73685` |
| `models/twin/member_0.pt` | 8 | `431cfe382623853fa4be6f7ae3f79f1b95efdd7997232e72dd720b652e8b49cf` |
| `models/twin/member_1.pt` | 8 | `e0b63aa789a630bde114e77232df64b5fa055b105dea6059b44a3c311ffb4485` |
| `models/twin/member_2.pt` | 8 | `302d96573f3a413fe9f174e1518756dd4ee8d5c7e6aec19e3b14f1bb3a88b5f5` |
| `models/twin/member_3.pt` | 8 | `da7e1a38fe5bf7001527e97e09efa597fe65033b264b517a46d013d77bbb346a` |
| `models/twin/member_4.pt` | 8 | `72dbdb62b242f6cf6f0df93435a09fc46eea5c642e58b43e0843bf31b6e03064` |
| `models/twin_degraded/ensemble_degraded_f10.pt` | 37 | `c03c7e1c60cc0a8a86ab8d54b6882fdc9d7afacc45195c407f5367e413c57a2d` |
| `models/twin_degraded/ensemble_degraded_f25.pt` | 37 | `19aafe59ffa625f553afc8a151b8902a425abb8ee5f3d2c383c2bd70882c55dc` |
| `models/twin_degraded/ensemble_degraded_f50.pt` | 37 | `3fd97018dc1d7aa7ef64b0b2f93d1ca6ee6e1229dac39c6c3ecd46eaa5c2306f` |
| `models/twin_improved/ensemble_physics_anchored.pt` | 38 | `466ac9dcd89d187199dd5483bc74dc66a85e76bd9e69e6a7a45eeddf9ef84633` |
| `models/twin_improved/ensemble_physics_anchored_conformal.pt` | 39 | `6833e4cfdb055e67cfa0d89cbe0d28ec946e361bd0ac21100f8a9eb148b46a3a` |
