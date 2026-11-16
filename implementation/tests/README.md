# Unit tests

The suite contains 98 tests on small fixtures and short runs; none of them performs a full experiment.

| File | Tests | Scope |
|---|---:|---|
| `test_preprocess.py` | 10 | Topology parsing, traffic validation, Yen paths (B4 matches the frozen P1 paths), bipartite graphs |
| `test_twin.py` | 3 | Twin physics and a one-step ensemble check on B4 |
| `test_twin_improvements.py` | 4 | Physics-anchored twin and conformal helpers |
| `test_safety.py` | 8 | Accept/reject cases of the original gate |
| `test_safety_gate_improvements.py` | 23 | Improved gate: UCB rule, safeguards, fallback selection, oracle labels |
| `test_safety_gate_eval_infra.py` | 9 | Policy backend, paired smoke run and decision logging |
| `test_experimental_protocol.py` | 9 | Metric denominators, pairing and failed-row handling |
| `test_framework.py` | 3 | Integrated pipeline |
| `test_general_policy.py` | 5 | Generalized actor and least-violating fallback controller |
| `test_train_policy.py` | 7 | Training controls, early stopping, checkpoint isolation |
| `test_scale_eval.py` | 3 | Scale evaluation and fallback classification |
| `test_robustness_eval.py` | 5 | Multi-seed robustness helpers |
| `test_ablation_study.py` | 5 | Audit helpers and offline ablation decision rules |
| `test_sndlib.py` | 4 | SNDlib native-format parser, demand-archive reader and stratified sampler |

The paired smoke test writes its outputs to `results/safety_gate/improvements/smoke_test_tmp/`.
