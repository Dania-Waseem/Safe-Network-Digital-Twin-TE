# GRL-TE reproduction (P1 baseline)

This folder reproduces GRL-TE from Lu, Tang, Ma and Xing (2025), *Graph-based reinforcement learning for
software-defined networking traffic engineering* (P1), which serves as the baseline of the project.

## Model

- The WAN is represented as a **link–path bipartite graph**. TopoFlowNet has **6 GINConv layers**, a
  path-collaboration MLP, a skip concatenation with \(h^0\) and a path embedding of dimension **7**.
- The actor heads map **28→4** (mean and log-std); the critic is **28→256→256→1**; training is
  **one-step A2C** with advantage \(A = r - V(s)\) and no discount.
- Training uses **3 episodes**, Adam, actor learning rate \(10^{-4}\) and critic learning rate \(10^{-3}\).
- Deployment applies the actor mean, then **ADMM** (\(\rho = 1\), 2 iterations on B4), then the constraint
  adjustment of eqs. 23–28.
- Traffic and capacities for B4, UsCarrier, Kdl and ASN come from **TEAL**; each node pair has 4 shortest
  candidate paths.

## Contents

| Path | Contents |
|---|---|
| `models/` | TopoFlowNet, actor, critic and the A2C model (`a2c.py`) |
| `training/` | Training loop and evaluation |
| `optimization/` | ADMM refinement (P1-order and Teal-order variants) and constraint adjustment |
| `preprocessing/` | Topology, traffic, gravity-model and candidate-path loading; bipartite graph construction |
| `experiments/` | B4 training (`b4.py`, `run_b4_seeds.py`), LP / ECMP / shortest-path baselines (`baselines.py`), link failures, and the multi-topology studies of Phase F |
| `config.py` | Hyperparameters, seeds and paths |
| `BASELINE_FREEZE/` | The frozen B4 protocol (`frozen_b4_config.py`, `P1_BASELINE_PROTOCOL_CARD.md`), its runner, the freeze decision, the ADMM variant comparison and the reproducibility report |
| `phase3_final/` | GRL-TE on the FIGRET GEANT 22/72 graph (`geant_figret.py`) and the P1 §5.5-comparable B4 link-failure table (`b4_link_failures.py`), with their results |
| `COMPLETE_P1_AUDIT/` | Component, mathematical and experimental audits of the reproduction |
| `*_REPORT.md`, `*AUDIT*.md`, `CORRECTIONS.md`, `FINAL_P1_*.md` | Reports of the reproduction phases (B–F), the corrections, the deviations from P1 and the methodology compliance |
| `results/` | Tables, graphs and logs of the reproduction phases |
| `data/` | Topologies, traffic matrices and candidate paths used by the baseline |

Checkpoints are stored in the repository-level `models/baseline_p1/` (Git LFS): `b4/` (10 archived
seeds), `b4_freeze/` (frozen seed 100), `geant_figret/` and `geant_figret_units/`. The GEANT 22/72
topology and traffic come from the repository-level `data/` folder.

## Deviations from the published description

P1 does not publish the following details, so the reproduction makes documented choices:

- The ADMM subproblem solver: closed forms are derived from eqs. 48–52.
- The traffic-matrix train/test cut: TEAL's 20/8/8 split of the 36 public B4 matrices.
- Input normalisation: the frozen protocol uses raw \(h^0\) (P1 eq. 5); ADMM works in raw units.
- The path set: k-shortest (Yen) paths rather than TEAL's edge-disjoint set; B4 has 4 simple paths for
  every pair.

## B4 results (10 seeds)

`CORRECTIONS.md` describes the audit fixes (ADMM, raw \(h^0\), \(r_t = F_{\text{total}}\), shared TopoFlowNet).

| Method | Demand satisfaction | Notes |
| --- | --- | --- |
| LP (ours) | **99.74%** | P1 LP = 99.83% |
| ECMP | 98.74% | |
| GRL-TE + ADMM (ours) | **97.44% ± 1.68%** | 10 seeds; P1 = 92.39% |
| GRL-TE, no ADMM | **96.51% ± 2.75%** | ADMM lift ≈ 0.93 pt (P1 B4 ablation 0.68) |
| P1 Teal (published) | 84.77% | not reproduced |

ADMM no longer diverges (previously 3.55× demand). The closed forms are **Teal-derived**, because P1
does not publish the argmins of (49)–(51); the numbers are therefore not an exact numerical match to
P1 Table 6. On the FIGRET GEANT 22/72 graph the reproduced GRL-TE reaches 94.44 ± 1.81 % (P1: 86.41 %),
and with three failed B4 links 80.61 % (P1: 77.25 %); see `phase3_final/results/`.
