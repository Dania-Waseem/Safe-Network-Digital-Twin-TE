# Safe Network Digital Twin for Traffic Engineering

This repository contains an AI-driven network digital twin for **safe and generalizable decision
testing** in traffic engineering (TE), together with a reproduction of the GRL-TE baseline (P1:
Lu, Tang, Ma and Xing, *Graph-based reinforcement learning for software-defined networking traffic
engineering*, J. King Saud Univ. – Comput. Inf. Sci. 37:119, 2025).

Authors: Dania Waseem (23I-2622), Taiba Tariq (23I-2618).

## Research problem

Learned TE policies, such as graph reinforcement-learning routers, can propose routings that overload
links, and they are usually evaluated on the topologies they were trained on. Applying such a
proposal directly to a production network is unsafe, and good results on seen topologies do not show
that a policy generalizes. The project studies whether a digital twin that predicts link utilisation,
with calibrated uncertainty, can screen a policy's proposals before deployment: unsafe proposals are
rejected and replaced by a safe fallback, both on the training topologies and on unseen ones, under
link failures and under twin error.

## Method

Each decision follows four stages (see the architecture diagrams in `diagrams/`):

1. **Policy.** A graph neural network policy proposes, for every origin–destination demand, a split
   over k = 4 Yen candidate paths. It is trained on B4, UsCarrier and Cogentco.
2. **Digital twin.** A five-member ensemble predicts per-link utilisation for the proposed routing. The
   improved twin is *physics-anchored*: it predicts a residual on top of the analytical flow model, and
   its ensemble standard deviation is scaled by a split-conformal factor (c = 0.955) for 90 % coverage.
3. **Safety gate.** The proposal is accepted only if the twin's upper confidence bound respects
   capacity on every link, μ + kσ ≤ capacity (k = 1, utilisation limit 1.0). Six gate variants are
   compared: **A** no gate, **B/C** original twin with mean-only / uncertainty-aware bound,
   **D/E** improved twin with mean-only / uncertainty-aware bound, and **F** a physics oracle.
4. **Fallback.** A rejected proposal is replaced by shortest-path or equal-split routing: a feasible
   candidate if one exists, otherwise the least-violating one.

The flow-level physics model is the ground truth for both the twin's training labels and the oracle.
All linear programs (min-MLU and max-throughput path LPs) are solved with SciPy HiGHS.

## Repository structure

| Folder | Contents |
|---|---|
| [`implementation/`](implementation/) | Proposed-method code. `preprocessing/` builds topologies, traffic matrices, candidate paths, graph tensors and splits; `twin/` holds the ensemble digital twin; `safety/` holds the safety gate, fallbacks and the A–F evaluation runner; `framework/` holds the GNN policy, its training and the end-to-end evaluations; `phase3_final/` holds the final experiments (calibrated load, E2–E4, ablations, statistics, figures, results summary); `tests/` holds 98 unit tests; `scripts/` holds the run scripts. `repo_paths.py` defines every repository location. |
| [`baseline_p1_reproduction/`](baseline_p1_reproduction/) | Reproduction of P1 GRL-TE: TopoFlowNet with one-step A2C, ADMM refinement and constraint adjustment, its configuration, the frozen B4 protocol (`BASELINE_FREEZE/`), audits of the reproduction, the GEANT 22/72 and B4 link-failure experiments (`phase3_final/`), and their results. |
| [`data/`](data/) | Raw datasets (`raw/`: TEAL, FIGRET, Topology Zoo), processed data (`processed/`: topologies, traffic matrices, candidate paths, graph tensors, normalisation statistics, train/validation/test splits), and scenarios (`scenarios/`: link failures, twin errors). `DATASET_CATALOG.md` documents every source. |
| [`results/`](results/) | Experiment outputs. `phase3_final/` holds the final results, figures, statistics and `RESULTS_SUMMARY.md`; the other folders hold the earlier development stages (twin, safety gate, policy, framework evaluations, audits). |
| [`models/`](models/) | Every trained checkpoint (twins, policies, P1 baseline seeds), stored with Git LFS. `CHECKPOINTS.md` lists each file with its producer, consumer and SHA256. |
| [`diagrams/`](diagrams/) | Architecture, operational-workflow, experimental-design and safety-gate diagrams in `.dot`, SVG, PDF, PNG and `.drawio` form, produced by `make_diagrams.py`. |
| [`docs/`](docs/) | The Phase 2 report. |
| [`reference/`](reference/) | Third-party reference code (`teal/`, the TEAL repository) and the P1 paper (`papers/p1.pdf`). |
| [`tools/`](tools/) | `make_submission.py`, which assembles the course submission package from the repository. |

## Datasets

| Topology | Nodes / directed links | Traffic | Role |
|---|---|---|---|
| B4 | 12 / 38 | TEAL real matrices (Yates/Abilene demands mapped onto B4) | train / validation |
| UsCarrier | 158 / 378 | TEAL synthetic toy template | train / validation |
| Cogentco | 197 / 486 | gravity reconstruction (Topology Zoo graph) | train / validation |
| GEANT | 22 / 72 | FIGRET real demand histories | unseen test |
| Kdl | 754 / 1790 | TEAL synthetic toy template | unseen test |
| ASN | 1739 / 8558 | TEAL synthetic toy template | excluded (path generation incomplete) |
| Abilene | 12 / 30 | SNDlib measured traffic (Zhang, 5-min, Mar–Sep 2004); related to the B4 training traffic, see below | unseen test (additional) |
| Germany50 | 50 / 176 | SNDlib measured traffic (DFN, 5-min, one day in Feb 2005) | unseen test (additional) |

Traffic-matrix indices 0–19 are used for training, 20–27 for validation and model selection, and
28–35 are held aside and never used; GEANT and Kdl are used only for the final evaluation. The
UsCarrier and Kdl traffic matrices are identical copies of one toy template, so additional indices add
no traffic diversity on these topologies. Load is studied at fixed scales and at *calibrated* levels,
where each matrix is scaled so that its reference MLU (path LP, or shortest path on Kdl) equals a
target between 0.5 and 1.1.

Abilene and Germany50 come from SNDlib (`data/raw/sndlib/`, with `SOURCE.md` and `QUALITY_REPORT.md`) and
are evaluated at the calibrated levels only, with 200 seeded matrices each. The B4 traffic is a gravity
estimate of the Abilene network on 1 March 2004 (00:00–02:55); those slots are excluded from the Abilene
sample, which therefore counts as an unseen topology with related traffic.

## Experiments

| Experiment | What it measures | Results |
|---|---|---|
| Calibrated load | How often fixed-scale versus calibrated traffic is feasible for each routing method | `results/phase3_final/calibrated_traffic/` |
| E1 development A–F | Gate variants A–F on the training topologies | `results/safety_gate/improvements/eval_runs/` |
| E2 holdout evaluation | Gate variants A–F on the unseen GEANT (1,600 cases) and Kdl (8 cases) | `results/phase3_final/e2_holdout_af/` |
| E3 link-failure experiment | Gate safety and satisfaction with 1–3 failed links on B4, GEANT and UsCarrier | `results/phase3_final/e3_link_failures/` |
| E4 twin-error sweep | Mean-only (D) versus uncertainty-aware (E) gating when the twin is perturbed or trained on less data, against a mean-only gate with a tuned safety margin (Dm) | `results/phase3_final/e4_twin_error/` |
| Policy generalization | Policy MLU relative to shortest path on seen and unseen topologies, and a second policy trained at calibrated load | `results/phase3_final/policy_v2/` |
| Ablations | Removing the gate, the uncertainty term, the fallback or the learned twin residual, and replacing the policy with the single-topology P1 policy | `results/phase3_final/ablations_v2/` |
| Statistics | Friedman/Nemenyi ranks, paired Wilcoxon and exact McNemar tests with Holm correction, Cliff's δ, run-time overhead and a failure taxonomy | `results/phase3_final/stats/` |
| SNDlib topologies | Holdout A–F, link failures and the twin-error sweep on Abilene and Germany50 with the frozen stack, and the statistics pooled over all four unseen topologies | `results/holdout_sndlib/`, `results/link_failures_sndlib/`, `results/twin_error_sndlib/`, `results/stats_sndlib/` |
| P1 baseline | GRL-TE on B4 (10 seeds), on the FIGRET GEANT 22/72 graph (10 seeds) and under B4 link failures, against LP, ECMP and shortest path | `baseline_p1_reproduction/BASELINE_FREEZE/`, `baseline_p1_reproduction/phase3_final/results/` |

Figures in `results/phase3_final/figures/` are produced by `implementation/phase3_final/figures.py`, and
`results/phase3_final/RESULTS_SUMMARY.md` is produced by `implementation/phase3_final/consolidate.py`
from the result files alone; every number in it cites its source file.

## Results

- **Safety.** Without a gate, 86.6 % (GEANT) and 87.5 % (Kdl) of the policy's proposals on unseen
  topologies overload a link. Every twin-based gate (B–E) reduces unsafe accepts to zero on these
  topologies, as on the training topologies; the improved-twin gates (D, E) also do so under 1–3 link
  failures.
- **Further unseen topologies (SNDlib).** On Abilene and Germany50 with measured traffic, the frozen gate
  removes every unsafe accept (policy alone: 97.5 % and 74.0 % of cases overload) without false rejects, and
  raises applied satisfaction from 0.655 to 0.911 (Abilene) and from 0.931 to 0.949 (Germany50), Holm-corrected
  p < 1e-90 on each. The framework stays below the LP (0.982 and 0.989), and on Germany50 it only matches
  ECMP, because the fallback is an equal split there; most rejected cases still overload after fallback.
  Details are in §15 of `RESULTS_SUMMARY.md`.
- **Applied quality.** On GEANT the framework raises mean demand satisfaction from 0.880 (policy alone)
  to 0.976 (paired difference +0.096, Holm-corrected p ≈ 7e-228) and has the best Friedman rank across
  40 blocks (χ² = 74.85, p ≈ 1e-14).
- **Uncertainty.** With an accurate twin, the mean-only and uncertainty-aware gates make identical
  decisions (D ≡ E ≡ F). When the twin is perturbed, the uncertainty-aware gate avoids unsafe accepts
  that the mean-only gate makes (e.g. 50 versus 0 at σ = 0.4, McNemar, Holm-corrected), at the cost of
  more false rejects; a mean-only gate with a safety margin tuned on validation data matches it, so the
  gain comes from conservatism rather than from the uncertainty estimate itself.
- **Twin.** The improved twin falsely rejects 0 of 214 safe proposals (original twin: 1–2), and a pure
  physics twin agrees with the oracle on every holdout case, so the learned residual adds no decision
  value on these topologies.
- **Fallback.** The fallback lowers overload from 0.825 to 0.681 of holdout decisions, but 68 % of
  GEANT cases still overload after gate and fallback; in 547 of 1,095 overloaded decisions a feasible
  k = 4 routing existed that the shortest/equal fallback does not find.
- **Generalization.** The learned policy's MLU on unseen topologies is 1.51× that of shortest path;
  training at calibrated load does not change this (validation score 1.1736 versus 1.1700).
- **P1 baseline.** The reproduced GRL-TE reaches 97.44 ± 1.68 % satisfaction on B4 (P1: 92.39 %) and
  94.44 ± 1.81 % on GEANT 22/72 (P1: 86.41 %); removing ADMM costs 12.7 points on GEANT (P1: 15.1).
  With three failed B4 links satisfaction falls to 80.61 % (P1: 77.25 %).

The claims ledger in `RESULTS_SUMMARY.md` (§11) classifies every hypothesis as demonstrated, partially
supported, not supported or not tested.

## Software

Dependencies are pinned in `requirements.txt` (Python 3.12, CPU only); Graphviz renders the diagrams.
Checkpoints in `models/` and `data/processed/graph_data/Kdl_bipartite.pt` are stored with Git LFS.
The full experiment sequence is listed in `implementation/scripts/run_all.sh` (PowerShell:
`run_all.ps1`); `implementation/scripts/smoke.sh` runs the unit tests and a two-case A–F evaluation.
Some JSON and Markdown records contain absolute paths of the machines on which they were produced;
they are provenance only and are not read by the code.
