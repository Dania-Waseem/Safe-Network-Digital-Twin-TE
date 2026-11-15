"""Build the Phase 3 submission folder and zip.

Layout (user-specified): two top-level folders, the P1 baseline reproduction and the
proposed method, plus a short top-level README.

submission/23i2622_23i2618_Phase3_Implementation/
├── README.md
├── Baseline_P1_Reproduction/        GRL_TE_Reproduction code + configs, BASELINE_FREEZE protocol card
│                                    and reports, new GEANT results, tables, figures, README (how to rerun)
└── Proposed_Method_Implementation/  README, requirements.txt, data/, src/, configs/, scripts/,
                                     results/{RESULTS_SUMMARY.md, tables, figures, reports}, diagrams/,
                                     CHECKPOINTS.md

Excluded everywhere: *.pt, *.npz, *.pkl, raw datasets, reference/ (third-party code), caches, decisions.jsonl / any
.jsonl larger than 5 MB, __pycache__. Checkpoints are listed (SHA256, repository path) in
CHECKPOINTS.md but not copied.

Usage: python tools/make_submission.py
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMPL = REPO / "implementation"          # proposed-method code
REPRO = REPO / "baseline_p1_reproduction"  # P1 baseline
MODELS = REPO / "models"                 # all checkpoints (Git LFS)
NAME = "23i2622_23i2618_Phase3_Implementation"
DEST = REPO / "submission" / NAME
GITHUB = "https://github.com/Dania-Waseem/Safe-Network-Digital-Twin-TE"
EXCLUDE_SUFFIX = {".pt", ".npz", ".pkl", ".pyc", ".pth", ".ckpt"}
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "cache", "raw", "_ref_teal", "smoke_test_tmp"}  # checkpoints: by suffix
MAX_JSONL = 5 * 1024 * 1024


def keep(path: Path, full: Path | None = None) -> bool:
    """path: relative path checked against the exclusion rules; full: the file on disk (for size)."""
    if path.suffix.lower() in EXCLUDE_SUFFIX:
        return False
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    if path.name == "decisions.jsonl":
        return False
    if path.suffix == ".jsonl" and (full or path).stat().st_size > MAX_JSONL:
        return False
    return True


def copy_tree(src: Path, dst: Path, patterns=("*",)) -> int:
    n = 0
    for pattern in patterns:
        for f in src.rglob(pattern):
            if f.is_file() and keep(f.relative_to(src.parent), f):
                target = dst / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                n += 1
    return n


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_requirements() -> str:
    import importlib.metadata as md
    pkgs = ["numpy", "scipy", "torch", "networkx", "pandas", "matplotlib", "scikit-posthocs", "statsmodels",
            "psutil", "pytest"]
    lines = [f"# Versions used for the Phase 3 final results (Python {sys.version.split()[0]}, CPU only).",
             "# Graphviz (dot) 16.1.1 is needed only to re-render the diagrams."]
    for p in pkgs:
        try:
            lines.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            lines.append(f"# {p}: not installed")
    return "\n".join(lines) + "\n"


def checkpoints_md() -> str:
    rows = []
    for root in (MODELS, REPO / "data" / "processed" / "graph_data"):
        for f in sorted(root.rglob("*.pt")):
            if "__pycache__" in f.parts:
                continue
            rel = f.relative_to(REPO).as_posix()
            rows.append(f"| `{rel}` | {f.stat().st_size / 1e6:.1f} | `{sha256(f)}` |")
    return "\n".join([
        "# Checkpoints (not included in this package)",
        "",
        f"All files are in the GitHub repository {GITHUB} (branch `main`) at the listed path. "
        "Every checkpoint is under `models/` and stored with Git LFS (as is `Kdl_bipartite.pt`).",
        "",
        "| Path (repository-relative) | Size (MB) | SHA256 |",
        "|---|---:|---|",
        *rows, ""])


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


TOP_README = f"""# Phase 3 Implementation — 23I-2622 Dania Waseem, 23I-2618 Taiba Tariq

**AI-Driven Network Digital Twin for Safe and Generalizable Network Decision Testing**
(FAST-NUCES Islamabad, Computer Networks, Phase 3: Research Implementation and Experimental Evaluation)

Repository: {GITHUB} (branch `phase3-completion`, commit `{{sha}}`)

| Folder | Contents |
|---|---|
| `Baseline_P1_Reproduction/` | Reproduction of the baseline paper P1 (Lu et al., GRL-TE, JKSU-CIS 37:119, 2025): code, configs, frozen B4 protocol and reports, new GEANT 22/72 results, B4 link-failure table, figures. See its README for how to rerun B4 and GEANT. |
| `Proposed_Method_Implementation/` | Our framework (graph-RL policy + physics-anchored ensemble digital twin + uncertainty-aware safety gate + fallback): source, configs, scripts, all Phase 3 results (`results/RESULTS_SUMMARY.md`), figures, architecture diagrams, checkpoint list. |

Model checkpoints, raw datasets and large decision logs are not included (size); see
`Proposed_Method_Implementation/CHECKPOINTS.md` and `Proposed_Method_Implementation/data/DOWNLOAD_DATA.md`.
"""

BASELINE_README = """# Baseline: P1 GRL-TE reproduction

Reproduction of Lu, Tang, Ma, Xing, "Graph-based reinforcement learning for software-defined networking traffic
engineering", *J. King Saud Univ. – Comput. Inf. Sci.* 37:119 (2025), doi 10.1007/s44443-025-00133-z.

## What is here

| Path | Contents |
|---|---|
| `code/` | TopoFlowNet + one-step A2C (`models/`), ADMM and constraint adjustment (`optimization/`), preprocessing, training, experiments, `config.py` |
| `code/BASELINE_FREEZE/` | Frozen B4 protocol: `frozen_b4_config.py`, runner, protocol card, reproducibility and freeze reports, logs |
| `code/phase3_final/` | New Phase 3 runners: `geant_figret.py` (GEANT 22/72), `b4_link_failures.py` (P1 §5.5-comparable table) |
| `reports/` | Reproduction audits, deviations, methodology compliance, experimental results (markdown) |
| `results/` | New GEANT results (raw h⁰ headline + FIGRET-units sensitivity), B4 link-failure table, earlier tables/logs |
| `figures/` | Baseline figures (link-failure curves) |

## Headline results (all numbers traceable to `results/`)

See `results/geant_figret/P1_GEANT_FIGRET_REPORT.md`, `results/geant_figret_units/P1_GEANT_FIGRET_REPORT.md`,
`results/b4_link_failures/B4_P1_SECTION55_TABLE.md` and `code/BASELINE_FREEZE/BASELINE_FREEZE_DECISION.md`.

## How to rerun

Requirements: Python 3.12 and the pinned packages in `../Proposed_Method_Implementation/requirements.txt`.
Data: TEAL B4 topology/TMs (`code/data/topologies`, `code/data/traffic_matrices/B4`) and the FIGRET GEANT data
processed by the proposed-method pipeline (see `../Proposed_Method_Implementation/data/DOWNLOAD_DATA.md`).
Run from `code/`:

```bash
# B4 frozen baseline (re-evaluates the 10 archived checkpoints, which are not in this zip: copy them from the
# repository paths listed in ../Proposed_Method_Implementation/CHECKPOINTS.md into ../models/baseline_p1/b4/ (repository layout); ~5 min)
python BASELINE_FREEZE/run_b4_baseline_freeze.py
# B4 link failures, P1 §5.5-comparable table (~30 s)
python -m phase3_final.b4_link_failures
# GEANT 22/72 with FIGRET TMs: 10 seeds x 3 episodes, raw h0 (P1 eq. 5) headline (~45 min CPU)
python -m phase3_final.geant_figret
# sensitivity channel: h0 in FIGRET units (/1e9)
python -m phase3_final.geant_figret --figret-units
```

Notes: all LPs use SciPy HiGHS (no Gurobi licence). GEANT training uses a seeded 1,000-TM subsample of the 8,080
FIGRET train TMs (runtime). Teal, FIGRET and NCFlow were not run; their numbers are quoted from P1 Table 6 only.
"""


def proposed_readme(sha: str) -> str:
    return f"""# Proposed method: safe graph-RL traffic engineering tested in a network digital twin

Repository: {GITHUB} (branch `phase3-completion`, commit `{sha}`). Start with `results/RESULTS_SUMMARY.md`.

## Layout

| Path | Contents |
|---|---|
| `src/preprocessing/` | dataset collection, topology validation, Yen k=4 paths, normalisation, bipartite graph, splits |
| `src/twin/` | flow-level physics simulator and the 5-member ensemble twin; `twin/improvements/` physics-anchored twin + conformal calibration |
| `src/framework/` | graph-RL policy (GNN), controller, earlier evaluations |
| `src/safety/` | safety gate; `safety/improvements/` utilisation-space UCB gate, A–F runner, fallback, conditional LP |
| `src/phase3_final/` | Phase 3 final experiments (calibrated load, E2 holdout A–F, E3 link failures, E4 twin error, policy v2, ablations, statistics, figures) |
| `src/tests/` | unit tests (`pytest src/tests` from `src/`) |
| `configs/` | JSON configs used for the final results |
| `scripts/` | `run_all.ps1` / `run_all.sh` (every final experiment in order) |
| `results/` | `RESULTS_SUMMARY.md`, `tables/` (CSV/JSON), `figures/` (300-dpi PNG + PDF), `reports/` (markdown reports) |
| `diagrams/` | architecture diagrams (.dot, SVG, PDF, PNG, .drawio) |
| `data/` | dataset catalog and download instructions |
| `CHECKPOINTS.md` | checkpoints with SHA256 and repository path (files not included) |

## Setup

```bash
python -m pip install -r requirements.txt     # CPU-only; Python 3.12
```

Place the processed data as described in `data/DOWNLOAD_DATA.md`, then run the experiments from `src/`
(the code expects the repository layout, i.e. `src/` = `implementation/`, with `data/`, `results/` and `models/` beside it).

## Reproducing each table and figure

| Result | Command (from `src/`) | Output |
|---|---|---|
| Calibrated load (calibrated-load study) | `python -m phase3_final.calibrate_load` | `results/phase3_final/calibrated_traffic/` |
| E2 holdout A–F | `python -m phase3_final.holdout_af --topologies GEANT` and `--topologies Kdl` | `results/phase3_final/e2_holdout_af/` |
| E4 twin error | `python -m phase3_final.train_degraded_twin`; `python -m phase3_final.twin_error_sweep cache --topologies B4 UsCarrier Cogentco GEANT`; `... cache --topologies Kdl`; `... sweep` | `results/phase3_final/e4_twin_error/` |
| Policy v2 + generalization gap | `python -m phase3_final.train_policy_v2 score-reference`; `... train`; holdout pass with `--policy-id v2_calload_normscore --checkpoint <best.pt>`; `python -m phase3_final.generalization_gap` | `results/phase3_final/policy_v2/` |
| E3 link failures | `python -m phase3_final.link_failure_eval --topologies B4 GEANT UsCarrier` | `results/phase3_final/e3_link_failures/` |
| Ablations | `python -m phase3_final.ablations_v2 --topologies B4 UsCarrier Cogentco GEANT`; `--topologies Kdl` | `results/phase3_final/ablations_v2/` |
| Statistics | `python -m phase3_final.stats_final grid --topologies B4 UsCarrier Cogentco GEANT`; `... grid --topologies Kdl`; `... analyze` | `results/phase3_final/stats/` |
| Figures | `python -m phase3_final.figures` | `results/phase3_final/figures/` |
| Summary | `python -m phase3_final.consolidate` | `results/phase3_final/RESULTS_SUMMARY.md` |
| Diagrams | `python ../../diagrams/make_diagrams.py` | `diagrams/` |

`scripts/run_all.ps1` / `scripts/run_all.sh` run these in order. Total CPU time on the development laptop was
roughly 3–4 hours; Kdl steps need about 5 GB of free RAM.

## Integrity notes

- Tuning, calibration and checkpoint selection use only the validation split (TM 20–27 of B4, UsCarrier, Cogentco).
  GEANT/Kdl test TMs and the held-aside TMs 28–35 were never used for any choice.
- UsCarrier and Kdl TMs are synthetic toy templates (identical copies); Cogentco TMs are gravity reconstructions.
  Calibrated load levels rescale these synthetic units by a method-independent LP/shortest reference.
- All LPs use SciPy HiGHS (no Gurobi licence).
- Negative results are reported as measured (see the claims ledger in `results/RESULTS_SUMMARY.md`).
"""


DOWNLOAD = f"""# Getting the data

Raw datasets are not included (size and licences). The pipeline expects them under
`data/raw/` in the repository ({GITHUB}).

| Source | What we use | Where to get it | Local folder |
|---|---|---|---|
| Teal (Xu et al., SIGCOMM 2023) | B4 topology + 36 real TMs; UsCarrier, Kdl, ASN topologies and toy TMs | https://github.com/harvard-cns/teal (`topologies/`, `traffic-matrices/`) | `data/raw/teal/` |
| FIGRET (Liu et al., SIGCOMM 2024) | GEANT 22/72 topology, real train/test TM histories, tunnels | https://github.com/FIGRET/figret (`Data/GEANT`) | `data/raw/figret/` |
| Internet Topology Zoo | Cogentco graph (GraphML); gravity TMs are reconstructed | http://www.topology-zoo.org/dataset.html | `data/raw/topology_zoo/` |

Scripts (from `src/`):

```bash
python -m preprocessing.collect_datasets      # download / verify raw sources
python -m preprocessing.preprocess --jobs 2   # topologies, Yen k=4 paths, normalisation, bipartite graphs
python -m preprocessing.split_datasets        # train / val / held-aside / test splits
```

ASN path generation is incomplete (39 of 1,739 shards missing) and ASN is excluded from all results. The processed
artefacts (`data/processed/**`, including the `.npz` demand slices) are in the GitHub repository;
`Kdl_bipartite.pt` uses Git LFS.
"""


RUN_ALL_PS1 = """# Run every Phase 3 final experiment in order (Windows PowerShell). From src/.
$ErrorActionPreference = "Stop"
python -m phase3_final.calibrate_load
python -m phase3_final.holdout_af --topologies GEANT
python -m phase3_final.holdout_af --topologies Kdl
python -m phase3_final.train_degraded_twin
python -m phase3_final.twin_error_sweep cache --topologies B4 UsCarrier Cogentco GEANT
python -m phase3_final.twin_error_sweep cache --topologies Kdl
python -m phase3_final.twin_error_sweep sweep
python -m phase3_final.train_policy_v2 score-reference
python -m phase3_final.train_policy_v2 train
python -m phase3_final.holdout_af --topologies GEANT --policy-id v2_calload_normscore --checkpoint ../models/policy_v2/v2_calload_normscore/best.pt
python -m phase3_final.holdout_af --topologies Kdl --policy-id v2_calload_normscore --checkpoint ../models/policy_v2/v2_calload_normscore/best.pt
python -m phase3_final.generalization_gap
python -m phase3_final.link_failure_eval --topologies B4 GEANT UsCarrier
python -m phase3_final.ablations_v2 --topologies B4 UsCarrier Cogentco GEANT
python -m phase3_final.ablations_v2 --topologies Kdl
python -m phase3_final.stats_final grid --topologies B4 UsCarrier Cogentco GEANT
python -m phase3_final.stats_final grid --topologies Kdl
python -m phase3_final.stats_final analyze
python -m phase3_final.figures
python -m phase3_final.consolidate
"""


def build() -> Path:
    if DEST.exists():
        shutil.rmtree(DEST)
    sha = git_sha()
    DEST.mkdir(parents=True)
    (DEST / "README.md").write_text(TOP_README.replace("{sha}", sha), encoding="utf-8")

    # ---------------- Baseline
    B = DEST / "Baseline_P1_Reproduction"
    for sub in ("models", "optimization", "preprocessing", "training", "experiments", "phase3_final", "BASELINE_FREEZE"):
        copy_tree(REPRO / sub, B / "code" / sub, ("*.py", "*.md", "*.json", "*.log"))
    for f in ("config.py", "README.md"):
        copy_file(REPRO / f, B / "code" / f)
    copy_tree(REPRO / "data" / "topologies", B / "code" / "data" / "topologies")
    for f in REPRO.glob("*.md"):
        copy_file(f, B / "reports" / f.name)
    copy_tree(REPRO / "COMPLETE_P1_AUDIT", B / "reports" / "COMPLETE_P1_AUDIT")
    copy_tree(REPRO / "phase3_final" / "results", B / "results")
    for sub in ("tables", "logs"):
        if (REPRO / "results" / sub).exists():
            copy_tree(REPRO / "results" / sub, B / "results" / "earlier" / sub)
    if (REPRO / "results" / "graphs").exists():
        copy_tree(REPRO / "results" / "graphs", B / "figures" / "earlier")
    for ext in ("png", "pdf"):
        p = REPO / "results" / "phase3_final" / "figures" / f"fig05_link_failure_satisfaction.{ext}"
        if p.exists():
            copy_file(p, B / "figures" / p.name)
    (B / "README.md").write_text(BASELINE_README, encoding="utf-8")

    # ---------------- Proposed method
    P = DEST / "Proposed_Method_Implementation"
    (P / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (P / "README.md").write_text(proposed_readme(sha), encoding="utf-8")
    (P / "requirements.txt").write_text(pinned_requirements(), encoding="utf-8")
    copy_file(REPO / "data" / "DATASET_CATALOG.md", P / "data" / "DATASET_CATALOG.md")
    (P / "data" / "DOWNLOAD_DATA.md").write_text(DOWNLOAD, encoding="utf-8")
    for sub in ("preprocessing", "twin", "framework", "safety", "tests", "phase3_final"):
        copy_tree(IMPL / sub, P / "src" / sub, ("*.py", "*.md", "*.json"))
    copy_file(IMPL / "repo_paths.py", P / "src" / "repo_paths.py")
    # configs used for the final results
    for f in (REPO / "results" / "phase3_final").rglob("config*.json"):
        rel = f.relative_to(REPO / "results" / "phase3_final")
        copy_file(f, P / "configs" / rel.as_posix().replace("/", "__"))
    for f in (REPO / "results" / "safety_gate" / "improvements" / "eval_runs" / "af_eval_k1_calibrated_devtopos_v1"
              / "frozen_gate_improvements_config.json", IMPL / "framework" / "policy_improved_config.json"):
        copy_file(f, P / "configs" / f.name)
    (P / "scripts").mkdir(parents=True, exist_ok=True)
    (P / "scripts" / "run_all.ps1").write_text(RUN_ALL_PS1, encoding="utf-8")
    (P / "scripts" / "run_all.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n# From src/.\n" +
                                              "\n".join(l for l in RUN_ALL_PS1.splitlines()[2:]) + "\n", encoding="utf-8")
    R = P / "results"
    copy_file(REPO / "results" / "phase3_final" / "RESULTS_SUMMARY.md", R / "RESULTS_SUMMARY.md")
    copy_tree(REPO / "results" / "phase3_final" / "figures", R / "figures")
    for f in (REPO / "results" / "phase3_final").rglob("*"):
        if f.is_file() and keep(f.relative_to(REPO), f) and "figures" not in f.parts:
            rel = f.relative_to(REPO / "results" / "phase3_final")
            if f.suffix == ".md":
                copy_file(f, R / "reports" / "phase3_final" / rel)
            elif f.suffix in (".csv", ".json", ".gz") or (f.suffix == ".jsonl"):
                copy_file(f, R / "tables" / rel)
            elif f.suffix in (".png", ".pdf"):
                copy_file(f, R / "figures" / "experiment_plots" / rel)
    for f in (REPO / "results").rglob("*.md"):
        if "phase3_final" not in f.parts and keep(f.relative_to(REPO), f):
            copy_file(f, R / "reports" / "earlier" / f.relative_to(REPO / "results"))
    copy_tree(REPO / "diagrams", P / "diagrams")
    (P / "CHECKPOINTS.md").write_text(checkpoints_md(), encoding="utf-8")
    return DEST


def make_zip(folder: Path) -> Path:
    zpath = folder.parent / f"{folder.name}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder.parent).as_posix())
    return zpath


def main() -> None:
    folder = build()
    z = make_zip(folder)
    n = sum(1 for f in folder.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    print(f"folder: {folder}  files: {n}  size: {size / 1e6:.1f} MB")
    print(f"zip:    {z}  size: {z.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
