"""Single source of truth for repository locations.

Layout (repository root):

    implementation/            proposed-method code (run commands from here)
    baseline_p1_reproduction/  P1 GRL-TE reproduction (code, BASELINE_FREEZE, audits, results)
    data/                      raw, processed, splits, scenarios
    results/                   every experiment output (reports, tables, figures, logs)
    models/                    every .pt checkpoint (Git LFS)

Checkpoints used to sit next to their run outputs under ``results/``. They now live
under ``models/``; :func:`model_path` maps an old ``results/...`` checkpoint location
to its ``models/...`` location so writers and readers agree.
"""

from __future__ import annotations

from pathlib import Path

IMPL = Path(__file__).resolve().parent
REPO = IMPL.parent
DATA = REPO / "data"
PROCESSED = DATA / "processed"
RESULTS = REPO / "results"
MODELS = REPO / "models"
BASELINE = REPO / "baseline_p1_reproduction"

# results/<prefix>  ->  models/<prefix>   (first match wins; anything else keeps its relative path)
_MODEL_PREFIXES = (
    ("twin/models/", "twin/"),
    ("digital_twin/improvements/models/", "twin_improved/"),
    ("phase3_final/e4_twin_error/degraded_twins/", "twin_degraded/"),
    ("phase3_final/policy_v2/runs/", "policy_v2/"),
    ("policy/", "policy/"),
)


def model_path(path: Path | str) -> Path:
    """Return where a checkpoint lives under ``models/``.

    ``path`` may be an old ``results/...`` location (absolute or relative to the repo
    root). Paths outside ``results/`` (for example a temporary test directory) are
    returned unchanged.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    try:
        rel = p.resolve().relative_to(RESULTS.resolve()).as_posix()
    except ValueError:
        return Path(path)
    for old, new in _MODEL_PREFIXES:
        if rel.startswith(old):
            return MODELS / (new + rel[len(old):])
    return MODELS / rel
