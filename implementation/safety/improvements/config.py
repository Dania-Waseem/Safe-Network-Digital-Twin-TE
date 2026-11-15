"""Load gate-improvement configuration from the results tree."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from twin.data import ROOT

DEFAULT_CONFIG_PATH = ROOT / "results" / "safety_gate" / "improvements" / "gate_improvements_config.json"
RESULT_DIR = ROOT / "results" / "safety_gate" / "improvements"


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_improvements_config(path: Path | None = None) -> dict[str, Any]:
    """Read JSON config and resolve relative artifact paths against project ROOT."""
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = deepcopy(payload)

    twins = cfg.setdefault("twins", {})
    for _name, twin_cfg in twins.items():
        if "path" in twin_cfg:
            twin_cfg["path"] = str(_resolve_path(twin_cfg["path"]))

    policy = cfg.setdefault("policy", {})
    if "default_checkpoint" in policy:
        policy["default_checkpoint"] = str(_resolve_path(policy["default_checkpoint"]))
    for cid, ckpt in list(policy.get("checkpoints", {}).items()):
        policy["checkpoints"][cid] = str(_resolve_path(ckpt))

    evaluation = cfg.setdefault("evaluation", {})
    if "output_dir" in evaluation:
        evaluation["output_dir"] = str(_resolve_path(evaluation["output_dir"]))
    else:
        evaluation["output_dir"] = str(RESULT_DIR)

    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def gate_params_from_config(cfg: dict[str, Any], variant_id: str) -> dict[str, Any]:
    """Merge global gate defaults with per-variant overrides."""
    base = dict(cfg.get("gate", {}))
    variants = cfg.get("variants", {})
    override = dict(variants.get(variant_id, {}).get("gate", {}))
    base.update(override)
    return base
