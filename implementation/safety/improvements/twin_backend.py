"""Twin backends for the improved Safety Gate (read-only over existing weights)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twin.improvements.infer import load_ensemble_checkpoint, predict_loads
from twin.model import EnsembleTwin


@dataclass
class TwinPrediction:
    true_load: np.ndarray
    mean_load: np.ndarray
    std_load: np.ndarray
    twin_ms: float
    backend: str
    checkpoint_path: str
    conformal_std_scale_applied: float
    conformal_applied: bool


class TwinBackend:
    """Lazy-loadable ensemble twin with optional conformal std scaling."""

    def __init__(
        self,
        name: str,
        path: Path | str,
        *,
        apply_conformal: bool = False,
        conformal_std_scale_override: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.name = name
        self.path = Path(path)
        self.apply_conformal = bool(apply_conformal)
        self.conformal_std_scale_override = conformal_std_scale_override
        self.device = device or torch.device("cpu")
        self._model: EnsembleTwin | None = None
        self._payload: dict[str, Any] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        model, payload = load_ensemble_checkpoint(self.path, self.device)
        self._model = model
        self._payload = payload

    @property
    def payload(self) -> dict[str, Any]:
        self.load()
        assert self._payload is not None
        return self._payload

    @property
    def model(self) -> EnsembleTwin:
        self.load()
        assert self._model is not None
        return self._model

    def resolved_conformal_scale(self) -> float:
        if not self.apply_conformal:
            return 1.0
        if self.conformal_std_scale_override is not None:
            return float(self.conformal_std_scale_override)
        return float(self.payload.get("conformal_std_scale", 1.0))

    def predict(self, topo, matrix: np.ndarray, action: np.ndarray) -> TwinPrediction:
        scale = self.resolved_conformal_scale()
        true_load, mean_load, std_load, twin_ms = predict_loads(
            self.model,
            topo,
            matrix,
            action,
            self.device,
            conformal_std_scale=scale,
        )
        return TwinPrediction(
            true_load=np.asarray(true_load, dtype=np.float64),
            mean_load=np.asarray(mean_load, dtype=np.float64),
            std_load=np.asarray(std_load, dtype=np.float64),
            twin_ms=float(twin_ms),
            backend=self.name,
            checkpoint_path=str(self.path),
            conformal_std_scale_applied=float(scale),
            conformal_applied=bool(self.apply_conformal) and float(scale) != 1.0,
        )


def backends_from_config(cfg: dict[str, Any], device: torch.device | None = None) -> dict[str, TwinBackend]:
    """Build named backends from the improvements config ``twins`` section."""
    device = device or torch.device(str(cfg.get("evaluation", {}).get("device", "cpu")))
    out: dict[str, TwinBackend] = {}
    for name, twin_cfg in cfg.get("twins", {}).items():
        out[name] = TwinBackend(
            name=name,
            path=twin_cfg["path"],
            apply_conformal=bool(twin_cfg.get("apply_conformal", False)),
            conformal_std_scale_override=twin_cfg.get("conformal_std_scale"),
            device=device,
        )
    return out
