"""Inference helpers compatible with Safety Gate / ablation predict_action shape.

Uses the same path features as ``twin.train._sample`` so an improved ensemble
with the original 3-D feature layout plugs into existing callers.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from twin.model import EnsembleTwin
from twin.train import _predict_numpy, _sample


def load_ensemble_checkpoint(path: Path, device: torch.device) -> tuple[EnsembleTwin, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = EnsembleTwin(
        n_members=int(payload.get("n_members", 5)),
        hidden=int(payload.get("hidden", 32)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def predict_loads(
    model: EnsembleTwin,
    topo,
    matrix: np.ndarray,
    action: np.ndarray,
    device: torch.device,
    conformal_std_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return true simulator load, mean, std (optionally conformal-scaled), twin_ms."""
    sample = _sample(topo, matrix, action)
    started = time.perf_counter()
    mean_load, std_load = _predict_numpy(model, sample, topo, device)
    twin_ms = (time.perf_counter() - started) * 1000.0
    std_load = np.asarray(std_load, dtype=np.float64) * float(conformal_std_scale)
    return sample["load"], mean_load, std_load, twin_ms
