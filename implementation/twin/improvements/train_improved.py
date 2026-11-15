"""Physics-anchored ensemble training and validation-only conformal calibration.

Does not read GEANT/Kdl during training or calibration. Does not overwrite
``results/twin/models/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from repo_paths import model_path
from twin.data import ROOT, load_split_demands, load_topology
from twin.model import EnsembleTwin
from twin.physics import dirichlet_split, equal_split, shortest_split
from twin.train import _predict_numpy, _sample, _tensors

from .metrics import aggregate_case_metrics, score_prediction

TRAIN_TOPOS = ("B4", "UsCarrier", "Cogentco")
HELD_OUT = ("GEANT", "Kdl")


def _actions(topo, matrix_index: int, seed: int):
    rng = np.random.default_rng(seed + matrix_index)
    return [
        ("equal", equal_split(topo.n_paths)),
        ("shortest", shortest_split(topo.n_paths, topo.hops)),
        ("dirichlet", dirichlet_split(topo.n_paths, rng)),
    ]


def evaluate_topologies(
    model: EnsembleTwin,
    names: tuple[str, ...],
    split: str,
    device: torch.device,
    *,
    stride: int = 1,
    conformal_std_scale: float = 1.0,
    scales: tuple[float, ...] = (1.0,),
    limit_matrices: int | None = None,
) -> dict:
    rows = []
    by_scale: dict[float, list] = {float(s): [] for s in scales}
    for name in names:
        if name in HELD_OUT and split != "test":
            raise RuntimeError(f"refusing to use held-out topology {name} on split={split}")
        topo = load_topology(name)
        demands, meta = load_split_demands(split, name)
        n = demands.shape[0] if limit_matrices is None else min(limit_matrices, demands.shape[0])
        indices = list(range(0, n, stride))
        for index in indices:
            for scale in scales:
                matrix = demands[index] * float(scale)
                for action_name, action in _actions(topo, index, 2026 + int(index)):
                    sample = _sample(topo, matrix, action)
                    mean_load, std_load = _predict_numpy(model, sample, topo, device)
                    std_load = std_load * float(conformal_std_scale)
                    stats = score_prediction(sample["load"], mean_load, std_load, topo.capacity)
                    stats.update(
                        {
                            "topology": name,
                            "split": split,
                            "matrix_index": int(meta["matrix_indices"][index]),
                            "scale": float(scale),
                            "action": action_name,
                            "unseen": name in HELD_OUT,
                        }
                    )
                    rows.append(stats)
                    by_scale[float(scale)].append(stats)
        del topo
    summary = aggregate_case_metrics(rows)
    summary["by_topology"] = {}
    for name in names:
        bundle = [r for r in rows if r["topology"] == name]
        if bundle:
            summary["by_topology"][name] = aggregate_case_metrics(bundle)
    summary["by_scale"] = {
        str(scale): aggregate_case_metrics(bundle) for scale, bundle in by_scale.items() if bundle
    }
    return {"summary": summary, "cases": rows}


@torch.no_grad()
def _member_mult_penalty(member, features: torch.Tensor) -> torch.Tensor:
    mult = member.multiplier(features)
    return torch.mean((mult - 1.0) ** 2)


def train_physics_anchored(
    *,
    out_dir: Path,
    seed: int = 2026,
    n_members: int = 5,
    hidden: int = 32,
    max_epochs: int = 16,
    patience: int = 4,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    anchor_weight: float = 1e-2,
    dropout: float = 0.1,
) -> dict:
    """Train on B4/UsCarrier/Cogentco train; early-stop on val utilisation MAE."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EnsembleTwin(n_members=n_members, hidden=hidden, dropout=dropout).to(device)
    optimisers = [
        torch.optim.Adam(member.parameters(), lr=lr, weight_decay=weight_decay) for member in model.members
    ]
    topologies = {name: load_topology(name) for name in TRAIN_TOPOS}
    train_demands = {name: load_split_demands("train", name)[0] for name in TRAIN_TOPOS}

    history = []
    best_val = None
    best_state = None
    bad = 0

    for epoch in range(max_epochs):
        losses = []
        anchors = []
        order = list(TRAIN_TOPOS)
        for member_index, (member, opt) in enumerate(zip(model.members, optimisers)):
            rng = np.random.default_rng(seed + epoch * 100 + member_index)
            member.train()
            rng.shuffle(order)
            for name in order:
                topo = topologies[name]
                # Bootstrap matrices; also use 0.5x traffic augmentation (train only).
                picks = rng.choice(train_demands[name].shape[0], size=train_demands[name].shape[0], replace=True)
                for matrix_index in picks:
                    scale = 1.0 if rng.random() < 0.7 else 0.5
                    action = _actions(topo, int(matrix_index), seed + member_index)[int(rng.integers(0, 3))][1]
                    sample = _sample(topo, train_demands[name][int(matrix_index)] * scale, action)
                    batch = _tensors(sample, topo, device)
                    pred = model.member_loads(
                        batch["features"],
                        batch["flow"],
                        batch["incidence_path"],
                        batch["incidence_link"],
                        topo.n_links,
                        member,
                    )
                    mse = torch.mean((pred - batch["target"]) ** 2)
                    anchor = _member_mult_penalty(member, batch["features"])
                    loss = mse + float(anchor_weight) * anchor
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(mse.detach()))
                    anchors.append(float(anchor.detach()))
        # Validation for early stopping (seen topologies only; B4-heavy for speed).
        model.eval()
        val_b4 = evaluate_topologies(model, ("B4",), "val", device, stride=1, scales=(1.0,))
        val_large = evaluate_topologies(
            model, ("UsCarrier", "Cogentco"), "val", device, scales=(1.0,), limit_matrices=1
        )
        val_rows = val_b4["cases"] + val_large["cases"]
        from .metrics import aggregate_case_metrics as _agg

        val_summary = _agg(val_rows)
        val_mae_util = float(val_summary["mae_utilisation"])
        row = {
            "epoch": epoch,
            "train_mse_normalised": float(np.mean(losses)),
            "train_anchor": float(np.mean(anchors)),
            "val_mae_utilisation": val_mae_util,
            "val_mae": float(val_summary["mae"]),
            "val_coverage_2std": float(val_summary["coverage_2std"]),
        }
        history.append(row)
        print(
            f"improved epoch {epoch + 1}/{max_epochs} "
            f"mse={row['train_mse_normalised']:.3e} anchor={row['train_anchor']:.3e} "
            f"val_mae_util={val_mae_util:.6g}",
            flush=True,
        )
        if best_val is None or val_mae_util < best_val:
            best_val = val_mae_util
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch + 1} (best val_mae_util={best_val:.6g})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    config = {
        "name": "physics_anchored_ensemble",
        "n_members": n_members,
        "hidden": hidden,
        "dropout": dropout,
        "max_epochs": max_epochs,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "anchor_weight": anchor_weight,
        "seed": seed,
        "train_topologies": list(TRAIN_TOPOS),
        "held_out": list(HELD_OUT),
        "augmentation": "70% native + 30% 0.5x traffic scales on train matrices",
        "early_stopping_metric": "val mae_utilisation on B4/UsCarrier/Cogentco",
        "features": "same as original twin.train._features (demand/denom, action, hops/16)",
    }
    ckpt = {
        **config,
        "state_dict": model.state_dict(),
        "best_val_mae_utilisation": best_val,
        "history": history,
        "conformal_std_scale": 1.0,
    }
    path = model_path(out_dir / "ensemble_physics_anchored.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    (out_dir / "train_config_physics_anchored.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "training_curve_physics_anchored.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model": model, "checkpoint_path": path, "config": config, "history": history, "best_val": best_val}


def fit_conformal_scale(
    model: EnsembleTwin,
    device: torch.device,
    *,
    target_coverage: float = 0.90,
    max_scale: float = 8.0,
) -> dict:
    """Choose std scale from validation residuals of seen topologies only."""
    # Grid search scales to hit target empirical coverage_2std on val.
    candidates = list(np.linspace(0.25, max_scale, 12))
    scored = []
    for scale in candidates:
        scored_b4 = evaluate_topologies(
            model, ("B4",), "val", device, scales=(1.0,), conformal_std_scale=float(scale)
        )
        scored_large = evaluate_topologies(
            model,
            ("UsCarrier", "Cogentco"),
            "val",
            device,
            scales=(1.0,),
            conformal_std_scale=float(scale),
            limit_matrices=1,
        )
        from .metrics import aggregate_case_metrics as _agg

        scored_val = _agg(scored_b4["cases"] + scored_large["cases"])
        cov = float(scored_val["coverage_2std"])
        width = float(scored_val["mean_interval_width_2std_util"])
        scored.append({"scale": float(scale), "coverage_2std": cov, "width_util": width})
    # Prefer smallest width among scales with coverage >= target; else closest coverage.
    feasible = [s for s in scored if s["coverage_2std"] >= target_coverage]
    if feasible:
        chosen = min(feasible, key=lambda s: s["width_util"])
    else:
        chosen = min(scored, key=lambda s: abs(s["coverage_2std"] - target_coverage))
    return {
        "conformal_std_scale": chosen["scale"],
        "target_coverage_2std": target_coverage,
        "val_coverage_2std": chosen["coverage_2std"],
        "val_width_util": chosen["width_util"],
        "grid": scored,
        "selection": "min util interval width subject to val coverage_2std >= target; train/val topologies only",
    }


def attach_conformal(checkpoint_path: Path, conformal: dict, out_path: Path) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["conformal_std_scale"] = float(conformal["conformal_std_scale"])
    payload["conformal"] = conformal
    torch.save(payload, out_path)
    return out_path
