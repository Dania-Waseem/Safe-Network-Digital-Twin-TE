"""E4 twin-error sweep, error model 2: train deliberately weaker physics-anchored twins.

Same architecture, loss, optimiser and seed as
``twin.improvements.train_improved.train_physics_anchored``; only the training
budget is reduced:

  degraded_f10 : first 10% of the train TMs per topology, 1 epoch
  degraded_f25 : first 25%,                                2 epochs
  degraded_f50 : first 50%,                                4 epochs

(The reference improved twin used 100% of the 20 train TMs and up to 16 epochs.)
UsCarrier train TMs are identical toy copies, so the fraction only changes
the number of steps there. GEANT/Kdl are never read. The conformal std scale of
each degraded twin is fitted on the validation split of B4/UsCarrier/Cogentco only,
with the unchanged ``fit_conformal_scale`` rule. New checkpoints are written under
``results/phase3_final/e4_twin_error/degraded_twins/``; nothing existing is touched.

Usage: python -m phase3_final.train_degraded_twin
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from twin.data import ROOT, load_split_demands, load_topology
from twin.improvements.train_improved import (
    TRAIN_TOPOS,
    _actions,
    _member_mult_penalty,
    evaluate_topologies,
    fit_conformal_scale,
)
from twin.improvements.metrics import aggregate_case_metrics
from twin.model import EnsembleTwin
from twin.train import _sample, _tensors

from .common import environment, out_dir, write_json

LEVELS = (("degraded_f10", 0.10, 1), ("degraded_f25", 0.25, 2), ("degraded_f50", 0.50, 4))
# Unchanged hyperparameters of the reference improved twin.
REF = {"seed": 2026, "n_members": 5, "hidden": 32, "dropout": 0.1, "lr": 5e-4,
       "weight_decay": 1e-4, "anchor_weight": 1e-2}
OUT = out_dir("e4_twin_error") / "degraded_twins"
MODEL_OUT = ROOT / "models" / "twin_degraded"  # checkpoints (Git LFS)


def train_one(name: str, fraction: float, epochs: int) -> dict:
    seed = REF["seed"]
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EnsembleTwin(n_members=REF["n_members"], hidden=REF["hidden"], dropout=REF["dropout"]).to(device)
    opts = [torch.optim.Adam(m.parameters(), lr=REF["lr"], weight_decay=REF["weight_decay"])
            for m in model.members]
    topologies = {n: load_topology(n) for n in TRAIN_TOPOS}
    demands = {}
    for n in TRAIN_TOPOS:
        d = load_split_demands("train", n)[0]
        demands[n] = d[: max(1, math.ceil(fraction * d.shape[0]))]
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        losses = []
        order = list(TRAIN_TOPOS)
        for mi, (member, opt) in enumerate(zip(model.members, opts)):
            rng = np.random.default_rng(seed + epoch * 100 + mi)
            member.train()
            rng.shuffle(order)
            for n in order:
                topo = topologies[n]
                picks = rng.choice(demands[n].shape[0], size=demands[n].shape[0], replace=True)
                for idx in picks:
                    scale = 1.0 if rng.random() < 0.7 else 0.5
                    action = _actions(topo, int(idx), seed + mi)[int(rng.integers(0, 3))][1]
                    sample = _sample(topo, demands[n][int(idx)] * scale, action)
                    batch = _tensors(sample, topo, device)
                    pred = model.member_loads(batch["features"], batch["flow"], batch["incidence_path"],
                                              batch["incidence_link"], topo.n_links, member)
                    mse = torch.mean((pred - batch["target"]) ** 2)
                    loss = mse + REF["anchor_weight"] * _member_mult_penalty(member, batch["features"])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(mse.detach()))
        model.eval()
        val = aggregate_case_metrics(
            evaluate_topologies(model, ("B4",), "val", device)["cases"]
            + evaluate_topologies(model, ("UsCarrier", "Cogentco"), "val", device, limit_matrices=1)["cases"]
        )
        history.append({"epoch": epoch, "train_mse_normalised": float(np.mean(losses)),
                        "val_mae_utilisation": float(val["mae_utilisation"]),
                        "val_coverage_2std": float(val["coverage_2std"]), "elapsed_s": time.time() - t0})
        print(f"{name} epoch {epoch + 1}/{epochs} val_mae_util={history[-1]['val_mae_utilisation']:.4g} "
              f"({time.time() - t0:.0f}s)", flush=True)
    conformal = fit_conformal_scale(model, device)
    payload = {
        "name": name, **REF, "data_fraction": fraction, "epochs": epochs,
        "train_matrices_per_topology": {n: int(demands[n].shape[0]) for n in TRAIN_TOPOS},
        "state_dict": model.state_dict(), "history": history,
        "conformal_std_scale": float(conformal["conformal_std_scale"]), "conformal": conformal,
        "note": "Deliberately weakened twin for E4; same recipe as ensemble_physics_anchored with a reduced budget.",
    }
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    path = MODEL_OUT / f"ensemble_{name}.pt"
    torch.save(payload, path)
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    meta["checkpoint"] = path.name
    meta["train_seconds"] = time.time() - t0
    write_json(OUT / f"{name}_meta.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", {"levels": LEVELS, "reference_hyperparameters": REF,
                                     "reference_checkpoint": "results/digital_twin/improvements/models/"
                                                             "ensemble_physics_anchored_conformal.pt",
                                     "environment": environment()})
    for name, frac, epochs in LEVELS:
        if args.only and name not in args.only:
            continue
        if (MODEL_OUT / f"ensemble_{name}.pt").exists():
            print(f"{name}: exists, skipping (resume)")
            continue
        train_one(name, frac, epochs)


if __name__ == "__main__":
    main()
