"""Train and evaluate the ensemble digital twin on the Step 4 splits.

Train topologies: B4, UsCarrier, Cogentco (early traffic indices).
Validation: the same topologies, later indices.
Test: unseen GEANT (FIGRET test histories) and Kdl.

Ground truth is the path-routing simulator in ``physics.link_load``.
Uncertainty is the variance across ensemble members.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import READY, ROOT, load_split_demands, load_topology
from .model import EnsembleTwin
from .physics import (
    demand_per_path,
    dirichlet_split,
    equal_split,
    link_load,
    od_demands,
    shortest_split,
)

RESULT_DIR = ROOT / "results" / "twin"
MODEL_DIR = ROOT / "models" / "twin"  # checkpoints (Git LFS)
PRED_DIR = RESULT_DIR / "predictions"

N_MEMBERS = 5
HIDDEN = 32
EPOCHS = 8
LR = 1e-3
BASE_SEED = 2026
# GEANT test is large but each graph is small. Kdl graphs are large, so the
# identical toy matrices are scored on a fixed action set once per matrix.
GEANT_EVAL_STRIDE = 10


def _actions(topo, matrix_index: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed + matrix_index)
    return [
        ("equal", equal_split(topo.n_paths)),
        ("shortest", shortest_split(topo.n_paths, topo.hops)),
        ("dirichlet", dirichlet_split(topo.n_paths, rng)),
    ]


def _features(demand_path: np.ndarray, action: np.ndarray, hops: np.ndarray, denom: float) -> np.ndarray:
    demand_norm = demand_path / denom
    hop_norm = hops.astype(np.float64) / 16.0
    return np.stack([demand_norm, action, hop_norm], axis=1).astype(np.float32)


def _sample(topo, matrix: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    od = od_demands(matrix)
    demand_path = demand_per_path(od, topo.demand_row)
    load = link_load(demand_path, action, topo.incidence_path, topo.incidence_link, topo.n_links)
    return {
        "features": _features(demand_path, action, topo.hops, topo.denominator),
        "flow": (demand_path * action / topo.denominator).astype(np.float32),
        "target": (load / topo.denominator).astype(np.float32),
        "load": load.astype(np.float64),
    }


def _tensors(sample: dict, topo, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(sample["features"]).to(device),
        "flow": torch.from_numpy(sample["flow"]).to(device),
        "target": torch.from_numpy(sample["target"]).to(device),
        "incidence_path": torch.from_numpy(topo.incidence_path).to(device),
        "incidence_link": torch.from_numpy(topo.incidence_link).to(device),
    }


def _metrics(true_load: np.ndarray, mean_load: np.ndarray, std_load: np.ndarray, capacity: np.ndarray) -> dict:
    err = mean_load - true_load
    util_true = true_load / np.maximum(capacity, 1e-12)
    util_hat = mean_load / np.maximum(capacity, 1e-12)
    std = np.maximum(std_load, 1e-12)
    abs_err = np.abs(err)
    coverage_1 = float(np.mean(abs_err <= std))
    coverage_2 = float(np.mean(abs_err <= 2.0 * std))
    if abs_err.std() < 1e-12 or std.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(std, abs_err)[0, 1])
    return {
        "links": int(true_load.size),
        "mae_load": float(np.mean(abs_err)),
        "rmse_load": float(np.sqrt(np.mean(err ** 2))),
        "mae_normalised": float(np.mean(np.abs(err) / max(float(np.max(capacity)), 1e-12))),
        "mae_utilisation": float(np.mean(np.abs(util_hat - util_true))),
        "mlu_true": float(np.max(util_true)) if util_true.size else None,
        "mlu_pred": float(np.max(util_hat)) if util_hat.size else None,
        "mlu_abs_error": float(abs(np.max(util_hat) - np.max(util_true))) if util_true.size else None,
        "mean_predictive_std": float(np.mean(std_load)),
        "coverage_1std": coverage_1,
        "coverage_2std": coverage_2,
        "std_error_correlation": corr,
        "finite": bool(np.isfinite(mean_load).all() and np.isfinite(std_load).all()),
    }


def _predict_numpy(model: EnsembleTwin, sample: dict, topo, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    batch = _tensors(sample, topo, device)
    model.eval()
    with torch.no_grad():
        mean, var, _ = model.predict(
            batch["features"],
            batch["flow"],
            batch["incidence_path"],
            batch["incidence_link"],
            topo.n_links,
        )
    mean_load = mean.cpu().numpy().astype(np.float64) * topo.denominator
    std_load = np.sqrt(np.maximum(var.cpu().numpy().astype(np.float64), 0.0)) * topo.denominator
    return mean_load, std_load


def _iter_cases(topo, demands: np.ndarray, seed: int, stride: int = 1):
    for index in range(0, demands.shape[0], stride):
        for name, action in _actions(topo, index, seed):
            yield index, name, action


def evaluate_split(model, topo, split: str, device: torch.device, stride: int = 1) -> dict:
    demands, meta = load_split_demands(split, topo.name)
    rows = []
    mean_rows = []
    std_rows = []
    true_rows = []
    for index, action_name, action in _iter_cases(topo, demands, seed=BASE_SEED + 17, stride=stride):
        sample = _sample(topo, demands[index], action)
        mean_load, std_load = _predict_numpy(model, sample, topo, device)
        stats = _metrics(sample["load"], mean_load, std_load, topo.capacity)
        stats.update({"matrix_index": int(meta["matrix_indices"][index]), "action": action_name})
        rows.append(stats)
        mean_rows.append(mean_load.astype(np.float32))
        std_rows.append(std_load.astype(np.float32))
        true_rows.append(sample["load"].astype(np.float32))
    summary = _aggregate(rows)
    summary["topology"] = topo.name
    summary["split"] = split
    summary["cases"] = len(rows)
    summary["stride"] = stride
    summary["traffic_kind"] = meta["traffic_kind"]
    np.savez_compressed(
        PRED_DIR / f"{split}_{topo.name}.npz",
        mean_load=np.stack(mean_rows),
        std_load=np.stack(std_rows),
        true_load=np.stack(true_rows),
        matrix_index=np.asarray([r["matrix_index"] for r in rows], dtype=np.int32),
        action=np.asarray([r["action"] for r in rows]),
    )
    return {"summary": summary, "cases": rows}


def _aggregate(rows: list[dict]) -> dict:
    keys = [
        "mae_load",
        "rmse_load",
        "mae_utilisation",
        "mlu_abs_error",
        "mean_predictive_std",
        "coverage_1std",
        "coverage_2std",
        "std_error_correlation",
    ]
    out = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    out["all_finite"] = all(row["finite"] for row in rows)
    return out


def train() -> dict:
    device = torch.device("cpu")
    torch.manual_seed(BASE_SEED)
    model = EnsembleTwin(n_members=N_MEMBERS, hidden=HIDDEN).to(device)
    optimisers = [
        torch.optim.Adam(member.parameters(), lr=LR) for member in model.members
    ]
    train_names = ("B4", "UsCarrier", "Cogentco")
    topologies = {name: load_topology(name) for name in train_names}
    demands = {name: load_split_demands("train", name)[0] for name in train_names}

    history = []
    for epoch in range(EPOCHS):
        losses = []
        order = list(train_names)
        for member_index, (member, opt) in enumerate(zip(model.members, optimisers)):
            rng = np.random.default_rng(BASE_SEED + epoch * 100 + member_index)
            member.train()
            rng.shuffle(order)
            for name in order:
                topo = topologies[name]
                matrix_ids = rng.choice(demands[name].shape[0], size=demands[name].shape[0], replace=True)
                for matrix_index in matrix_ids:
                    # Each member sees a bootstrap of matrices and one random action family.
                    action_name, action = _actions(topo, int(matrix_index), BASE_SEED + member_index)[
                        int(rng.integers(0, 3))
                    ]
                    sample = _sample(topo, demands[name][int(matrix_index)], action)
                    batch = _tensors(sample, topo, device)
                    pred = model.member_loads(
                        batch["features"],
                        batch["flow"],
                        batch["incidence_path"],
                        batch["incidence_link"],
                        topo.n_links,
                        member,
                    )
                    target = batch["target"]
                    loss = torch.mean((pred - target) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                    del batch, pred, loss
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "mse_normalised_load": mean_loss})
        print(f"epoch {epoch + 1}/{EPOCHS} mse={mean_loss:.6e}", flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for index, member in enumerate(model.members):
        torch.save(member.state_dict(), MODEL_DIR / f"member_{index}.pt")
    torch.save(
        {
            "n_members": N_MEMBERS,
            "hidden": HIDDEN,
            "epochs": EPOCHS,
            "state_dict": model.state_dict(),
        },
        MODEL_DIR / "ensemble.pt",
    )

    # Drop training graphs before loading the large test topologies.
    del topologies, demands

    reports = []
    for name in train_names:
        topo = load_topology(name)
        reports.append(evaluate_split(model, topo, "train", device))
        reports.append(evaluate_split(model, topo, "val", device))
        del topo
    geant = load_topology("GEANT")
    reports.append(evaluate_split(model, geant, "test", device, stride=GEANT_EVAL_STRIDE))
    del geant
    kdl = load_topology("Kdl")
    reports.append(evaluate_split(model, kdl, "test", device, stride=1))
    del kdl

    payload = {
        "architecture": {
            "type": "ensemble of path-multiplier MLPs with link scatter",
            "members": N_MEMBERS,
            "hidden": HIDDEN,
            "features": ["demand/max_capacity", "routing_fraction", "hop_count/16"],
            "prediction": "link load = scatter(demand * action * softplus(MLP(features)))",
            "uncertainty": "variance across ensemble members, reported as std of link load",
            "label": "simulator link load from the same routing action",
        },
        "training": {
            "topologies": list(train_names),
            "epochs": EPOCHS,
            "lr": LR,
            "seed": BASE_SEED,
            "loss": "mean squared error on load / max(link capacity)",
            "bootstrap": "each member resamples training matrices with replacement each epoch",
            "actions": ["equal split", "shortest path", "Dirichlet split"],
            "history": history,
        },
        "evaluation": {
            "train_and_val": "all matrices, three actions",
            "GEANT_test_stride": GEANT_EVAL_STRIDE,
            "Kdl_test": "all Step 4 test matrices (identical toy traffic), three actions",
            "unseen_topologies": ["GEANT", "Kdl"],
            "ready_topologies": list(READY),
            "excluded": "ASN (path merge incomplete; shards not read)",
        },
        "splits": [item["summary"] for item in reports],
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULT_DIR / "TWIN_REPORT.md").write_text(_report(payload), encoding="utf-8")
    print("Wrote", RESULT_DIR / "metrics.json", flush=True)
    return payload


def _report(payload: dict) -> str:
    lines = [
        "# Phase 3 Step 5 — digital twin",
        "",
        "The twin predicts the link load that follows a proposed routing action and",
        "reports uncertainty as the standard deviation across an ensemble.",
        "It trains on B4, UsCarrier and Cogentco. GEANT and Kdl are unseen test topologies.",
        "ASN checkpoints are not used.",
        "",
        "## Architecture",
        "",
        "Each ensemble member is a two-layer MLP on three path features:",
        "normalised demand, the routing fraction on that path, and hop count.",
        "The network outputs a positive multiplier. Link load is the sum, over paths",
        "that traverse the link, of `demand * routing fraction * multiplier`.",
        "The initial multiplier is 1, which is the exact flow simulator.",
        "Training lets the multiplier depend on the features; disagreement across",
        "five independently bootstrapped members is the uncertainty.",
        "",
        "The same scatter works for every graph size, so a model trained on the",
        "smaller topologies can be applied to GEANT and Kdl without a new head.",
        "",
        "## Training",
        "",
        "- Loss: MSE of predicted versus simulated load, divided by the topology's maximum capacity.",
        "- Actions: equal split, shortest-path split, and Dirichlet splits.",
        "- Labels come from `twin.physics.link_load`, not from a measured counter.",
        "- Optimiser: Adam, learning rate 0.001, 8 epochs, 5 members.",
        "",
        "## Results",
        "",
        "| Split | Topology | Cases | MAE utilisation | MLU abs error | 1-std coverage | Std/error corr | Finite |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["splits"]:
        lines.append(
            "| {split} | {topology} | {cases} | {mae_utilisation:.6g} | {mlu_abs_error:.6g} | "
            "{coverage_1std:.3f} | {std_error_correlation:.3f} | {finite} |".format(
                split=row["split"],
                topology=row["topology"],
                cases=row["cases"],
                mae_utilisation=row["mae_utilisation"],
                mlu_abs_error=row["mlu_abs_error"],
                coverage_1std=row["coverage_1std"],
                std_error_correlation=row["std_error_correlation"],
                finite="yes" if row["all_finite"] else "NO",
            )
        )
    lines.extend(
        [
            "",
            "Coverage near 0.68 means the ensemble std matches a normal error scale.",
            "GEANT test matrices are scored with stride "
            f"{payload['evaluation']['GEANT_test_stride']} so the run stays within memory;",
            "every selected matrix still uses the published FIGRET test histories only.",
            "",
            "Saved artifacts:",
            "",
            "- `results/twin/models/ensemble.pt`",
            "- `results/twin/models/member_*.pt`",
            "- `results/twin/metrics.json`",
            "- `results/twin/predictions/<split>_<topology>.npz`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
