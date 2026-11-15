"""Train the new actor on train splits and keep the best validation checkpoint.

GEANT, Kdl, and the test slices are not read. The frozen B4 file is not read.
UsCarrier and Cogentco train matrices are used at 0.25x and 0.5x so the loss
sees traffic that can fit. B4 uses native and 0.5x. Validation for checkpoint
selection and hyperparameter tuning uses B4 native plus UsCarrier/Cogentco at
0.25x.

Extended runs write under ``results/policy/improved/`` and never overwrite the
Phase-3 baseline checkpoint at ``results/policy/general_grl.pt``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repo_paths import model_path
from twin.data import ROOT, load_split_demands, load_topology

from .general_policy import CHECKPOINT, GeneralPolicy

# Legacy 3-epoch artifact — read-only for this module; never written here.
LEGACY_CHECKPOINT = CHECKPOINT
EXPERIMENT_ROOT = ROOT / "results" / "policy" / "improved"

TRAIN_TOPOLOGIES = ("B4", "UsCarrier", "Cogentco")
HELD_OUT = ("GEANT", "Kdl")
FORBIDDEN_SPLITS = ("test",)


@dataclass
class TrainConfig:
    run_id: str = "default"
    max_epochs: int = 20
    patience: int = 5
    actor_lr: float = 1e-4
    batch_size: int = 1
    seed: int = 2026
    large_train_matrices: int = 4
    large_val_matrices: int = 2
    # Gradient clipping keeps large-graph steps from exploding.
    grad_clip: float = 1.0
    # Optional hard cap on optimiser steps (None = epochs × data only).
    max_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Small grid: LR × batch × duration. Tuned only on train/val of B4/UsCarrier/Cogentco.
SEARCH_GRID: tuple[TrainConfig, ...] = (
    TrainConfig(run_id="lr1e4_bs1_e20", actor_lr=1e-4, batch_size=1, max_epochs=20, patience=5),
    TrainConfig(run_id="lr3e4_bs2_e16", actor_lr=3e-4, batch_size=2, max_epochs=16, patience=4),
    TrainConfig(run_id="lr5e5_bs1_e24", actor_lr=5e-5, batch_size=1, max_epochs=24, patience=6),
    TrainConfig(run_id="lr1e4_bs4_e16", actor_lr=1e-4, batch_size=4, max_epochs=16, patience=4),
)


def _loss(util: torch.Tensor, fractions: torch.Tensor) -> tuple[torch.Tensor, dict]:
    overflow = torch.relu(util - 1.0).mean()
    mlu = util.max()
    clipped = fractions.clamp_min(1e-8)
    entropy = -(clipped * clipped.log()).sum(dim=-1).mean()
    loss = overflow + 0.1 * mlu - 0.001 * entropy
    return loss, {
        "loss": float(loss.detach()),
        "overflow": float(overflow.detach()),
        "mlu": float(mlu.detach()),
        "entropy": float(entropy.detach()),
    }


def train_plan(config: TrainConfig) -> tuple[tuple[str, int | None, tuple[float, ...]], ...]:
    return (
        ("B4", None, (1.0, 0.5)),
        ("UsCarrier", config.large_train_matrices, (0.25, 0.5)),
        ("Cogentco", config.large_train_matrices, (0.25, 0.5)),
    )


def val_plan(config: TrainConfig) -> tuple[tuple[str, int | None, float], ...]:
    return (
        ("B4", None, 1.0),
        ("UsCarrier", config.large_val_matrices, 0.25),
        ("Cogentco", config.large_val_matrices, 0.25),
    )


def assert_safe_plan(plan_names: list[str]) -> None:
    bad = [name for name in plan_names if name in HELD_OUT]
    if bad:
        raise RuntimeError(f"held-out topologies must not appear in training/tuning: {bad}")


class EarlyStopper:
    """Stop when validation mean MLU fails to improve for ``patience`` epochs."""

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.best: float | None = None
        self.bad_epochs = 0
        self.best_epoch = -1

    def step(self, epoch: int, val_mean_mlu: float) -> tuple[bool, bool]:
        """Return (is_best, should_stop)."""
        if self.best is None or val_mean_mlu < self.best:
            self.best = float(val_mean_mlu)
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def _mean_mlu(policy: GeneralPolicy, name: str, split: str, limit: int | None, scale: float) -> float:
    if split in FORBIDDEN_SPLITS:
        raise RuntimeError(f"refusing to read split={split} during training/tuning")
    if name in HELD_OUT:
        raise RuntimeError(f"refusing to evaluate held-out topology {name} during training/tuning")
    topo = load_topology(name)
    demands, _meta = load_split_demands(split, name)
    if limit is not None:
        demands = demands[:limit]
    values = []
    policy.model.eval()
    with torch.no_grad():
        for matrix in demands:
            values.append(float(policy.utilisation(topo, matrix * scale)[0].max()))
    policy.release(name)
    del topo
    return float(np.mean(values))


def _iter_train_jobs(
    plan: tuple[tuple[str, int | None, tuple[float, ...]], ...],
) -> list[tuple[str, int, float]]:
    jobs: list[tuple[str, int, float]] = []
    for name, limit, scales in plan:
        demands, _meta = load_split_demands("train", name)
        count = demands.shape[0] if limit is None else min(limit, demands.shape[0])
        for index in range(count):
            for scale in scales:
                jobs.append((name, index, float(scale)))
    return jobs


def _run_dir(root: Path, run_id: str) -> Path:
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_one(config: TrainConfig, experiment_root: Path | None = None) -> dict:
    """Train a single configuration; write curves and the best checkpoint under experiment_root."""
    root = Path(experiment_root or EXPERIMENT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    out = _run_dir(root, config.run_id)
    plan = train_plan(config)
    vplan = val_plan(config)
    assert_safe_plan([name for name, *_ in plan] + [name for name, *_ in vplan])

    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    policy = GeneralPolicy(device)
    optimiser = torch.optim.Adam(policy.model.parameters(), lr=config.actor_lr)
    stopper = EarlyStopper(config.patience)
    history: list[dict] = []
    best_payload: dict | None = None
    steps_done = 0
    jobs = _iter_train_jobs(plan)
    # Cache train demands per topology for the epoch loop.
    demand_cache: dict[str, np.ndarray] = {}
    for name, limit, _scales in plan:
        mats, _ = load_split_demands("train", name)
        demand_cache[name] = mats if limit is None else mats[:limit]

    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    topo_cache = {name: load_topology(name) for name, _limit, _scales in plan}

    for epoch in range(config.max_epochs):
        epoch_rows: list[dict] = []
        policy.model.train()
        optimiser.zero_grad()
        batch_losses: list[torch.Tensor] = []
        order = list(range(len(jobs)))
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(order)

        for job_i, job_idx in enumerate(order):
            if config.max_steps is not None and steps_done >= config.max_steps:
                break
            name, index, scale = jobs[job_idx]
            topo = topo_cache[name]
            matrix = demand_cache[name][index] * scale
            util, fractions = policy.utilisation(topo, matrix)
            loss, stats = _loss(util, fractions)
            batch_losses.append(loss)
            stats = {
                **stats,
                "topology": name,
                "matrix_index": index,
                "scale": scale,
                "epoch": epoch,
            }
            epoch_rows.append(stats)
            steps_done += 1

            if len(batch_losses) >= config.batch_size or job_i == len(order) - 1:
                stacked = torch.stack(batch_losses).mean()
                stacked.backward()
                if config.grad_clip is not None and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config.grad_clip)
                optimiser.step()
                optimiser.zero_grad()
                batch_losses = []

            print(
                f"{config.run_id} epoch {epoch + 1}/{config.max_epochs} "
                f"{name} tm {index} x{scale:.2f} mlu={stats['mlu']:.4f} loss={stats['loss']:.4f}",
                flush=True,
            )

        val = {
            name: _mean_mlu(policy, name, "val", limit, scale)
            for name, limit, scale in vplan
        }
        val_mean = float(np.mean(list(val.values())))
        row = {
            "epoch": epoch,
            "train_mlu": float(np.mean([item["mlu"] for item in epoch_rows])) if epoch_rows else None,
            "train_loss": float(np.mean([item["loss"] for item in epoch_rows])) if epoch_rows else None,
            "val_mlu": val,
            "val_mean_mlu": val_mean,
            "steps_done": steps_done,
        }
        history.append(row)
        print(f"{config.run_id} epoch {epoch + 1} val_mean_mlu={val_mean:.4f} {val}", flush=True)

        is_best, should_stop = stopper.step(epoch, val_mean)
        if is_best:
            best_payload = {
                "epoch": epoch,
                "val_mean_mlu": val_mean,
                "val_mlu": val,
                "normalization": "capacity and demand divided by max capacity",
                "train_topologies": list(TRAIN_TOPOLOGIES),
                "train_scales": {
                    "B4": [1.0, 0.5],
                    "UsCarrier": [0.25, 0.5],
                    "Cogentco": [0.25, 0.5],
                },
                "val_used_for_selection": val,
                "test_topologies_not_used": list(HELD_OUT),
                "config": config.to_dict(),
            }
            ckpt_path = model_path(out / "best.pt")
            # Never touch the legacy baseline path.
            if ckpt_path.resolve() == LEGACY_CHECKPOINT.resolve():
                raise RuntimeError("refusing to overwrite the existing generalized-policy checkpoint")
            policy.save(ckpt_path, best_payload)

        curve = {
            "history": history,
            "best_val_mean_mlu": stopper.best,
            "best_epoch": stopper.best_epoch,
            "stopped_early": should_stop,
            "checkpoint": str(model_path(out / "best.pt")),
            "config": config.to_dict(),
        }
        (out / "training_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
        (out / "val_metrics.json").write_text(
            json.dumps({"history": history, "best": best_payload}, indent=2),
            encoding="utf-8",
        )

        if should_stop:
            print(
                f"{config.run_id} early stop at epoch {epoch + 1} "
                f"(best epoch {stopper.best_epoch + 1}, val={stopper.best:.4f})",
                flush=True,
            )
            break
        if config.max_steps is not None and steps_done >= config.max_steps:
            print(f"{config.run_id} hit max_steps={config.max_steps}", flush=True)
            break

    summary = {
        "run_id": config.run_id,
        "best_val_mean_mlu": stopper.best,
        "best_epoch": stopper.best_epoch,
        "epochs_ran": len(history),
        "stopped_early": bool(history) and stopper.bad_epochs >= config.patience,
        "checkpoint": str(model_path(out / "best.pt")),
        "config": config.to_dict(),
        "legacy_checkpoint_untouched": str(LEGACY_CHECKPOINT),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_search(
    grid: tuple[TrainConfig, ...] | list[TrainConfig] | None = None,
    experiment_root: Path | None = None,
) -> dict:
    """Tune hyperparameters on validation MLU; keep the best run under experiment_root."""
    root = Path(experiment_root or EXPERIMENT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    configs = list(grid or SEARCH_GRID)
    results = []
    for cfg in configs:
        print(f"=== starting run {cfg.run_id} ===", flush=True)
        results.append(train_one(cfg, root))

    ranked = sorted(results, key=lambda row: (row["best_val_mean_mlu"] is None, row["best_val_mean_mlu"]))
    best = ranked[0]
    best_src = Path(best["checkpoint"])
    best_dst = model_path(root / "best_checkpoint.pt")
    if best_src.exists():
        payload = torch.load(best_src, map_location="cpu", weights_only=False)
        # Re-save via GeneralPolicy guard so frozen repo paths stay blocked.
        policy = GeneralPolicy()
        policy.model.load_state_dict(payload["model"])
        meta = {k: v for k, v in payload.items() if k != "model"}
        meta["selected_from_search"] = True
        meta["search_ranking"] = [
            {"run_id": r["run_id"], "best_val_mean_mlu": r["best_val_mean_mlu"], "best_epoch": r["best_epoch"]}
            for r in ranked
        ]
        policy.save(best_dst, meta)

    search_summary = {
        "experiment_root": str(root),
        "runs": results,
        "best_run_id": best["run_id"],
        "best_val_mean_mlu": best["best_val_mean_mlu"],
        "best_checkpoint": str(best_dst),
        "legacy_checkpoint": str(LEGACY_CHECKPOINT),
        "held_out": list(HELD_OUT),
        "train_topologies": list(TRAIN_TOPOLOGIES),
        "normalization": "capacity and demand divided by max capacity",
        "path_padding": "four slots per demand; collapsed onto real paths",
    }
    (root / "search_summary.json").write_text(json.dumps(search_summary, indent=2), encoding="utf-8")
    (root / "search_config.json").write_text(
        json.dumps([c.to_dict() for c in configs], indent=2),
        encoding="utf-8",
    )
    return search_summary


def train() -> dict:
    """Back-compat entry: run the improved search into the separate experiment directory."""
    return run_search()


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Improved generalized Graph RL policy training")
    parser.add_argument("--single", action="store_true", help="train one configuration instead of the search grid")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--large-train-matrices", type=int, default=None)
    parser.add_argument("--large-val-matrices", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="directory for new artifacts (default: results/policy/improved)",
    )
    args = parser.parse_args(argv)
    if args.single:
        cfg = TrainConfig(run_id=args.run_id)
        overrides = {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "actor_lr": args.actor_lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "large_train_matrices": args.large_train_matrices,
            "large_val_matrices": args.large_val_matrices,
            "grad_clip": args.grad_clip,
            "max_steps": args.max_steps,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return train_one(cfg, args.experiment_root)
    return run_search(experiment_root=args.experiment_root)


if __name__ == "__main__":
    main()
