"""Policy v2: one policy training run with calibrated-load TMs and a
per-topology normalised validation score.

Two changes relative to ``framework/train_policy.py`` run ``lr5e5_bs1_e24``
(the run copied to ``recommended_stable.pt``); everything else is identical
(GeneralPolicy GNN, ``_loss``, Adam lr 5e-5, batch 1, <= 24 epochs, patience 6,
grad clip 1.0, seed 2026, 4 large-graph train TMs, 2 large-graph val TMs):

1. Training TMs at **calibrated load** (calibrated-load alpha, train split only): each epoch,
   2 of the targets {0.5, 0.7, 0.9, 1.1} are sampled per TM (seeded), instead of the
   fixed scales B4 {1.0, 0.5} and UsCarrier/Cogentco {0.25, 0.5}. Same step budget
   (56 steps/epoch).
2. Selection / early stopping on the **per-topology normalised** validation score
       S = mean_{topology in B4, UsCarrier, Cogentco} mean_{val TM, target} MLU_policy / MLU_shortest
   at calibrated targets {0.5, 0.7, 0.9, 1.1}, instead of mean validation MLU (which
   Cogentco dominated). Lower is better; S < 1 means better than shortest path.

GEANT/Kdl and the held-aside TM indices 28-35 are never read. The new checkpoint is
written under ``results/phase3_final/policy_v2/`` only. ``recommended_stable.pt`` is
scored on the identical validation set for reference.

Usage:
  python -m phase3_final.train_policy_v2 train [--max-epochs N]
  python -m phase3_final.train_policy_v2 score-reference
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from framework.general_policy import GeneralPolicy
from framework.train_policy import HELD_OUT, _loss
from twin.data import ROOT, load_split_demands, load_topology

from . import routing as R
from .common import append_jsonl, environment, fmt_eta, out_dir, read_jsonl, write_json

OUT = out_dir("policy_v2")
RUN = OUT / "runs" / "v2_calload_normscore"
MODEL_RUN = ROOT / "models" / "policy_v2" / "v2_calload_normscore"  # checkpoints (Git LFS)
CAL = ROOT / "results" / "phase3_final" / "calibrated_traffic" / "scaled_tm_index.json"
REFERENCE = ROOT / "models" / "policy" / "improved" / "recommended_stable.pt"
TARGETS = (0.5, 0.7, 0.9, 1.1)
TRAIN = (("B4", None), ("UsCarrier", 4), ("Cogentco", 4))
VAL = (("B4", None), ("UsCarrier", 2), ("Cogentco", 2))
HP = {"actor_lr": 5e-5, "batch_size": 1, "max_epochs": 24, "patience": 6, "grad_clip": 1.0, "seed": 2026,
      "targets_per_tm_per_epoch": 2}


def _alphas(split: str, name: str, n: int) -> list[dict[str, float]]:
    cal = json.loads(CAL.read_text(encoding="utf-8"))["alpha"]
    return [cal[f"{name}/{split}/{i}"] for i in range(n)]


def _data(split: str, plan) -> dict[str, tuple]:
    out = {}
    for name, limit in plan:
        if name in HELD_OUT:
            raise RuntimeError(f"held-out topology {name} must not be used")
        d, meta = load_split_demands(split, name)
        if split == "val" and any(int(i) >= 28 for i in meta["matrix_indices"]):
            raise RuntimeError("held-aside TM indices must not be used")
        d = d if limit is None else d[:limit]
        out[name] = (d, _alphas(split, name, d.shape[0]))
    return out


@torch.no_grad()
def val_score(policy: GeneralPolicy, topos: dict, val: dict) -> dict:
    policy.model.eval()
    per_topo, per_topo_mlu, per_target = {}, {}, {f"{t:.1f}": [] for t in TARGETS}
    for name, (demands, alphas) in val.items():
        topo = topos[name]
        sp = R.shortest_action(topo)
        ratios, mlus = [], []
        for matrix, alpha in zip(demands, alphas):
            for t in TARGETS:
                m = matrix * alpha[f"{t:.1f}"]
                util, _ = policy.utilisation(topo, m)
                mlu = float(util.max())
                ratio = mlu / R.mlu(topo, m, sp)
                ratios.append(ratio)
                mlus.append(mlu)
                per_target[f"{t:.1f}"].append(ratio)
        per_topo[name] = float(np.mean(ratios))
        per_topo_mlu[name] = float(np.mean(mlus))
    return {"S": float(np.mean(list(per_topo.values()))), "ratio_by_topology": per_topo,
            "mlu_by_topology": per_topo_mlu,
            "ratio_by_target": {k: float(np.mean(v)) for k, v in per_target.items()}}


def train(max_epochs: int | None) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    hp = dict(HP)
    if max_epochs is not None:
        hp["max_epochs"] = max_epochs
    write_json(RUN / "config.json", {
        **hp, "train_plan": TRAIN, "val_plan": VAL, "targets": TARGETS,
        "changes_vs_recommended_stable": ["calibrated-load training TMs", "per-topology normalised val score S"],
        "reference_run": "results/policy/improved/runs/lr5e5_bs1_e24 (recommended_stable.pt)",
        "selection": "lowest S on validation (B4/UsCarrier/Cogentco val split only)",
        "environment": environment()})
    device = torch.device("cpu")
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    policy = GeneralPolicy(device)
    opt = torch.optim.Adam(policy.model.parameters(), lr=hp["actor_lr"])
    topos = {name: load_topology(name) for name, _ in TRAIN}
    train_d, val_d = _data("train", TRAIN), _data("val", VAL)
    curve_path = RUN / "training_curve.jsonl"
    history = read_jsonl(curve_path)
    best = min((h["val"]["S"] for h in history), default=None)
    bad = 0
    if history:  # resume: restore the last epoch state
        state = torch.load(MODEL_RUN / "last.pt", map_location="cpu", weights_only=False)
        policy.model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        best_epoch = min(history, key=lambda h: h["val"]["S"])["epoch"]
        bad = history[-1]["epoch"] - best_epoch
        print(f"resuming after epoch {history[-1]['epoch'] + 1}; best S={best:.4f}", flush=True)
    else:
        v0 = val_score(policy, topos, val_d)
        append_jsonl(curve_path, {"epoch": -1, "val": v0, "train_loss": None, "train_mlu": None,
                                  "seconds": 0.0, "note": "untrained initialisation"})
        print(f"init S={v0['S']:.4f} {v0['ratio_by_topology']}", flush=True)
    start = 1 + max((h["epoch"] for h in history), default=-1)
    for epoch in range(start, hp["max_epochs"]):
        if bad >= hp["patience"]:
            break
        t0 = time.time()
        rng = np.random.default_rng(hp["seed"] + epoch)
        jobs = []
        for name, (demands, alphas) in train_d.items():
            for i in range(demands.shape[0]):
                for t in rng.choice(TARGETS, size=hp["targets_per_tm_per_epoch"], replace=False):
                    jobs.append((name, i, float(alphas[i][f"{t:.1f}"]), float(t)))
        rng.shuffle(jobs)
        policy.model.train()
        losses, mlus = [], []
        for name, i, alpha, _t in jobs:
            util, frac = policy.utilisation(topos[name], train_d[name][0][i] * alpha)
            loss, stats = _loss(util, frac)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), hp["grad_clip"])
            opt.step()
            losses.append(stats["loss"])
            mlus.append(stats["mlu"])
        v = val_score(policy, topos, val_d)
        secs = time.time() - t0
        append_jsonl(curve_path, {"epoch": epoch, "val": v, "train_loss": float(np.mean(losses)),
                                  "train_mlu": float(np.mean(mlus)), "seconds": secs, "steps": len(jobs)})
        torch.save({"model": policy.model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
                   MODEL_RUN / "last.pt")
        if best is None or v["S"] < best:
            best, bad = v["S"], 0
            policy.save(MODEL_RUN / "best.pt", {"epoch": epoch, "val_score": v, "selection": "min S on validation",
                                          "config": hp, "test_topologies_not_used": list(HELD_OUT)})
        else:
            bad += 1
        remaining = (hp["max_epochs"] - epoch - 1) * secs
        print(f"epoch {epoch + 1}/{hp['max_epochs']} S={v['S']:.4f} best={best:.4f} bad={bad} "
              f"{v['ratio_by_topology']} {secs:.0f}s (max remaining {fmt_eta(remaining)})", flush=True)
    hist = read_jsonl(curve_path)
    trained = [h for h in hist if h["epoch"] >= 0]
    b = min(trained, key=lambda h: h["val"]["S"])
    write_json(RUN / "summary.json", {"best_epoch": b["epoch"], "best_val": b["val"], "epochs_ran": len(trained),
                                      "stopped_early": bad >= hp["patience"], "checkpoint": "best.pt",
                                      "total_seconds": float(sum(h["seconds"] for h in hist))})


def score_reference() -> None:
    policy = GeneralPolicy(torch.device("cpu"))
    payload = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    policy.model.load_state_dict(payload["model"])
    topos = {name: load_topology(name) for name, _ in VAL}
    v = val_score(policy, topos, _data("val", VAL))
    write_json(OUT / "reference_recommended_stable_val.json", {"checkpoint": "results/policy/improved/recommended_stable.pt",
                                                                 "val": v, "same_val_set_as_v2": True})
    print(json.dumps(v, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--max-epochs", type=int, default=None)
    sub.add_parser("score-reference")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.max_epochs)
    else:
        score_reference()


if __name__ == "__main__":
    main()
