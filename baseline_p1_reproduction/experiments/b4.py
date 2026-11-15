"""Milestone run: reproduce GRL-TE on Google B4 (P1's smallest WAN)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ADMM_ITERS,
    EPISODES,
    GRAPH_DIR,
    LOG_DIR,
    MODEL_DIR,
    P1_PUBLISHED,
    TABLE_DIR,
)
from experiments.baselines import (  # noqa: E402
    ecmp_allocate,
    evaluate_baseline,
    lp_allocate,
    shortest_path_allocate,
)
from models.a2c import GRLTEModel  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402
from training.train import train_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(history, comparison_rows, test_rows, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = [h.get("satisfaction", h["reward"]) for h in history]
    ax.plot([h["step"] for h in history], y, color="#1f77b4")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Demand satisfaction (train TM)")
    ax.set_title("B4 one-step A2C training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "b4_training_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["method"] for r in comparison_rows]
    values = [100.0 * r["demand_satisfaction"] if r["demand_satisfaction"] <= 1.1 else r["demand_satisfaction"] for r in comparison_rows]
    colors = ["#4c78a8" if "GRL-TE (ours)" in n else "#f58518" if "P1" in n else "#54a24b" for n in names]
    ax.bar(range(len(names)), values, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Demand satisfaction (%)")
    ax.set_title("B4 test-set comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "b4_method_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([100.0 * r["demand_satisfaction"] for r in test_rows], marker="o")
    ax.set_xlabel("Test TM index")
    ax.set_ylabel("Demand satisfaction (%)")
    ax.set_title("GRL-TE on held-out B4 traffic matrices")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "b4_per_tm_satisfaction.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce GRL-TE on B4")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to verify the pipeline")
    parser.add_argument("--skip-lp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}  seed={args.seed}")

    topology = load_topology("B4")
    tms = load_traffic_matrices("B4")
    splits = split_indices(len(tms), "B4")
    print(
        f"B4: {topology.num_nodes} nodes, {topology.num_edges} directed links, "
        f"{len(tms)} TMs  splits={splits}"
    )

    path_dict = generate_candidate_paths(topology)
    n_paths = {len(v) for v in path_dict.values()}
    print(f"Candidate paths: {len(path_dict)} demands, path-counts={n_paths}")

    problem = build_bipartite_graph(topology, path_dict, device=device)
    print(
        f"Bipartite graph: {problem.num_edges} link nodes + "
        f"{problem.num_demands * problem.num_paths} path nodes, "
        f"{problem.p2e.size(1)} incidences"
    )

    train_tms = tms[splits["train"][0] : splits["train"][1]]
    val_tms = tms[splits["val"][0] : splits["val"][1]]
    test_tms = tms[splits["test"][0] : splits["test"][1]]
    if args.smoke:
        train_tms = train_tms[:3]
        val_tms = val_tms[:2]
        test_tms = test_tms[:2]
        args.episodes = min(args.episodes, 1)
        print("SMOKE MODE: 3 train TMs, 1 episode")

    model = GRLTEModel().to(device)
    dummy_h0 = problem.initial_features(problem.demand_vector(train_tms[0], device))
    path_emb = model.topoflownet(dummy_h0, problem.bipartite_index, problem.num_edges)
    print(f"TopoFlowNet path embedding: {tuple(path_emb.shape)} (expect (*, 7))")
    if path_emb.size(-1) != 7:
        raise RuntimeError("TopoFlowNet output dim is not 7")

    print("Training one-step A2C...")
    train_info = train_model(model, problem, train_tms, episodes=args.episodes)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_DIR / f"grl_te_b4_seed{args.seed}.pt"
    torch.save({"model": model.state_dict(), "seed": args.seed}, ckpt)
    print(f"Saved {ckpt}  train_time={train_info['train_time_s']:.2f}s")

    print("Evaluating GRL-TE on test TMs...")
    ours = evaluate_model(model, problem, test_tms, use_admm=True, num_admm_iters=ADMM_ITERS["B4"])
    ours_no_admm = evaluate_model(model, problem, test_tms, use_admm=False)
    print(
        f"GRL-TE + ADMM  sat={100*ours['demand_satisfaction']:.2f}%  "
        f"MLU={ours['max_link_utilization']:.3f}  "
        f"viol={ours['capacity_violations']:.2f}  "
        f"time={ours['runtime_s']:.4f}s"
    )
    print(
        f"GRL-TE no ADMM sat={100*ours_no_admm['demand_satisfaction']:.2f}%  "
        f"(P1 B4 ablation drop was 0.68 points)"
    )

    print("Evaluating baselines on the same test TMs...")
    ecmp = evaluate_baseline(ecmp_allocate, problem, test_tms)
    sp = evaluate_baseline(shortest_path_allocate, problem, test_tms)
    lp = {"demand_satisfaction": float("nan"), "max_link_utilization": float("nan"), "runtime_s": float("nan"), "capacity_violations": float("nan")}
    if not args.skip_lp:
        lp = evaluate_baseline(lp_allocate, problem, test_tms)
        print(f"LP             sat={100*lp['demand_satisfaction']:.2f}%  time={lp['runtime_s']:.4f}s")
    print(f"ECMP           sat={100*ecmp['demand_satisfaction']:.2f}%")
    print(f"Shortest path  sat={100*sp['demand_satisfaction']:.2f}%")

    p1_b4 = P1_PUBLISHED["B4"]
    comparison = [
        {"method": "Shortest path", "demand_satisfaction": sp["demand_satisfaction"], "source": "this run"},
        {"method": "ECMP", "demand_satisfaction": ecmp["demand_satisfaction"], "source": "this run"},
        {"method": "LP (ours)", "demand_satisfaction": lp["demand_satisfaction"], "source": "this run"},
        {"method": "GRL-TE no ADMM", "demand_satisfaction": ours_no_admm["demand_satisfaction"], "source": "this run"},
        {"method": "GRL-TE (ours)", "demand_satisfaction": ours["demand_satisfaction"], "source": "this run"},
        {"method": "P1 Teal", "demand_satisfaction": p1_b4["Teal"] / 100.0, "source": "P1 Table 6"},
        {"method": "P1 GRL-TE", "demand_satisfaction": p1_b4["GRL-TE"] / 100.0, "source": "P1 Table 6"},
        {"method": "P1 LP", "demand_satisfaction": p1_b4["LP"] / 100.0, "source": "P1 Table 6"},
    ]
    for row in comparison:
        print(f"  {row['method']:18s} {100*row['demand_satisfaction']:6.2f}%  ({row['source']})")

    table_rows = [
        {
            "method": "GRL-TE (ours)",
            "demand_satisfaction": ours["demand_satisfaction"],
            "max_link_utilization": ours["max_link_utilization"],
            "capacity_violations": ours["capacity_violations"],
            "runtime_s": ours["runtime_s"],
        },
        {
            "method": "GRL-TE no ADMM",
            "demand_satisfaction": ours_no_admm["demand_satisfaction"],
            "max_link_utilization": ours_no_admm["max_link_utilization"],
            "capacity_violations": ours_no_admm["capacity_violations"],
            "runtime_s": ours_no_admm["runtime_s"],
        },
        {
            "method": "LP",
            "demand_satisfaction": lp["demand_satisfaction"],
            "max_link_utilization": lp["max_link_utilization"],
            "capacity_violations": lp.get("capacity_violations", float("nan")),
            "runtime_s": lp["runtime_s"],
        },
        {
            "method": "ECMP",
            "demand_satisfaction": ecmp["demand_satisfaction"],
            "max_link_utilization": ecmp["max_link_utilization"],
            "capacity_violations": ecmp["capacity_violations"],
            "runtime_s": ecmp["runtime_s"],
        },
        {
            "method": "Shortest path",
            "demand_satisfaction": sp["demand_satisfaction"],
            "max_link_utilization": sp["max_link_utilization"],
            "capacity_violations": sp["capacity_violations"],
            "runtime_s": sp["runtime_s"],
        },
    ]
    save_csv(TABLE_DIR / "b4_test_metrics.csv", table_rows)

    test_tm_rows = [
        {
            "tm_index": i,
            "demand_satisfaction": row["demand_satisfaction"],
            "max_link_utilization": row["max_link_utilization"],
            "runtime_s": row["runtime_s"],
        }
        for i, row in enumerate(ours["per_tm"])
    ]
    save_csv(TABLE_DIR / "b4_per_tm.csv", test_tm_rows)
    save_csv(TABLE_DIR / "b4_training_history.csv", train_info["history"])

    plot_results(train_info["history"], comparison, ours["per_tm"], GRAPH_DIR)

    summary = {
        "seed": args.seed,
        "episodes": args.episodes,
        "device": str(device),
        "train_time_s": train_info["train_time_s"],
        "ours": {k: v for k, v in ours.items() if k != "per_tm"},
        "ours_no_admm": {k: v for k, v in ours_no_admm.items() if k != "per_tm"},
        "lp": {k: v for k, v in lp.items() if k != "per_tm"},
        "ecmp": {k: v for k, v in ecmp.items() if k != "per_tm"},
        "shortest_path": {k: v for k, v in sp.items() if k != "per_tm"},
        "p1_published_b4": p1_b4,
        "notes": [
            "P1 does not publish the TM split; we use TEAL's 20/8/8 split of the 36 public B4 matrices.",
            "P1 ADMM subproblems (49)-(51) are implemented as closed-form minimizers of (48) plus ReLU.",
            "GNN inputs are scaled by max capacity; ADMM/reward use physical units.",
            "Reward is F_total / total_demand, a positive constant times P1 eq. (30) within each TM.",
        ],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"b4_seed{args.seed}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote tables to {TABLE_DIR} and figures to {GRAPH_DIR}")


if __name__ == "__main__":
    main()
