"""Validate one WAN: graph stats, paths, LP/ECMP, optional 1-seed GRL-TE.

Usage:
  python experiments/phase_f_eval_topology.py --name B4 --skip-train
  python experiments/phase_f_eval_topology.py --name GEANT --train --seeds 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ADMM_ITERS, EPISODES, P1_PUBLISHED  # noqa: E402
from experiments.baselines import ecmp_allocate, evaluate_baseline, lp_allocate  # noqa: E402
from models.a2c import GRLTEModel  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402
from training.train import train_model  # noqa: E402

LOG = ROOT / "results" / "logs" / "phase_f"

P1_SIZE = {
    "B4": (12, 38),
    "GEANT": (22, 72),
    "UsCarrier": (158, 378),
    "Cogentco": (197, 486),
    "Kdl": (754, 1790),
    "ASN": (1739, 8558),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--skip-lp", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=[100])
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--max-test-tms", type=int, default=8)
    args = parser.parse_args()
    name = args.name
    LOG.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    topo = load_topology(name)
    tms = load_traffic_matrices(name)
    splits = split_indices(len(tms), name)
    test = tms[splits["test"][0] : splits["test"][1]][: args.max_test_tms]
    train = tms[splits["train"][0] : splits["train"][1]]

    print(f"Generating paths for {name} ({topo.num_nodes} nodes)...")
    paths = generate_candidate_paths(topo)
    problem = build_bipartite_graph(topo, paths)
    totals = [float(tm.sum()) for tm in tms]
    rec = {
        "name": name,
        "nodes": topo.num_nodes,
        "edges": topo.num_edges,
        "p1_table5": list(P1_SIZE.get(name, (None, None))),
        "size_match": (topo.num_nodes, topo.num_edges) == P1_SIZE.get(name),
        "n_tms": len(tms),
        "tm_shape": list(tms[0].shape),
        "mean_demand": float(np.mean(totals)),
        "test_mean_demand": float(np.mean([float(tm.sum()) for tm in test])),
        "n_demands": problem.num_demands,
        "n_path_nodes": problem.num_demands * problem.num_paths,
        "splits": {k: list(v) for k, v in splits.items()},
        "admm_iters": ADMM_ITERS.get(name, 2),
        "p1_table6": P1_PUBLISHED.get(name),
        "path_gen": "Yen k-shortest min-hop (P1: '4 shortest paths')",
        "setup_s": time.perf_counter() - t0,
    }

    if not args.skip_lp:
        print("LP/ECMP on test...")
        rec["ecmp"] = evaluate_baseline(ecmp_allocate, problem, test)
        rec["ecmp"].pop("per_tm", None)
        try:
            rec["lp"] = evaluate_baseline(lp_allocate, problem, test)
            rec["lp"].pop("per_tm", None)
        except Exception as exc:
            rec["lp_error"] = str(exc)

    rec["grl_te"] = []
    if args.train:
        for seed in args.seeds:
            print(f"Train GRL-TE seed={seed} episodes={args.episodes}")
            set_seed(seed)
            model = GRLTEModel()
            info = train_model(model, problem, train, episodes=args.episodes, log_every=0)
            teal = evaluate_model(
                model, problem, test, use_admm=True, admm_variant="teal",
                num_admm_iters=ADMM_ITERS.get(name, 2),
            )
            noad = evaluate_model(model, problem, test, use_admm=False)
            rec["grl_te"].append(
                {
                    "seed": seed,
                    "train_time_s": info["train_time_s"],
                    "sat_teal_admm": teal["demand_satisfaction"],
                    "sat_no_admm": noad["demand_satisfaction"],
                    "mlu_teal": teal["max_link_utilization"],
                    "viol_teal": teal["capacity_violations"],
                    "runtime_s": teal["runtime_s"],
                }
            )
            print("  sat teal", teal["demand_satisfaction"], "no_admm", noad["demand_satisfaction"])

    out = LOG / f"{name}_eval.json"
    # drop huge tensors
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
