"""Link-failure robustness on B4 (P1 Section 5.5)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MODEL_DIR  # noqa: E402
from models.a2c import GRLTEModel  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402


def fail_links(problem, n_fail: int, rng: random.Random) -> None:
    idx = rng.sample(range(problem.num_edges), n_fail)
    problem.capacities[idx] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--failures", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    topology = load_topology("B4")
    tms = load_traffic_matrices("B4")
    splits = split_indices(len(tms), "B4")
    test_tms = tms[splits["test"][0] : splits["test"][1]]
    problem = build_bipartite_graph(topology, generate_candidate_paths(topology))
    model = GRLTEModel()
    ckpt = Path(args.ckpt) if args.ckpt else MODEL_DIR / f"grl_te_b4_seed{args.seed}.pt"
    if not ckpt.exists():
        raise SystemExit(f"Train first: missing {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])

    rng = random.Random(args.seed)
    original_cap = problem.capacities.clone()
    for n_fail in args.failures:
        problem.capacities = original_cap.clone()
        if n_fail:
            fail_links(problem, n_fail, rng)
        summary = evaluate_model(model, problem, test_tms, use_admm=True)
        print(
            f"{n_fail} failed links: sat={100*summary['demand_satisfaction']:.2f}%  "
            f"MLU={summary['max_link_utilization']:.3f}"
        )


if __name__ == "__main__":
    main()
