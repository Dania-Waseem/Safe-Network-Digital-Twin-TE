"""Compare P1 ADMM vs Teal ADMM on identical actor warm-starts (B4).

Does not retrain. Uses a fixed seed-100 checkpoint if present; otherwise a
fresh deterministic actor (documented). Same fractions fed to both solvers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ADMM_ITERS, MODEL_DIR  # noqa: E402
from models.a2c import GRLTEModel  # noqa: E402
from optimization.admm import admm_refine_p1, admm_refine_teal  # noqa: E402
from optimization.constraint_adjustment import (  # noqa: E402
    adjust_allocation,
    evaluate_allocation,
)
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402


def main():
    topo = load_topology("B4")
    problem = build_bipartite_graph(topo, generate_candidate_paths(topo))
    tms = load_traffic_matrices("B4")
    lo, hi = split_indices(len(tms), "B4")["test"]
    test_tms = tms[lo:hi]
    iters = ADMM_ITERS["B4"]

    model = GRLTEModel()
    ckpt = MODEL_DIR / "grl_te_b4_seed100.pt"
    source = "untrained_seed100_init"
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "model" in blob:
            model.load_state_dict(blob["model"])
        elif isinstance(blob, dict) and "state_dict" in blob:
            model.load_state_dict(blob["state_dict"])
        else:
            model.load_state_dict(blob)
        source = str(ckpt.name)

    model.eval()
    rows = []
    with torch.no_grad():
        for i, tm in enumerate(test_tms):
            demands = problem.demand_vector(tm)
            h0 = problem.initial_features(demands)
            out = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)
            frac = out["fractions"]
            raw = (frac * demands.unsqueeze(-1)).reshape(-1)

            x_p1, tr_p1 = admm_refine_p1(frac, demands, problem, num_iters=iters, return_trace=True)
            x_teal, tr_teal = admm_refine_teal(frac, demands, problem, num_iters=iters, return_trace=True)

            m_raw = evaluate_allocation(adjust_allocation(raw, demands, problem), demands, problem)
            m_p1 = evaluate_allocation(adjust_allocation(x_p1, demands, problem), demands, problem)
            m_teal = evaluate_allocation(adjust_allocation(x_teal, demands, problem), demands, problem)

            rows.append(
                {
                    "tm_index": lo + i,
                    "sat_no_admm": m_raw["demand_satisfaction"],
                    "sat_p1": m_p1["demand_satisfaction"],
                    "sat_teal": m_teal["demand_satisfaction"],
                    "p1_flow_over_demand": tr_p1[-1]["flow_over_demand"],
                    "teal_flow_over_demand": tr_teal[-1]["flow_over_demand"],
                    "p1_g1": tr_p1[-1]["g1_abs_mean"],
                    "p1_g3": tr_p1[-1]["g3_abs_mean"],
                    "p1_g4": tr_p1[-1]["g4_abs_mean"],
                    "teal_g1": tr_teal[-1]["g1_abs_mean"],
                    "teal_g3": tr_teal[-1]["g3_abs_mean"],
                    "teal_g4": tr_teal[-1]["g4_abs_mean"],
                    "flow_l1_diff": float((x_p1.double() - x_teal.double()).abs().sum()),
                }
            )

    def mean(key):
        return float(sum(r[key] for r in rows) / len(rows))

    summary = {
        "checkpoint": source,
        "n_test_tms": len(rows),
        "admm_iters": iters,
        "mean_sat_no_admm": mean("sat_no_admm"),
        "mean_sat_p1": mean("sat_p1"),
        "mean_sat_teal": mean("sat_teal"),
        "mean_p1_flow_over_demand": mean("p1_flow_over_demand"),
        "mean_teal_flow_over_demand": mean("teal_flow_over_demand"),
        "mean_p1_g3": mean("p1_g3"),
        "mean_teal_g3": mean("teal_g3"),
        "mean_flow_l1_diff": mean("flow_l1_diff"),
        "per_tm": rows,
        "p1_published_b4": 0.9239,
        "note": "No retuning; identical actor fractions for both solvers.",
    }
    out_path = Path(__file__).with_name("admm_p1_vs_teal.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_tm"}, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
