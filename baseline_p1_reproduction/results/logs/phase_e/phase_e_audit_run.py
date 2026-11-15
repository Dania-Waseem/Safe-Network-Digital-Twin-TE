"""Phase E: B4 experimental fidelity audit vs P1 + TEAL reference data."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REF = ROOT.parent.parent / "reference" / "teal"

from config import P1_PUBLISHED, B4_SPLITS, ADMM_ITERS, RHO, EPISODES, SEEDS, ACTOR_LR, CRITIC_LR
from preprocessing.load_topology import load_topology
from preprocessing.load_traffic import load_traffic_matrices, split_indices
from preprocessing.generate_paths import generate_candidate_paths
from preprocessing.build_bipartite_graph import build_bipartite_graph
from experiments.baselines import lp_allocate, ecmp_allocate, evaluate_baseline
from optimization.constraint_adjustment import evaluate_allocation, adjust_allocation

OUT = ROOT / "results" / "logs" / "phase_e"
OUT.mkdir(parents=True, exist_ok=True)


def topo_audit():
    topo = load_topology("B4")
    caps = topo.capacities
    # diameter / avg shortest path
    try:
        und = topo.graph.to_undirected()
        asp = nx.average_shortest_path_length(und)
        diam = nx.diameter(und)
    except Exception as e:
        asp, diam = None, None
        err = str(e)
    else:
        err = None
    # compare to TEAL ref if present
    ref_path = REF / "topologies" / "B4.json"
    ref_match = None
    if ref_path.exists():
        ours = json.loads((ROOT / "data" / "topologies" / "B4.json").read_text(encoding="utf-8"))
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_match = ours == ref
    return {
        "name": topo.name,
        "num_nodes": topo.num_nodes,
        "num_edges": topo.num_edges,
        "p1_table5": {"nodes": 12, "edges": 38, "avg_sp": 2.3, "diameter": 5},
        "ours_avg_sp_undirected": asp,
        "ours_diameter_undirected": diam,
        "diameter_error": err,
        "capacity_min": float(caps.min()),
        "capacity_max": float(caps.max()),
        "capacity_mean": float(caps.mean()),
        "capacity_unique": sorted(set(float(c) for c in caps)),
        "matches_p1_nodes_edges": topo.num_nodes == 12 and topo.num_edges == 38,
        "identical_to_ref_teal_json": ref_match,
    }


def tm_audit():
    tms = load_traffic_matrices("B4")
    totals = [float(tm.sum()) for tm in tms]
    shapes = [tm.shape for tm in tms]
    # nonzero OD pairs
    nnz = [int((tm > 0).sum()) for tm in tms]
    splits = split_indices(len(tms), "B4")
    return {
        "n_matrices": len(tms),
        "shape": shapes[0],
        "all_shapes_equal": all(s == shapes[0] for s in shapes),
        "filename_pattern": "B4.json_real_{i}_1.0_traffic-matrix.pkl",
        "scale_suffix": "1.0",
        "traffic_type": "TEAL 'real' (measured/public TEAL dump), NOT gravity-synthetic",
        "mean_total_demand": float(np.mean(totals)),
        "std_total_demand": float(np.std(totals)),
        "min_total_demand": float(np.min(totals)),
        "max_total_demand": float(np.max(totals)),
        "p1_cited_b4_mean_demand": 27980.0,  # "27.98 thousand"
        "mean_vs_p1_ratio": float(np.mean(totals) / 27980.0),
        "mean_nnz_od_pairs": float(np.mean(nnz)),
        "max_possible_od": 12 * 11,
        "splits_used": splits,
        "split_source": "TEAL-style 20/8/8 — NOT published in P1",
        "train_mean_demand": float(np.mean(totals[splits["train"][0]:splits["train"][1]])),
        "test_mean_demand": float(np.mean(totals[splits["test"][0]:splits["test"][1]])),
    }


def path_audit():
    topo = load_topology("B4")
    paths = generate_candidate_paths(topo)
    # uniqueness and hop stats
    n_unique = []
    hops = []
    for (s, t), plist in paths.items():
        uniq = len({tuple(p) for p in plist})
        n_unique.append(uniq)
        for p in plist:
            hops.append(max(len(p) - 1, 0))
    # TEAL edge-disjoint pickle if available
    teal_pkl = REF / "topologies" / "paths" / "path-form" / "B4.json-4-paths_edge-disjoint-True_dist-metric-min-hop-dict.pkl"
    teal_cmp = None
    if teal_pkl.exists():
        with open(teal_pkl, "rb") as f:
            teal_paths = pickle.load(f)
        # TEAL keys may differ; try to compare overlap
        # normalize keys
        def norm_key(k):
            if isinstance(k, tuple) and len(k) == 2:
                return (int(k[0]), int(k[1]))
            return k
        teal_norm = {norm_key(k): v for k, v in teal_paths.items()}
        ours_keys = set(paths.keys())
        teal_keys = set(teal_norm.keys())
        common = ours_keys & teal_keys
        identical = 0
        partial = 0
        for k in list(common)[:132]:
            op = [tuple(p) for p in paths[k]]
            tp = teal_norm[k]
            if isinstance(tp, dict):
                tp = list(tp.values()) if tp else []
            tp = [tuple(p) for p in tp]
            if op == tp:
                identical += 1
            elif set(op) & set(tp):
                partial += 1
        teal_cmp = {
            "teal_pickle": str(teal_pkl.name),
            "ours_pairs": len(ours_keys),
            "teal_pairs": len(teal_keys),
            "common_pairs": len(common),
            "identical_path_lists_sampled": identical,
            "partial_overlap_sampled": partial,
            "note": "P1: '4 shortest paths'; TEAL public often edge-disjoint; we use Yen min-hop",
        }
    return {
        "num_od_pairs": len(paths),
        "paths_per_pair": 4,
        "algorithm": "Yen k-shortest min-hop (networkx)",
        "p1_text": "precompute 4 shortest paths",
        "mean_unique_paths_per_pair": float(np.mean(n_unique)),
        "min_unique": int(np.min(n_unique)),
        "max_unique": int(np.max(n_unique)),
        "mean_hops": float(np.mean(hops)),
        "teal_comparison": teal_cmp,
    }


def metric_audit():
    return {
        "demand_satisfaction_ours": "sum(allocated_path_flows) / sum(demands) after eqs 23-28",
        "p1_term": "average demand satisfaction rate / total satisfied demand (Fig 6, Table 6)",
        "p1_exact_formula": "NOT algebraically printed; inferred as satisfied throughput / offered demand",
        "reward_train": "r_t = F_total = sum of feasible allocations (eq 29-30)",
        "mlu": "max_e load(e)/c(e)",
        "congestion": "capacity_violations count if load > 1.001*c (ours); P1 discusses violations qualitatively",
        "comparable_if": "same TMs, capacities, path set, and post-processing feasibility",
    }


def baseline_numbers():
    topo = load_topology("B4")
    tms = load_traffic_matrices("B4")
    problem = build_bipartite_graph(topo, generate_candidate_paths(topo))
    splits = split_indices(len(tms), "B4")
    test = tms[splits["test"][0]:splits["test"][1]]
    train = tms[splits["train"][0]:splits["train"][1]]
    lp_test = evaluate_baseline(lp_allocate, problem, test)
    ecmp_test = evaluate_baseline(ecmp_allocate, problem, test)
    lp_all = evaluate_baseline(lp_allocate, problem, tms)
    return {
        "test_lp_sat": lp_test["demand_satisfaction"],
        "test_ecmp_sat": ecmp_test["demand_satisfaction"],
        "all36_lp_sat": lp_all["demand_satisfaction"],
        "p1_table6_b4": P1_PUBLISHED["B4"],
        "phase_d_grl_te_no_admm_20ep": 0.9772930398583413,
        "phase_d_grl_te_teal_admm_20ep": 0.9813108533620835,
        "archived_10seed_3ep_grl_teal": 0.9744480364024639,
        "archived_10seed_3ep_no_admm": 0.9651111453771591,
        "lp_gap_to_p1_pp": (lp_test["demand_satisfaction"] * 100) - P1_PUBLISHED["B4"]["LP"],
        "grl_gap_to_p1_pp_3ep_teal": (0.9744480364024639 * 100) - P1_PUBLISHED["B4"]["GRL-TE"],
    }


def methodology_checklist():
    return {
        "topo_nodes_edges": {"p1": "12/38", "ours": "12/38", "status": "MATCH"},
        "capacities_source": {"p1": "Teal", "ours": "TEAL B4.json", "status": "MATCH_SOURCE"},
        "tm_source": {"p1": "Teal", "ours": "TEAL real_*_1.0 pkl", "status": "MATCH_SOURCE"},
        "tm_mean_demand": {"p1": "27.98k", "ours": "~26.9k", "status": "CLOSE_NOT_IDENTICAL"},
        "candidate_paths": {"p1": "4 shortest", "ours": "Yen min-hop (not edge-disjoint)", "status": "AMBIGUOUS"},
        "train_episodes": {"p1": 3, "ours_table6_protocol": 3, "phase_d_extra": 20, "status": "MATCH_FOR_3EP"},
        "seeds": {"p1": list(range(100,1100,100)), "ours": SEEDS, "status": "MATCH"},
        "actor_lr": {"p1": 1e-4, "ours": ACTOR_LR, "status": "MATCH"},
        "critic_lr": {"p1": 1e-3, "ours": CRITIC_LR, "status": "MATCH"},
        "admm_rho": {"p1": 1.0, "ours": RHO, "status": "MATCH"},
        "admm_iters_b4": {"p1": 2, "ours": ADMM_ITERS["B4"], "status": "MATCH"},
        "admm_solver": {"p1": "unpublished argmins order F-z-xi-lambda", "ours": "Teal order for stable deploy; P1-order available but unstable", "status": "DEVIATES"},
        "tm_split": {"p1": "unspecified", "ours": "20/8/8 TEAL", "status": "UNSPECIFIED_ASSUMPTION"},
        "grad_clip": {"p1": "not stated", "ours": 1.0, "status": "EXTRA"},
        "hardware": {"p1": "RTX 4090", "ours": "local CPU/GPU may differ", "status": "DEVIATES_ENV"},
    }


def comparability_verdict(base, tm):
    # LP close => same problem family; GRL gap large
    lp_close = abs(base["lp_gap_to_p1_pp"]) < 0.5
    grl_far = abs(base["grl_gap_to_p1_pp_3ep_teal"]) > 3.0
    if lp_close and grl_far:
        return {
            "directly_comparable_to_table6_92_39": False,
            "reason": (
                "LP nearly matches P1 (same TE instance family), but GRL-TE "
                f"differs by {base['grl_gap_to_p1_pp_3ep_teal']:.2f} pp. "
                "Likely causes: path-set ambiguity, unpublished TM split, "
                "ADMM numerical solver deviation, and/or unstated eval details. "
                "Our higher sat must NOT be claimed as beating P1."
            ),
            "same_data_family": True,
            "tm_mean_ratio_to_p1": tm["mean_vs_p1_ratio"],
        }
    return {
        "directly_comparable_to_table6_92_39": False,
        "reason": "Insufficient exact protocol match",
        "same_data_family": lp_close,
        "tm_mean_ratio_to_p1": tm["mean_vs_p1_ratio"],
    }


def main():
    print("Phase E audit...")
    topo = topo_audit()
    print("topo done", topo["num_nodes"], topo["num_edges"])
    tm = tm_audit()
    print("tm mean", tm["mean_total_demand"])
    paths = path_audit()
    print("paths", paths["num_od_pairs"])
    metrics = metric_audit()
    meth = methodology_checklist()
    print("running LP/ECMP on test...")
    base = baseline_numbers()
    print("LP test", base["test_lp_sat"])
    comp = comparability_verdict(base, tm)

    # Corrections needed?
    corrections = {
        "applied": [],
        "recommended_not_applied": [
            {
                "item": "Do not retune to 92.39%",
                "why": "Would violate fidelity rules; gap is protocol/solver ambiguity",
            },
            {
                "item": "Optional path-set A/B with TEAL edge-disjoint",
                "why": "P1 says shortest not edge-disjoint; A/B would diagnose gap but is not a silent 'fix'",
            },
            {
                "item": "Publish both 3-ep (P1 protocol) and 20-ep (Phase D) numbers separately",
                "why": "Already done; Table-6 comparisons must use 3-ep + documented ADMM variant",
            },
        ],
        "necessary_code_corrections": "NONE for B4 data integrity — topology/TM source match TEAL as P1 cites; deviations are documented assumptions",
    }

    report = {
        "topology": topo,
        "traffic": tm,
        "paths": paths,
        "metrics": metrics,
        "methodology": meth,
        "baselines_and_gaps": base,
        "comparability": comp,
        "corrections": corrections,
        "overall_verdict": "PARTIALLY VERIFIED",
        "verdict_reason": (
            "B4 topology size, TEAL TM/capacity source, hyperparams, and metric "
            "definition are largely faithful; TM mean is close but not identical to "
            "P1's 27.98k; path algorithm and TM split are underspecified; ADMM "
            "solver is not exact P1; therefore 97–98% is NOT directly comparable "
            "to published 92.39% as an improvement claim."
        ),
    }
    (OUT / "phase_e_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": report["overall_verdict"], "comparability": comp, "lp": base["test_lp_sat"], "tm_mean": tm["mean_total_demand"]}, indent=2))
    print("wrote", OUT / "phase_e_audit.json")


if __name__ == "__main__":
    main()
