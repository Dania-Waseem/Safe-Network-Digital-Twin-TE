"""B4 baseline-freeze runner: explicit ADMM channels, residual logging, no archive overwrite.

Usage (from baseline_p1_reproduction/):
  python BASELINE_FREEZE/run_b4_baseline_freeze.py
  python BASELINE_FREEZE/run_b4_baseline_freeze.py --retrain-seed 100

Outputs only under BASELINE_FREEZE/{logs,models,results}/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

FREEZE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FREEZE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FREEZE_DIR) not in sys.path:
    sys.path.insert(0, str(FREEZE_DIR))

import frozen_b4_config as cfg  # noqa: E402
from experiments.baselines import (  # noqa: E402
    ecmp_allocate,
    evaluate_baseline,
    lp_allocate,
    shortest_path_allocate,
)
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
from training.train import train_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


@torch.no_grad()
def evaluate_channel(
    model: GRLTEModel,
    problem,
    tms: list[np.ndarray],
    admm_variant: str | None,
) -> dict:
    """Deploy eval with explicit ADMM channel.

    admm_variant: 'teal' | 'p1' | None (no ADMM).
    Logs pre-adjust ADMM residuals when ADMM is used.
    """
    model.eval()
    device = problem.capacities.device
    rows = []
    for tm in tms:
        demands = problem.demand_vector(tm, device=device)
        h0 = problem.initial_features(demands, scale=cfg.H0_SCALING)
        out = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)
        flow = (out["fractions"] * demands.unsqueeze(-1)).reshape(-1)
        residual = None
        if admm_variant == "teal":
            flow, trace = admm_refine_teal(
                out["fractions"], demands, problem, num_iters=cfg.ADMM_ITERS_B4, return_trace=True
            )
            residual = trace[-1] if trace else None
        elif admm_variant == "p1":
            flow, trace = admm_refine_p1(
                out["fractions"], demands, problem, num_iters=cfg.ADMM_ITERS_B4, return_trace=True
            )
            residual = trace[-1] if trace else None
        elif admm_variant is not None:
            raise ValueError(f"Unknown ADMM variant: {admm_variant}")

        flow = adjust_allocation(flow, demands, problem)
        metrics = evaluate_allocation(flow, demands, problem)
        if residual is not None:
            metrics["admm_pre_adjust"] = {
                "flow_over_demand": residual.get("flow_over_demand"),
                "g1_abs_mean": residual.get("g1_abs_mean"),
                "g3_abs_mean": residual.get("g3_abs_mean"),
                "g4_abs_mean": residual.get("g4_abs_mean"),
                "update_order": residual.get("update_order"),
            }
        rows.append(metrics)

    sat = [r["demand_satisfaction"] for r in rows]
    fod = [
        r["admm_pre_adjust"]["flow_over_demand"]
        for r in rows
        if "admm_pre_adjust" in r and r["admm_pre_adjust"]["flow_over_demand"] is not None
    ]
    summary = {
        "admm_variant": admm_variant or "none",
        "n": len(rows),
        "demand_satisfaction_mean": float(np.mean(sat)),
        "demand_satisfaction_std": float(np.std(sat, ddof=0)),
        "max_link_utilization_mean": float(np.mean([r["max_link_utilization"] for r in rows])),
        "capacity_violations_mean": float(np.mean([r["capacity_violations"] for r in rows])),
        "admm_flow_over_demand_mean": float(np.mean(fod)) if fod else None,
        "per_tm_satisfaction": sat,
    }
    return summary


def load_problem(device: torch.device):
    topology = load_topology(cfg.TOPOLOGY)
    tms = load_traffic_matrices(cfg.TOPOLOGY)
    splits = split_indices(len(tms), cfg.TOPOLOGY)
    assert splits == cfg.TM_SPLIT, f"Split mismatch: {splits} vs {cfg.TM_SPLIT}"
    path_dict = generate_candidate_paths(
        topology, k=cfg.PATH_K, metric=cfg.PATH_METRIC, cache=True
    )
    problem = build_bipartite_graph(topology, path_dict, k=cfg.PATH_K, device=device)
    return topology, tms, splits, problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain-seed",
        type=int,
        default=None,
        help="If set, train that seed into BASELINE_FREEZE/models (does not overwrite archive).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg.FREEZE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FREEZE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FREEZE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    topology, tms, splits, problem = load_problem(device)
    train_tms = tms[splits["train"][0] : splits["train"][1]]
    test_tms = tms[splits["test"][0] : splits["test"][1]]

    print(f"Protocol {cfg.PROTOCOL_ID}")
    print(
        f"B4: {topology.num_nodes}n/{topology.num_edges}e  "
        f"TMs={len(tms)}  split={splits}  "
        f"frozen_admm={cfg.FROZEN_ADMM_VARIANT}  diagnostic={cfg.DIAGNOSTIC_ADMM_VARIANT}"
    )

    # Baselines once on identical test TMs / paths
    print("Baselines (LP / ECMP / SP)...")
    lp = evaluate_baseline(lp_allocate, problem, test_tms)
    ecmp = evaluate_baseline(ecmp_allocate, problem, test_tms)
    sp = evaluate_baseline(shortest_path_allocate, problem, test_tms)

    archive_summary = json.loads(
        (cfg.ARCHIVE_LOG_DIR / "b4_all_seeds_summary.json").read_text(encoding="utf-8")
    )

    per_seed = []
    for seed in cfg.SEEDS:
        ckpt_path = cfg.ARCHIVE_MODEL_DIR / f"grl_te_b4_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        print(f"\n=== seed {seed}  ckpt={ckpt_path.name} ===")
        set_seed(seed)
        model = GRLTEModel(
            num_layers=cfg.NUM_LAYERS, num_paths=cfg.NUM_PATHS
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

        no_admm = evaluate_channel(model, problem, test_tms, admm_variant=None)
        teal = evaluate_channel(model, problem, test_tms, admm_variant="teal")
        p1 = evaluate_channel(model, problem, test_tms, admm_variant="p1")

        archived = next(r for r in archive_summary["per_seed"] if r["seed"] == seed)
        row = {
            "seed": seed,
            "ckpt": str(ckpt_path.relative_to(cfg.REPO_ROOT)),
            "grl_te_no_admm": no_admm["demand_satisfaction_mean"],
            "grl_te_teal_admm": teal["demand_satisfaction_mean"],
            "grl_te_p1_admm": p1["demand_satisfaction_mean"],
            "teal_flow_over_demand_mean": teal["admm_flow_over_demand_mean"],
            "p1_flow_over_demand_mean": p1["admm_flow_over_demand_mean"],
            "archived_grl_te": archived["grl_te"],
            "archived_grl_te_no_admm": archived["grl_te_no_admm"],
            "teal_minus_archive": teal["demand_satisfaction_mean"] - archived["grl_te"],
            "no_admm_minus_archive": no_admm["demand_satisfaction_mean"]
            - archived["grl_te_no_admm"],
            "channels": {"none": no_admm, "teal": teal, "p1": p1},
        }
        per_seed.append(row)
        print(
            f"  noADMM={100*row['grl_te_no_admm']:.2f}%  "
            f"teal={100*row['grl_te_teal_admm']:.2f}%  "
            f"p1={100*row['grl_te_p1_admm']:.2f}%  "
            f"| archive teal-era={100*archived['grl_te']:.2f}%  "
            f"Δteal={100*row['teal_minus_archive']:+.4f}pp  "
            f"fod teal/p1={row['teal_flow_over_demand_mean']:.3f}/"
            f"{row['p1_flow_over_demand_mean']:.3f}"
        )

    def channel_stats(key: str):
        vals = [r[key] for r in per_seed]
        m, s = _mean_std(vals)
        return {"mean": m, "std": s, "n": len(vals)}

    teal_stats = channel_stats("grl_te_teal_admm")
    p1_stats = channel_stats("grl_te_p1_admm")
    none_stats = channel_stats("grl_te_no_admm")
    archive_match = all(abs(r["teal_minus_archive"]) < 1e-6 for r in per_seed)

    retrain_info = None
    if args.retrain_seed is not None:
        seed = int(args.retrain_seed)
        print(f"\nFresh retrain seed={seed} -> BASELINE_FREEZE/models (archive untouched)")
        set_seed(seed)
        model = GRLTEModel(num_layers=cfg.NUM_LAYERS, num_paths=cfg.NUM_PATHS).to(device)
        train_info = train_model(model, problem, train_tms, episodes=cfg.EPISODES)
        out_ckpt = cfg.FREEZE_MODEL_DIR / f"grl_te_b4_freeze_seed{seed}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "seed": seed,
                "protocol_id": cfg.PROTOCOL_ID,
                "frozen_admm_variant": cfg.FROZEN_ADMM_VARIANT,
                "episodes": cfg.EPISODES,
            },
            out_ckpt,
        )
        teal = evaluate_channel(model, problem, test_tms, "teal")
        p1 = evaluate_channel(model, problem, test_tms, "p1")
        none = evaluate_channel(model, problem, test_tms, None)
        retrain_info = {
            "seed": seed,
            "ckpt": str(out_ckpt.relative_to(cfg.REPO_ROOT)),
            "train_time_s": train_info["train_time_s"],
            "steps": train_info["steps"],
            "grl_te_teal_admm": teal["demand_satisfaction_mean"],
            "grl_te_p1_admm": p1["demand_satisfaction_mean"],
            "grl_te_no_admm": none["demand_satisfaction_mean"],
            "teal_flow_over_demand_mean": teal["admm_flow_over_demand_mean"],
            "p1_flow_over_demand_mean": p1["admm_flow_over_demand_mean"],
        }
        print(
            f"  retrain teal={100*retrain_info['grl_te_teal_admm']:.2f}%  "
            f"p1={100*retrain_info['grl_te_p1_admm']:.2f}%  "
            f"noADMM={100*retrain_info['grl_te_no_admm']:.2f}%"
        )

    report = {
        "protocol_id": cfg.PROTOCOL_ID,
        "protocol_date": cfg.PROTOCOL_DATE,
        "disclaimer": cfg.DISCLAIMER,
        "claims_forbidden": list(cfg.CLAIMS_FORBIDDEN),
        "topology": {
            "name": cfg.TOPOLOGY,
            "nodes": topology.num_nodes,
            "edges": topology.num_edges,
        },
        "traffic": {
            "source": cfg.TRAFFIC_SOURCE,
            "n_tms": len(tms),
            "split": cfg.TM_SPLIT,
            "split_source": cfg.TM_SPLIT_SOURCE,
            "n_test": len(test_tms),
        },
        "paths": {
            "generator": cfg.PATH_GENERATOR,
            "k": cfg.PATH_K,
            "metric": cfg.PATH_METRIC,
            "edge_disjoint": cfg.EDGE_DISJOINT,
        },
        "admm": {
            "frozen_reference_variant": cfg.FROZEN_ADMM_VARIANT,
            "frozen_order": cfg.FROZEN_ADMM_ORDER,
            "diagnostic_variant": cfg.DIAGNOSTIC_ADMM_VARIANT,
            "diagnostic_order": cfg.DIAGNOSTIC_ADMM_ORDER,
            "rho": cfg.RHO,
            "iters": cfg.ADMM_ITERS_B4,
        },
        "seeds": cfg.SEEDS,
        "episodes": cfg.EPISODES,
        "baselines": {
            "lp": {
                "demand_satisfaction": lp["demand_satisfaction"],
                "max_link_utilization": lp["max_link_utilization"],
            },
            "ecmp": {
                "demand_satisfaction": ecmp["demand_satisfaction"],
                "max_link_utilization": ecmp["max_link_utilization"],
            },
            "shortest_path": {
                "demand_satisfaction": sp["demand_satisfaction"],
                "max_link_utilization": sp["max_link_utilization"],
            },
        },
        "channels": {
            "teal_admm": {
                **teal_stats,
                "pct_mean": 100.0 * teal_stats["mean"],
                "pct_std": 100.0 * teal_stats["std"],
                "role": "FROZEN reference baseline",
            },
            "p1_admm": {
                **p1_stats,
                "pct_mean": 100.0 * p1_stats["mean"],
                "pct_std": 100.0 * p1_stats["std"],
                "role": "DIAGNOSTIC only — do not mix with teal",
            },
            "no_admm": {
                **none_stats,
                "pct_mean": 100.0 * none_stats["mean"],
                "pct_std": 100.0 * none_stats["std"],
                "role": "ablation",
            },
        },
        "archive_comparison": {
            "archived_teal_era_mean_std_pct": [
                100.0 * archive_summary["grl_te_mean_std"][0],
                100.0 * archive_summary["grl_te_mean_std"][1],
            ],
            "reeval_teal_matches_archive_per_seed": archive_match,
            "max_abs_teal_minus_archive": max(abs(r["teal_minus_archive"]) for r in per_seed),
            "p1_published_b4_grl_te_pct": 92.39,
            "note": "92.39 is P1 Table 6 — NOT claimed as matched",
        },
        "per_seed": [
            {k: v for k, v in r.items() if k != "channels"} | {"channels_summary": {
                "none_sat": r["grl_te_no_admm"],
                "teal_sat": r["grl_te_teal_admm"],
                "p1_sat": r["grl_te_p1_admm"],
                "teal_fod": r["teal_flow_over_demand_mean"],
                "p1_fod": r["p1_flow_over_demand_mean"],
            }}
            for r in per_seed
        ],
        "retrain": retrain_info,
    }

    # Full per-seed channel dumps (large) in separate file
    detailed = {
        "protocol_id": cfg.PROTOCOL_ID,
        "per_seed_channels": [
            {"seed": r["seed"], "channels": r["channels"]} for r in per_seed
        ],
    }

    out_summary = cfg.FREEZE_LOG_DIR / "b4_freeze_summary.json"
    out_detail = cfg.FREEZE_LOG_DIR / "b4_freeze_per_seed_channels.json"
    out_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_detail.write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    print("\n=== FROZEN CHANNEL (Teal ADMM) ===")
    print(f"  {100*teal_stats['mean']:.2f}% ± {100*teal_stats['std']:.2f}%")
    print("=== DIAGNOSTIC CHANNEL (P1 ADMM) ===")
    print(f"  {100*p1_stats['mean']:.2f}% ± {100*p1_stats['std']:.2f}%")
    print("=== NO ADMM ===")
    print(f"  {100*none_stats['mean']:.2f}% ± {100*none_stats['std']:.2f}%")
    print(f"LP={100*lp['demand_satisfaction']:.2f}%  ECMP={100*ecmp['demand_satisfaction']:.2f}%")
    print(f"Teal re-eval matches archive per seed: {archive_match}")
    print(f"Wrote {out_summary}")
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
