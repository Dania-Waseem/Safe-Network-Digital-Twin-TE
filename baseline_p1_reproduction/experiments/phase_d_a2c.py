"""Phase D: extended A2C training + critic validation on B4.

Extends beyond P1's published 3 episodes as a *validation* experiment while
keeping methodology fixed (LRs, reward F_total, no γ, critic 28→256→256→1,
grad clip 1.0). Does not modify train.py, ADMM, TopoFlowNet, or Phase C files.

Outputs → results/logs/phase_d/ and results/graphs/phase_d/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ACTOR_LR,
    CRITIC_LR,
    LOG_DIR,
    RESULT_DIR,
    SEEDS,
)
from models.a2c import GRLTEModel  # noqa: E402
from optimization.constraint_adjustment import adjust_allocation  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402

PHASE_D_LOG = RESULT_DIR / "logs" / "phase_d"
PHASE_D_GRAPH = RESULT_DIR / "graphs" / "phase_d"
CLIP = 1.0
# Validation horizon beyond P1's published 3 episodes (documented in report).
DEFAULT_EPISODES = 20
EVAL_EVERY = 1  # evaluate test sat after every episode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def train_extended(model, problem, train_tms, test_tms, episodes: int) -> dict:
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.topoflownet.parameters()) + list(model.actor.parameters()),
                "lr": ACTOR_LR,
            },
            {"params": list(model.critic.parameters()), "lr": CRITIC_LR},
        ]
    )
    history = []
    eval_rows = []
    model.train()
    step = 0
    nan_events = 0

    for episode in range(episodes):
        order = np.random.permutation(len(train_tms))
        ep_rows = []
        for idx in order:
            tm = train_tms[int(idx)]
            demands = problem.demand_vector(tm, device=problem.capacities.device)
            h0 = problem.initial_features(demands)
            out = model.act(
                h0, problem.bipartite_index, problem.num_edges, deterministic=False
            )
            flow = (out["fractions"] * demands.unsqueeze(-1)).reshape(-1)
            feasible = adjust_allocation(flow, demands, problem)
            reward = feasible.sum()
            total_demand = demands.sum().clamp_min(1e-6)
            value = out["value"]
            advantage = (reward - value).detach()
            log_prob = out["log_prob"].sum()
            actor_loss = -(advantage * log_prob)
            critic_loss = F.mse_loss(value, reward.detach())

            optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            pre_all = _grad_norm(model.parameters())
            pre_crit = _grad_norm(model.critic.parameters())
            pre_actor = _grad_norm(model.actor.parameters())
            pre_tfn = _grad_norm(model.topoflownet.parameters())
            clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
            optimizer.step()

            r = float(reward.item())
            v = float(value.item())
            mse = float(critic_loss.item())
            rel = abs(r - v) / max(abs(r), 1e-9)
            sat = float((reward / total_demand).item())
            nonfinite = not all(
                math.isfinite(x) for x in (r, v, mse, sat, pre_all, clip_norm)
            )
            if nonfinite:
                nan_events += 1

            row = {
                "episode": episode,
                "step": step,
                "reward": r,
                "value": v,
                "mse": mse,
                "rel_err": rel,
                "advantage": float(advantage.item()),
                "satisfaction": sat,
                "actor_loss": float(actor_loss.item()),
                "grad_pre_all": pre_all,
                "grad_pre_critic": pre_crit,
                "grad_pre_actor": pre_actor,
                "grad_pre_tfn": pre_tfn,
                "clip_total_norm": clip_norm,
                "clipped": clip_norm > CLIP,
                "nonfinite": nonfinite,
            }
            history.append(row)
            ep_rows.append(row)
            step += 1

        # Episode aggregate
        def mean(key):
            return float(np.mean([x[key] for x in ep_rows]))

        r_arr = np.array([x["reward"] for x in ep_rows])
        v_arr = np.array([x["value"] for x in ep_rows])
        corr = (
            float(np.corrcoef(r_arr, v_arr)[0, 1])
            if r_arr.std() > 1e-12 and v_arr.std() > 1e-12
            else 0.0
        )

        # Test routing: policy only (no ADMM) isolates A2C; Teal ADMM for deploy continuity
        model.eval()
        with torch.no_grad():
            ev_no = evaluate_model(model, problem, test_tms, use_admm=False)
            ev_teal = evaluate_model(
                model, problem, test_tms, use_admm=True, admm_variant="teal"
            )
        model.train()

        eval_rows.append(
            {
                "episode": episode,
                "train_mean_sat": mean("satisfaction"),
                "train_mean_r": mean("reward"),
                "train_mean_v": mean("value"),
                "train_mean_mse": mean("mse"),
                "train_mean_rel_err": mean("rel_err"),
                "train_corr_r_v": corr,
                "train_mean_grad_pre_all": mean("grad_pre_all"),
                "train_frac_clipped": float(np.mean([1.0 if x["clipped"] else 0.0 for x in ep_rows])),
                "test_sat_no_admm": ev_no["demand_satisfaction"],
                "test_sat_teal_admm": ev_teal["demand_satisfaction"],
                "test_mlu_no_admm": ev_no["max_link_utilization"],
            }
        )
        print(
            f"  ep {episode+1:02d}/{episodes} train_sat={mean('satisfaction'):.4f} "
            f"rel={mean('rel_err'):.3f} corr={corr:.3f} "
            f"test_noADMM={ev_no['demand_satisfaction']:.4f} "
            f"test_teal={ev_teal['demand_satisfaction']:.4f}"
        )

    return {"history": history, "per_episode": eval_rows, "nan_events": nan_events, "steps": step}


def plot_seed(per_episode: list[dict], seed: int, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    eps = [r["episode"] + 1 for r in per_episode]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(eps, [r["test_sat_no_admm"] for r in per_episode], label="test sat (no ADMM)", color="#1f77b4")
    axes[0].plot(eps, [r["test_sat_teal_admm"] for r in per_episode], label="test sat (Teal ADMM)", color="#ff7f0e")
    axes[0].plot(eps, [r["train_mean_sat"] for r in per_episode], label="train sat", color="#2ca02c", alpha=0.7)
    axes[0].axvline(3, color="gray", ls="--", alpha=0.6, label="P1 / Phase C (3 eps)")
    axes[0].set_ylabel("demand satisfaction")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Phase D B4 seed {seed}")

    axes[1].plot(eps, [r["train_mean_rel_err"] for r in per_episode], color="#d62728", label="rel |r−V|/|r|")
    axes[1].plot(eps, [r["train_corr_r_v"] for r in per_episode], color="#9467bd", label="corr(r,V)")
    axes[1].axvline(3, color="gray", ls="--", alpha=0.6)
    axes[1].set_ylabel("critic metrics")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(eps, [max(r["train_mean_mse"], 1e-12) for r in per_episode], label="MSE", color="#8c564b")
    axes[2].semilogy(
        eps,
        [max(r["train_mean_grad_pre_all"], 1e-12) for r in per_episode],
        label="‖g‖ pre-clip",
        color="#e377c2",
    )
    axes[2].axvline(3, color="gray", ls="--", alpha=0.6)
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("log scale")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"phase_d_seed{seed}.png", dpi=150)
    plt.close(fig)


def plot_aggregate(all_seed_eps: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # all_seed_eps: seed -> per_episode list
    seeds = sorted(all_seed_eps.keys())
    n_ep = len(next(iter(all_seed_eps.values())))
    eps = np.arange(1, n_ep + 1)

    def stack(key):
        return np.array([[all_seed_eps[s][i][key] for i in range(n_ep)] for s in seeds])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for key, ax, title in [
        ("test_sat_no_admm", axes[0], "Test sat (no ADMM) mean±std"),
        ("train_mean_rel_err", axes[1], "Train critic rel err mean±std"),
    ]:
        mat = stack(key)
        mu, sd = mat.mean(0), mat.std(0)
        ax.plot(eps, mu, color="#1f77b4")
        ax.fill_between(eps, mu - sd, mu + sd, color="#1f77b4", alpha=0.25)
        ax.axvline(3, color="gray", ls="--", alpha=0.6)
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out_dir / "phase_d_aggregate.png", dpi=150)
    plt.close(fig)


def load_phase_c_baseline() -> dict | None:
    path = LOG_DIR / "critic_diagnosis_seed100.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=SEEDS,
        help="Random seeds (default: P1 100..1000)",
    )
    args = parser.parse_args()

    PHASE_D_LOG.mkdir(parents=True, exist_ok=True)
    PHASE_D_GRAPH.mkdir(parents=True, exist_ok=True)

    topo = load_topology("B4")
    problem = build_bipartite_graph(topo, generate_candidate_paths(topo))
    tms = load_traffic_matrices("B4")
    splits = split_indices(len(tms), "B4")
    train_tms = tms[splits["train"][0] : splits["train"][1]]
    test_tms = tms[splits["test"][0] : splits["test"][1]]

    phase_c = load_phase_c_baseline()
    all_seed_eps = {}
    seed_summaries = []

    for seed in args.seeds:
        print(f"\n======== PHASE D SEED {seed} ({args.episodes} episodes) ========")
        set_seed(seed)
        model = GRLTEModel()
        result = train_extended(model, problem, train_tms, test_tms, args.episodes)
        all_seed_eps[seed] = result["per_episode"]
        plot_seed(result["per_episode"], seed, PHASE_D_GRAPH)

        # Snapshots at ep 3 (0-index 2), mid, final
        pe = result["per_episode"]
        snap = {
            "at_ep3": pe[2] if len(pe) > 2 else pe[-1],
            "at_final": pe[-1],
            "at_mid": pe[len(pe) // 2],
        }
        seed_summaries.append(
            {
                "seed": seed,
                "nan_events": result["nan_events"],
                "steps": result["steps"],
                "snapshots": snap,
                "test_sat_no_admm_ep3": snap["at_ep3"]["test_sat_no_admm"],
                "test_sat_no_admm_final": snap["at_final"]["test_sat_no_admm"],
                "test_sat_teal_ep3": snap["at_ep3"]["test_sat_teal_admm"],
                "test_sat_teal_final": snap["at_final"]["test_sat_teal_admm"],
                "rel_err_ep3": snap["at_ep3"]["train_mean_rel_err"],
                "rel_err_final": snap["at_final"]["train_mean_rel_err"],
                "corr_ep3": snap["at_ep3"]["train_corr_r_v"],
                "corr_final": snap["at_final"]["train_corr_r_v"],
                "frac_clipped_final": snap["at_final"]["train_frac_clipped"],
                "improved_test_sat_no_admm": snap["at_final"]["test_sat_no_admm"]
                >= snap["at_ep3"]["test_sat_no_admm"] - 1e-6,
                "improved_rel_err": snap["at_final"]["train_mean_rel_err"]
                <= snap["at_ep3"]["train_mean_rel_err"] + 1e-6,
            }
        )
        # Per-seed JSON (history can be large — store episode aggregates + light history stats)
        (PHASE_D_LOG / f"seed{seed}.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "episodes": args.episodes,
                    "nan_events": result["nan_events"],
                    "per_episode": result["per_episode"],
                    "history_tail": result["history"][-40:],  # last 2 episodes steps
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    plot_aggregate(all_seed_eps, PHASE_D_GRAPH)

    def stats(key):
        vals = np.array([s[key] for s in seed_summaries], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}

    # Verdict logic
    sat_improve_frac = float(np.mean([1.0 if s["improved_test_sat_no_admm"] else 0.0 for s in seed_summaries]))
    rel_improve_frac = float(np.mean([1.0 if s["improved_rel_err"] else 0.0 for s in seed_summaries]))
    any_nan = any(s["nan_events"] > 0 for s in seed_summaries)
    final_sat = stats("test_sat_no_admm_final")
    ep3_sat = stats("test_sat_no_admm_ep3")
    final_rel = stats("rel_err_final")
    ep3_rel = stats("rel_err_ep3")

    if any_nan:
        verdict = "FAILED"
        verdict_reason = "NaN/Inf observed during extended training"
    elif (
        sat_improve_frac >= 0.6
        and rel_improve_frac >= 0.6
        and final_rel["mean"] < ep3_rel["mean"]
        and not any_nan
    ):
        verdict = "MOSTLY VERIFIED"
        verdict_reason = (
            "Multi-seed longer training: critic rel-err improves vs ep3 and "
            "test routing (no ADMM) is stable/improved for majority of seeds; "
            "always-on grad clip and P1 3-ep protocol remain qualifications"
        )
    elif not any_nan and final_sat["mean"] >= 0.9:
        verdict = "PARTIALLY VERIFIED"
        verdict_reason = (
            "No numerical failure and high test sat, but learning gains beyond "
            "3 episodes are inconsistent across seeds"
        )
    else:
        verdict = "FAILED"
        verdict_reason = "Routing or critic metrics degraded / unstable under longer training"

    aggregate = {
        "phase": "D",
        "topology": "B4",
        "episodes": args.episodes,
        "seeds": list(args.seeds),
        "methodology_note": (
            "P1 publishes 3 episodes; Phase D extends to longer training as "
            "validation only — LRs, reward, critic arch, no γ unchanged"
        ),
        "data_note": (
            "Existing TEAL B4 TMs (36 matrices, 20/8/8 split); no new download"
        ),
        "phase_c_baseline_seed100": {
            "rel_err_ep0_to_ep2": phase_c["learning_signals"]["rel_err_ep0_to_last"]
            if phase_c
            else None,
            "classification": phase_c.get("classification") if phase_c else None,
        },
        "stats": {
            "test_sat_no_admm_ep3": ep3_sat,
            "test_sat_no_admm_final": final_sat,
            "test_sat_teal_ep3": stats("test_sat_teal_ep3"),
            "test_sat_teal_final": stats("test_sat_teal_final"),
            "rel_err_ep3": ep3_rel,
            "rel_err_final": final_rel,
            "corr_ep3": stats("corr_ep3"),
            "corr_final": stats("corr_final"),
        },
        "frac_seeds_improved_test_sat_no_admm": sat_improve_frac,
        "frac_seeds_improved_rel_err": rel_improve_frac,
        "any_nan": any_nan,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "per_seed": seed_summaries,
    }
    (PHASE_D_LOG / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("\n=== PHASE D AGGREGATE ===")
    print(json.dumps({k: aggregate[k] for k in ("verdict", "verdict_reason", "stats", "frac_seeds_improved_test_sat_no_admm", "frac_seeds_improved_rel_err")}, indent=2))
    print("Wrote", PHASE_D_LOG / "aggregate.json")


if __name__ == "__main__":
    main()
