"""Phase C: Critic numerical-stability diagnosis (B4, seed 100).

Mirrors ``training.train.train_model`` exactly (same losses, LRs, clip=1.0,
no γ / bootstrap) while logging r, V, MSE, relative error, advantage, and
per-module gradient norms before/after clipping.

Does NOT modify production training code or ADMM solvers.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ACTOR_LR, CRITIC_LR, EPISODES, GRAPH_DIR, LOG_DIR  # noqa: E402
from models.a2c import GRLTEModel  # noqa: E402
from optimization.constraint_adjustment import adjust_allocation  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402


SEED = 100
CLIP = 1.0


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _has_nonfinite_grads(params) -> bool:
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def _clip_by_module(model: GRLTEModel, max_norm: float) -> dict:
    """Compute pre-clip norms, apply global clip like train.py, report post norms.

    train.py uses a single ``clip_grad_norm_(model.parameters(), 1.0)``.
    We measure module norms before that global clip, apply the same global
    clip, then re-measure module norms.
    """
    pre = {
        "topoflownet": _grad_norm(model.topoflownet.parameters()),
        "actor": _grad_norm(model.actor.parameters()),
        "critic": _grad_norm(model.critic.parameters()),
        "all": _grad_norm(model.parameters()),
    }
    nonfinite_pre = _has_nonfinite_grads(model.parameters())
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    post = {
        "topoflownet": _grad_norm(model.topoflownet.parameters()),
        "actor": _grad_norm(model.actor.parameters()),
        "critic": _grad_norm(model.critic.parameters()),
        "all": _grad_norm(model.parameters()),
    }
    return {
        "pre": pre,
        "post": post,
        "clip_total_norm": float(total_norm) if torch.is_tensor(total_norm) else float(total_norm),
        "nonfinite_grads_pre": nonfinite_pre,
    }


def verify_architecture(model: GRLTEModel) -> dict:
    critic = model.critic.net
    linears = [m for m in critic if isinstance(m, torch.nn.Linear)]
    shapes = [(int(m.in_features), int(m.out_features)) for m in linears]
    relus = sum(1 for m in critic if isinstance(m, torch.nn.ReLU))
    return {
        "p1_spec": "28→256→256→1 with ReLU on first two layers",
        "implemented_linear_shapes": shapes,
        "n_relu": relus,
        "matches_p1": shapes == [(28, 256), (256, 256), (256, 1)] and relus == 2,
        "aggregation": "sum of per-demand critic heads → scalar V(s) (not stated in P1)",
        "target": "immediate r_t = F_total (eq. 30); no γ; no V(s') bootstrap",
    }


def train_instrumented(model, problem, train_tms, episodes=EPISODES):
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
    model.train()
    step = 0
    nan_events = 0

    for episode in range(episodes):
        order = np.random.permutation(len(train_tms))
        for idx in order:
            tm = train_tms[int(idx)]
            demands = problem.demand_vector(tm, device=problem.capacities.device)
            h0 = problem.initial_features(demands)
            out = model.act(
                h0, problem.bipartite_index, problem.num_edges, deterministic=False
            )
            flow = (out["fractions"] * demands.unsqueeze(-1)).reshape(-1)
            feasible = adjust_allocation(flow, demands, problem)
            reward = feasible.sum()  # r_t = F_total
            total_demand = demands.sum().clamp_min(1e-6)
            value = out["value"]
            advantage = (reward - value).detach()

            log_prob = out["log_prob"].sum()
            actor_loss = -(advantage * log_prob)
            critic_loss = F.mse_loss(value, reward.detach())  # target = r only

            # Confirm no bootstrap in this step's target
            target_is_r = True
            uses_gamma = False

            optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            grad_info = _clip_by_module(model, CLIP)
            optimizer.step()

            r = float(reward.item())
            v = float(value.item())
            mse = float(critic_loss.item())
            rel_err = abs(r - v) / max(abs(r), 1e-9)
            adv = float(advantage.item())
            sat = float((reward / total_demand).item())

            nonfinite = (
                (not math.isfinite(r))
                or (not math.isfinite(v))
                or (not math.isfinite(mse))
                or (not math.isfinite(adv))
                or grad_info["nonfinite_grads_pre"]
            )
            if nonfinite:
                nan_events += 1

            history.append(
                {
                    "episode": episode,
                    "step": step,
                    "reward": r,
                    "value": v,
                    "mse": mse,
                    "abs_err": abs(r - v),
                    "rel_err": rel_err,
                    "advantage": adv,
                    "satisfaction": sat,
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": mse,
                    "log_prob_sum": float(log_prob.item()),
                    "grad_pre_tfn": grad_info["pre"]["topoflownet"],
                    "grad_pre_actor": grad_info["pre"]["actor"],
                    "grad_pre_critic": grad_info["pre"]["critic"],
                    "grad_pre_all": grad_info["pre"]["all"],
                    "grad_post_tfn": grad_info["post"]["topoflownet"],
                    "grad_post_actor": grad_info["post"]["actor"],
                    "grad_post_critic": grad_info["post"]["critic"],
                    "grad_post_all": grad_info["post"]["all"],
                    "clip_total_norm": grad_info["clip_total_norm"],
                    "clipped": grad_info["clip_total_norm"] > CLIP,
                    "nonfinite": nonfinite,
                    "target_is_immediate_r": target_is_r,
                    "uses_gamma": uses_gamma,
                }
            )
            step += 1
            if step % 10 == 0:
                print(
                    f"  step {step:3d} r={r:.1f} V={v:.1f} rel={rel_err:.3f} "
                    f"mse={mse:.3e} |g|_pre={grad_info['pre']['all']:.3e} "
                    f"clip_norm={grad_info['clip_total_norm']:.3e}"
                )

    return history, nan_events


def summarize(history: list[dict]) -> dict:
    def mean(key, rows):
        return float(np.mean([r[key] for r in rows]))

    def corr_r_v(rows):
        r = np.array([x["reward"] for x in rows], dtype=np.float64)
        v = np.array([x["value"] for x in rows], dtype=np.float64)
        if r.std() < 1e-12 or v.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(r, v)[0, 1])

    by_ep = {}
    for ep in sorted({h["episode"] for h in history}):
        rows = [h for h in history if h["episode"] == ep]
        # first / last quartile within episode for learning trend
        n = len(rows)
        first = rows[: max(1, n // 4)]
        last = rows[-(max(1, n // 4)) :]
        by_ep[str(ep)] = {
            "mean_r": mean("reward", rows),
            "mean_v": mean("value", rows),
            "mean_mse": mean("mse", rows),
            "mean_rel_err": mean("rel_err", rows),
            "mean_abs_err": mean("abs_err", rows),
            "corr_r_v": corr_r_v(rows),
            "mean_grad_pre_all": mean("grad_pre_all", rows),
            "mean_grad_pre_critic": mean("grad_pre_critic", rows),
            "frac_clipped": float(np.mean([1.0 if h["clipped"] else 0.0 for h in rows])),
            "first_quartile_mean_rel_err": mean("rel_err", first),
            "last_quartile_mean_rel_err": mean("rel_err", last),
            "first_quartile_mean_mse": mean("mse", first),
            "last_quartile_mean_mse": mean("mse", last),
            "first_quartile_mean_v": mean("value", first),
            "last_quartile_mean_v": mean("value", last),
        }

    all_rows = history
    r_scale = mean("reward", all_rows)
    mse_scale = mean("mse", all_rows)
    # If V tracks r, mse ≈ (scale * rel_err)^2; expected mse from mean |r| and mean rel
    expected_mse_from_rel = float(
        np.mean([(h["reward"] * h["rel_err"]) ** 2 for h in all_rows])
    )

    # Learning: relative error and |V-r|/|r| should drop; corr should rise; V should approach r scale
    ep0 = by_ep["0"]
    ep_last = by_ep[str(max(int(k) for k in by_ep))]
    learning = {
        "rel_err_ep0_to_last": [ep0["mean_rel_err"], ep_last["mean_rel_err"]],
        "rel_err_decreased": ep_last["mean_rel_err"] < ep0["mean_rel_err"],
        "corr_ep0_to_last": [ep0["corr_r_v"], ep_last["corr_r_v"]],
        "corr_improved": ep_last["corr_r_v"] > ep0["corr_r_v"],
        "v_mean_ep0_to_last": [ep0["mean_v"], ep_last["mean_v"]],
        "r_mean_ep0_to_last": [ep0["mean_r"], ep_last["mean_r"]],
        "v_approaches_r_scale": abs(ep_last["mean_v"] - ep_last["mean_r"])
        < abs(ep0["mean_v"] - ep0["mean_r"]),
        "within_ep0_rel_err_first_to_last_q": [
            ep0["first_quartile_mean_rel_err"],
            ep0["last_quartile_mean_rel_err"],
        ],
    }

    # Scale-effect test: mse / r^2 ≈ rel_err^2 if error is proportional
    mse_over_r2 = float(np.mean([h["mse"] / max(h["reward"] ** 2, 1e-12) for h in all_rows]))
    mean_rel2 = float(np.mean([h["rel_err"] ** 2 for h in all_rows]))

    verdict = {
        "large_absolute_mse": mse_scale > 1e6,
        "mean_r": r_scale,
        "mean_mse": mse_scale,
        "mean_rel_err": mean("rel_err", all_rows),
        "mse_over_mean_r_squared": mse_scale / max(r_scale**2, 1e-12),
        "mean_mse_over_r2_per_step": mse_over_r2,
        "mean_rel_err_squared": mean_rel2,
        "expected_mse_from_abs_rel": expected_mse_from_rel,
        "scale_effect_consistent": abs(mse_over_r2 - mean_rel2) < 0.05
        or abs(mse_over_r2 - mean_rel2) / max(mean_rel2, 1e-12) < 0.25,
        "critic_learning": bool(
            learning["rel_err_decreased"]
            or learning["corr_improved"]
            or learning["v_approaches_r_scale"]
        ),
        "nan_inf_present": any(h["nonfinite"] for h in all_rows),
        "grad_explosion_unclipped_mean": mean("grad_pre_all", all_rows),
        "frac_steps_clipped": float(np.mean([1.0 if h["clipped"] else 0.0 for h in all_rows])),
    }

    # Final classification
    if verdict["nan_inf_present"]:
        classification = "FAILED_NONFINITE"
    elif verdict["critic_learning"] and verdict["mean_rel_err"] < 0.5:
        classification = "LEARNING_OK_LARGE_MSE_IS_SCALE"
    elif verdict["critic_learning"]:
        classification = "LEARNING_PARTIAL_LARGE_MSE_MOSTLY_SCALE"
    else:
        classification = "NOT_LEARNING_NEEDS_ADAPTATION"

    return {
        "per_episode": by_ep,
        "learning_signals": learning,
        "verdict": verdict,
        "classification": classification,
    }


def plot_curves(history: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [h["step"] for h in history]
    r = [h["reward"] for h in history]
    v = [h["value"] for h in history]
    rel = [h["rel_err"] for h in history]
    mse = [h["mse"] for h in history]
    gpre = [h["grad_pre_all"] for h in history]
    gcrit = [h["grad_pre_critic"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(steps, r, label="r = F_total", color="#1f77b4")
    axes[0].plot(steps, v, label="V(s)", color="#ff7f0e", alpha=0.85)
    axes[0].set_ylabel("reward / value")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("B4 seed 100 — critic diagnosis (3 episodes)")

    axes[1].plot(steps, rel, color="#2ca02c")
    axes[1].set_ylabel("|r−V| / |r|")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(steps, mse, label="MSE", color="#d62728")
    axes[2].semilogy(steps, gpre, label="‖g‖ all pre-clip", color="#9467bd", alpha=0.8)
    axes[2].semilogy(steps, gcrit, label="‖g‖ critic pre-clip", color="#8c564b", alpha=0.8)
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("log scale")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "critic_diagnosis_seed100.png", dpi=150)
    plt.close(fig)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    topo = load_topology("B4")
    problem = build_bipartite_graph(topo, generate_candidate_paths(topo))
    tms = load_traffic_matrices("B4")
    lo, hi = split_indices(len(tms), "B4")["train"]
    train_tms = tms[lo:hi]

    model = GRLTEModel()
    arch = verify_architecture(model)

    print(f"Training B4 seed={SEED}, episodes={EPISODES}, n_train_tm={len(train_tms)}")
    history, nan_events = train_instrumented(model, problem, train_tms, episodes=EPISODES)
    summary = summarize(history)
    summary["architecture"] = arch
    summary["nan_events"] = nan_events
    summary["seed"] = SEED
    summary["episodes"] = EPISODES
    summary["n_steps"] = len(history)
    summary["clip"] = CLIP
    summary["actor_lr"] = ACTOR_LR
    summary["critic_lr"] = CRITIC_LR
    summary["notes"] = {
        "grad_clip": "clip_grad_norm_=1.0 is engineering (not in P1); measured only",
        "no_changes": "No reward norm, architecture, LR, or optimizer changes applied",
    }

    # Proposed adaptation only if not learning — recorded, not applied
    if summary["classification"].startswith("NOT_LEARNING"):
        summary["proposed_adaptation_not_applied"] = {
            "option": "Document-only: optional value/reward scaling for logging; or init critic bias toward E[r]",
            "justification": "Only if classification stays NOT_LEARNING; must stay labeled adaptation",
        }
    else:
        summary["proposed_adaptation_not_applied"] = None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_json = LOG_DIR / "critic_diagnosis_seed100.json"
    # Store compact history + summary
    payload = {"summary": summary, "history": history}
    out_json.write_text(json.dumps(payload, indent=2))

    plot_curves(history, GRAPH_DIR)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {GRAPH_DIR / 'critic_diagnosis_seed100.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
