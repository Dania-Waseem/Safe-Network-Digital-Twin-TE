"""One-step A2C training (P1 Section 4.3.3 and 5.3).

Training: actor sample → constraint-aware adjustment → reward r_t = F_total
(eq. 30). ADMM is deployment-only (P1 Section 4).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from config import ACTOR_LR, CRITIC_LR, EPISODES
from models.a2c import GRLTEModel
from optimization.constraint_adjustment import adjust_allocation
from preprocessing.build_bipartite_graph import TEProblem


def train_model(
    model: GRLTEModel,
    problem: TEProblem,
    train_tms: list[np.ndarray],
    episodes: int = EPISODES,
    actor_lr: float = ACTOR_LR,
    critic_lr: float = CRITIC_LR,
    log_every: int = 5,
) -> dict:
    # Shared TopoFlowNet (P1 §4.3.1): one backbone; both losses contribute
    # gradients. Param groups keep P1's two learning rates.
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.topoflownet.parameters()) + list(model.actor.parameters()),
                "lr": actor_lr,
            },
            {"params": list(model.critic.parameters()), "lr": critic_lr},
        ]
    )

    history = []
    model.train()
    t_start = time.perf_counter()
    step = 0
    for episode in range(episodes):
        order = np.random.permutation(len(train_tms))
        ep_reward = []
        ep_sat = []
        for idx in order:
            tm = train_tms[int(idx)]
            demands = problem.demand_vector(tm, device=problem.capacities.device)
            h0 = problem.initial_features(demands)  # raw h^0, eq. (5)
            out = model.act(
                h0, problem.bipartite_index, problem.num_edges, deterministic=False
            )
            flow = (out["fractions"] * demands.unsqueeze(-1)).reshape(-1)
            feasible = adjust_allocation(flow, demands, problem)
            # P1 eqs. (29)-(30): r_t = F_total
            reward = feasible.sum()
            total_demand = demands.sum().clamp_min(1e-6)
            value = out["value"]
            advantage = (reward - value).detach()

            # Joint log π(a|s) = sum over demands of path-Gaussian log-probs
            # (P1 eq. 38 uses ∇ log π(a_t|s_t) for the full action).
            log_prob = out["log_prob"].sum()
            actor_loss = -(advantage * log_prob)
            # MSE(V, r) is the squared residual of A = r - V; same fixed point
            # as the advantage-weighted critic update (39).
            critic_loss = F.mse_loss(value, reward.detach())

            optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sat = float((reward / total_demand).item())
            ep_reward.append(float(reward.item()))
            ep_sat.append(sat)
            history.append(
                {
                    "episode": episode,
                    "step": step,
                    "reward": float(reward.item()),
                    "satisfaction": sat,
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "value": float(value.item()),
                }
            )
            step += 1
            if log_every and step % log_every == 0:
                print(
                    f"  episode {episode+1}/{episodes} step {step} "
                    f"r={reward.item():.1f} sat={sat:.3f} V={value.item():.1f} "
                    f"a_loss={actor_loss.item():.4f} c_loss={critic_loss.item():.4f}"
                )
        print(
            f"Episode {episode+1}/{episodes}: "
            f"mean sat={np.mean(ep_sat):.4f} mean r={np.mean(ep_reward):.1f}"
        )

    return {
        "history": history,
        "train_time_s": time.perf_counter() - t_start,
        "steps": step,
    }
