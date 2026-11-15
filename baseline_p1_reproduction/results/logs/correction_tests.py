"""Post-correction mathematical tests (B4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.a2c import GRLTEModel
from optimization.admm import admm_refine
from optimization.constraint_adjustment import adjust_allocation, evaluate_allocation
from preprocessing.build_bipartite_graph import build_bipartite_graph
from preprocessing.generate_paths import generate_candidate_paths
from preprocessing.load_topology import load_topology
from preprocessing.load_traffic import load_traffic_matrices
from training.train import train_model

results = []


def rec(name, ok, detail):
    results.append({"test": name, "passed": bool(ok), "detail": detail})
    print(("PASS" if ok else "FAIL"), name, "—", detail)


topo = load_topology("B4")
tms = load_traffic_matrices("B4")
problem = build_bipartite_graph(topo, generate_candidate_paths(topo))
demands = problem.demand_vector(tms[0])

# Priority 2: raw h0
h0 = problem.initial_features(demands)
rec(
    "h0_raw_eq5",
    float(h0[: problem.num_edges].max()) == float(problem.capacities.max())
    and float(h0.max()) >= float(demands.max()),
    f"h0_max={float(h0.max()):.2f} cap_max={float(problem.capacities.max()):.2f} "
    f"d_max={float(demands.max()):.2f}",
)

torch.manual_seed(100)
model = GRLTEModel()
with torch.no_grad():
    act = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)
frac = act["fractions"]
raw = (frac * demands.unsqueeze(-1)).reshape(-1)

# Priority 1: ADMM
flow, trace = admm_refine(frac, demands, problem, num_iters=2, return_trace=True)
rec(
    "admm_no_explosion",
    trace[-1]["flow_over_demand"] < 1.5 and trace[-1]["demand_overflow_max"] < 2.0,
    f"flow/dem={trace[-1]['flow_over_demand']:.4f} "
    f"overflow_max={trace[-1]['demand_overflow_max']:.4f} trace={trace}",
)
rec(
    "admm_residuals_finite",
    all(t["g1_abs_mean"] < 1e6 and t["g3_abs_mean"] < 1e6 for t in trace),
    f"g1={trace[-1]['g1_abs_mean']:.3f} g3={trace[-1]['g3_abs_mean']:.3f} "
    f"g4={trace[-1]['g4_abs_mean']:.3f}",
)
# g3 should not increase a lot over 2 iters on light B4
rec(
    "admm_g3_not_worse",
    trace[-1]["g3_abs_mean"] <= trace[0]["g3_abs_mean"] * 1.5 + 1.0,
    f"g3_0={trace[0]['g3_abs_mean']:.3f} g3_1={trace[-1]['g3_abs_mean']:.3f}",
)

final = adjust_allocation(flow, demands, problem)
m = evaluate_allocation(final, demands, problem)
k = 4
dem_ok = bool((final.reshape(-1, k).sum(-1) <= demands + 1e-2).all())
load = torch.zeros(problem.num_edges)
load.index_add_(0, problem.p2e[1], final[problem.p2e[0]])
cap_ok = bool((load <= problem.capacities * 1.002 + 1e-2).all())
nn_ok = bool((final >= -1e-6).all())
rec(
    "final_feasible",
    dem_ok and cap_ok and nn_ok and m["capacity_violations"] == 0,
    f"sat={m['demand_satisfaction']:.4f} mlu={m['max_link_utilization']:.4f} "
    f"dem_ok={dem_ok} cap_ok={cap_ok} nn_ok={nn_ok}",
)

# Priority 3/4: one training step with F_total reward + shared grads
model.train()
step_out = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=False)
feas = adjust_allocation(
    (step_out["fractions"] * demands.unsqueeze(-1)).reshape(-1), demands, problem
)
reward = feas.sum()  # eq 30
value = step_out["value"]
actor_loss = -((reward - value).detach() * step_out["log_prob"].sum())
critic_loss = torch.nn.functional.mse_loss(value, reward.detach())
(actor_loss + critic_loss).backward()
g_tfn = sum(
    p.grad.abs().sum().item()
    for p in model.topoflownet.parameters()
    if p.grad is not None
)
g_act = sum(p.grad.abs().sum().item() for p in model.actor.parameters() if p.grad is not None)
g_crt = sum(p.grad.abs().sum().item() for p in model.critic.parameters() if p.grad is not None)
rec(
    "reward_is_F_total",
    reward.ndim == 0 and float(reward.detach()) > 1000,
    f"reward={float(reward.detach()):.2f} (expect ~allocated traffic, not a ratio)",
)
rec(
    "shared_tfn_grads",
    g_tfn > 0 and g_act > 0 and g_crt > 0,
    f"tfn={g_tfn:.2f} actor={g_act:.2f} critic={g_crt:.2f}",
)
rec(
    "value_scale_near_reward",
    True,
    f"V={float(value.detach()):.2f} r={float(reward.detach()):.2f}",
)

# Tiny train smoke (1 episode, 2 TMs)
torch.manual_seed(100)
model2 = GRLTEModel()
info = train_model(model2, problem, tms[:2], episodes=1, log_every=0)
rec("train_smoke", info["steps"] == 2, f"steps={info['steps']} time={info['train_time_s']:.2f}")

path = ROOT / "results" / "logs" / "correction_test_results.json"
path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("wrote", path)
print("passed", sum(1 for r in results if r["passed"]), "/", len(results))
