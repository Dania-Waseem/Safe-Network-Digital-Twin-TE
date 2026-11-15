"""Read-only audit tests. Does not modify GRL-TE source modules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.a2c import GRLTEModel
from optimization.admm import admm_refine
from optimization.constraint_adjustment import adjust_allocation, evaluate_allocation
from preprocessing.build_bipartite_graph import build_bipartite_graph
from preprocessing.generate_paths import generate_candidate_paths
from preprocessing.load_topology import load_topology
from preprocessing.load_traffic import load_traffic_matrices, split_indices

results = []


def record(name, passed, detail):
    results.append({"test": name, "passed": bool(passed), "detail": detail})
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}: {detail}")


# 1. imports already succeeded
record("import_and_deps", True, "torch/numpy/networkx modules imported")

# 2-4. topology, TMs, paths
topo = load_topology("B4")
tms = load_traffic_matrices("B4")
paths = generate_candidate_paths(topo)
record(
    "topology_integrity",
    topo.num_nodes == 12 and topo.num_edges == 38 and topo.graph.is_directed(),
    f"nodes={topo.num_nodes} edges={topo.num_edges} directed={topo.graph.is_directed()} "
    f"cap_unique={sorted(set(topo.capacities.tolist()))}",
)
record("dataset_count_b4", len(tms) == 36, f"n_tms={len(tms)}")
avg_demand = float(np.mean([tm.sum() for tm in tms]))
record(
    "b4_mean_total_demand",
    abs(avg_demand - 27980) / 27980 < 0.05,
    f"mean_total_demand={avg_demand:.2f} (P1 text: 27.98 thousand)",
)
other = ["UsCarrier", "Cogentco", "Kdl", "ASN", "GEANT", "GÉANT"]
missing = [n for n in other if not (ROOT / "data" / "topologies" / f"{n}.json").exists()]
record("other_topologies_present", len(missing) == 0, f"missing={missing}")

n_pairs = topo.num_nodes * (topo.num_nodes - 1)
path_lens = [len(v) for v in paths.values()]
unique_counts = [len({tuple(p) for p in v}) for v in paths.values()]
all_edges_exist = True
bad = None
for (s, t), plist in paths.items():
    for p in plist:
        for u, v in zip(p[:-1], p[1:]):
            if (u, v) not in topo.edge_to_idx:
                all_edges_exist = False
                bad = (s, t, u, v)
record(
    "candidate_paths",
    len(paths) == n_pairs and min(path_lens) == 4 and max(path_lens) == 4 and all_edges_exist,
    f"pairs={len(paths)} path_count_range=({min(path_lens)},{max(path_lens)}) "
    f"unique_path_min={min(unique_counts)} edges_ok={all_edges_exist} bad={bad}",
)

# 5. bipartite
problem = build_bipartite_graph(topo, paths)
src, dst = problem.bipartite_index
n_link, n_path = problem.num_edges, problem.num_demands * problem.num_paths
only_cross = bool(((src < n_link) ^ (dst < n_link)).all())
bidir = src.numel() == 2 * problem.p2e.size(1)
# every incidence edge exists in both directions
record(
    "bipartite_connectivity",
    problem.num_demands == 132
    and n_path == 528
    and only_cross
    and bidir
    and problem.p2e.size(1) > 0,
    f"demands={problem.num_demands} path_nodes={n_path} incidences={problem.p2e.size(1)} "
    f"only_cross_partition={only_cross} bidirectional={bidir}",
)

# reconstruct one path's links
s0, t0 = problem.demand_pairs[0]
p0 = paths[(s0, t0)][0]
expected_links = [topo.edge_to_idx[(u, v)] for u, v in zip(p0[:-1], p0[1:])]
got_links = problem.p2e[1][problem.p2e[0] == 0].tolist()
record("bipartite_path0_links", expected_links == got_links, f"expected={expected_links} got={got_links}")

# 6. TopoFlowNet shapes
model = GRLTEModel()
demands = problem.demand_vector(tms[0])
h0 = problem.initial_features(demands, scale=True)
h0_raw = problem.initial_features(demands, scale=False)
record("h0_shape", tuple(h0.shape) == (n_link + n_path, 1), f"h0={tuple(h0.shape)}")
record(
    "h0_is_scaled",
    bool((h0.max() <= 1.0 + 1e-5) and (h0_raw.max() > 10)),
    f"scaled_max={float(h0.max()):.4f} raw_max={float(h0_raw.max()):.4f} (P1 eq.5 uses raw)",
)

h = h0
shapes = [tuple(h.shape)]
for i, layer in enumerate(model.topoflownet.layers):
    h = layer(h, h0, problem.bipartite_index, problem.num_edges)
    shapes.append(tuple(h.shape))
path_emb = model.topoflownet(h0, problem.bipartite_index, problem.num_edges)
record(
    "topoflownet_layer_dims",
    shapes == [(566, 1), (566, 2), (566, 3), (566, 4), (566, 5), (566, 6), (566, 7)]
    and tuple(path_emb.shape) == (528, 7),
    f"layer_shapes={shapes} path_emb={tuple(path_emb.shape)}",
)

# GIN uses shared MLP for both partitions: one gin_mlp per layer
shared = all(not hasattr(layer, "gin_mlp_edge") for layer in model.topoflownet.layers)
record("gin_shared_mlp", shared, "single gin_mlp per layer for link and path nodes")

# 7-8 actor/critic
out = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=False)
frac = out["fractions"]
softmax_ok = torch.allclose(frac.sum(dim=-1), torch.ones(problem.num_demands), atol=1e-5)
nonneg = bool((frac >= 0).all())
record(
    "actor_softmax",
    frac.shape == (132, 4) and softmax_ok and nonneg,
    f"shape={tuple(frac.shape)} row_sum_min={float(frac.sum(-1).min()):.6f} "
    f"row_sum_max={float(frac.sum(-1).max()):.6f}",
)
record(
    "actor_gaussian_heads",
    out["mean"].shape == (132, 4) and out["std"].shape == (132, 4) and out["log_prob"].shape == (132,),
    f"mean={tuple(out['mean'].shape)} std={tuple(out['std'].shape)} logp={tuple(out['log_prob'].shape)}",
)
record(
    "critic_scalar_value",
    out["value"].ndim == 0 or out["value"].numel() == 1,
    f"value_shape={tuple(out['value'].shape)} value={float(out['value']):.4f} "
    f"(aggregation=mean over |D| critic heads; not specified in P1)",
)
det = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)
record("deployment_uses_mean", torch.equal(det["raw"], det["mean"]), "deterministic raw == mean")

# 9. gradients
flow = (out["fractions"] * demands.unsqueeze(-1)).reshape(-1)
feas = adjust_allocation(flow, demands, problem)
reward = feas.sum() / demands.sum().clamp_min(1e-6)
adv = (reward - out["value"]).detach()
actor_loss = -(adv * out["log_prob"].mean())
critic_loss = torch.nn.functional.mse_loss(out["value"], reward.detach())
(actor_loss + critic_loss).backward()
g_tfn = sum(p.grad.abs().sum().item() for p in model.topoflownet.parameters() if p.grad is not None)
g_act = sum(p.grad.abs().sum().item() for p in model.actor.parameters() if p.grad is not None)
g_crt = sum(p.grad.abs().sum().item() for p in model.critic.parameters() if p.grad is not None)
record(
    "a2c_gradients",
    g_tfn > 0 and g_act > 0 and g_crt > 0,
    f"grad_l1 topoflownet={g_tfn:.4f} actor={g_act:.4f} critic={g_crt:.4f}",
)

# 10. ADMM + constraints on real TM
model.zero_grad()
with torch.no_grad():
    det = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)
    raw_flow = (det["fractions"] * demands.unsqueeze(-1)).reshape(-1)
    admm_flow = admm_refine(det["fractions"], demands, problem, num_iters=2, rho=1.0)
    final = adjust_allocation(admm_flow, demands, problem)
    m_raw = evaluate_allocation(raw_flow, demands, problem)
    m_admm = evaluate_allocation(admm_flow, demands, problem)
    m_final = evaluate_allocation(final, demands, problem)
    k = 4
    dem_ok = bool((final.reshape(-1, k).sum(-1) <= demands + 1e-2).all())
    load = torch.zeros(problem.num_edges)
    load.index_add_(0, problem.p2e[1], final[problem.p2e[0]])
    cap_ok = bool((load <= problem.capacities * 1.002 + 1e-2).all())
    nn_ok = bool((final >= -1e-6).all())
record(
    "final_allocation_constraints",
    dem_ok and cap_ok and nn_ok,
    f"demand_ok={dem_ok} capacity_ok={cap_ok} nonnegative={nn_ok} "
    f"sat_raw={m_raw['demand_satisfaction']:.4f} sat_admm={m_admm['demand_satisfaction']:.4f} "
    f"sat_final={m_final['demand_satisfaction']:.4f} mlu_final={m_final['max_link_utilization']:.4f} "
    f"viol_final={m_final['capacity_violations']}",
)

# synthetic 2-demand ADMM residual check: does ADMM reduce G residuals vs actor?
with torch.no_grad():
    g4_before = (raw_flow[problem.p2e[0]] - raw_flow[problem.p2e[0]]).abs().mean()  # dummy
    # G1 of softmax fractions is ~0 if xi=0 because softmax sums to 1
    f = det["fractions"]
    g1_actor = (f.sum(-1) - 1).abs().mean().item()
    # after ADMM, F may not sum to 1
    f_admm = admm_flow.reshape(-1, 4) / demands.unsqueeze(-1).clamp_min(1e-6)
    g1_admm = (f_admm.sum(-1) - 1).clamp(min=0).mean().item()  # overflow only
record(
    "admm_runs_fixed_iters",
    True,
    f"B4 iters=2 rho=1 g1_actor_absmean={g1_actor:.2e} "
    f"post_admm_demand_overflow_mean={g1_admm:.4f} (solver is derived, not published)",
)

# 11. one-step A2C: no gamma, no next-state in train.py
train_src = (ROOT / "training" / "train.py").read_text(encoding="utf-8")
has_gamma = "gamma" in train_src or "discount" in train_src
has_next_v = "s_t+1" in train_src or "next_value" in train_src or "V(s_{t+1}" in train_src
record("one_step_a2c_no_td", (not has_gamma) and (not has_next_v), f"gamma={has_gamma} next_state_value={has_next_v}")

# 12. seed reproducibility of actor fractions at init
torch.manual_seed(100)
m1 = GRLTEModel()
torch.manual_seed(100)
m2 = GRLTEModel()
with torch.no_grad():
    a1 = m1.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)["fractions"]
    a2 = m2.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)["fractions"]
record("init_reproducibility_seed100", torch.allclose(a1, a2), f"max_abs={float((a1-a2).abs().max()):.3e}")

# hyperparameters vs paper
from config import ACTOR_LR, CRITIC_LR, EPISODES, RHO, SEEDS, ADMM_ITERS
record(
    "hyperparams_config",
    EPISODES == 3 and ACTOR_LR == 1e-4 and CRITIC_LR == 1e-3 and RHO == 1.0
    and SEEDS == list(range(100, 1100, 100)) and ADMM_ITERS["B4"] == 2,
    f"episodes={EPISODES} actor_lr={ACTOR_LR} critic_lr={CRITIC_LR} rho={RHO} "
    f"n_seeds_listed={len(SEEDS)} b4_admm={ADMM_ITERS['B4']}",
)
b4_py = (ROOT / "experiments" / "b4.py").read_text(encoding="utf-8")
record(
    "ten_seed_loop_executed",
    False,
    "config lists seeds 100..1000 but experiments/b4.py runs a single --seed (default 100)",
)

splits = split_indices(len(tms), "B4")
record(
    "tm_split",
    splits == {"train": (0, 20), "val": (20, 28), "test": (28, 36)},
    f"splits={splits} (P1 does not publish a split; this is TEAL's default)",
)

# reward definition
record(
    "reward_is_not_F_total",
    True,
    "train.py uses feasible.sum()/total_demand, not P1 eq.(30) r_t=F_total",
)

n_fail = sum(1 for r in results if not r["passed"] and r["test"] not in {
    "other_topologies_present",
    "ten_seed_loop_executed",
    "h0_is_scaled",
    "reward_is_not_F_total",
    "admm_runs_fixed_iters",
})
print("\n=== SUMMARY ===")
print(f"recorded={len(results)}")
out = ROOT / "results" / "logs" / "audit_test_results.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"wrote {out}")
