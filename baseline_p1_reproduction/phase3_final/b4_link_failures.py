"""E3 link-failure experiment: P1 Section 5.5-comparable B4 link-failure table for the frozen P1 baseline.

Protocol (frozen B4 baseline, BASELINE_FREEZE/frozen_b4_config.py):
* B4 (TEAL 12/38), TEAL real TMs, test split TMs 28-35, Yen k=4 min-hop paths (fixed),
* the 10 archived GRL-TE checkpoints (seeds 100..1000) re-evaluated by the freeze,
* deployment = actor -> Teal-order ADMM (rho 1, 2 iterations) -> constraint adjustment (23)-(28),
* failure case = ``data/scenarios/link_failures/B4.json`` (P1 §5.5 sampler,
  0-3 failed directed edges per seed); failed edges matched by (u, v) tuple; capacity set to 0;
  candidate paths are NOT recomputed (as in the reproduction's ``experiments/link_failures.py``):
  flow on a path through a failed link is removed by the adjustment step.
* failure seed s is evaluated with checkpoint seed s.
Also: GRL-TE without ADMM, LP (max throughput, SciPy HiGHS), ECMP and shortest path on the same failed problems.

Usage (from baseline_p1_reproduction/): python -m phase3_final.b4_link_failures
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO, REPO / "BASELINE_FREEZE"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import frozen_b4_config as cfg  # noqa: E402
from experiments.baselines import ecmp_allocate, evaluate_baseline, lp_allocate, shortest_path_allocate  # noqa: E402
from models.a2c import GRLTEModel  # noqa: E402
from run_b4_baseline_freeze import evaluate_channel, load_problem, set_seed  # noqa: E402

SCEN = REPO.parent / "data" / "scenarios" / "link_failures" / "B4.json"
OUT = HERE / "results" / "b4_link_failures"
P1_REPORTED = {0: "92.39 (Table 6, intact)", 3: "77.25 (§5.5)"}  # §5.5 quoted as 90.80 -> 77.25; the 90.80 start point is not assigned to a row


def failed_problem(problem, topology, edges):
    prob = copy.copy(problem)
    cap = problem.capacities.clone()
    idx = [topology.edge_to_idx[tuple(e)] for e in edges]
    cap[idx] = 0.0
    prob.capacities = cap
    return prob, idx


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    topology, tms, splits, problem = load_problem(device)
    test = tms[splits["test"][0]: splits["test"][1]]
    cases = json.loads(SCEN.read_text(encoding="utf-8"))["cases"]
    rows = []
    for c in cases:
        seed, nf = int(c["seed"]), int(c["n_failed"])
        prob, idx = failed_problem(problem, topology, c["edges"])
        set_seed(seed)
        model = GRLTEModel(num_layers=cfg.NUM_LAYERS, num_paths=cfg.NUM_PATHS).to(device)
        state = torch.load(cfg.ARCHIVE_MODEL_DIR / f"grl_te_b4_seed{seed}.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        teal = evaluate_channel(model, prob, test, "teal")
        none = evaluate_channel(model, prob, test, None)
        row = {"seed": seed, "n_failed": nf, "edges": c["edges"], "edge_idx_repro": idx,
               "grl_te_teal_admm": teal["demand_satisfaction_mean"], "grl_te_no_admm": none["demand_satisfaction_mean"],
               "grl_te_mlu": teal["max_link_utilization_mean"]}
        for name, fn in (("lp", lp_allocate), ("ecmp", ecmp_allocate), ("shortest_path", shortest_path_allocate)):
            row[name] = evaluate_baseline(fn, prob, test)["demand_satisfaction"]
        rows.append(row)
        print(f"seed {seed} f={nf}: GRL-TE {row['grl_te_teal_admm']:.4f} LP {row['lp']:.4f} ECMP {row['ecmp']:.4f} "
              f"SP {row['shortest_path']:.4f}", flush=True)
    (OUT / "per_case.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    methods = ("grl_te_teal_admm", "grl_te_no_admm", "lp", "ecmp", "shortest_path")
    table = {}
    for nf in range(4):
        sub = [r for r in rows if r["n_failed"] == nf]
        table[nf] = {m: (float(np.mean([r[m] for r in sub]) * 100), float(np.std([r[m] for r in sub]) * 100)) for m in methods}
        table[nf]["n_seeds"] = len(sub)
    (OUT / "summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines = ["# B4 link failures — P1 §5.5-comparable table (frozen P1 baseline)", "",
             "Frozen B4 protocol (`BASELINE_FREEZE`): TEAL real TMs, test TMs 28–35, Yen k=4, 10 archived checkpoints "
             "(seeds 100–1000), Teal-order ADMM (ρ=1, 2 iterations) + constraint adjustment. Failure cases from "
             "`data/scenarios/link_failures/B4.json` (P1 §5.5 sampler); failure seed s uses checkpoint seed s. "
             "Failed links get capacity 0; paths are not recomputed (P1-style). LP = max-throughput path LP, SciPy "
             "HiGHS (no Gurobi licence).", "",
             "Satisfaction (%), mean ± SD over the 10 seeds (each seed = mean over the 8 test TMs):", "",
             "| Failed links | GRL-TE + Teal ADMM | GRL-TE no ADMM | LP | ECMP | Shortest path | P1 reported GRL-TE |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for nf in range(4):
        t = table[nf]
        cells = " | ".join(f"{t[m][0]:.2f} ± {t[m][1]:.2f}" for m in methods)
        lines.append(f"| {nf} | {cells} | {P1_REPORTED.get(nf, '–')} |")
    d_ours = table[3]["grl_te_teal_admm"][0] - table[0]["grl_te_teal_admm"][0]
    d13 = table[3]["grl_te_teal_admm"][0] - table[1]["grl_te_teal_admm"][0]
    lines += ["", f"Change in GRL-TE satisfaction: 0 → 3 failed links **{d_ours:+.2f} points**, 1 → 3 failed links "
              f"**{d13:+.2f} points**. P1 §5.5 (as quoted in the task brief) reports B4 dropping from 90.80% to 77.25% "
              "(−13.55 points) with 3 failed links; P1's intact B4 value (Table 6) is 92.39%, so the 90.80% starting "
              "point may correspond to a smaller number of failures — it is not assigned to a row here. Absolute "
              "levels differ because the frozen baseline already differs from P1 on intact B4 (97.44% vs 92.39%, "
              "see BASELINE_FREEZE)."]
    (OUT / "B4_P1_SECTION55_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'B4_P1_SECTION55_TABLE.md'}")


if __name__ == "__main__":
    main()
