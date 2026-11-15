"""B4 path-set A/B: Yen min-hop (P1 text) vs TEAL edge-disjoint pickle.

Does not change the default path generator. LP + ECMP on identical TMs.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines import ecmp_allocate, evaluate_baseline, lp_allocate  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402

TEAL_PKL = (
    ROOT.parent
    / "reference"
    / "teal"
    / "topologies"
    / "paths"
    / "path-form"
    / "B4.json-4-paths_edge-disjoint-True_dist-metric-min-hop-dict.pkl"
)


def _norm_teal(raw: dict, k: int = 4) -> dict:
    out = {}
    for key, val in raw.items():
        if isinstance(key, tuple) and len(key) == 2:
            s, t = int(key[0]), int(key[1])
        else:
            continue
        if isinstance(val, dict):
            paths = list(val.values())
        else:
            paths = list(val)
        paths = [list(p) for p in paths]
        if not paths:
            continue
        while len(paths) < k:
            paths.append(paths[0])
        out[(s, t)] = paths[:k]
    return out


def main() -> None:
    topo = load_topology("B4")
    tms = load_traffic_matrices("B4")
    lo, hi = split_indices(len(tms), "B4")["test"]
    test = tms[lo:hi]
    yen = generate_candidate_paths(topo)
    with open(TEAL_PKL, "rb") as handle:
        teal = _norm_teal(pickle.load(handle))

    identical = sum(1 for k in yen if yen[k] == teal.get(k))
    first_path_same = sum(1 for k in yen if teal.get(k) and yen[k][0] == teal[k][0])
    n_teal_unique = []
    for k, ps in teal.items():
        n_teal_unique.append(len({tuple(p) for p in ps}))

    rec = {
        "n_pairs": len(yen),
        "identical_lists": identical,
        "first_path_identical": first_path_same,
        "teal_mean_unique_among_4_slots": float(sum(n_teal_unique) / max(len(n_teal_unique), 1)),
        "p1_text": "4 shortest paths (no edge-disjoint)",
        "decision": "KEEP Yen as default — switching to TEAL edge-disjoint would follow Teal, not P1 text",
    }

    for label, pdict in [("yen_minhop", yen), ("teal_edge_disjoint", teal)]:
        problem = build_bipartite_graph(topo, pdict)
        lp = evaluate_baseline(lp_allocate, problem, test)
        ecmp = evaluate_baseline(ecmp_allocate, problem, test)
        rec[label] = {
            "lp_sat": lp["demand_satisfaction"],
            "ecmp_sat": ecmp["demand_satisfaction"],
            "lp_mlu": lp["max_link_utilization"],
        }
        print(label, rec[label])

    out = ROOT / "results" / "logs" / "phase_f" / "path_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in rec if k not in ()}, indent=2)[:2000])
    print("wrote", out)


if __name__ == "__main__":
    main()
