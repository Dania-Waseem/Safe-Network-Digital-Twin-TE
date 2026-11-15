"""E3 link-failure experiment: link failures on B4, GEANT (50 TMs) and UsCarrier.

Failure cases: ``data/scenarios/link_failures`` (seeds 100..1000 x 0-3 failed directed
edges); paths rebuilt per case by ``failures.apply_failure`` (dead paths dropped, Yen
k=4 for pairs left without a path, disconnected pairs unservable).

TMs: B4 val (8 TMs, indices 20-27), UsCarrier val (1 TM; toy TMs are identical copies),
GEANT 50 seeded test TMs (calibrated-load study). Load levels: fixed {0.5, 1.0}, calibrated {0.7, 0.9}
(alpha from the *intact* topology, so failures add load on top of the calibrated level).

Methods on identical (failed topology, routed matrix):
  shortest, equal (ECMP over surviving candidates), LP (min-MLU path LP, SciPy HiGHS),
  frozen P1 actor (B4 freeze seed 100, actor only),
  policy actors: ``stable`` (recommended_stable.pt) and ``v2`` (policy v2 checkpoint),
  framework per learned policy: A no gate, D mean-only gate, E calibrated-uncertainty gate
  (k=1, frozen), F oracle; rejected -> frozen shortest/equal least-violating fallback.
Satisfaction = P1 definition, counting unservable demand as undelivered.

The P1 Section 5.5-comparable table (frozen P1 with its own ADMM pipeline) is produced in
``baseline_p1_reproduction/phase3_final/b4_link_failures.py``.

Usage: python -m phase3_final.link_failure_eval --topologies B4 GEANT UsCarrier
       python -m phase3_final.link_failure_eval --report-only
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from framework.general_policy import GeneralPolicy
from framework.policy_adapter import FrozenP1Policy
from safety.improvements.decision import decide_variant
from safety.improvements.twin_backend import backends_from_config
from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import od_demands

from . import routing as R
from .common import LP_SOLVER_NOTE, TOY_TM_NOTE, append_jsonl, environment, fmt_eta, out_dir, read_jsonl, write_json
from .failures import apply_failure, load_cases, routable_matrix
from .holdout_af import CALIBRATION, holdout_config

OUT = out_dir("e3_link_failures")
ROWS = OUT / "case_rows.jsonl"
POLICIES = {
    "stable": ROOT / "models" / "policy" / "improved" / "recommended_stable.pt",
    "v2": ROOT / "models" / "policy_v2" / "v2_calload_normscore" / "best.pt",
}
FROZEN_P1 = ROOT / "models" / "baseline_p1" / "b4_freeze" / "grl_te_b4_freeze_seed100.pt"
VARIANTS = {"A": "A_policy_only", "D": "D_improved_twin_mean", "E": "E_improved_twin_calibrated_uncertainty",
            "F": "F_physics_oracle"}
LEVELS = (("fixed", 0.5), ("fixed", 1.0), ("calibrated", 0.7), ("calibrated", 0.9))


def tms_for(name: str) -> tuple[str, list[int]]:
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if name == "GEANT":
        return "test", cal["geant_e3_positions_50"]
    if name == "UsCarrier":
        return "val", [0]
    return "val", list(range(8))


def load_policies() -> dict:
    pols = {}
    for pid, path in POLICIES.items():
        if not path.exists():
            print(f"policy {pid} missing at {path}; skipped", flush=True)
            continue
        p = GeneralPolicy(torch.device("cpu"))
        p.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        p.model.eval()
        pols[pid] = p
    return pols


def sat_eff(topo, routed, original, action) -> float:
    total = float(od_demands(original).sum())
    return R.satisfaction_p1(topo, routed, action) * float(od_demands(routed).sum()) / total if total > 0 else 1.0


def run(names: list[str]) -> None:
    cfg = holdout_config(None, None)
    backends = backends_from_config(cfg, torch.device("cpu"))
    pols = load_policies()
    frozen = FrozenP1Policy(FROZEN_P1)
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    done = {r["key"] for r in read_jsonl(ROWS)}
    write_json(OUT / "config.json", {"policies": {k: str(v.relative_to(ROOT)) for k, v in POLICIES.items() if k in pols},
                                     "frozen_p1": str(FROZEN_P1.relative_to(ROOT).as_posix()), "levels": LEVELS,
                                     "gate": cfg["gate"], "variants": VARIANTS, "lp_solver": LP_SOLVER_NOTE,
                                     "toy_tm_note": TOY_TM_NOTE, "environment": environment()})
    for name in names:
        base = load_topology(name)
        split, positions = tms_for(name)
        demands = load_split_demands(split, name)[0]
        cases = load_cases(name)
        todo = [(c, pos, mode, lvl) for c in cases for pos in positions for mode, lvl in LEVELS
                if f"{c.key}|pos={pos}|{mode}={lvl}" not in done]
        print(f"{name}: {len(todo)} cases to run", flush=True)
        t0 = time.time()
        k = 0
        for case in cases:
            mine = [x for x in todo if x[0] is case]
            if not mine:
                continue
            ft = apply_failure(base, case)
            topo = ft.topo
            sp, eq = R.shortest_action(topo), R.equal_action(topo)
            lp_cache: dict[int, np.ndarray | None] = {}
            for _c, pos, mode, lvl in mine:
                scale = lvl if mode == "fixed" else cal["alpha"][f"{name}/{split}/{pos}"][f"{lvl:.1f}"]
                original = demands[pos] * scale
                routed = routable_matrix(ft, original)
                if pos not in lp_cache:
                    sol = R.lp_min_mlu(topo, routable_matrix(ft, demands[pos]))
                    lp_cache[pos] = sol["action"]
                row = {"key": f"{case.key}|pos={pos}|{mode}={lvl}", "topology": name, "seed": case.seed,
                       "n_failed": case.n_failed, "failed_edges": case.edges, "position": pos, "load_mode": mode,
                       "load_level": lvl, "scale": scale, "unservable_pairs": int(ft.unservable_pairs.size),
                       "unservable_demand_frac": 1 - float(od_demands(routed).sum()) / float(od_demands(original).sum()),
                       "pairs_rerouted": ft.n_pairs_rerouted, "methods": {}}
                acts = {"shortest": sp, "equal": eq, "lp": lp_cache[pos]}
                tp = time.perf_counter()
                acts["frozen_p1"] = frozen.propose(topo, od_demands(routed))
                row["frozen_p1_ms"] = (time.perf_counter() - tp) * 1000
                for pid, pol in pols.items():
                    tp = time.perf_counter()
                    with torch.no_grad():
                        acts[f"actor_{pid}"] = pol.propose(topo, od_demands(routed))
                    row[f"actor_{pid}_ms"] = (time.perf_counter() - tp) * 1000
                for mname, a in acts.items():
                    if a is None:
                        continue
                    load = R.loads(topo, routed, a)
                    row["methods"][mname] = {"mlu": float(R.utilisation(load, topo.capacity).max()),
                                             "sat": sat_eff(topo, routed, original, a),
                                             "overload": bool(np.any(load > topo.capacity + 1e-6))}
                for pid in pols:
                    pa = acts[f"actor_{pid}"]
                    for short, vid in VARIANTS.items():
                        rec = decide_variant(vid, topo=topo, matrix=routed, policy_action=pa, backends=backends, cfg=cfg)
                        applied = {"policy": pa, "shortest": sp, "equal": eq}[rec["applied_action_source"]]
                        row["methods"][f"fw_{pid}_{short}"] = {
                            "accept": rec["acceptance"], "unsafe_accept": rec["unsafe_accept"],
                            "false_reject": rec["false_reject"], "fallback": rec["fallback_used"],
                            "applied": rec["applied_action_source"], "mlu": rec["mlu_applied"],
                            "sat": sat_eff(topo, routed, original, applied),
                            "overload": rec["capacity_violations_applied"] > 0,
                            "proposal_unsafe": not rec["oracle_feasible_proposed"],
                            "decision_ms": rec["decision_runtime_ms"]}
                append_jsonl(ROWS, row)
                k += 1
                if k % 50 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"{name} [{k}/{len(todo)}] elapsed {fmt_eta(el)} eta {fmt_eta(el / k * (len(todo) - k))}", flush=True)
            for p in list(pols.values()) + [frozen]:
                p.release(topo.name)


def summarize() -> dict:
    rows = read_jsonl(ROWS)
    groups = defaultdict(list)
    for r in rows:
        for mode in (r["load_mode"], "all"):
            groups[(r["topology"], mode, r["load_level"] if mode != "all" else "all", r["n_failed"])].append(r)
    table = []
    for (topo, mode, lvl, nf), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        methods = sorted({m for r in rs for m in r["methods"]})
        for m in methods:
            vs = [r["methods"][m] for r in rs if m in r["methods"]]
            rec = {"topology": topo, "load_mode": mode, "load_level": lvl, "n_failed": nf, "method": m, "n": len(vs),
                   "sat_mean": float(np.mean([v["sat"] for v in vs])), "mlu_mean": float(np.mean([min(v["mlu"], 1e6) for v in vs])),
                   "violation_rate": float(np.mean([v["overload"] for v in vs]))}
            if m.startswith("fw_"):
                unsafe = [v for v in vs if v["proposal_unsafe"]]
                safe = [v for v in vs if not v["proposal_unsafe"]]
                rec.update({"accept_rate": float(np.mean([v["accept"] for v in vs])),
                            "unsafe_accept_rate": float(np.mean([v["unsafe_accept"] for v in vs])),
                            "unsafe_accept_given_unsafe": float(np.mean([v["accept"] for v in unsafe])) if unsafe else None,
                            "false_reject_given_safe": float(np.mean([v["false_reject"] for v in safe])) if safe else None,
                            "fallback_rate": float(np.mean([v["fallback"] for v in vs]))})
            table.append(rec)
    unserv = defaultdict(list)
    for r in rows:
        unserv[(r["topology"], r["n_failed"])].append(r["unservable_demand_frac"])
    out = {"table": table, "n_rows": len(rows),
           "unservable": {f"{t}|{n}": float(np.mean(v)) for (t, n), v in unserv.items()}}
    write_json(OUT / "summary.json", out)
    return out


def report(summary: dict) -> None:
    t = summary["table"]
    def get(topo, mode, nf, m):
        return next((r for r in t if r["topology"] == topo and r["load_mode"] == mode and r["load_level"] == "all"
                     and r["n_failed"] == nf and r["method"] == m), None)
    f = lambda x, nd=3: "–" if x is None else f"{x:.{nd}f}"  # noqa: E731
    lines = ["# E3 link-failure experiment (B4, GEANT, UsCarrier)", "",
             "Failure cases: `data/scenarios/link_failures` (P1 §5.5 sampler; seeds 100–1000; 0–3 failed directed "
             "edges). Per case, paths through failed links are dropped and Yen k=4 is recomputed for pairs left "
             "without a path; demand of pairs disconnected by the failure is unservable and counted as undelivered.",
             "", "TMs: B4 val (8), UsCarrier val (1; identical toy copies), GEANT 50 seeded test TMs. Load: fixed "
             "{0.5, 1.0} and calibrated {0.7, 0.9} (alpha from the intact topology). "
             f"{LP_SOLVER_NOTE} LP here is the min-MLU LP over the surviving k=4 candidates.", "",
             "Framework rows (`fw_<policy>_<variant>`): A no gate, D mean-only gate, E calibrated-uncertainty gate "
             "(k=1, frozen), F physics oracle; rejected proposals fall back to shortest/equal (least-violating).", "",
             "The P1 §5.5-comparable table for B4 (frozen P1 with its own ADMM pipeline) is in "
             "`baseline_p1_reproduction/phase3_final/results/b4_link_failures/`.", ""]
    order = ["lp", "shortest", "equal", "frozen_p1", "actor_stable", "actor_v2", "fw_stable_A", "fw_stable_D",
             "fw_stable_E", "fw_stable_F", "fw_v2_A", "fw_v2_D", "fw_v2_E", "fw_v2_F"]
    for topo in ("B4", "GEANT", "UsCarrier"):
        if not any(r["topology"] == topo for r in t):
            continue
        unserv = ", ".join(f"{n}: {summary['unservable'].get(f'{topo}|{n}', 0) * 100:.2f}%" for n in range(4))
        lines += [f"## {topo} (all load levels; unservable demand share by failed links — {unserv})", "",
                  "| Method | Sat. 0 | Sat. 1 | Sat. 2 | Sat. 3 | MLU 0 | MLU 3 | Viol. 0 | Viol. 3 | "
                  "Unsafe acc. (0/1/2/3) | False rej. given safe (0/1/2/3) | Fallback 3 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
        for m in order:
            rs = [get(topo, "all", n, m) for n in range(4)]
            if rs[0] is None:
                continue
            ua = "/".join(f(r.get("unsafe_accept_rate"), 2) if r and "unsafe_accept_rate" in r else "–" for r in rs)
            fr = "/".join(f(r.get("false_reject_given_safe"), 2) if r and "accept_rate" in r else "–" for r in rs)
            lines.append(f"| {m} | " + " | ".join(f(r["sat_mean"] * 100, 2) if r else "–" for r in rs) +
                         f" | {f(rs[0]['mlu_mean'])} | {f(rs[3]['mlu_mean'] if rs[3] else None)} | "
                         f"{f(rs[0]['violation_rate'], 2)} | {f(rs[3]['violation_rate'] if rs[3] else None, 2)} | {ua} | {fr} | "
                         f"{f(rs[3].get('fallback_rate') if rs[3] else None, 2)} |")
        lines.append("")
    lines += ["Satisfaction in %, P1 definition; MLU capped at 1e6 for averaging (a loaded failed link would be inf). "
              "Violation = any link above capacity after the method's routing (for `fw_*`: after gate + fallback).", "",
              "## Artifacts", "", "- `case_rows.jsonl` (one row per failure case × TM × load), `summary.json`, `config.json`"]
    (OUT / "E3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topologies", nargs="+", default=["B4", "GEANT", "UsCarrier"], choices=["B4", "GEANT", "UsCarrier"])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        run(args.topologies)
    report(summarize())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
