"""Framework ablations on identical paired cases.

Cases: dev = B4 val (8 TMs), UsCarrier val (1 TM; identical toy copies), Cogentco val (8 TMs);
holdout = GEANT (200 seeded test TMs) and Kdl (1 TM); 8 load levels each (fixed 0.25-1.0,
calibrated 0.5-1.1). Within each (topology, load level) the TMs are processed in time order,
so "keep the previous routing" is well defined.

Configurations (all with the same proposals and the frozen gate settings):
  full     policy (recommended_stable) + improved twin + calibrated-uncertainty gate E (k=1)
           + shortest/equal least-violating fallback            [reference]
  (a) no_gate        always apply the policy proposal (variant A)
  (b) mean_gate      gate without uncertainty (variant D, k=0)
  (c) no_fallback    gate E, but a rejection keeps the previously applied routing
                     (initial routing before the first TM of a sequence = shortest path)
  (d) b4_only_policy frozen P1 actor trained on B4 only (BASELINE_FREEZE seed 100) through the
                     same twin + gate E + fallback. CONFOUNDED: differs from recommended_stable in
                     training topologies *and* model/trainer (TopoFlowNet A2C vs Phase-3 GNN).
  (e) physics_twin   twin without the learned residual: multipliers fixed to 1 and zero spread,
                     i.e. predicted load = flow-level physics. Gate E then reduces to the
                     capacity check on simulator loads.
Metrics: applied MLU, unsafe accept, false reject, P1 satisfaction, decision time.

Usage: python -m phase3_final.ablations_v2 --topologies B4 UsCarrier Cogentco GEANT
       python -m phase3_final.ablations_v2 --topologies Kdl
       python -m phase3_final.ablations_v2 --report-only
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

from framework.policy_adapter import FrozenP1Policy
from safety.improvements.decision import decide_variant
from safety.improvements.gate_logic import ImprovedGateConfig, inspect_utilisation_ucb
from safety.improvements.graph_cache import TopologyGraphCache
from safety.improvements.oracle import physics_violations
from safety.improvements.policy_backend import policy_backend_from_config
from safety.improvements.twin_backend import backends_from_config
from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import od_demands

from . import routing as R
from .common import LP_SOLVER_NOTE, TOY_TM_NOTE, append_jsonl, environment, fmt_eta, out_dir, read_jsonl, write_json
from .holdout_af import CALIBRATION, FIXED_SCALES, TARGETS, holdout_config

OUT = out_dir("ablations_v2")
ROWS = OUT / "sequence_rows.jsonl"
FROZEN_P1 = ROOT / "models" / "baseline_p1" / "b4_freeze" / "grl_te_b4_freeze_seed100.pt"
CONFIGS = ("full", "no_gate", "mean_gate", "no_fallback", "b4_only_policy", "physics_twin")


def sequences(name: str):
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if name in ("B4", "Cogentco"):
        split, positions = "val", list(range(8))
    elif name == "UsCarrier":
        split, positions = "val", [0]
    elif name == "GEANT":
        split, positions = "test", cal["geant_test_positions_200"]
    else:
        split, positions = "test", [0]
    levels = [("fixed", s) for s in FIXED_SCALES] + [("calibrated", t) for t in TARGETS]
    for mode, lvl in levels:
        scales = [lvl if mode == "fixed" else cal["alpha"][f"{name}/{split}/{p}"][f"{lvl:.1f}"] for p in positions]
        yield split, mode, lvl, list(zip(positions, scales))


def _apply(topo, matrix, action, ms):
    load = R.loads(topo, matrix, action)
    return {"mlu": float(R.utilisation(load, topo.capacity).max()), "sat": R.satisfaction_p1(topo, matrix, action),
            "overload": physics_violations(load, topo.capacity) > 0, "ms": ms}


def run(names: list[str]) -> None:
    cfg = holdout_config(None, None)
    dev = torch.device("cpu")
    policy = policy_backend_from_config(cfg, device=dev, graph_cache=TopologyGraphCache(dev))
    policy.load()
    frozen = FrozenP1Policy(FROZEN_P1)
    backends = backends_from_config(cfg, dev)
    gate_e = ImprovedGateConfig(k=float(cfg["gate"]["k"]), utilisation_limit=1.0, max_relative_std=None)
    done = {r["key"] for r in read_jsonl(ROWS)}
    write_json(OUT / "config.json", {"configs": CONFIGS, "gate": cfg["gate"], "policy": cfg["policy"],
                                     "frozen_p1": str(FROZEN_P1.relative_to(ROOT).as_posix()), "twins": cfg["twins"],
                                     "no_fallback_initial_routing": "shortest path", "lp_solver": LP_SOLVER_NOTE,
                                     "toy_tm_note": TOY_TM_NOTE, "environment": environment()})
    for name in names:
        topo = load_topology(name)
        sp = R.shortest_action(topo)
        t0 = time.time()
        for split, mode, lvl, seq in sequences(name):
            demands = load_split_demands(split, name)[0]
            prev = sp  # routing in place before the first TM (no_fallback)
            for pos, scale in seq:
                key = f"{name}|{mode}={lvl}|pos={pos}"
                m = demands[pos] * scale
                tp = time.perf_counter()
                pa = policy.propose(topo, m).action
                pol_ms = (time.perf_counter() - tp) * 1000
                tp = time.perf_counter()
                fa = frozen.propose(topo, od_demands(m))
                frz_ms = (time.perf_counter() - tp) * 1000
                recs = {v: decide_variant(v, topo=topo, matrix=m, policy_action=pa, backends=backends, cfg=cfg)
                        for v in ("A_policy_only", "D_improved_twin_mean", "E_improved_twin_calibrated_uncertainty")}
                rec_frz = decide_variant("E_improved_twin_calibrated_uncertainty", topo=topo, matrix=m,
                                         policy_action=fa, backends=backends, cfg=cfg)
                actions = {"policy": pa, "shortest": sp, "equal": R.equal_action(topo)}
                row = {"key": key, "topology": name, "load_mode": mode, "load_level": lvl, "position": pos,
                       "policy_ms": pol_ms, "frozen_ms": frz_ms, "configs": {}}
                for cname, rec in (("no_gate", recs["A_policy_only"]), ("mean_gate", recs["D_improved_twin_mean"]),
                                   ("full", recs["E_improved_twin_calibrated_uncertainty"])):
                    out = _apply(topo, m, actions[rec["applied_action_source"]], rec["decision_runtime_ms"] + pol_ms)
                    out.update({"accept": rec["acceptance"], "unsafe_accept": rec["unsafe_accept"],
                                "false_reject": rec["false_reject"], "proposal_unsafe": not rec["oracle_feasible_proposed"]})
                    row["configs"][cname] = out
                # (c) no fallback: rejection keeps the previously applied routing
                e = recs["E_improved_twin_calibrated_uncertainty"]
                applied = pa if e["acceptance"] else prev
                out = _apply(topo, m, applied, e["decision_runtime_ms"] + pol_ms)
                out.update({"accept": e["acceptance"], "unsafe_accept": e["unsafe_accept"],
                            "false_reject": e["false_reject"], "proposal_unsafe": not e["oracle_feasible_proposed"],
                            "kept_previous": not e["acceptance"]})
                row["configs"]["no_fallback"] = out
                prev = applied
                # (d) B4-only frozen P1 actor through the same gate + fallback
                frz_actions = {"policy": fa, "shortest": sp, "equal": actions["equal"]}
                out = _apply(topo, m, frz_actions[rec_frz["applied_action_source"]], rec_frz["decision_runtime_ms"] + frz_ms)
                out.update({"accept": rec_frz["acceptance"], "unsafe_accept": rec_frz["unsafe_accept"],
                            "false_reject": rec_frz["false_reject"],
                            "proposal_unsafe": not rec_frz["oracle_feasible_proposed"]})
                row["configs"]["b4_only_policy"] = out
                # (e) physics twin: mean = simulator load, std = 0
                tp = time.perf_counter()
                true_load = R.loads(topo, m, pa)
                ok = inspect_utilisation_ucb(true_load, np.zeros_like(true_load), topo.capacity, gate_e).accepted
                ph_ms = (time.perf_counter() - tp) * 1000
                fb_src = e["selected_fallback"] or recs["D_improved_twin_mean"]["selected_fallback"]
                if not ok and fb_src is None:  # E accepted but physics rejects: compute the same fallback
                    from safety.improvements.fallback import select_physics_fallback
                    fb_src = select_physics_fallback(topo, m).selected
                out = _apply(topo, m, pa if ok else actions[fb_src], ph_ms + pol_ms)
                unsafe_prop = not e["oracle_feasible_proposed"]
                out.update({"accept": bool(ok), "unsafe_accept": bool(ok and unsafe_prop),
                            "false_reject": bool((not ok) and not unsafe_prop), "proposal_unsafe": unsafe_prop,
                            "agrees_with_oracle": bool(ok) == (not unsafe_prop)})
                row["configs"]["physics_twin"] = out
                append_jsonl(ROWS, row)
            print(f"{name} {mode}={lvl} done ({fmt_eta(time.time() - t0)})", flush=True)
        policy_cache_release = getattr(frozen, "release", None)
        if policy_cache_release:
            policy_cache_release()


def summarize() -> dict:
    rows = read_jsonl(ROWS)
    groups = defaultdict(list)
    dev = {"B4", "UsCarrier", "Cogentco"}
    for r in rows:
        scope = "dev" if r["topology"] in dev else "holdout"
        for key in (r["topology"], scope):
            groups[key].append(r)
    table = []
    for scope, rs in sorted(groups.items()):
        for c in CONFIGS:
            vs = [r["configs"][c] for r in rs]
            unsafe = [v for v in vs if v["proposal_unsafe"]]
            safe = [v for v in vs if not v["proposal_unsafe"]]
            table.append({"scope": scope, "config": c, "n": len(vs),
                          "applied_mlu": float(np.mean([min(v["mlu"], 1e6) for v in vs])),
                          "applied_sat": float(np.mean([v["sat"] for v in vs])),
                          "applied_overload_rate": float(np.mean([v["overload"] for v in vs])),
                          "accept_rate": float(np.mean([v["accept"] for v in vs])),
                          "unsafe_accept_rate": float(np.mean([v["unsafe_accept"] for v in vs])),
                          "unsafe_accept_given_unsafe": float(np.mean([v["accept"] for v in unsafe])) if unsafe else None,
                          "false_reject_given_safe": float(np.mean([v["false_reject"] for v in safe])) if safe else None,
                          "n_unsafe_proposals": len(unsafe),
                          "ms_median": float(np.median([v["ms"] for v in vs])),
                          "physics_agrees_with_oracle": (float(np.mean([v["agrees_with_oracle"] for v in vs]))
                                                         if c == "physics_twin" else None)})
    out = {"table": table, "n_rows": len(rows)}
    write_json(OUT / "summary.json", out)
    return out


def report(s: dict) -> None:
    f = lambda x, nd=3: "–" if x is None else f"{x:.{nd}f}"  # noqa: E731
    labels = {"full": "Full framework (E gate + fallback)", "no_gate": "(a) No gate", "mean_gate": "(b) Gate without uncertainty",
              "no_fallback": "(c) No fallback (keep previous routing)", "b4_only_policy": "(d) B4-only policy (frozen P1)†",
              "physics_twin": "(e) Twin without learned residual (physics)"}
    lines = ["# Framework ablations", "",
             "Same paired cases for every configuration: dev = B4/UsCarrier/Cogentco val, holdout = GEANT (200) + Kdl (1), "
             "8 load levels each; TMs processed in time order within each (topology, level) so that (c) can keep the "
             f"previous routing (initial routing = shortest path). Gate settings frozen (k=1, limit 1.0). {TOY_TM_NOTE}", "",
             "† (d) is confounded: the frozen P1 actor differs from `recommended_stable` in training topology (B4 only) "
             "**and** in model/trainer (TopoFlowNet + A2C vs Phase-3 GNN with a differentiable loss). It is the only "
             "single-topology policy available without new training.", ""]
    for scope in ("dev", "holdout", "B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"):
        sub = [r for r in s["table"] if r["scope"] == scope]
        if not sub:
            continue
        lines += [f"## {scope} (n = {sub[0]['n']} cases)", "",
                  "| Configuration | Applied MLU | Applied satisfaction | Applied overload | Accept | Unsafe accept | "
                  "Unsafe acc. given unsafe | False rej. given safe | Median ms |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sub:
            lines.append(f"| {labels[r['config']]} | {f(r['applied_mlu'])} | {f(r['applied_sat'], 4)} | "
                         f"{f(r['applied_overload_rate'])} | {f(r['accept_rate'])} | {f(r['unsafe_accept_rate'])} | "
                         f"{f(r['unsafe_accept_given_unsafe'])} | {f(r['false_reject_given_safe'])} | {f(r['ms_median'], 1)} |")
        ph = next(r for r in sub if r["config"] == "physics_twin")
        lines += ["", f"Physics twin agrees with the oracle on {f(ph['physics_agrees_with_oracle'] * 100, 1)}% of cases.", ""]
    lines += ["## Artifacts", "", "- `sequence_rows.jsonl`, `summary.json`, `config.json`"]
    (OUT / "ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topologies", nargs="+", default=["B4", "UsCarrier", "Cogentco", "GEANT"],
                    choices=["B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        run(args.topologies)
    report(summarize())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
